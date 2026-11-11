"""One forced compression event, four ways to pay for it, on real anonymised sessions.

The event is explicit (a user asking to compact), so the window is fixed here instead of waiting for a
threshold: everything up to the protected tail is the material, and every arm receives exactly that
material. Only two things vary: where the instruction sits, and whether the session prefix is in cache.

    A   auxiliary model     instruction first, then the material                     cold
    B   plugin              material as messages, instruction appended last          warm session
    C   main model          instruction first, then the material                     warm session
    B'  main model          same bytes as B, in a session that never saw them        cold

B against C isolates the position of the instruction at constant model, constant content and constant
session. B against B' isolates what reusing the session buys. C against A isolates the model at constant
request shape.

The router's ``cached_tokens`` is not used as evidence anywhere: on the routes available here it moves
with the session identifier and not with the bytes. Cost is read from what the router bills, and the
cache question is answered by B minus B'.

Speed is decomposed rather than reported as one number: a ``max_tokens=1`` probe on the exact prompt gives
the prefill, the real call gives prefill plus decoding, and the difference is the decoding.

Quality is measured by containment: anchors (paths with line numbers, identifiers, numbers with units,
decision sentences) are drawn deterministically from the material, then looked for in the summary. No judge.

Every call is also reported to Langfuse through its ingestion API, so the run can be costed three ways:
what the router bills, what Langfuse recomputes from usage, and what the tokens imply at list prices.

Usage:

    BENCH_BASE=https://opencode.ai/inference/go/openai/v1 BENCH_MODEL=deepseek-v4.1-flash \
    BENCH_API_KEY=... ALBERT_KEY=... python bench/compaction_arms.py --sessions 12 --repeats 2
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize  # noqa: E402
import bench as harness  # noqa: E402
import quality_probe  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
ALBERT_BASE = os.environ.get("ALBERT_BASE", "http://127.0.0.1:8060/v1")
AUX_MODEL = os.environ.get("ALBERT_MODEL", "gemma-4-31b-it")
ALBERT_KEY = os.environ.get("ALBERT_KEY", "")
LANGFUSE_BASE = os.environ.get("LANGFUSE_BASE", "http://localhost:3000")
LANGFUSE_PUBLIC = os.environ.get("LANGFUSE_PUBLIC", "")
LANGFUSE_SECRET = os.environ.get("LANGFUSE_SECRET", "")
PROTECTED_TAIL = 6
MAX_OUTPUT = 1_500
#: USD per million tokens: (input, output, cache-read). Public list prices, 2026-11-11.
PRICES = {"main": (0.25, 0.95, 0.13), "aux": (0.20, 0.40, 0.0)}
ARMS = ("A", "B", "C", "B'")

INSTRUCTION = (
    "Summarise the conversation above for an engineer who has to resume the work. Keep every file path "
    "with its line number, every identifier, every measured number with its unit, the decisions taken and "
    "what is still open. No preamble. At most 250 words."
)

PATH = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|ts|tsx|rs|js|md|yaml|yml|json))(?::(\d+))?")
NUMBER = re.compile(r"\b(\d+(?:[.,]\d+)?)\s?(ms|s|MB|GB|KB|%|USD|tokens?)\b")
DECISION = re.compile(r"(?i)\b(décision|decision|retenu|choisi|reste à|todo|open question)\b")


def anchors(material: list[dict], count: int = 9, seed: int = 7) -> list[dict]:
    """Anchors drawn deterministically from the material, stratified by kind.

    Three kinds, because they fail differently: a path can survive as a bare file name, a number has to
    keep its unit to mean anything, and a decision can be paraphrased. Each is scored on its own terms in
    ``retained``, so a summary that drops every path is not hidden by one that keeps the numbers.
    """
    paths: list[tuple[str, str]] = []
    numbers: list[str] = []
    decisions: list[str] = []
    for message in material:
        content = message.get("content") or ""
        for match in PATH.finditer(content):
            paths.append((match.group(1), match.group(2) or ""))
        numbers.extend(match.group(0) for match in NUMBER.finditer(content))
        decisions.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", content)
                         if DECISION.search(sentence) and 20 <= len(sentence) <= 160)

    def unique(items):
        seen, kept = set(), []
        for item in items:
            key = (item[0].lower(), item[1]) if isinstance(item, tuple) else item.lower()[:60]
            if key not in seen:
                seen.add(key)
                kept.append(item)
        return kept

    paths, numbers, decisions = unique(paths), unique(numbers), unique(decisions)
    rng = random.Random(seed)
    share = max(count // 3, 1)
    picked: list[dict] = []
    if paths:
        picked += [{"kind": "path", "file": p, "line": line}
                   for p, line in rng.sample(paths, min(share, len(paths)))]
    if numbers:
        picked += [{"kind": "number", "text": n} for n in rng.sample(numbers, min(share, len(numbers)))]
    if decisions:
        picked += [{"kind": "decision", "text": d} for d in rng.sample(decisions, min(share, len(decisions)))]
    return picked


def retained(summary: str, wanted: list[dict]) -> dict[str, int]:
    """Count what survived, each kind on its own terms."""
    lowered = summary.lower()
    score = {"path": 0, "number": 0, "decision": 0}
    for anchor in wanted:
        if anchor["kind"] == "path":
            name = anchor["file"].lower()
            if name in lowered or name.rsplit("/", 1)[-1] in lowered:
                # the line number matters as much as the file: without it the anchor is unusable
                if not anchor["line"] or anchor["line"] in lowered:
                    score["path"] += 1
        elif anchor["kind"] == "number":
            if anchor["text"].lower() in lowered:
                score["number"] += 1
        elif anchor["text"].lower()[:40] in lowered:
            score["decision"] += 1
    return score


class Transport:
    """Posts an OpenAI-shaped payload and keeps what the endpoint reported."""

    def __init__(self, base: str, model: str, key: str, body_extra: dict | None = None,
                 session: str = "") -> None:
        self.base, self.model, self.key = base, model, key
        self.session = session or f"arms-{os.getpid()}"
        self.body_extra = dict(body_extra or {})
        self.calls: list[dict] = []
        self.last_text = ""

    def __call__(self, **kwargs) -> object:
        body: dict = {"model": self.model, "messages": kwargs["messages"]}
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        body.update(self.body_extra)
        payload: dict = {}
        request = urllib.request.Request(
            f"{self.base}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "User-Agent": UA, "x-opencode-session": self.session})
        transient = (400, 408, 409, 425, 429, 500, 502, 503, 504, 529)
        attempts, served = 0, 0.0
        for attempt in range(8):
            attempts = attempt + 1
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.loads(response.read())
                served = time.monotonic() - started
                break
            except urllib.error.HTTPError as exc:
                # This route answers 400 as well as 429 when it is unhappy; both are replayed.
                if exc.code not in transient or attempt == 7:
                    raise
                wait = exc.headers.get("retry-after") if exc.headers else None
                pause = max(float(wait) if wait and wait.isdigit() else 0.0, 15 * (attempt + 1))
                time.sleep(min(pause, 90) + random.uniform(0, 5))
        usage = payload.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        self.last_text = text
        self.calls.append({
            "model": self.model,
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "cached_tokens": cached,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "finish_reason": choice.get("finish_reason"),
            "service_s": round(served, 2),
            "attempts": attempts,
            "cost_router_usd": (usage.get("opencode") or {}).get("list_cost"),
        })
        return SimpleNamespace(
            model=payload.get("model"),
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens") or 0,
                                  completion_tokens=usage.get("completion_tokens") or 0,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=cached)))


def estimate(call: dict, kind: str) -> float:
    incoming, outgoing, cache = PRICES[kind]
    fresh = max(call["prompt_tokens"] - call["cached_tokens"], 0)
    return round((fresh * incoming + call["cached_tokens"] * cache
                  + call["completion_tokens"] * outgoing) / 1_000_000, 6)


def ingest(events: list[dict]) -> str:
    """Report the generations to Langfuse; the router's own figures stay the reference."""
    if not (LANGFUSE_PUBLIC and LANGFUSE_SECRET and events):
        return "desactive"
    auth = base64.b64encode(f"{LANGFUSE_PUBLIC}:{LANGFUSE_SECRET}".encode()).decode()
    request = urllib.request.Request(
        f"{LANGFUSE_BASE}/api/public/ingestion", data=json.dumps({"batch": events}).encode(),
        method="POST", headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.loads(response.read())
        errors = reply.get("errors") or []
        return f"ok ({len(events)} evenements, {len(errors)} erreurs)"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return f"echec ({exc})"


def event(call: dict, arm: str, session: str, repeat: int, kind: str) -> dict:
    start = time.time() - call["service_s"]
    return {"type": "generation-create", "id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "body": {
                "id": str(uuid.uuid4()), "traceId": f"arms-{session}-{repeat}",
                "name": f"compaction-{arm}", "model": call["model"],
                "startTime": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(start)),
                "endTime": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "usageDetails": {"input": call["prompt_tokens"], "output": call["completion_tokens"],
                                 "total": call["prompt_tokens"] + call["completion_tokens"]},
                "metadata": {"arm": arm, "kind": kind, "repeat": repeat,
                             "cached_tokens": call["cached_tokens"],
                             "cost_estimate_usd": estimate(call, "aux" if arm == "A" else "main")}}}


def calibrate(material: list[dict], pause: float) -> float | None:
    """Seconds per output token, from two output budgets on the same prompt.

    Timing one call cannot separate reading from writing: a retry, a queue, a cold prefix all land in the
    same number. Two budgets on the same bytes give a slope that none of those move. The second call also
    runs warm, which removes the prefill from the difference entirely.
    """
    session = f"cal-{uuid.uuid4().hex[:6]}"
    long_task = {"role": "user", "content": "List the integers from 1 to 400, one per line, nothing else."}
    points: list[tuple[int, float]] = []
    for budget in (64, 128, 256):
        best: tuple[int, float] | None = None
        for _ in range(2):
            transport = Transport(harness.BASE, harness.MODEL, harness.KEY,
                                  {"max_tokens": budget}, session=session)
            try:
                transport(messages=[*material, long_task], tools=harness.TOOLS)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
            call = transport.calls[-1]
            reading = (call["completion_tokens"], call["service_s"])
            # The minimum over two tries drops the queue: a backoff or a slow instance can only add time.
            best = reading if best is None else (best[0], min(best[1], reading[1]))
            time.sleep(pause)
        if best:
            points.append(best)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = numerator / denominator
    return round(slope, 5) if slope > 0 else None


def sanitize(material: list[dict]) -> list[dict]:
    """Drop the tool-call pairing this conversation cannot satisfy.

    The upstream rejects any assistant message whose ``tool_calls`` is not answered by a matching tool
    message. Anonymisation and head-and-tail capping both break pairs, and the rejection only appears when
    the instruction is placed first, which would silently disqualify one arm. Removing the dangling
    ``tool_calls`` keeps the text and makes every arm equally valid.
    """
    kept: list[dict] = []
    for index, message in enumerate(material):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            wanted = {call.get("id") for call in message["tool_calls"] if isinstance(call, dict)}
            answered = {later.get("tool_call_id") for later in material[index + 1:]
                        if later.get("role") == "tool"}
            if not wanted <= answered:
                message = {key: value for key, value in message.items() if key != "tool_calls"}
        kept.append(message)
    referenced = {call.get("id") for message in kept for call in (message.get("tool_calls") or [])
                  if isinstance(call, dict)}
    return [message for message in kept
            if message.get("role") != "tool" or message.get("tool_call_id") in referenced]


def build(arm: str, material: list[dict]) -> tuple[list[dict], str]:
    """Messages for one arm and the kind of endpoint it talks to.

    B and C carry the same content in the same message format and differ only by where the instruction
    sits. B' is B in a session that never saw the material, which is what an auxiliary call looks like
    from the router's point of view.
    """
    instruction = {"role": "user", "content": INSTRUCTION}
    if arm == "A":
        return [instruction, *material], "aux"
    if arm == "C":
        return [instruction, *material], "main"
    return [*material, instruction], "main"      # B and B'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument("--out", default=os.path.join(harness.REPO, "bench", "results"))
    parser.add_argument("--per-token", type=float, default=0.0,
                        help="secondes par token de sortie, mesurees par --self-test ou par calibrate()")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = [{"role": "user", "content": "voir src/a.py:12 et 3 ms, decision: garder la politique"}]
        picked = anchors(sample)
        print("anchors:", picked, "->", retained("On garde src/a.py:12 et 3 ms.", picked))
        return 0

    if anonymize.canary() != 0:
        print("refusing to run: the anonymizer failed its canary", file=sys.stderr)
        return 2

    sessions = anonymize.load_sessions(args.sessions, args.max_tokens)
    print(f"{len(sessions)} sessions anonymisees, "
          f"{sum(s['approx_tokens'] for s in sessions):,} tokens de transcript")

    if not args.per_token:
        sample = sanitize(sessions[0]["messages"][:-PROTECTED_TAIL])
        args.per_token = calibrate(sample, args.pause) or 0.0
        print(f"calibration: {args.per_token} s par token de sortie")

    window_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict = {"date": window_start,
                    "model_main": harness.MODEL, "model_aux": AUX_MODEL,
                    "prices": PRICES, "sessions": []}
    events: list[dict] = []
    kicked = 0.0

    for repeat in range(1, args.repeats + 1):
        for session in sessions:
            material = sanitize(session["messages"][:-PROTECTED_TAIL])
            if len(material) < 4:
                continue
            wanted = anchors(material)
            record: dict = {"session": session["session"], "repeat": repeat,
                            "messages": session["message_count"], "anchors": len(wanted), "arms": {}}

            warm_session = f"warm-{session['session']}-r{repeat}"
            fresh_session = f"cold-{session['session']}-r{repeat}-{uuid.uuid4().hex[:6]}"
            aux_session = f"aux-{session['session']}-r{repeat}-{uuid.uuid4().hex[:6]}"

            # Warm the session the way a live turn does, then B and C reuse it and B' does not.
            warm = Transport(harness.BASE, harness.MODEL, harness.KEY, {"max_tokens": 1},
                             session=warm_session)
            try:
                warm(messages=material, tools=harness.TOOLS)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                print(f"r{repeat} {session['session']} echauffement echoue: {exc}")
            time.sleep(args.pause)

            for arm in ARMS:
                record["arms"][arm] = None
                try:
                    messages, kind = build(arm, material)
                    base = ALBERT_BASE if kind == "aux" else harness.BASE
                    model = AUX_MODEL if kind == "aux" else harness.MODEL
                    key = ALBERT_KEY if kind == "aux" else harness.KEY
                    chosen = {"A": aux_session, "B": warm_session, "C": warm_session,
                              "B'": fresh_session}[arm]

                    call_model = {"messages": messages}
                    if kind == "main":
                        call_model["tools"] = harness.TOOLS

                    full = Transport(base, model, key, {"max_tokens": MAX_OUTPUT}, session=chosen)
                    full(**call_model)
                    time.sleep(args.pause)

                    reading, answer = full.calls[-1], full.last_text
                    kept = retained(answer, wanted)
                    questions = quality_probe.build(material, wanted)

                    def grade(model=None, messages=None, max_tokens=24, temperature=0.0):
                        aux = Transport(ALBERT_BASE, AUX_MODEL, ALBERT_KEY,
                                        {"max_tokens": max_tokens, "temperature": temperature},
                                        session=f"grade-{session['session']}")
                        aux(messages=messages)
                        return SimpleNamespace(choices=[SimpleNamespace(
                            message=SimpleNamespace(content=aux.last_text))])

                    graded = quality_probe.ask(answer, questions, grade, AUX_MODEL)
                    prefill = (round(reading["service_s"] - reading["completion_tokens"] * args.per_token, 2)
                               if args.per_token else None)
                    entry = {
                        "prompt_tokens": reading["prompt_tokens"],
                        "cached_tokens": reading["cached_tokens"],
                        "hit": round(reading["cached_tokens"] / reading["prompt_tokens"], 4)
                        if reading["prompt_tokens"] else 0.0,
                        "completion_tokens": reading["completion_tokens"],
                        "finish_reason": reading["finish_reason"],
                        "service_s": reading["service_s"],
                        "attempts": reading["attempts"],
                        "prefill_est_s": prefill,
                        "cost_router_usd": reading["cost_router_usd"],
                        "cost_estimate_usd": estimate(reading, kind),
                        "anchors_total": len(wanted),
                        "anchors_kept": sum(kept.values()),
                        "anchors_by_kind": kept,
                        "mcq_asked": graded["asked"],
                        "mcq_correct": graded["correct"],
                    }
                    record["arms"][arm] = entry
                    events.append(event(reading, arm, session["session"], repeat, "full"))
                    print(f"r{repeat} {session['session']} {arm:2s} prompt={entry['prompt_tokens']:6d} "
                          f"hit={entry['hit']:.2f} out={entry['completion_tokens']:4d} "
                          f"anc={entry['anchors_kept']}/{entry['anchors_total']} "
                          f"qcm={entry['mcq_correct']}/{entry['mcq_asked']} "
                          f"service={entry['service_s']:5.1f}s essais={entry['attempts']} "
                          f"prefill~{entry['prefill_est_s']} facture={entry['cost_router_usd']}")
                    kicked += entry["cost_estimate_usd"]
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                    print(f"r{repeat} {session['session']} {arm:2s} echec apres reprises: {exc}")
            report["sessions"].append(record)

    report["window"] = [window_start, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())]
    report["langfuse"] = ingest(events)
    report["totals"] = {
        arm: {"runs": len(done),
              "cost_estimate_usd": round(sum(r["cost_estimate_usd"] for r in done), 6),
              "cost_router_usd": round(sum(r["cost_router_usd"] or 0.0 for r in done), 6),
              "anchors_kept": sum(r["anchors_kept"] for r in done),
              "anchors_by_kind": {k: sum(r["anchors_by_kind"][k] for r in done)
                                  for k in ("path", "number", "decision")},
              "anchors_total": sum(r["anchors_total"] for r in done),
              "service_s": round(sum(r["service_s"] for r in done) / len(done), 2),
              "prefill_est_s": round(sum(r["prefill_est_s"] or 0.0 for r in done) / len(done), 2),
              "mcq_asked": sum(r["mcq_asked"] for r in done),
              "mcq_correct": sum(r["mcq_correct"] for r in done)}
        for arm in ARMS
        for done in [[s["arms"][arm] for s in report["sessions"] if s["arms"].get(arm)]]}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("arms-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nlangfuse: {report['langfuse']}")
    print(f"cout estime total: {kicked:.5f} USD")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

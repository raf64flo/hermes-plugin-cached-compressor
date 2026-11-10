"""Fact recall after compaction: does each engine keep what must not be lost?

The bench harness scores summary quality with a rubric judge. This probe measures something the
judge cannot: whether specific facts survive the compaction and can still be answered from the
compacted transcript alone.

Three families, because they do not measure the same thing:

- ``tail``: facts in the protected tail. Both engines must keep them; a miss is a bug, not a
  quality difference.
- ``middle``: facts in ordinary conversation, inside the summarized window. This is the family
  that measures the summarizer.
- ``pruned``: facts inside a large tool output, which the parent prunes by design. Both engines
  lose them; the family is a control, and a hit there means the summary happened to quote the log.

The reader is the session model, shown only the compacted transcript, asked one JSON question per
fact. Output: a JSON record with per-family recall and per-arm cost. No transcript content is
written anywhere.

    BENCH_API_KEY=... python bench/fact_recall.py --tokens 20000
    python bench/fact_recall.py --self-test          # no network
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from typing import Any

REPO = os.environ.get("PLUGIN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# bench.py reads the key at import time; the self-test must run without one (and without network).
SELF_TEST = "--self-test" in sys.argv
if SELF_TEST:
    os.environ.setdefault("BENCH_API_KEY", "offline-self-test")

#: Kept local so ``--self-test`` runs without importing the Hermes package at all (CI has no
#: ``run_agent`` dependencies). The live path imports the real harness below.
CHARS_PER_TOKEN = 3.6
MODULES = ("alignment", "summary", "state", "cacheability", "plugin", "transport")
harness: Any = None
if not SELF_TEST:
    _spec = importlib.util.spec_from_file_location(
        "bench_harness", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py"))
    assert _spec and _spec.loader
    harness = importlib.util.module_from_spec(_spec)
    sys.modules["bench_harness"] = harness
    _spec.loader.exec_module(harness)

#: (family, id, token the answer must contain, question put to the reader)
FACTS = (
    ("tail", "T-1", "ZX-4711", "the batch identifier given in the final instruction"),
    ("tail", "T-2", "QA-2260", "the ticket number given in the final instruction"),
    ("tail", "T-3", "17 mm", "the spacer thickness given in the final instruction"),
    ("middle", "M-1", "QTX-88", "the reagent lot agreed early in the session"),
    ("middle", "M-2", "4 C", "the storage temperature agreed early in the session"),
    ("middle", "M-3", "2026-12-04", "the freeze date decided early in the session"),
    ("pruned", "P-1", "PLT-90312", "the plate identifier printed in the machine log"),
    ("pruned", "P-2", "0.42", "the purity ratio printed in the machine log"),
    ("pruned", "P-3", "run 7", "the run number printed in the machine log"),
)

FILLER = ("    record = _enrich(payload, key='field')\n"
          "    return _commit(record, retries=3)\n") * 12


def build_transcript(target_tokens: int, salt: str = "") -> list[dict]:
    """Deterministic transcript with the three fact families planted."""
    budget = int(target_tokens * CHARS_PER_TOKEN)
    messages: list[dict] = []

    def add(role: str, content: str, **extra: object) -> None:
        message: dict = {"role": role, "content": content}
        message.update(extra)
        messages.append(message)

    def turn(index: int) -> None:
        module = MODULES[index % len(MODULES)]
        add("user", f"Task {index}: audit src/{module}.py for unbounded retries.{salt}")
        add("assistant", "", tool_calls=[{
            "id": f"call_{index}", "type": "function",
            "function": {"name": "read_file", "arguments": '{ "path" : "src/%s.py" }' % module}}])
        add("tool", f"# src/{module}.py\n" + FILLER, tool_call_id=f"call_{index}", name="read_file")
        add("assistant", f"src/{module}.py:14 retries is hard-coded to 3. "
                         f"Decision: bound it and re-raise. Next: patch both sites.")

    add("user", "Start the audit; I will hand you the machine log first.")
    add("assistant", "Ready. Reading the instrument log.")
    # The prunable family: inside a long tool output the parent replaces on compaction.
    add("tool", "instrument.log\n" + ("calibration sweep " + "x" * 60 + "\n") * 120 +
        "plate PLT-90312 | purity 0.42 | run 7\n" + ("readback ok\n") * 120,
        tool_call_id="call_log", name="read_file")

    index = 0
    middle_done = 0
    while sum(len(m.get("content") or "") for m in messages) < budget * 0.7:
        turn(index)
        index += 1
        # Plant the middle facts one per turn, early, so they sit in the summarized window.
        if middle_done < 3 and index % 3 == 0:
            fact_id = FACTS[3 + middle_done][1]
            if fact_id == "M-1":
                add("user", "Note for the protocol: use reagent lot QTX-88 for every plate.")
            elif fact_id == "M-2":
                add("assistant", "Storing the reagent at 4 C, as the vendor sheet says.")
            else:
                add("user", "The freeze date is 2026-12-04, put it in the plan.")
            middle_done += 1

    while sum(len(m.get("content") or "") for m in messages) < budget * 0.9:
        turn(index)
        index += 1

    # The protected tail: the last instruction carries three facts.
    add("user", "Final instruction: batch ZX-4711, ticket QA-2260, spacer thickness 17 mm. "
                "Write them down before we compact.")
    add("assistant", "Noted: ZX-4711, QA-2260, 17 mm.")
    add("user", "Now compact the session and keep those details.")
    add("assistant", "Compacting now.")
    add("user", "One last check: the audit is paused until the summary lands.")
    add("assistant", "Understood, waiting for the summary.")
    add("user", "Go ahead.")
    add("assistant", "Proceeding.")
    return messages


def _reader_prompt(compacted: list[dict], facts: tuple) -> str:
    transcript = "\n".join(
        f"[{m.get('role')}] {m.get('content') or ''}"
        for m in compacted if (m.get("content") or "").strip())
    questions = "\n".join(f"- {fact_id}: {question}" for _, fact_id, _, question in facts)
    return (f"Compacted agent transcript:\n\n{transcript}\n\n"
            f"Questions:\n{questions}\n\n"
            "Answer with JSON only, no prose: {\"answers\": {\"T-1\": \"...\", ...}}. "
            "Use null when the transcript does not contain the answer.")


def ask_reader(model: str, compacted: list[dict], facts: tuple) -> dict[str, str | None]:
    prompt = _reader_prompt(compacted, facts)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0}
    request = urllib.request.Request(
        harness.BASE + "/chat/completions", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {harness.KEY}", "Content-Type": "application/json",
                 "User-Agent": harness.USER_AGENT, "x-opencode-session": harness.SESSION})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    answers = data.get("answers") if isinstance(data, dict) else None
    return answers if isinstance(answers, dict) else {}


def score(answers: dict, facts: tuple) -> dict:
    per_family: dict[str, dict] = {}
    details: list[dict] = []
    for family, fact_id, token, _ in facts:
        answer = answers.get(fact_id)
        hit = isinstance(answer, str) and token.casefold() in answer.casefold()
        per_family.setdefault(family, {"hits": 0, "total": 0})
        per_family[family]["total"] += 1
        per_family[family]["hits"] += int(hit)
        details.append({"id": fact_id, "family": family, "hit": hit})
    return {"families": per_family, "details": details}


def _compacted_of(kind: str, messages: list[dict], model: str) -> tuple[list[dict], dict]:
    """Run one engine's real compress() and keep the transcript it returns."""
    import agent.context_compressor as context_module

    transport = harness.Transport(model=model)
    original_context = context_module.call_llm
    original_plugin = harness.cc.call_llm
    context_module.call_llm = transport
    harness.cc.call_llm = transport
    engine = harness._make_engine(kind, model)
    try:
        started = time.monotonic()
        compacted = engine.compress(list(messages), force=True)
        elapsed = time.monotonic() - started
    finally:
        context_module.call_llm = original_context
        harness.cc.call_llm = original_plugin
    prompt = sum(call["prompt_tokens"] for call in transport.calls)
    cached = sum(call["cached_tokens"] for call in transport.calls)
    costs = [call["cost_usd"] for call in transport.calls]
    stats = {"calls": len(transport.calls), "prompt_tokens": prompt, "cached_tokens": cached,
             "cache_hit": round(cached / prompt, 4) if prompt else None,
             "cost_usd": round(sum(costs), 6) if all(c is not None for c in costs) else None,
             "latency_s": round(elapsed, 1), "messages_before": len(messages),
             "messages_after": len(compacted)}
    return compacted, stats


def self_test() -> int:
    """No network: the transcript plants every fact, scoring separates hits from misses."""
    messages = build_transcript(20_000, salt="self-test")
    joined = "\n".join(m.get("content") or "" for m in messages)
    assert all(token in joined for _, _, token, _ in FACTS), "a fact is missing from the transcript"
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert any("PLT-90312" in (m.get("content") or "") for m in tool_messages), "P facts not in a tool output"
    tail = messages[-8:]
    assert any("ZX-4711" in (m.get("content") or "") for m in tail), "T facts not in the protected tail"

    answers = {"T-1": "batch ZX-4711", "T-2": None, "M-1": "QTX-99",
               "M-2": "4 c", "M-3": None, "P-1": "PLT-90312", "P-2": "0.42", "P-3": None}
    result = score(answers, FACTS)
    assert result["families"]["tail"] == {"hits": 1, "total": 3}, result["families"]["tail"]
    assert result["families"]["middle"] == {"hits": 1, "total": 3}, result["families"]["middle"]
    assert result["families"]["pruned"] == {"hits": 2, "total": 3}, result["families"]["pruned"]
    print(f"self-test ok: {len(messages)} messages, "
          f"{len(joined) // CHARS_PER_TOKEN} tokens, scoring as expected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=20_000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    messages = build_transcript(args.tokens)
    agent = harness.AIAgent(api_key=harness.KEY, base_url=harness.BASE, provider=harness.PROVIDER,
                            model=harness.MODEL, quiet_mode=True, skip_context_files=True,
                            skip_memory=True)
    agent._cached_system_prompt = harness.SYSTEM
    agent.tools = harness.TOOLS
    wire = harness._build_api_kwargs_for_mode(
        agent, harness._iteration_summary_api_messages(agent, messages))
    warming = harness.warm(harness.Transport(), wire)
    print(f"prefix warm: {warming['hit']:.1%} after {len(warming['attempts'])} attempt(s)")
    harness.cc.state.record_payload(
        request=wire, request_messages=wire["messages"], model=harness.MODEL,
        provider=harness.PROVIDER, base_url=harness.BASE, api_mode="chat_completions",
        conversation_history=messages, session_id="fact-recall")

    report = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "model": harness.MODEL, "target_tokens": args.tokens,
              "facts": [{"id": f[1], "family": f[0]} for f in FACTS], "arms": {}}
    for kind in ("builtin", "plugin"):
        compacted, stats = _compacted_of(kind, messages, harness.MODEL)
        answers = ask_reader(harness.MODEL, compacted, FACTS)
        result = score(answers, FACTS)
        report["arms"][kind] = {**stats, **result}
        families = " ".join(f"{name}={value['hits']}/{value['total']}"
                            for name, value in result["families"].items())
        print(f"{kind:8s} {families}  messages {stats['messages_before']}->{stats['messages_after']} "
              f"cost={stats['cost_usd']}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("fact-recall-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

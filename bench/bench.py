"""Head-to-head benchmark: built-in compressor against cached-compressor.

Both arms run their real ``compress()`` over the same transcript, with the same model and the same
transport, so the only variable left is the summary payload each engine decides to send:

- built-in: a serialized copy of the window, billed as fresh input;
- cached-compressor: the session prefix the main loop already sent, billed as a cache read.

The transport posts exactly the messages an engine builds and reads ``usage`` back, including the
provider's own billed cost, so no price table is invented here. Routing (the auxiliary model the
built-in path uses in a real deployment) is a configuration choice measured separately; here both
arms call the same model, which isolates the cache.

A rubric judge then scores both summaries blind on the same criteria.

Usage:

    BENCH_API_KEY=... python bench/bench.py --tokens 20000 --tokens 100000

Output: a table on stdout and a JSON record under ``bench/results/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.environ.get("HERMES_SOURCE", os.path.expanduser("~/.hermes/hermes-agent")))
import hermes_bootstrap  # noqa: F401

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="bench-home-"))

from agent.chat_completion_helpers import (  # noqa: E402
    _build_api_kwargs_for_mode, _iteration_summary_api_messages)
from agent.context_compressor import ContextCompressor  # noqa: E402
from run_agent import AIAgent  # noqa: E402

REPO = os.environ.get("PLUGIN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "cc_bench", f"{REPO}/__init__.py", submodule_search_locations=[REPO])
cc = importlib.util.module_from_spec(_spec)
sys.modules["cc_bench"] = cc
_spec.loader.exec_module(cc)

BASE = os.environ.get("BENCH_BASE", "https://opencode.ai/inference/go/openai/v1")
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash")
PROVIDER = os.environ.get("BENCH_PROVIDER", "opencode-go")
KEY = os.environ.get("BENCH_API_KEY") or os.environ["HERMES_API_KEY"]
SESSION = f"bench-{os.getpid()}"
#: The router asks for a browser user agent and a session id, and reports the billed cost.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/131.0 Safari/537.36")
CHARS_PER_TOKEN = 3.6
WARM_ATTEMPTS = 4
WARM_SLEEP_S = 8
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", MODEL)
#: Varying the salt gives each repeat a transcript the provider cache has never seen, for both arms.
SALT = os.environ.get("BENCH_SALT", "")

SYSTEM = (
    "You are Hermes, an agentic assistant operating in a terminal. "
    "House conventions: minimal diffs, evidence with path:line, no secrets in output. "
    "Tools: read_file, terminal. "
) * 45

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file. " * 30,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command. " * 30,
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]

MODULES = ("alignment", "summary", "state", "cacheability", "plugin", "transport")

RUBRIC = """You score two summaries of the same agent transcript. Neither is labelled.

Criteria, 1 to 5 each:
- fidelity: no invented content, nothing attributed that is not in the transcript;
- decisions: the decisions taken are stated;
- open_items: what is still open is stated;
- evidence: file paths and identifiers are preserved (e.g. src/state.py:22);
- usefulness: an engineer could resume the work from this summary alone.

Answer with JSON only, no prose:
{"A": {"fidelity": n, "decisions": n, "open_items": n, "evidence": n, "usefulness": n},
 "B": {"fidelity": n, "decisions": n, "open_items": n, "evidence": n, "usefulness": n},
 "better": "A" or "B" or "tie"}"""


def build_history(target_tokens: int) -> list[dict]:
    """Deterministic agent transcript: tool calls, file excerpts, decisions with evidence."""
    budget = int(target_tokens * CHARS_PER_TOKEN)
    body = "".join(
        f"def handler_{line}(payload):\n"
        f"    record = _enrich(payload, key='field_{line}')\n"
        f"    return _commit(record, retries=3)\n"
        for line in range(30))
    messages: list[dict] = []
    index = 0
    while sum(len(m.get("content") or "") for m in messages) < budget:
        module = MODULES[index % len(MODULES)]
        messages.append({"role": "user",
                         "content": f"Task {index}: audit src/{module}.py for unbounded retries, "
                                    f"report with path:line evidence.{(' Run ' + SALT) if SALT else ''}"})
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{index}", "type": "function",
            "function": {"name": "read_file",
                         "arguments": '{ "path" : "src/%s.py" }' % module}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{index}", "name": "read_file",
                         "content": f"# src/{module}.py\n" + body})
        messages.append({"role": "assistant",
                         "content": f"src/{module}.py:14 retries is hard-coded to 3, "
                                    f"src/{module}.py:22 does not propagate the failure. "
                                    f"Decision: add a bound and re-raise. Next: patch both sites."})
        index += 1
    messages.append({"role": "user",
                     "content": "Before you continue, summarize what we decided and what is still "
                                "open, across every module audited so far."})
    return messages


def _cost(usage: dict) -> float | None:
    """The router bills in ``usage.opencode.list_cost``; plain OpenAI-compatible APIs use ``cost``."""
    router = usage.get("opencode") or {}
    value = router.get("list_cost")
    if value is None:
        value = usage.get("cost")
    return float(value) if value is not None else None


class Transport:
    """Posts exactly the messages an engine builds and returns an OpenAI-shaped response."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or MODEL
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> object:
        started = time.monotonic()
        body: dict = {"model": self.model, "messages": kwargs["messages"]}
        for key in ("tools", "tool_choice", "temperature"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]
        request = urllib.request.Request(
            BASE + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                     "User-Agent": USER_AGENT, "x-opencode-session": SESSION})
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
        usage = payload.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self.calls.append({
            "messages": len(kwargs["messages"]),
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "cached_tokens": cached,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "cost_usd": _cost(usage),
            "duration_s": round(time.monotonic() - started, 2),
        })
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        return SimpleNamespace(
            model=payload.get("model"),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=message.get("content") or "",
                                        reasoning_content=message.get("reasoning_content"),
                                        reasoning=message.get("reasoning")),
                finish_reason=(payload.get("choices") or [{}])[0].get("finish_reason"))],
            usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens") or 0,
                                  completion_tokens=usage.get("completion_tokens") or 0,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=cached)))


def _make_engine(kind: str) -> object:
    cls = cc.CachedCompressor if kind == "plugin" else ContextCompressor
    return cls(model=MODEL, provider=PROVIDER, api_key=KEY, base_url=BASE,
               api_mode="chat_completions", config_context_length=200_000,
               quiet_mode=True, protect_last_n=20)


def _summary_of(compacted: list[dict]) -> str:
    for message in compacted:
        content = message.get("content") or ""
        if "Historical Task Snapshot" in content or "PRIOR CONTEXT" in content:
            return content
    return ""


def run_arm(kind: str, messages: list[dict]) -> dict:
    import agent.context_compressor as context_module

    transport = Transport()
    original_context = context_module.call_llm
    original_plugin = cc.call_llm
    # Both entry points resolve ``call_llm`` from their own module globals.
    context_module.call_llm = transport
    cc.call_llm = transport
    engine = _make_engine(kind)
    try:
        started = time.monotonic()
        compacted = engine.compress(list(messages), force=True)
        elapsed = time.monotonic() - started
    finally:
        context_module.call_llm = original_context
        cc.call_llm = original_plugin

    prompt = sum(call["prompt_tokens"] for call in transport.calls)
    cached = sum(call["cached_tokens"] for call in transport.calls)
    costs = [call["cost_usd"] for call in transport.calls]
    return {
        "arm": kind,
        "calls": len(transport.calls),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit": round(cached / prompt, 4) if prompt else None,
        "completion_tokens": sum(call["completion_tokens"] for call in transport.calls),
        "cost_usd": round(sum(costs), 6) if all(c is not None for c in costs) else None,
        "latency_s": round(elapsed, 1),
        "messages_before": len(messages),
        "messages_after": len(compacted),
        "summary": _summary_of(compacted),
        "call_details": transport.calls,
    }


def warm(transport: Transport, wire: dict) -> dict:
    """Send the session prefix until the provider reports it as cached, as a session would."""
    attempts: list[dict] = []
    for _ in range(WARM_ATTEMPTS):
        transport(messages=wire["messages"], tools=wire.get("tools"))
        detail = transport.calls[-1]
        attempts.append({key: detail[key] for key in ("prompt_tokens", "cached_tokens")})
        hit = (detail["cached_tokens"] / detail["prompt_tokens"]) if detail["prompt_tokens"] else 0.0
        if hit >= 0.9:
            break
        time.sleep(WARM_SLEEP_S)
    last = attempts[-1]
    return {"attempts": attempts,
            "hit": round(last["cached_tokens"] / last["prompt_tokens"], 4)
            if last["prompt_tokens"] else 0.0}


def judge(builtin: dict, plugin: dict) -> dict:
    """Score both summaries blind, once in each label order, and average the two votes."""
    votes = [_judge_once(builtin, plugin, flip) for flip in (False, True)]
    totals = {}
    for arm in ("builtin", "plugin"):
        per_vote = []
        for vote in votes:
            label = next((name for name in ("A", "B") if vote["labels"][name] == arm), None)
            if label and isinstance(vote.get(label), dict):
                per_vote.append(vote[label])
        totals[arm] = ({key: round(sum(score[key] for score in per_vote) / len(per_vote), 1)
                        for key in per_vote[0]} if per_vote else None)
        if totals[arm]:
            totals[arm]["total"] = round(sum(totals[arm].values()), 1)
    order_votes = [vote.get("better_arm") for vote in votes]
    winner = "tie" if len(set(order_votes)) > 1 else order_votes[0]
    return {"votes": votes, "totals": totals, "better_arm": winner,
            "label_order_agrees": len(set(order_votes)) == 1}


def _judge_once(builtin: dict, plugin: dict, flip: bool) -> dict:
    first, second = (plugin, builtin) if flip else (builtin, plugin)
    labels = {"A": "plugin" if flip else "builtin", "B": "builtin" if flip else "plugin"}
    content = f"{RUBRIC}\n\nSummary A:\n{first['summary']}\n\nSummary B:\n{second['summary']}"
    transport = Transport(model=JUDGE_MODEL)
    response = transport(messages=[{"role": "user", "content": content}], temperature=0.0)
    text = response.choices[0].message.content or ""
    start, end = text.find("{"), text.rfind("}")
    scores = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    better = scores.pop("better", None)
    if better in labels:
        scores["better_arm"] = labels[better]
    else:
        scores["better_arm"] = None
    scores["labels"] = labels
    scores["cost_usd"] = transport.calls[-1]["cost_usd"]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, action="append", required=True,
                        help="transcript size in tokens, repeatable")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    args = parser.parse_args()

    agent = AIAgent(api_key=KEY, base_url=BASE, provider=PROVIDER, model=MODEL,
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = SYSTEM
    agent.tools = TOOLS

    report = {"model": MODEL, "provider": PROVIDER, "runs": []}
    for target in args.tokens:
        messages = build_history(target)
        wire = _build_api_kwargs_for_mode(agent, _iteration_summary_api_messages(agent, messages))
        warming = warm(Transport(), wire)
        print(f"\n=== transcript ~{target} tokens === (prefix warm: {warming['hit']:.1%} "
              f"after {len(warming['attempts'])} attempt(s))")
        if warming["hit"] < 0.9:
            print("  warning: the prefix never entered the provider cache, the prefix arm would be")
            print("           measured without a cache: the run is not a valid comparison")
        cc.state.record_payload(
            request=wire, request_messages=wire["messages"], model=MODEL, provider=PROVIDER,
            base_url=BASE, api_mode="chat_completions", conversation_history=messages,
            session_id="bench")
        for _ in range(args.repeats):
            builtin = run_arm("builtin", messages)
            plugin = run_arm("plugin", messages)
            run = {"target_tokens": target, "warm": warming,
                   "arms": {"builtin": builtin, "plugin": plugin}}
            if not args.no_judge:
                try:
                    run["judge"] = judge(builtin, plugin)
                except Exception as error:  # the judge must never discard the measurement
                    run["judge_error"] = repr(error)
            report["runs"].append(run)
            print(f"{'arm':8s} {'calls':>5s} {'prompt':>8s} {'cached':>8s} {'hit':>7s} "
                  f"{'out':>5s} {'cost_usd':>9s} {'latency':>8s} {'msgs':>9s}")
            for name in ("builtin", "plugin"):
                arm = run["arms"][name]
                print(f"{name:8s} {arm['calls']:5d} {arm['prompt_tokens']:8d} "
                      f"{arm['cached_tokens']:8d} {str(arm['cache_hit']):>7s} "
                      f"{arm['completion_tokens']:5d} {str(arm['cost_usd']):>9s} "
                      f"{arm['latency_s']:7.1f}s {arm['messages_before']:4d}->"
                      f"{arm['messages_after']}")
            if run.get("judge"):
                verdict = run["judge"]
                totals = verdict.get("totals") or {}
                builtin_total = (totals.get("builtin") or {}).get("total")
                plugin_total = (totals.get("plugin") or {}).get("total")
                print(f"judge: builtin={builtin_total}/25 plugin={plugin_total}/25 "
                      f"-> {verdict.get('better_arm')} "
                      f"(label order stable: {verdict.get('label_order_agrees')})")
            elif run.get("judge_error"):
                print("judge failed:", run["judge_error"])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("bench-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

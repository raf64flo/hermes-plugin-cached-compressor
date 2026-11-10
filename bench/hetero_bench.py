"""Head-to-head on real, anonymized local transcripts. Publishes counters only.

The synthetic benchmark uses one transcript shape. Real sessions do not: short and long, tool-heavy
or not, with and without compaction history. This probe runs both engines' real ``compress()`` over
a stratified sample of the local Hermes session store, after masking (``bench/anonymize.py``), and
records cost, cache hit, latency and size ratios.

What leaves this process: aggregate counters. What never leaves it: message content, identifiers,
paths. The probe asserts that property on its own report before writing it: every 30-character
window of every transcript is checked against the serialized report.

    BENCH_API_KEY=... python bench/hetero_bench.py --sessions 4
    python bench/hetero_bench.py --self-test        # no network, no real data
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PLUGIN_ROOT", os.path.dirname(HERE))
sys.path.insert(0, HERE)

if "--self-test" in sys.argv:
    os.environ.setdefault("BENCH_API_KEY", "offline-self-test")
    SELF_TEST = True
else:
    SELF_TEST = False

#: The self-test runs without the Hermes package: CI has no ``run_agent`` dependencies, and the
#: leak guard must be testable on its own.
harness: Any = None
if not SELF_TEST:
    _spec = importlib.util.spec_from_file_location("bench_harness", os.path.join(HERE, "bench.py"))
    assert _spec and _spec.loader
    harness = importlib.util.module_from_spec(_spec)
    sys.modules["bench_harness"] = harness
    _spec.loader.exec_module(harness)

_aspec = importlib.util.spec_from_file_location("bench_anonymize", os.path.join(HERE, "anonymize.py"))
assert _aspec and _aspec.loader
anonymize = importlib.util.module_from_spec(_aspec)
sys.modules["bench_anonymize"] = anonymize
_aspec.loader.exec_module(anonymize)

WARM_ATTEMPTS = 4
WARM_SLEEP_S = 5


def warm(transport, wire: dict) -> dict:
    """Send the prefix until it is served from cache twice in a row (bounded)."""
    attempts: list[dict] = []
    streak = 0
    for _ in range(WARM_ATTEMPTS):
        transport(messages=wire["messages"], tools=wire.get("tools"))
        detail = transport.calls[-1]
        hit = (detail["cached_tokens"] / detail["prompt_tokens"]) if detail["prompt_tokens"] else 0.0
        attempts.append({"prompt_tokens": detail["prompt_tokens"],
                         "cached_tokens": detail["cached_tokens"]})
        streak = streak + 1 if hit >= 0.9 else 0
        if streak >= 2:
            break
        time.sleep(WARM_SLEEP_S)
    last = attempts[-1] if attempts else {"prompt_tokens": 0, "cached_tokens": 0}
    return {"attempts": len(attempts),
            "hit": round(last["cached_tokens"] / last["prompt_tokens"], 4)
            if last["prompt_tokens"] else 0.0}


def _leak_windows(sessions: list[dict]) -> list[str]:
    windows: list[str] = []
    for session in sessions:
        for message in session["messages"]:
            content = message.get("content") or ""
            for start in range(0, max(1, len(content) - 30), 120):
                window = content[start:start + 30]
                if len(window) == 30 and window.strip():
                    windows.append(window)
    return windows


def self_test() -> int:
    """No network, no real data: the leak guard catches a deliberate leak and passes a clean report."""
    sessions = [{"session": "self",
                 "messages": [{"role": "user", "content": "calibration sweep " + "x" * 60},
                              {"role": "assistant", "content": "y" * 60}]}]
    clean = json.dumps({"sessions": [{"session": s["session"], "message_count": len(s["messages"])}
                                     for s in sessions]})
    assert not [w for w in _leak_windows(sessions) if w in clean], "clean report flagged as leaking"
    dirty = json.dumps({"sessions": [{"leak": sessions[0]["messages"][0]["content"][:50]}]})
    assert [w for w in _leak_windows(sessions) if w in dirty], "the leak guard missed a real leak"
    print(f"self-test ok: leak guard flags a planted leak, clears a clean report "
          f"({len(sessions)} session(s), {len(_leak_windows(sessions))} windows)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    parser.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    if anonymize.canary() != 0:
        print("refusing to run: the anonymizer failed its canary", file=sys.stderr)
        return 2

    sessions = anonymize.load_sessions(args.sessions, args.max_tokens)
    print(f"selected {len(sessions)} masked sessions "
          f"({sum(s['approx_tokens'] for s in sessions)} tokens total)")

    agent = harness.AIAgent(api_key=harness.KEY, base_url=harness.BASE, provider=harness.PROVIDER,
                            model=harness.MODEL, quiet_mode=True, skip_context_files=True,
                            skip_memory=True)
    agent._cached_system_prompt = harness.SYSTEM
    agent.tools = harness.TOOLS

    report: dict = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": harness.MODEL, "sessions": []}
    for session in sessions:
        messages = session["messages"]
        wire = harness._build_api_kwargs_for_mode(
            agent, harness._iteration_summary_api_messages(agent, messages))
        warming = warm(harness.Transport(), wire)
        cc = harness.cc
        cc.state.record_payload(
            request=wire, request_messages=wire["messages"], model=harness.MODEL,
            provider=harness.PROVIDER, base_url=harness.BASE, api_mode="chat_completions",
            conversation_history=messages, session_id=f"hetero-{session['session']}")
        record: dict = {"session": session["session"], "messages": session["message_count"],
                        "approx_tokens": session["approx_tokens"], "warm": warming, "arms": {}}
        for kind in ("builtin", "plugin"):
            arm = harness.run_arm(kind, messages, harness.MODEL)
            arm.pop("summary", None)          # content must not reach the report
            record["arms"][kind] = arm
        report["sessions"].append(record)
        builtin, plugin = record["arms"]["builtin"], record["arms"]["plugin"]
        print(f"{record['session']} msgs={record['messages']:4d} "
              f"cost builtin={builtin['cost_usd']} plugin={plugin['cost_usd']} "
              f"hit={plugin['cache_hit']} latency={plugin['latency_s']}s")

    serialized = json.dumps(report)
    leaked = [window for window in _leak_windows(sessions) if window in serialized]
    if leaked:
        print(f"REFUSING TO WRITE: {len(leaked)} transcript window(s) would leak into the report",
              file=sys.stderr)
        return 3

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("hetero-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {path} (counters only, leak check passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

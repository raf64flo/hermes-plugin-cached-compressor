"""Input-only variant: pin the summary length, then compare the cost of the call itself.

The head-to-head benchmark (`bench.py`) measures whole compactions, so output length leaks into the
bill: the built-in writes 4 500 to 12 600 tokens of summary, the plugin 2 300 to 6 300. This variant
removes that variable.

Each engine's summary payload is captured with a stub transport (no network), then both payloads are
posted to the same model with the output capped at the same 256 tokens. With the output pinned, the
difference in billed cost comes from the prompt alone: the plugin's cache-priced prefix against the
built-in's serialized sample. The summaries themselves are truncated and are not scored here.

Usage:

    BENCH_API_KEY=... python bench/input_only.py --tokens 20000 --tokens 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

import agent.context_compressor as context_module  # noqa: E402

OUTPUT_CAP = int(os.environ.get("BENCH_OUTPUT_CAP", "256"))


def _capture(kind: str, messages: list[dict], wire: dict) -> dict:
    """Run ``compress`` with a stub transport and keep the summary payload it tried to send."""
    captured: dict = {}
    stub_calls: list[dict] = []

    def stub(**kwargs) -> object:
        captured.setdefault("messages", kwargs["messages"])
        captured.setdefault("tools", kwargs.get("tools"))
        stub_calls.append(kwargs)
        return SimpleNamespace(
            model=bench.MODEL,
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="RESUME", reasoning_content=None, reasoning=None),
                finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)))

    if kind == "plugin":
        bench.cc.state.record_payload(
            request=wire, request_messages=wire["messages"], model=bench.MODEL,
            provider=bench.PROVIDER, base_url=bench.BASE, api_mode="chat_completions",
            conversation_history=messages, session_id="bench")
    original_context = context_module.call_llm
    original_plugin = bench.cc.call_llm
    context_module.call_llm = stub
    bench.cc.call_llm = stub
    try:
        bench._make_engine(kind).compress(list(messages), force=True)
    finally:
        context_module.call_llm = original_context
        bench.cc.call_llm = original_plugin
    if not captured:
        raise RuntimeError(f"{kind}: no summary call was captured")
    captured["stub_calls"] = len(stub_calls)
    return captured


def _post(payload: dict) -> dict:
    transport = bench.Transport(body_extra={"max_tokens": OUTPUT_CAP})
    kwargs: dict = {"messages": payload["messages"]}
    if payload.get("tools"):
        kwargs["tools"] = payload["tools"]
    transport(**kwargs)
    call = transport.calls[-1]
    call["chars"] = sum(len(message.get("content") or "") for message in payload["messages"])
    return call


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, action="append", required=True)
    parser.add_argument("--out", default=os.path.join(bench.REPO, "bench", "results"))
    args = parser.parse_args()

    agent = bench.AIAgent(api_key=bench.KEY, base_url=bench.BASE, provider=bench.PROVIDER,
                          model=bench.MODEL, quiet_mode=True, skip_context_files=True,
                          skip_memory=True)
    agent._cached_system_prompt = bench.SYSTEM
    agent.tools = bench.TOOLS

    report = {"model": bench.MODEL, "output_cap": OUTPUT_CAP, "runs": []}
    for target in args.tokens:
        messages = bench.build_history(target)
        wire = bench._build_api_kwargs_for_mode(
            agent, bench._iteration_summary_api_messages(agent, messages))
        warming = bench.warm(bench.Transport(), wire)
        print(f"\n=== {target} tokens (prefix warm: {warming['hit']:.1%}) ===")
        if warming["hit"] < 0.9:
            print("  warning: prefix not cached, the plugin arm would be measured cold")

        payloads = {kind: _capture(kind, messages, wire) for kind in ("builtin", "plugin")}
        arms = {}
        # Each payload is posted once: the built-in's serialized window is new text every time (cold,
        # as in production), the plugin's prefix was just sent by the main loop (warm, as in
        # production). Repeating the built-in payload would measure a state production never reaches,
        # and the provider's cache write lands too late for a repeat a few seconds later.
        for kind in ("builtin", "plugin"):
            arms[kind] = _post(payloads[kind])
            time.sleep(3)
        ratio = (arms["plugin"]["cost_usd"] / arms["builtin"]["cost_usd"]
                 if arms["builtin"]["cost_usd"] else None)
        report["runs"].append({"target_tokens": target, "warm": warming, "arms": arms,
                               "cost_ratio": round(ratio, 3) if ratio else None})
        print(f"{'arm':8s} {'prompt':>8s} {'cached':>8s} {'hit':>7s} {'out':>5s} "
              f"{'cost_usd':>10s} {'chars':>9s}")
        for kind in ("builtin", "plugin"):
            arm = arms[kind]
            hit = (f"{arm['cached_tokens'] / arm['prompt_tokens']:.1%}"
                   if arm["prompt_tokens"] else "-")
            print(f"{kind:8s} {arm['prompt_tokens']:8d} {arm['cached_tokens']:8d} {hit:>7s} "
                  f"{arm['completion_tokens']:5d} {str(arm['cost_usd']):>10s} {arm['chars']:9d}")
        print(f"cost ratio (plugin / built-in): {report['runs'][-1]['cost_ratio']}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("input-only-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

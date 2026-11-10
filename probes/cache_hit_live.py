"""End to end: the compaction goes through the plugin and the prefix is served from cache.

Writer  = the session's real request (Hermes' assembly path), which warms the cache.
Reader  = `CachedCompressor._call_summary_llm`, the method actually used in production
          (reader_messages -> call_llm), on the same route.
We read the `cached_tokens` the provider reports, not an estimate.
"""

import json
import os
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.environ.get("HERMES_SOURCE", os.path.expanduser("~/.hermes/hermes-agent")))

REPO = os.environ.get("PLUGIN_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hermes_bootstrap  # noqa: F401

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="probe-home-"))

from run_agent import AIAgent
from agent.chat_completion_helpers import _build_api_kwargs_for_mode, _iteration_summary_api_messages

import importlib.util

spec = importlib.util.spec_from_file_location(
    "cc_probe", f"{REPO}/__init__.py",
    submodule_search_locations=[REPO])
cc = importlib.util.module_from_spec(spec)
sys.modules["cc_probe"] = cc
spec.loader.exec_module(cc)

BASE = "https://inference-api.nousresearch.com/v1"
MODEL = "deepseek/deepseek-v4.1-flash-1017"
KEY = os.environ["HERMES_API_KEY"]

SYSTEM = (
    "You are Hermes, an agentic assistant operating in a terminal. "
    "House conventions: minimal diffs, evidence with path:line, no secrets in output. "
    "Tools: read_file, search_files, patch, terminal. "
) * 45

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


def history(n=40):
    msgs = []
    body = ("    out.append(_enrich(r))\n    return out\n") * 30
    for i in range(n):
        msgs.append({"role": "user", "content": f"  Task {i}: check src/module_{i}.py  "})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "read_file", "arguments": '{ "path" : "src/module_%d.py" }' % i}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "read_file",
                     "content": f"# src/module_{i}.py\n" + body})
        msgs.append({"role": "assistant", "content": f"Module {i} unchanged, next step: patch."})
    msgs.append({"role": "user", "content": "Continue."})
    return msgs


def post(body, max_tokens):
    body = dict(body)
    body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def usage(payload):
    u = payload["usage"]
    p = u["prompt_tokens"]
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return p, c, u.get("completion_tokens", 0)


agent = AIAgent(api_key=KEY, base_url=BASE, provider="nous", model=MODEL,
                quiet_mode=True, skip_context_files=True, skip_memory=True)
agent._cached_system_prompt = SYSTEM
agent.tools = TOOLS

msgs = history()
cut = len(msgs) - 12                      # protected tail, like the compressor

# --- 1. writer: the session's real request, which warms the cache
wire_full = _iteration_summary_api_messages(agent, msgs)
writer_body = _build_api_kwargs_for_mode(agent, wire_full)
for attempt in (1, 2):
    w = post(writer_body, max_tokens=16)
    pw, cw, _ = usage(w)
    print(f"writer{attempt} prompt={pw:6d} cached={cw:6d} hit={cw / pw:5.1%}")
    time.sleep(2)

time.sleep(3)

# --- 2. the hook captures that payload as is (same dicts, shallow copy)
cc.state.record_payload(
    request=writer_body, request_messages=writer_body["messages"],
    model=MODEL, provider="nous", base_url=BASE, api_mode="chat_completions",
    conversation_history=msgs, session_id="probe")
slot = cc.state.SLOT.snapshot()
print(f"slot    messages={len(slot['messages'])} history_len={slot['history_len']} offset={slot['message_count'] - slot['history_len']}")

# --- 3. reader: the plugin's method, on the same route
engine = cc.CachedCompressor(model=MODEL, provider="nous", api_key=KEY, base_url=BASE,
                             api_mode="chat_completions", config_context_length=200_000,
                             quiet_mode=True)
engine._pristine = list(msgs)
engine._window = list(msgs[:cut])
engine._active_compression_telemetry = {}

prompt = engine._build_summary_prompt("irrelevant", 900, None, "", True)
print(f"prompt  pointer={'yes' if 'earlier messages of this same request' in prompt else 'no'} chars={len(prompt)}")

started = time.monotonic()
real_call_llm = cc.call_llm


def spy(**kwargs):
    print(f"sent    messages={len(kwargs['messages'])} "
          f"tail_roles={[m.get('role') for m in kwargs['messages'][-3:]]} "
          f"runtime_model={kwargs.get('main_runtime', {}).get('model')} "
          f"runtime_session={kwargs.get('main_runtime', {}).get('session_id')}")
    response = real_call_llm(**kwargs)
    usage_obj = getattr(response, "usage", None)
    print(f"got     model={getattr(response, 'model', None)} usage={usage_obj}")
    return response


cc.call_llm = spy
summary = engine._call_summary_llm(prompt, started)
elapsed = time.monotonic() - started
cc.call_llm = real_call_llm

telemetry = engine._active_compression_telemetry
print()
print(f"reader  prompt={telemetry.get('aux_prompt_tokens')} "
      f"cached={telemetry.get('cache_read_tokens')} "
      f"hit={telemetry.get('cache_read_ratio')} elapsed={elapsed:.1f}s")
print()
print("--- produced summary ---")
print(summary[:1200])

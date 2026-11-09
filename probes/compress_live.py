"""Bout en bout : `compress()` complet du plugin, pas seulement l'appel de résumé.

Ce qui est vérifié, sur le vrai provider :
- le cycle entier du parent tourne (élagage, tête/queue protégées, résumé, nettoyage) ;
- le résumé a bien été produit depuis le préfixe fil (cached_tokens > 0 dans la télémétrie) ;
- le transcript rendu est compact et contient la synthèse.
Si l'alignement échoue, le parent reprend la main : on le voit à l'absence de cache_read_tokens.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.environ.get("HERMES_SOURCE", os.path.expanduser("~/.hermes/hermes-agent")))
import hermes_bootstrap  # noqa: F401

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="probe-home-"))

from run_agent import AIAgent
from agent.chat_completion_helpers import _build_api_kwargs_for_mode, _iteration_summary_api_messages

REPO = os.environ.get("PLUGIN_ROOT",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "cc_probe", f"{REPO}/__init__.py", submodule_search_locations=[REPO])
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
        msgs.append({"role": "user", "content": f"  Tâche {i}: vérifie src/module_{i}.py  "})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "read_file", "arguments": '{ "path" : "src/module_%d.py" }' % i}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "read_file",
                     "content": f"# src/module_{i}.py\n" + body})
        msgs.append({"role": "assistant", "content": f"Module {i} inchangé, prochaine étape: patch."})
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
    return p, c


agent = AIAgent(api_key=KEY, base_url=BASE, provider="nous", model=MODEL,
                quiet_mode=True, skip_context_files=True, skip_memory=True)
agent._cached_system_prompt = SYSTEM
agent.tools = TOOLS

msgs = history()

# 1. la session écrit deux fois le même préfixe : le cache se remplit au deuxième passage
wire_full = _iteration_summary_api_messages(agent, msgs)
writer_body = _build_api_kwargs_for_mode(agent, wire_full)
for attempt in (1, 2):
    p, c = usage(post(writer_body, max_tokens=16))
    print(f"writer{attempt} prompt={p:6d} cached={c:6d} hit={(c / p):5.1%}")
    time.sleep(2)

# 2. le hook capture le payload tel quel
cc.state.record_payload(
    request=writer_body, request_messages=writer_body["messages"],
    model=MODEL, provider="nous", base_url=BASE, api_mode="chat_completions",
    conversation_history=msgs, session_id="probe-compress")

# 3. compress() complet, sur le transcript vivant
engine = cc.CachedCompressor(model=MODEL, provider="nous", api_key=KEY, base_url=BASE,
                             api_mode="chat_completions", config_context_length=200_000,
                             quiet_mode=True, protect_last_n=20)

sent: list[int] = []
real_call_llm = cc.call_llm


def spy(**kwargs):
    sent.append(len(kwargs["messages"]))
    return real_call_llm(**kwargs)


cc.call_llm = spy
started = time.monotonic()
compacted = engine.compress(list(msgs), force=True)
elapsed = time.monotonic() - started
cc.call_llm = real_call_llm

telemetry = (getattr(engine, "_last_compression_telemetry", None)
             or getattr(engine, "_active_compression_telemetry", None) or {})
print()
print(f"messages  avant={len(msgs)} après={len(compacted)}")
print(f"appels auxiliaires={len(sent)} messages_envoyés={sent}")
print(f"télémétrie aux_prompt_tokens={telemetry.get('aux_prompt_tokens')} "
      f"cache_read_tokens={telemetry.get('cache_read_tokens')} "
      f"cache_read_ratio={telemetry.get('cache_read_ratio')} durée={elapsed:.1f}s")
print(f"rôles rendus={[m.get('role') for m in compacted][:6]} ... "
      f"{[m.get('role') for m in compacted][-3:]}")

body = next((m.get("content") or "" for m in compacted
             if m.get("role") == "user" and "Historical Task Snapshot" in (m.get("content") or "")), "")
print()
print("--- structure du transcript compacté ---")
for i, m in enumerate(compacted[:5]):
    print(f"  [{i}] {m.get('role'):9s} {(m.get('content') or '')[:90]!r}")
biggest = max(compacted, key=lambda m: len(m.get("content") or ""))
print(f"  plus long message : role={biggest.get('role')} chars={len(biggest.get('content') or '')}")
print()
print("--- début de la synthèse insérée ---")
print((biggest.get("content") or "")[:700])
print()
with open(f"{REPO}/probes/last_compress.json", "w", encoding="utf-8") as fh:
    json.dump(compacted, fh, ensure_ascii=False, indent=1)
print("transcript écrit dans probes/last_compress.json")
print()
print("VERDICT:", "préfixe réutilisé" if (telemetry.get("cache_read_tokens") or 0) > 0
      else "repli parent (aucun hit)")

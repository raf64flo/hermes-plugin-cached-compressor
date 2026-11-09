"""Le slot doit survivre à un double chargement du paquet par Hermes."""

from __future__ import annotations

import importlib.util
import sys

from support import REPO, load_plugin


def _second_copy() -> object:
    name = "cached_compressor_second_copy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_deux_imports_partagent_le_meme_slot() -> None:
    first = load_plugin()
    second = _second_copy()
    assert first.state.SLOT is second.state.SLOT


def test_le_hook_ecrit_dans_le_slot_lu_par_le_moteur() -> None:
    first = load_plugin()
    second = _second_copy()
    first.state.SLOT.clear()
    messages = [{"role": "user", "content": "capture"}]
    second.state.record_payload(
        request={"messages": messages, "tools": []}, request_messages=messages,
        model="m", provider="p", base_url="", api_mode="", conversation_history=messages, session_id="s")
    slot = first.state.SLOT.snapshot()
    assert slot is not None
    assert slot["messages"] == messages
    assert slot["model"] == "m"

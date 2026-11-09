"""Contrat de chargement : découverte du moteur, ``register(ctx)`` sous les deux collecteurs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERMES = Path.home() / ".hermes" / "hermes-agent"
MODULE_NAME = "cached_compressor_under_test"


def load_plugin() -> object:
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    if str(HERMES) not in sys.path:
        sys.path.insert(0, str(HERMES))
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class PluginCtx:
    """Contexte plugin complet : enregistre un moteur et des hooks."""

    def __init__(self) -> None:
        self.engine = None
        self.hooks: dict[str, object] = {}

    def register_context_engine(self, engine: object) -> None:
        self.engine = engine

    def register_hook(self, name: str, callback: object, **_: object) -> None:
        self.hooks[name] = callback


class EngineCollector:
    """Collecteur du moteur de contexte : il ne connaît que ``register_context_engine``."""

    def __init__(self) -> None:
        self.engine = None

    def register_context_engine(self, engine: object) -> None:
        self.engine = engine


class HookOnlyCtx:
    """Chargement comme plugin ordinaire : pas de moteur à enregistrer."""

    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}

    def register_hook(self, name: str, callback: object, **_: object) -> None:
        self.hooks[name] = callback


def test_register_pose_le_moteur_et_le_hook() -> None:
    module = load_plugin()
    ctx = PluginCtx()
    module.register(ctx)  # type: ignore[attr-defined]
    assert ctx.engine is not None
    assert ctx.engine.name == "cached-compressor"  # type: ignore[attr-defined]
    assert "pre_api_request" in ctx.hooks


def test_collecteur_du_moteur_ne_leve_pas() -> None:
    module = load_plugin()
    ctx = EngineCollector()
    module.register(ctx)  # type: ignore[attr-defined]
    assert ctx.engine is not None
    assert ctx.engine.name == "cached-compressor"  # type: ignore[attr-defined]


def test_contexte_plugin_ne_pose_que_les_hooks() -> None:
    module = load_plugin()
    ctx = HookOnlyCtx()
    module.register(ctx)  # type: ignore[attr-defined]
    assert set(ctx.hooks) == {"pre_api_request", "post_api_request"}


def test_decouverte_par_le_chargeur_reel() -> None:
    if str(HERMES) not in sys.path:
        sys.path.insert(0, str(HERMES))
    from plugins.context_engine import discover_context_engines, load_context_engine

    if "cached-compressor" not in [name for name, _, _ in discover_context_engines()]:
        pytest.skip("~/.hermes/plugins/cached-compressor absent")
    engine = load_context_engine("cached-compressor")
    assert engine is not None
    assert engine.name == "cached-compressor"
    engine.update_model("deepseek-v4.1-flash", 131072, provider="nous")
    assert engine.threshold_tokens == int(131072 * engine.threshold_percent)


def test_le_slot_garde_le_payload_fil() -> None:
    module = load_plugin()
    module.state.SLOT.clear()  # type: ignore[attr-defined]
    wire = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    module.state.record_payload(  # type: ignore[attr-defined]
        request={"messages": wire, "tools": [{"type": "function"}]},
        request_messages=wire, model="m", provider="p", base_url="u",
        api_mode="chat_completions", session_id="s1",
        conversation_history=[{"role": "user", "content": "q"}])
    snap = module.state.SLOT.snapshot()  # type: ignore[attr-defined]
    assert snap is not None
    assert snap["model"] == "m"
    assert snap["provider"] == "p"
    assert snap["messages"] == wire
    assert snap["tools"] == [{"type": "function"}]
    assert snap["history_len"] == 1

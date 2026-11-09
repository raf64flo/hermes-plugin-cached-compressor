"""Confidentialité et mémoire du slot de payload.

Le slot retient le dernier payload fil par session, donc une conversation entière, y compris ce que
l'utilisateur n'a pas l'intention de voir recopié ailleurs. Quatre propriétés à tenir : rien du contenu
ne part dans les journaux, la fin de session libère, le budget mémoire est borné, et une session ne
peut pas lire le fil d'une autre.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

import pytest
from support import HERMES, MODULE_NAME, REPO, load_plugin

_plugin = load_plugin()
CachedCompressor = _plugin.CachedCompressor  # type: ignore[attr-defined]
state = _plugin.state  # type: ignore[attr-defined]
SLOT = state.SLOT
SECRET = "sk-live-DO-NOT-LEAK-9f3a"


def _record(session_id: str, content: str, size: int = 0, model: str = "m",
            provider: str = "p") -> None:
    messages = [{"role": "user", "content": content + ("x" * size)}]
    SLOT.record(
        request={"messages": messages, "tools": []}, request_messages=messages,
        model=model, provider=provider, base_url="", api_mode="",
        conversation_history=messages, session_id=session_id)


@pytest.fixture(autouse=True)
def _clean_slot():
    SLOT.clear()
    yield
    SLOT.clear()


def test_journalisation_ne_contient_jamais_le_contenu(caplog):
    """Les journaux portent des routes et des compteurs, jamais le texte de la conversation."""
    caplog.set_level(logging.DEBUG)
    _record("s1", f"clé de test {SECRET}")
    engine = CachedCompressor(model="m", provider="p")
    engine._session_id = "s1"
    # Deux mesures sans cache ferment la route : c'est le chemin qui journalise le repli.
    route = _plugin.cacheability.route_key("p", "m", "")
    for _ in range(2):
        state.TRACKER.record(route, 5_000, 0)
    assert engine._prefix_allowed() is False
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert text, "le test doit produire au moins un journal"
    assert SECRET not in text
    assert "clé de test" not in text
    state.TRACKER.clear()


def test_fin_de_session_libere_le_slot():
    """``on_session_end`` supprime le transcript : c'est le seul moment où la mémoire se libère."""
    _record("s2", "conversation terminée")
    assert SLOT.snapshot("s2") is not None
    CachedCompressor(model="m", provider="p").on_session_end("s2", [])
    assert SLOT.snapshot("s2") is None


def test_budget_memoire_borne(monkeypatch):
    """Au-delà du budget, la plus ancienne session cède la place, même sous le plafond de sessions."""
    monkeypatch.setattr(state, "MAX_BYTES", 4_000)
    monkeypatch.setattr(state, "MAX_SESSIONS", 64)
    _record("vieux", "a", size=3_000)
    _record("recent", "b", size=3_000)
    assert SLOT.snapshot("vieux") is None
    assert SLOT.snapshot("recent") is not None


def test_le_slot_ne_suit_pas_une_mutation_du_fil():
    """L'élagage du compresseur réécrit ``content`` en place ; le slot doit garder ce qui a été envoyé."""
    message = {"role": "tool", "content": "sortie d'outil originale"}
    messages = [message]
    SLOT.record(request={"messages": messages, "tools": []}, request_messages=messages,
                model="m", provider="p", base_url="", api_mode="",
                conversation_history=messages, session_id="s3")
    message["content"] = "[Old tool output cleared to save context space]"
    assert SLOT.snapshot("s3")["messages"][0]["content"] == "sortie d'outil originale"


def test_sessions_concurrentes_et_import_double():
    """Deux sessions ne se volent pas leur fil, et deux imports du paquet partagent le même slot."""
    for name in ("a", "b", "c"):
        _record(name, f"fil de {name}")
    for name in ("a", "b", "c"):
        assert SLOT.snapshot(name)["messages"][0]["content"] == f"fil de {name}"
    SLOT.clear("b")
    assert SLOT.snapshot("b") is None
    assert SLOT.snapshot("a") is not None and SLOT.snapshot("c") is not None

    if str(HERMES) not in sys.path:
        sys.path.insert(0, str(HERMES))
    second = "cached_compressor_second_import"
    if second in sys.modules:
        assert sys.modules[second].state.SLOT is SLOT
        return
    spec = importlib.util.spec_from_file_location(
        second, REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[second] = module
    spec.loader.exec_module(module)
    assert module.state.SLOT is SLOT
    assert module.state.TRACKER is state.TRACKER
    assert MODULE_NAME in sys.modules

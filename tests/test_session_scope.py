"""Le slot doit être cadré par session : un processus peut servir plusieurs conversations.

Dans le gateway ou une flotte, plusieurs sessions partagent le même processus et le même couple
modèle/provider. Sans cadrage, la dernière requête émise écrase le slot des autres et la compaction
d'une session peut se servir du fil d'une autre.
"""

from __future__ import annotations

from support import load_plugin

_plugin = load_plugin()
CachedCompressor = _plugin.CachedCompressor  # type: ignore[attr-defined]
SLOT = _plugin.state.SLOT  # type: ignore[attr-defined]
MAX_SESSIONS = _plugin.state.MAX_SESSIONS  # type: ignore[attr-defined]
summary = _plugin.summary  # type: ignore[attr-defined]


def _record(session_id: str, content: str, model: str = "m", provider: str = "p") -> None:
    messages = [{"role": "user", "content": content}]
    SLOT.record(
        request={"messages": messages, "tools": []}, request_messages=messages,
        model=model, provider=provider, base_url="", api_mode="",
        conversation_history=messages, session_id=session_id)


def _engine(session_id: str) -> object:
    engine = CachedCompressor(model="m", provider="p")
    engine._session_id = session_id
    return engine


def setup_function() -> None:
    SLOT.clear()


def test_deux_sessions_ne_secrasent_pas() -> None:
    _record("s-A", "alpha")
    _record("s-B", "beta")
    assert SLOT.snapshot("s-A")["messages"][0]["content"] == "alpha"
    assert SLOT.snapshot("s-B")["messages"][0]["content"] == "beta"


def test_session_inconnue_ne_rend_rien() -> None:
    _record("s-A", "alpha")
    assert SLOT.snapshot("s-inconnue") is None


def test_snapshot_sans_argument_rend_la_plus_recente() -> None:
    _record("s-A", "alpha")
    _record("s-B", "beta")
    assert SLOT.snapshot()["messages"][0]["content"] == "beta"


def test_clear_une_session_laisse_les_autres() -> None:
    _record("s-A", "alpha")
    _record("s-B", "beta")
    SLOT.clear("s-A")
    assert SLOT.snapshot("s-A") is None
    assert SLOT.snapshot("s-B")["messages"][0]["content"] == "beta"


def test_borne_du_nombre_de_sessions() -> None:
    for index in range(MAX_SESSIONS + 2):
        _record(f"s-{index}", f"msg-{index}")
    assert SLOT.snapshot("s-0") is None
    assert SLOT.snapshot(f"s-{MAX_SESSIONS + 1}") is not None


def test_route_refuse_le_fil_dune_autre_session() -> None:
    _record("s-A", "alpha")
    wire = SLOT.snapshot("s-A")
    assert summary.route_matches(wire, "m", "p", "s-B") is False


def test_route_accepte_le_fil_de_la_meme_session() -> None:
    _record("s-A", "alpha")
    wire = SLOT.snapshot("s-A")
    assert summary.route_matches(wire, "m", "p", "s-A") is True


def test_route_tolere_un_fil_sans_session() -> None:
    messages = [{"role": "user", "content": "alpha"}]
    wire = {"messages": messages, "model": "m", "provider": "p", "session_id": ""}
    assert summary.route_matches(wire, "m", "p", "s-A") is True


def test_moteur_cadre_ne_lit_pas_le_slot_dune_autre_session() -> None:
    _record("s-A", "alpha")
    assert _engine("s-B")._wire_for_session() is None
    assert _engine("s-A")._wire_for_session() is not None


def test_moteur_non_cadre_garde_le_comportement_mono_session() -> None:
    _record("s-A", "alpha")
    engine = CachedCompressor(model="m", provider="p")
    assert engine._wire_for_session() is not None

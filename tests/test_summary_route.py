"""La requête de résumé doit repartir du payload fil capturé, et retomber sur le parent sinon."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from support import load_plugin

_plugin = load_plugin()
CachedCompressor = _plugin.CachedCompressor  # type: ignore[attr-defined]
summary = _plugin.summary  # type: ignore[attr-defined]
SLOT = _plugin.state.SLOT  # type: ignore[attr-defined]


WIRE = {
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ],
    "tools": [{"type": "function", "function": {"name": "read_file"}}],
    "model": "m-session",
    "provider": "p-session",
    "base_url": "https://exemple/v1",
    "api_mode": "chat_completions",
}
WINDOW = [{"role": "user", "content": "u2"}]


@pytest.fixture(autouse=True)
def _clean_slot():
    SLOT.clear()
    yield
    SLOT.clear()


def _engine() -> CachedCompressor:
    return CachedCompressor(model="m-session", provider="p-session")


def _response(content: str = "RÉSUMÉ", finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=None, reasoning=None),
            finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None),
    )


def test_reader_messages_rend_le_prefixe_et_l_instruction() -> None:
    messages = summary.reader_messages(WIRE, WINDOW, "INSTRUCTION")
    assert messages is not None
    assert messages[:-1] == WIRE["messages"]
    assert messages[-1] == {"role": "user", "content": "INSTRUCTION"}


def test_reader_messages_sans_alignement_rend_none() -> None:
    absent = [{"role": "user", "content": "jamais vu"}]
    assert summary.reader_messages(WIRE, absent, "INSTRUCTION") is None
    assert summary.reader_messages(None, WINDOW, "INSTRUCTION") is None
    assert summary.reader_messages(WIRE, [], "INSTRUCTION") is None


def test_alignement_par_index_malgre_contenu_elague() -> None:
    # Le transcript élagué porte un contenu réécrit ; le fil garde les octets envoyés. L'index tranche.
    pristine = [{"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1 [elided]"},
                {"role": "user", "content": "u2"}]
    window = [pristine[1], pristine[2]]
    wire = {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u1"},
                         {"role": "assistant", "content": "a1 complet"},
                         {"role": "user", "content": "u2"}],
            "message_count": 4, "history_len": 3, "model": "m-session", "provider": "p-session"}
    messages = summary.reader_messages(wire, window, "INSTRUCTION", window_end=2)
    assert messages is not None
    assert messages[2] == {"role": "assistant", "content": "a1 complet"}
    assert messages[-1]["content"] == "INSTRUCTION"
    assert len(messages) == 5          # 4 messages du fil, plus l'instruction


def test_index_rejete_si_le_role_ne_correspond_pas() -> None:
    wire = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}],
            "message_count": 2, "history_len": 1, "model": "m-session", "provider": "p-session"}
    window = [{"role": "assistant", "content": "a1"}]
    # cut calculé = 2, mais messages[1] est user : l'index est refusé, wire_cut ne trouve rien.
    assert summary.reader_messages(wire, window, "INSTRUCTION", window_end=1) is None


def test_index_of_par_identite() -> None:
    transcript = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    copie = dict(transcript[1])
    assert summary.index_of(transcript, [transcript[0], transcript[1]]) == 1
    assert summary.index_of(transcript, [copie]) is None      # copie, pas le même objet
    assert summary.index_of(transcript, []) is None


def test_index_de_la_liste_de_travail_alimente_l_alignement(monkeypatch) -> None:
    # L'élagage a réécrit les contenus : seule la position dans la liste de travail est fiable.
    wire = {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u1"},
                         {"role": "assistant", "content": "a1"},
                         {"role": "user", "content": "u2"}],
            "message_count": 4, "history_len": 3, "model": "m-session", "provider": "p-session"}
    SLOT.record(request={"messages": wire["messages"]}, request_messages=wire["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="", session_id="s")
    seen: dict = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr("cached_compressor_under_test.call_llm", fake_call_llm)
    engine = _engine()
    engine._window = [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    engine._window_end = 2      # fin de fenêtre dans la liste de travail
    engine._same_shape = True
    engine._call_summary_llm("INSTRUCTION", 0.0)
    # cut = (4 - 3) + 2 + 1 = 4 : les quatre messages du fil, puis l'instruction
    assert len(seen["messages"]) == 5
    assert seen["messages"][-1] == {"role": "user", "content": "INSTRUCTION"}


def test_main_runtime_porte_la_session() -> None:
    runtime = summary.main_runtime({**WIRE, "session_id": "s9"}, "sk-test")
    assert runtime["model"] == "m-session"
    assert runtime["provider"] == "p-session"
    assert runtime["session_id"] == "s9"
    assert runtime["api_key"] == "sk-test"
    assert summary.main_runtime(None, None)["session_id"] == ""


def test_le_contexte_de_session_part_avec_l_appel(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="https://exemple/v1",
                api_mode="chat_completions", session_id="s9")
    seen: dict = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr("cached_compressor_under_test.call_llm", fake_call_llm)
    engine = _engine()
    engine._window = WINDOW
    engine._call_summary_llm("INSTRUCTION", 0.0)
    assert seen["main_runtime"]["session_id"] == "s9"
    assert seen["main_runtime"]["model"] == "m-session"


def test_route_doit_correspondre_a_la_session() -> None:
    assert summary.route_matches(WIRE, "m-session", "p-session")
    assert not summary.route_matches(WIRE, "autre-modele", "p-session")
    assert not summary.route_matches(WIRE, "m-session", "autre-provider")
    assert not summary.route_matches(None, "m-session", "p-session")


def test_call_summary_llm_epingle_la_route_et_le_prefixe(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"], "tools": WIRE["tools"]},
                request_messages=WIRE["messages"], model="m-session", provider="p-session",
                base_url="https://exemple/v1", api_mode="chat_completions", session_id="s1")
    captured: dict = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response()

    monkeypatch.setattr("cached_compressor_under_test.call_llm", fake_call_llm)
    engine = _engine()
    engine._window = WINDOW

    assert engine._call_summary_llm("INSTRUCTION", 0.0) == "RÉSUMÉ"
    assert captured["provider"] == "p-session"
    assert captured["model"] == "m-session"
    assert captured["base_url"] == "https://exemple/v1"
    assert captured["api_mode"] == "chat_completions"
    assert captured["messages"][:-1] == WIRE["messages"]
    assert captured["messages"][-1]["content"] == "INSTRUCTION"
    assert captured["tools"] == WIRE["tools"]
    assert "max_tokens" not in captured


def test_call_summary_llm_repli_sur_le_parent_si_route_change(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="")
    monkeypatch.setattr("cached_compressor_under_test.call_llm",
                        lambda **kw: pytest.fail("le préfixe ne doit pas être envoyé ici"))

    engine = _engine()
    engine.model = "modele-change"          # bascule de secours depuis la capture
    engine._window = WINDOW
    monkeypatch.setattr(
        "agent.context_compressor.ContextCompressor._call_summary_llm",
        lambda self, prompt, started: f"PARENT:{prompt}")
    assert engine._call_summary_llm("INSTRUCTION", 0.0) == "PARENT:INSTRUCTION"


def test_contenu_vide_leve_pour_declencher_le_repli(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="")
    monkeypatch.setattr("cached_compressor_under_test.call_llm", lambda **kw: _response(""))
    engine = _engine()
    engine._window = WINDOW
    with pytest.raises(RuntimeError, match="empty content"):
        engine._call_summary_llm("INSTRUCTION", 0.0)


def test_resume_tronque_leve(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="")
    monkeypatch.setattr("cached_compressor_under_test.call_llm",
                        lambda **kw: _response("partiel", finish_reason="length"))
    engine = _engine()
    engine._window = WINDOW
    with pytest.raises(RuntimeError, match="length-truncated"):
        engine._call_summary_llm("INSTRUCTION", 0.0)


def test_prompt_pointe_vers_la_requete_quand_la_capture_est_utilisable() -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="")
    engine = _engine()
    prompt = engine._build_summary_prompt("TOURS ENORME " * 500, 500, None, "", True)
    assert "The turns to summarize are the earlier messages" in prompt
    assert "TOURS ENORME" not in prompt


def test_prompt_garde_le_contenu_sans_capture() -> None:
    engine = _engine()
    prompt = engine._build_summary_prompt("TOURS ORIGINAUX", 500, None, "", True)
    assert "TOURS ORIGINAUX" in prompt


def test_le_hit_de_cache_est_trace(monkeypatch) -> None:
    SLOT.record(request={"messages": WIRE["messages"]}, request_messages=WIRE["messages"],
                model="m-session", provider="p-session", base_url="", api_mode="")
    response = _response()
    response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5,
                                     prompt_tokens_details={"cached_tokens": 94})
    monkeypatch.setattr("cached_compressor_under_test.call_llm", lambda **kw: response)
    engine = _engine()
    engine._window = WINDOW
    engine._active_compression_telemetry = {}
    assert engine._call_summary_llm("INSTRUCTION", 0.0) == "RÉSUMÉ"
    assert engine._active_compression_telemetry["cache_read_tokens"] == 94
    assert engine._active_compression_telemetry["cache_read_ratio"] == 0.94

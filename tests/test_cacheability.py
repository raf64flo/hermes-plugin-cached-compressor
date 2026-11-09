"""Une route qui ne rapporte pas de lecture de cache ne doit pas payer le prefixe.

Le mode prefixe suppose que le prefixe est deja en cache chez le provider : on renvoie le fil
tronque et seule la lecture de cache est facturee. Certains backends ne renvoient aucun
``prompt_tokens_details`` ; la lecture y est alors au plein tarif, et le prefixe complet coute plus
cher que la fenetre serialisee du compresseur integre. Mesure : 2 445 tokens de prompt renvoyes deux
fois sans ``cached_tokens``, contre 11 264/11 383 sur une route qui rapporte le cache.
"""

from __future__ import annotations

import time

from support import load_plugin

_plugin = load_plugin()
CachedCompressor = _plugin.CachedCompressor  # type: ignore[attr-defined]
ContextCompressor = _plugin.ContextCompressor  # type: ignore[attr-defined]
TRACKER = _plugin.state.TRACKER  # type: ignore[attr-defined]
MAX_STRIKES = _plugin.cacheability.MAX_STRIKES  # type: ignore[attr-defined]
MIN_PROMPT_TOKENS = _plugin.cacheability.MIN_PROMPT_TOKENS  # type: ignore[attr-defined]

ROUTE = ("p-session", "m-session", "https://exemple/v1")
AUTRE = ("p-autre", "m-autre", "https://autre/v1")
GROS = MIN_PROMPT_TOKENS + 1


def setup_function() -> None:
    TRACKER.clear()
    _plugin.state.SLOT.clear()


def _sans_cache(route: tuple[str, str, str] = ROUTE) -> None:
    for _ in range(MAX_STRIKES):
        TRACKER.record(route, GROS, 0)


def test_route_inconnue_garde_le_mode_prefixe() -> None:
    assert TRACKER.prefix_allowed(ROUTE) is True


def test_deux_mesures_sans_cache_desactivent_la_route() -> None:
    _sans_cache()
    assert TRACKER.prefix_allowed(ROUTE) is False


def test_une_seule_mesure_sans_cache_ne_suffit_pas() -> None:
    TRACKER.record(ROUTE, GROS, 0)
    assert TRACKER.prefix_allowed(ROUTE) is True


def test_prompt_court_ignore() -> None:
    for _ in range(MAX_STRIKES + 2):
        TRACKER.record(ROUTE, MIN_PROMPT_TOKENS - 1, 0)
    assert TRACKER.prefix_allowed(ROUTE) is True


def test_un_hit_remet_le_compteur_a_zero() -> None:
    _sans_cache()
    TRACKER.record(ROUTE, GROS, GROS // 2)
    assert TRACKER.prefix_allowed(ROUTE) is True


def test_les_routes_sont_independantes() -> None:
    _sans_cache()
    assert TRACKER.prefix_allowed(AUTRE) is True


def test_borne_du_nombre_de_routes() -> None:
    for index in range(_plugin.cacheability.MAX_ROUTES + 5):
        _sans_cache((f"p{index}", "m", "u"))
    assert len(TRACKER._strikes) <= _plugin.cacheability.MAX_ROUTES


def _engine_with_wire() -> object:
    messages = [{"role": "user", "content": "u1"}]
    _plugin.state.SLOT.record(
        request={"messages": messages, "tools": []}, request_messages=messages,
        model="m-session", provider="p-session", base_url="https://exemple/v1",
        api_mode="chat_completions", conversation_history=messages, session_id="s")
    engine = CachedCompressor(model="m-session", provider="p-session",
                              base_url="https://exemple/v1")
    engine._window = list(messages)
    engine._window_end = 0
    engine._same_shape = True
    return engine


def test_moteur_retombe_sur_le_parent_quand_la_route_est_froide(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("cached_compressor_under_test.call_llm", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ContextCompressor, "_call_summary_llm", lambda self, prompt, started: "PARENT")
    _sans_cache()
    assert _engine_with_wire()._call_summary_llm("INSTRUCTION", time.monotonic()) == "PARENT"
    assert calls == []


def test_moteur_sert_le_prefixe_tant_que_la_route_repond_au_cache(monkeypatch) -> None:
    from types import SimpleNamespace

    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="RESUME", reasoning_content=None, reasoning=None),
                finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5000, completion_tokens=10,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=4800)))

    monkeypatch.setattr("cached_compressor_under_test.call_llm", fake_call_llm)
    assert _engine_with_wire()._call_summary_llm("INSTRUCTION", time.monotonic()) == "RESUME"
    assert len(calls) == 1

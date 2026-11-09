"""La garde de cacheabilité, durcie : pré-diagnostic gratuit, seuil, repli forcé.

Une route marquée « sans cache » après deux compactions coûte jusqu'à deux fois le prix du
compresseur intégré au plein tarif. Trois leviers ajoutés, tous optionnels : lire les usages de la
boucle principale (gratuit), exiger un taux de hit observé, et forcer le repli sur le natif. Rien
n'est configuré, le comportement de base ne change pas.
"""

from __future__ import annotations

import time

import pytest
from support import load_plugin

_plugin = load_plugin()
CachedCompressor = _plugin.CachedCompressor  # type: ignore[attr-defined]
ContextCompressor = _plugin.ContextCompressor  # type: ignore[attr-defined]
TRACKER = _plugin.state.TRACKER  # type: ignore[attr-defined]
MIN_PROMPT_TOKENS = _plugin.cacheability.MIN_PROMPT_TOKENS  # type: ignore[attr-defined]

ROUTE = ("p", "m", "https://exemple/v1")
GROS = MIN_PROMPT_TOKENS + 10


def setup_function() -> None:
    TRACKER.clear()
    _plugin.state.SLOT.clear()  # type: ignore[attr-defined]
    _plugin.cacheability.reset_tuning()  # type: ignore[attr-defined]


def test_par_defaut_rien_ne_change() -> None:
    """Sans configuration, seul le compteur de strikes décide, comme avant."""
    assert TRACKER.prefix_allowed(ROUTE) is True
    TRACKER.record(ROUTE, GROS, 0)
    assert TRACKER.prefix_allowed(ROUTE) is True
    TRACKER.record(ROUTE, GROS, 0)
    assert TRACKER.prefix_allowed(ROUTE) is False


def test_seuil_observed_cache_read() -> None:
    """Avec un seuil, le taux de hit des dernières observations décide aussi."""
    _plugin.cacheability.set_min_ratio(0.9)
    assert TRACKER.prefix_allowed(ROUTE) is True  # aucune observation : on laisse la chance
    TRACKER.record(ROUTE, GROS, 0)  # strike 1, et 0 % observé
    assert TRACKER.prefix_allowed(ROUTE) is False  # le seuil refuse avant le deuxième strike
    for _ in range(8):  # huit hits à 95 % remplacent la fenêtre
        TRACKER.record(ROUTE, GROS, int(GROS * 0.95))
    assert TRACKER.prefix_allowed(ROUTE) is True  # 95 % observé >= 90 %


def test_seuil_invalide_est_ignoré() -> None:
    for bad in (None, "vague", -1, 1.5, object()):
        _plugin.cacheability.set_min_ratio(bad)
        assert _plugin.cacheability._MIN_RATIO is None


def test_forcer_le_repli() -> None:
    """force_builtin : même une route qui répond toujours au cache n'utilise pas le mode préfixe."""
    _plugin.cacheability.force_builtin(True)
    TRACKER.record(ROUTE, GROS, GROS)  # hits à 100 %
    assert TRACKER.prefix_allowed(ROUTE) is False
    assert _plugin.cacheability.is_forced_builtin() is True
    _plugin.cacheability.reset_tuning()
    assert TRACKER.prefix_allowed(ROUTE) is True


def _usage(prompt: int, cached: int) -> dict:
    return {"input_tokens": prompt - cached, "output_tokens": 10,
            "cache_read_tokens": cached, "prompt_tokens": prompt, "total_tokens": prompt + 10}


def test_observe_usage_alimente_la_garde() -> None:
    """Le pré-diagnostic gratuit : l'usage de la boucle principale suffit, aucune requête en plus."""
    _plugin.state.observe_usage(usage=_usage(8000, 7600), provider="p", model="m",
                                base_url="https://exemple/v1")
    _plugin.state.observe_usage(usage=_usage(8000, 7700), provider="p", model="m",
                                base_url="https://exemple/v1")
    ratio = TRACKER.observed_ratio(ROUTE)
    assert ratio is not None and ratio > 0.94
    _plugin.cacheability.set_min_ratio(0.95)
    assert TRACKER.prefix_allowed(ROUTE) is True
    _plugin.cacheability.set_min_ratio(0.99)
    assert TRACKER.prefix_allowed(ROUTE) is False


def test_observe_usage_ignore_le_court_et_le_muet() -> None:
    """Un prompt trop court n'a jamais été mis en cache : l'observation ne compte pas."""
    _plugin.state.observe_usage(usage=_usage(500, 0), provider="p", model="m",
                                base_url="https://exemple/v1")
    assert TRACKER.observed_ratio(ROUTE) is None
    _plugin.state.observe_usage(usage=None, provider="p", model="m",
                                base_url="https://exemple/v1")
    assert TRACKER.observed_ratio(ROUTE) is None


def test_register_lit_la_config(ctx) -> None:
    """Les deux réglages se lisent depuis ``plugins.entries.cached-compressor.settings``."""
    ctx.config["min_cache_ratio"] = 0.9
    ctx.config["force_builtin"] = True
    _plugin.register(ctx)
    assert _plugin.cacheability._MIN_RATIO == 0.9
    assert _plugin.cacheability.is_forced_builtin() is True
    assert "post_api_request" in [call[0] for call in ctx.hooks]
    _plugin.cacheability.reset_tuning()


def test_register_pour_la_config_invalide() -> None:
    """Un réglage illisible ne casse pas l'installation : le comportement de base s'applique."""
    ctx = _SimpleCtx({"min_cache_ratio": "vague", "force_builtin": "oui"})
    _plugin.register(ctx)
    assert _plugin.cacheability._MIN_RATIO is None
    assert _plugin.cacheability.is_forced_builtin() is False
    assert "pre_api_request" in [call[0] for call in ctx.hooks]


@pytest.fixture()
def ctx() -> _SimpleCtx:
    return _SimpleCtx()


def test_moteur_observe_force(monkeypatch) -> None:
    """Moteur forcé : le parent est appelé sans lire le slot."""
    monkeypatch.setattr(ContextCompressor, "_call_summary_llm", lambda self, prompt, started: "PARENT")
    _plugin.cacheability.force_builtin(True)
    messages = [{"role": "user", "content": "u1"}]
    _plugin.state.SLOT.record(
        request={"messages": messages, "tools": []}, request_messages=messages,
        model="m", provider="p", base_url="https://exemple/v1", api_mode="chat_completions",
        conversation_history=messages, session_id="s")
    engine = CachedCompressor(model="m", provider="p", base_url="https://exemple/v1")
    engine._window = list(messages)
    engine._window_end = 0
    engine._same_shape = True
    assert engine._call_summary_llm("INSTRUCTION", time.monotonic()) == "PARENT"


class _SimpleCtx:
    """Double du ``PluginContext`` : les trois méthodes que ``register`` appelle."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.hooks: list[tuple[str, object]] = []
        self.engines: list[object] = []

    def register_context_engine(self, engine: object) -> None:
        self.engines.append(engine)

    def register_hook(self, name: str, callback: object) -> None:
        self.hooks.append((name, callback))

    def get_config(self, key: str, default: object = None) -> object:
        return self.config.get(key, default)

"""cached-compressor : compacter le contexte en réutilisant le cache de préfixe du provider.

Le paquet a deux points d'entrée, tous deux via ``register(ctx)`` :

- moteur de contexte (``context.engine: cached-compressor``) : le collecteur du moteur expose
  ``register_context_engine``, on lui passe l'instance ci-dessous ;
- plugin ordinaire (``plugins.enabled``) : le contexte plugin expose ``register_hook``, on y
  enregistre ``pre_api_request``, seul endroit où Hermes livre le payload fil final.

Le hook n'est pas un confort. ``select_context`` reçoit la requête **avant** le plan de cache et
les sanitizers (``agent/context_engine.py:132-136``), donc reconstruire le fil depuis là ne partage
pas les octets envoyés : mesuré à 0 % de hit, contre 99,5 % en repartant du payload réel.

Limite connue : ``plugins.isolation: host`` interdit de sous-classer ``ContextCompressor``, le
chargement échoue alors proprement (``load_context_engine`` rend ``None``). Le défaut
``in_process`` (``hermes_cli/config_defaults.py:1730``) est requis.
"""

from __future__ import annotations

from typing import Any

from agent.context_compressor import ContextCompressor

from . import state

__all__ = ["CachedCompressor", "register"]


class CachedCompressor(ContextCompressor):
    """Le compresseur intégré, avec la requête de résumé servie depuis le payload fil capturé."""

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    @property
    def name(self) -> str:
        return "cached-compressor"

    def compress(
        self, messages: list[dict[str, Any]], current_tokens: int | None = None,
        focus_topic: str | None = None, force: bool = False, memory_context: str = "",
    ) -> list[dict[str, Any]]:
        # Transcript vierge conservé pour aligner la fenêtre de compaction sur la liste fil (tâche 3.2).
        self._pristine = list(messages)
        try:
            return super().compress(
                messages, current_tokens=current_tokens, focus_topic=focus_topic,
                force=force, memory_context=memory_context)
        finally:
            self._pristine = None


def register(ctx: Any) -> None:
    """Point d'entrée unique : moteur de contexte, ou plugin porteur du hook."""
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(CachedCompressor())
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("pre_api_request", state.record_payload)

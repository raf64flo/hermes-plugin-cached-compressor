"""Slot du dernier payload fil observé, rempli par le hook ``pre_api_request``.

Le hook reçoit ``request`` (le payload complet envoyé au provider, après assemblage, sanitizers et
plan de cache) et ``request_messages`` (sa liste de messages). On garde des copies de surface : le
hook documente lui-même que ``deepcopy`` coûte trop cher sur un transcript de session, et le
compresseur ne relit ce slot qu'au tour suivant.

Une seule session écrit ce slot : il est remplacé à chaque requête, jamais empilé.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any


class PayloadSlot:
    """Dernier payload fil, plus la route qui l'a produit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None

    def record(self, **hook_kwargs: Any) -> None:
        request = hook_kwargs.get("request")
        payload = request if isinstance(request, dict) else {}
        messages = hook_kwargs.get("request_messages")
        if not isinstance(messages, list):
            messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return
        tools = payload.get("tools")
        with self._lock:
            self._data = {
                "messages": list(messages),
                "tools": list(tools) if isinstance(tools, list) else [],
                "model": hook_kwargs.get("model"),
                "provider": hook_kwargs.get("provider"),
                "base_url": hook_kwargs.get("base_url"),
                "api_mode": hook_kwargs.get("api_mode"),
                "session_id": hook_kwargs.get("session_id"),
                "history_len": len(hook_kwargs.get("conversation_history") or []),
                "message_count": len(messages),
                "recorded_at": time.time(),
            }

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._data) if self._data else None

    def clear(self) -> None:
        with self._lock:
            self._data = None


_SHARED_HOLDER = "_hermes_cached_compressor_shared"


def _shared_slot() -> "PayloadSlot":
    """Slot unique du processus, même si le paquet est importé deux fois.

    Hermes peut charger ce paquet par deux chemins distincts : le moteur de contexte
    (``plugins/context_engine`` -> ``_hermes_user_context_engine.cached-compressor``) et le plugin
    ordinaire (``plugins.enabled`` -> ``plugins.cached_compressor``). Deux imports, deux modules,
    donc deux ``SLOT`` : le hook écrirait dans l'un et le moteur lirait l'autre, sans rien dire.
    Le slot est donc rangé dans un module sentinelle de ``sys.modules``.
    """
    holder = sys.modules.get(_SHARED_HOLDER)
    if holder is None or getattr(holder, "slot", None) is None:
        # Pas de isinstance : la classe PayloadSlot diffère entre deux imports du paquet, et le
        # contrôle recréait alors un slot au lieu de partager celui du premier import.
        holder = types.ModuleType(_SHARED_HOLDER)
        holder.slot = PayloadSlot()  # type: ignore[attr-defined]
        sys.modules[_SHARED_HOLDER] = holder
    return holder.slot  # type: ignore[attr-defined]


SLOT = _shared_slot()


def record_payload(**hook_kwargs: Any) -> None:
    """Callback enregistré sur ``pre_api_request`` ; ne lève jamais vers la boucle de l'agent."""
    try:
        SLOT.record(**hook_kwargs)
    except Exception:
        return

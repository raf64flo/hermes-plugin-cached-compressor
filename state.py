"""Slots des derniers payloads fil observés, remplis par le hook ``pre_api_request``.

Le hook reçoit ``request`` (le payload complet envoyé au provider, après assemblage, sanitizers et
plan de cache) et ``request_messages`` (sa liste de messages). On garde des copies de surface : le
hook documente lui-même que ``deepcopy`` coûte trop cher sur un transcript de session, et le
compresseur ne relit ce slot qu'au tour suivant.

Un slot par session, et non un slot unique : un processus peut servir plusieurs conversations
(gateway, agents de flotte, sous-agents) avec le même couple modèle/provider. Avec un slot unique, la
dernière requête émise écrasait le slot des autres, et la compaction d'une session pouvait se servir
du fil d'une autre. Les sessions sont bornées en nombre, la plus ancienne écriture cédant la place.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

#: Nombre de sessions gardées en mémoire dans un même processus.
MAX_SESSIONS = 8


class PayloadSlot:
    """Dernier payload fil par session, plus la route qui l'a produit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, dict[str, Any]] = {}
        self._recency: list[str] = []

    def record(self, **hook_kwargs: Any) -> None:
        request = hook_kwargs.get("request")
        payload = request if isinstance(request, dict) else {}
        messages = hook_kwargs.get("request_messages")
        if not isinstance(messages, list):
            messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return
        tools = payload.get("tools")
        key = str(hook_kwargs.get("session_id") or "")
        with self._lock:
            self._slots[key] = {
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
            if key in self._recency:
                self._recency.remove(key)
            self._recency.append(key)
            while len(self._recency) > MAX_SESSIONS:
                self._slots.pop(self._recency.pop(0), None)

    def snapshot(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Payload d'une session, ou de la plus récente si ``session_id`` est omis."""
        with self._lock:
            key = None
            if session_id is None:
                key = self._recency[-1] if self._recency else None
            else:
                key = str(session_id)
            data = self._slots.get(key) if key is not None else None
            return dict(data) if data else None

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._slots.clear()
                self._recency.clear()
                return
            key = str(session_id)
            self._slots.pop(key, None)
            if key in self._recency:
                self._recency.remove(key)


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

"""Slots des derniers payloads fil observés, remplis par le hook ``pre_api_request``.

Le hook reçoit ``request`` (le payload complet envoyé au provider, après assemblage, sanitizers et
plan de cache) et ``request_messages`` (sa liste de messages). On garde des copies de surface : le
hook documente lui-même que ``deepcopy`` coûte trop cher sur un transcript de session, et le
compresseur ne relit ce slot qu'au tour suivant.

Un slot par session, et non un slot unique : un processus peut servir plusieurs conversations
(gateway, agents de flotte, sous-agents) avec le même couple modèle/provider. Avec un slot unique, la
dernière requête émise écrasait le slot des autres, et la compaction d'une session pouvait se servir
du fil d'une autre. Les sessions sont bornées en nombre et en octets, la plus ancienne écriture cédant
la place, et la fin de session libère son slot (``CachedCompressor.on_session_end``).

Confidentialité : ce module ne journalise rien. Ni le contenu des messages, ni les extraits d'outils ne
sortent d'ici ; les journaux du plugin ne portent que des routes et des compteurs. Les messages sont
copiés en surface à l'enregistrement, donc une mutation du dictionnaire d'origine après le hook ne
change pas ce que le slot renvoie. Les valeurs imbriquées (``tool_calls``) restent partagées : on ne
les copie pas, le coût d'un ``deepcopy`` sur un transcript de session étant disqualifiant.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

from . import cacheability

#: Nombre de sessions gardées en mémoire dans un même processus.
MAX_SESSIONS = 8
#: Budget mémoire des slots, en octets estimés (contenu des messages, plus un forfait par message).
MAX_BYTES = 64 * 1024 * 1024
#: Forfait par message pour l'enveloppe (rôle, identifiants, clés), qui n'est pas mesurée.
_MESSAGE_OVERHEAD = 128


def _entry_bytes(messages: list[dict[str, Any]], tools: list[Any]) -> int:
    """Estimation volontairement grossière : le contenu, plus un forfait par message et par outil."""
    total = 0
    for message in messages:
        content = message.get("content")
        total += len(content) if isinstance(content, str) else 0
        total += _MESSAGE_OVERHEAD
        total += len(str(message.get("tool_calls") or ""))
    total += sum(len(str(tool)) for tool in tools)
    return total


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
        stored = [dict(message) for message in messages if isinstance(message, dict)]
        frozen_tools = [dict(tool) if isinstance(tool, dict) else tool for tool in tools] \
            if isinstance(tools, list) else []
        with self._lock:
            self._slots[key] = {
                "messages": stored,
                "tools": frozen_tools,
                "model": hook_kwargs.get("model"),
                "provider": hook_kwargs.get("provider"),
                "base_url": hook_kwargs.get("base_url"),
                "api_mode": hook_kwargs.get("api_mode"),
                "session_id": hook_kwargs.get("session_id"),
                "history_len": len(hook_kwargs.get("conversation_history") or []),
                "message_count": len(stored),
                "bytes": _entry_bytes(stored, frozen_tools),
                "recorded_at": time.time(),
            }
            if key in self._recency:
                self._recency.remove(key)
            self._recency.append(key)
            self._evict(key)

    def _evict(self, keep: str) -> None:
        """Ramène le nombre de sessions et le budget mémoire sous leurs plafonds.

        La session qui vient d'être écrite est conservée : un seul transcript plus lourd que tout le
        budget reste utile au tour qui suit, et l'écraser ne libérerait rien.
        """
        while len(self._recency) > 1:
            total = sum(entry["bytes"] for entry in self._slots.values())
            if len(self._recency) <= MAX_SESSIONS and total <= MAX_BYTES:
                break
            oldest = next((name for name in self._recency if name != keep), None)
            if oldest is None:
                break
            self._recency.remove(oldest)
            self._slots.pop(oldest, None)

    def snapshot(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Payload d'une session, ou de la plus récente si ``session_id`` est omis."""
        with self._lock:
            key = None
            if session_id is None:
                key = self._recency[-1] if self._recency else None
            else:
                key = str(session_id)
            data = self._slots.get(key) if key is not None else None
            if not data:
                return None
            copy = dict(data)
            copy["messages"] = list(data["messages"])
            copy["tools"] = list(data["tools"])
            return copy

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


def _shared_holder() -> types.ModuleType:
    """État partagé du processus, même si le paquet est importé deux fois.

    Hermes peut charger ce paquet par deux chemins distincts : le moteur de contexte
    (``plugins/context_engine`` -> ``_hermes_user_context_engine.cached-compressor``) et le plugin
    ordinaire (``plugins.enabled`` -> ``plugins.cached_compressor``). Deux imports, deux modules,
    donc deux ``SLOT`` : le hook écrirait dans l'un et le moteur lirait l'autre, sans rien dire.
    L'état est donc rangé dans un module sentinelle de ``sys.modules``, avec la mémoire de cache des
    routes, qui doit suivre le même sort.
    """
    holder = sys.modules.get(_SHARED_HOLDER)
    if holder is None:
        holder = types.ModuleType(_SHARED_HOLDER)
        sys.modules[_SHARED_HOLDER] = holder
    # Pas de isinstance : la classe diffère entre deux imports du paquet, et le contrôle recréait
    # alors un objet au lieu de partager celui du premier import.
    if getattr(holder, "slot", None) is None:
        holder.slot = PayloadSlot()  # type: ignore[attr-defined]
    if getattr(holder, "tracker", None) is None:
        holder.tracker = cacheability.RouteCacheTracker()  # type: ignore[attr-defined]
    return holder


def _shared_slot() -> "PayloadSlot":
    """Slot unique du processus (compatibilité avec les premiers appels)."""
    return _shared_holder().slot  # type: ignore[attr-defined]


SLOT = _shared_slot()
TRACKER = _shared_holder().tracker  # type: ignore[attr-defined]


def record_payload(**hook_kwargs: Any) -> None:
    """Callback enregistré sur ``pre_api_request`` ; ne lève jamais vers la boucle de l'agent."""
    try:
        SLOT.record(**hook_kwargs)
    except Exception:
        return


def _non_negative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def observe_usage(**hook_kwargs: Any) -> None:
    """Callback enregistré sur ``post_api_request`` : pré-diagnostic gratuit de la route.

    L'usage de la réponse de la boucle principale est déjà là, normalisé (``cache_read_tokens``
    couvre les variantes OpenAI, Anthropic et DeepSeek) : observer une lecture de cache nulle sur un
    prompt assez grand coûte zéro, au lieu d'attendre une compaction pour découvrir que le backend
    ne sert rien depuis le cache. Les appels d'appoint du plugin ne passent pas par ce hook, il n'y
    a donc pas de boucle de rétroaction. Ne lève jamais.
    """
    try:
        usage = hook_kwargs.get("usage")
        if not isinstance(usage, dict):
            return
        prompt = _non_negative_int(usage.get("prompt_tokens"))
        if prompt is None:
            input_tokens = _non_negative_int(usage.get("input_tokens"))
            cache_read = _non_negative_int(usage.get("cache_read_tokens"))
            if input_tokens is not None and cache_read is not None:
                prompt = input_tokens + cache_read
        if not prompt:
            return
        cached = _non_negative_int(usage.get("cache_read_tokens")) or 0
        route = cacheability.route_key(
            hook_kwargs.get("provider"), hook_kwargs.get("model"), hook_kwargs.get("base_url"))
        TRACKER.record(route, prompt, cached)
    except Exception:
        return

"""Construction de la requête de résumé à partir du payload fil capturé.

Le principe : ne jamais reconstruire le préfixe, toujours repartir des messages réellement envoyés
au provider (capturés par le hook ``pre_api_request``), les tronquer à la frontière de la fenêtre de
compaction, et ajouter l'instruction de synthèse du parent.
"""

from __future__ import annotations

from typing import Any, Optional

from .alignment import wire_cut


def route_matches(wire: Optional[dict[str, Any]], model: str | None, provider: str | None) -> bool:
    """La route capturée doit être celle de la session courante.

    Un changement de modèle ou une bascule de secours depuis la capture rend le cache froid : mieux
    vaut alors passer par le modèle auxiliaire, moins cher, que d'envoyer la session entière au
    plein tarif.
    """
    if not wire:
        return False
    if not wire.get("messages") or not wire.get("model"):
        return False
    return (wire.get("model") or "") == (model or "") and (wire.get("provider") or "") == (provider or "")


def index_of(transcript: Optional[list[dict[str, Any]]], window: Optional[list[dict[str, Any]]]) -> Optional[int]:
    """Index de ``window[-1]`` dans le transcript, par identité d'objet uniquement.

    Le compresseur découpe la fenêtre dans la liste qu'on lui a passée : les dicts sont les mêmes
    objets, donc l'identité suffit et elle est insensible à l'élagage qui réécrit des contenus.
    """
    if not transcript or not window:
        return None
    last = window[-1]
    for index, message in enumerate(transcript):
        if message is last:
            return index
    return None


def _same_role(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a.get("role") or "") == str(b.get("role") or "")


def reader_messages(
    wire: Optional[dict[str, Any]], window: Optional[list[dict[str, Any]]], prompt: str,
    window_end: Optional[int] = None,
) -> Optional[list[dict[str, Any]]]:
    """Messages du résumé : préfixe fil jusqu'à la fenêtre, plus l'instruction. ``None`` si indéterminable.

    ``None`` est le signal de repli : l'appelant doit alors laisser le parent construire sa requête.

    Deux chemins d'alignement, du plus sûr au plus permissif :

    1. par index, quand la position de la fenêtre dans le transcript est connue : le fil vaut
       ``[système][préfills] + transcript``, donc ``cut = (len(fil) - len(transcript)) + window_end + 1``.
       Le décalage est vérifié par le rôle du message frontière.
    2. par séquence, via ``wire_cut``, quand le transcript d'origine n'est plus disponible.
    """
    if not wire or not window or not prompt.strip():
        return None
    messages = wire.get("messages") or []
    if window_end is not None:
        offset = int(wire.get("message_count") or 0) - int(wire.get("history_len") or 0)
        cut = offset + window_end + 1
        if 0 < cut <= len(messages) and _same_role(messages[cut - 1], window[-1]):
            return list(messages[:cut]) + [{"role": "user", "content": prompt}]
    cut = wire_cut(messages, window)
    if cut is None or cut <= 0 or cut > len(messages):
        return None
    return list(messages[:cut]) + [{"role": "user", "content": prompt}]

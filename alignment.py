"""Cale la fenêtre de résumé sur le fil capturé (payload réel envoyé au provider).

Le hook ``pre_api_request`` mémorise dans ``state.PayloadSlot`` la liste fil ``messages`` du
dernier payload. Pour réutiliser le cache de préfixe du provider, on envoie un vrai préfixe de la
requête précédente au lieu d'une reconstruction. ``wire_cut`` fournit, pour une fenêtre à résumer,
l'index de coupe dans ce fil : tout ce qui précède (fenêtre incluse) sera remplacé par le résumé.

Aucune exception ne sort de ``wire_cut`` : on traite avec prudence les entrées mal formées
(None, dicts vides, contenus bizarres) et on rend ``None`` quand la séquence n'est pas retrouvée.
"""

from __future__ import annotations

from typing import Any, Callable


def _project_text(content: Any) -> str:
    """Projection texte d'un contenu de message, sans jamais lever.

    Accepte un ``str``, ``None`` (rend ``""``) ou une liste de blocs (dicts portant une clé
    ``text``, ou de type ``"text"``/``"input_text"``). Rend ``""`` pour tout le reste.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            if block.get("type") in ("text", "input_text"):
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
        return "".join(parts)
    return ""


def _identity_index(wire_messages: list[dict], target: Any) -> int | None:
    """Indice (par identité d'objet) de la dernière apparition de ``target`` dans le fil."""
    found: int | None = None
    for i, w in enumerate(wire_messages):
        if w is target:
            found = i
    return found


def _level_uid(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    uid_a = a.get("message_uid")
    uid_b = b.get("message_uid")
    return uid_a is not None and uid_b is not None and uid_a == uid_b


def _level_tool(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    role_a, role_b = a.get("role"), b.get("role")
    if role_a is None or role_b is None or role_a != role_b:
        return False
    tc_a, tc_b = a.get("tool_call_id"), b.get("tool_call_id")
    return tc_a is not None and tc_b is not None and tc_a == tc_b


def _level_text(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    role_a, role_b = a.get("role"), b.get("role")
    if role_a is None or role_b is None or role_a != role_b:
        return False
    return _project_text(a.get("content"))[:80] == _project_text(b.get("content"))[:80]


def _find_sequence(
    wire_messages: list[dict], window: list[dict], level: Callable[[Any, Any], bool],
) -> int | None:
    """Cherche ``window`` en partant de la fin ; rend la coupe (dernier + 1) ou ``None``."""
    n = len(wire_messages)
    m = len(window)
    for start in range(n - m, -1, -1):
        if all(level(wire_messages[start + k], window[k]) for k in range(m)):
            return start + m
    return None


def _cut_by_identity(wire_messages: list[dict], window: list[dict]) -> int | None:
    """Cascade niveau 1 : tout se retrouve par identité d'objet, index croissants et contigus."""
    indices: list[int | None] = [_identity_index(wire_messages, wm) for wm in window]
    if any(i is None for i in indices):
        return None
    idx = [i for i in indices if i is not None]
    for k in range(len(idx) - 1):
        if idx[k + 1] != idx[k] + 1:
            return None
    return idx[-1] + 1


def wire_cut(wire_messages: list[dict], window: list[dict]) -> int | None:
    """Index exclusif dans ``wire_messages`` couvrant jusqu'à ``window[-1]`` inclus, ou ``None``.

    La coupe est l'index du message suivant le dernier message de la fenêtre dans le fil : la
    sous-liste couverte va de la première apparition retenue jusqu'à ``window[-1]`` inclus.

    La localisation suit une cascade de priorité stricte : dès qu'un niveau rend une séquence
    complète, il gagne et les suivants ne sont jamais consultés.

    1. **Identité d'objet** : si *tous* les messages de ``window`` se retrouvent par ``is`` dans
       le fil avec des index croissants et contigus, rendre ``index_du_dernier + 1`` immédiatement,
       sans jamais considérer le contenu. C'est le cas réel : la fenêtre est une tranche du
       transcript et le fil tient des copies de surface (``list(messages)``, pas ``deepcopy`` — cf.
       ``state``), donc les mêmes objets. Cela désambiguïse les fils où tous les contenus sont
       identiques : la coupe est exacte, pas « n'importe où où le texte coïncide ».
    2. **``message_uid``** présent des deux côtés, identique, sur toute la séquence.
    3. **Couple ``(role, tool_call_id)``** quand il existe, sur toute la séquence.
    4. **En dernier recours**, la séquence par ``(role, 80 premiers caractères de la projection
       texte)``, cherchée en partant de la fin (dernière occurrence gagnante) : dans une session,
       des messages identiques se répètent (deux « Continue. » successifs), la vraie frontière est
       la dernière occurrence, contiguë à la suite encore fraîche du transcript.

    Rend ``None`` si la fenêtre ou le fil est vide, si la séquence n'est pas retrouvée, ou dès
    qu'une entrée invalide empêche un alignement fiable. Ne lève jamais.
    """
    if not isinstance(wire_messages, list) or not isinstance(window, list):
        return None
    if not wire_messages or not window:
        return None
    n = len(wire_messages)
    m = len(window)
    if m > n:
        return None

    # Niveau 1 : identité d'objet, en priorité absolue.
    cut = _cut_by_identity(wire_messages, window)
    if cut is not None:
        return cut

    # Niveaux 2 à 4, chacun cherché en partant de la fin.
    for level in (_level_uid, _level_tool, _level_text):
        cut = _find_sequence(wire_messages, window, level)
        if cut is not None:
            return cut
    return None

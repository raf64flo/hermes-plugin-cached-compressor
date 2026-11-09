"""``wire_cut`` : retrouver la position du dernier message de la fenêtre dans le fil capturé.

La coupe est exclusive : index (dans le fil) du message suivant la fin de la fenêtre.
Ces tests couvrent la règle de priorité de comparaison et la recherche par la fin.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alignment import wire_cut  # noqa: E402


def msg(role: str, content: object, **extra: object) -> dict:
    d: dict = {"role": role, "content": content}
    d.update(extra)
    return d


def test_nominal_coupe_decalee_du_message_systeme() -> None:
    # fil = [system] + transcript de 10 messages ; fenêtre = les 4 derniers.
    transcript = [msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(10)]
    wire = [msg("system", "sys")] + transcript
    window = transcript[-4:]
    # transcript commence à l'index 1 ; les 4 derniers sont aux index 7..10 -> coupe 11.
    assert wire_cut(wire, window) == 11


def test_identite_objet_coupe_exacte_contenus_identiques() -> None:
    # Contenus identiques partout : seule l'identité d'objet désambiguïse.
    m0, m1, m2, m3, m4 = [msg("user", "x") for _ in range(5)]
    wire = [m0, m1, m2, m3, m4]
    window = [m2, m3]
    assert wire_cut(wire, window) == 4


def test_prefill_insere_decale_la_coupe_de_deux() -> None:
    transcript = [msg("user", f"q{i}") for i in range(5)]
    wire = [msg("system", "sys"), msg("user", "prefill")] + transcript
    window = transcript[-2:]
    # transcript aux index 2..6 ; les 2 derniers aux index 5..6 -> coupe 7.
    assert wire_cut(wire, window) == 7


def test_frontiere_absente_du_fil_rend_none() -> None:
    u0 = msg("user", "u0")
    a_empty = msg("assistant", "")  # écarte par le sanitizer, absent du fil
    u1 = msg("user", "u1")
    a1 = msg("assistant", "a1")
    u2 = msg("user", "u2")
    wire = [u0, u1, a1, u2]
    # frontière de la fenêtre = message tombé, absent du fil -> None
    assert wire_cut(wire, [u1, a1, a_empty]) is None
    # fenêtre qui débute par le message tombé -> introuvable -> None
    assert wire_cut(wire, [a_empty, u1, a1, u2]) is None


def test_occurence_dupliquee_vise_la_derniere() -> None:
    m_a = msg("user", "a")
    dup1 = msg("user", "same")
    m_b = msg("user", "b")
    dup2 = msg("user", "same")
    m_c = msg("user", "c")
    wire = [m_a, dup1, m_b, dup2, m_c]
    # fenêtre réduite à l'occurrence : la dernière gagne (index 3 -> coupe 4).
    assert wire_cut(wire, [dup2]) == 4
    # fenêtre suivie par les messages suivants : seule la seconde est candidate (coupe 5).
    assert wire_cut(wire, [dup2, m_c]) == 5


def test_entrees_degenerees_ne_levent_pas() -> None:
    wire = [msg("user", "u1"), msg("assistant", "a1")]
    assert wire_cut([], [msg("user", "x")]) is None
    assert wire_cut(wire, []) is None
    assert wire_cut([], []) is None
    assert wire_cut(None, [msg("user", "x")]) is None  # type: ignore[arg-type]
    assert wire_cut(wire, None) is None  # type: ignore[arg-type]
    # fenêtre plus longue que le fil : impossible
    long_window = [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u3")]
    assert wire_cut(wire, long_window) is None
    # contenus None et listes de blocs texte
    m_none = msg("user", None)
    m_blocks = msg("user", [{"type": "text", "text": "bonjour"}, {"text": " monde"}])
    wire2 = [m_none, m_blocks]
    assert wire_cut(wire2, [m_none]) == 1
    assert wire_cut(wire2, [m_blocks]) == 2
    # contenu totalement bizarre ne lève pas (même objet -> identité)
    weird = msg("user", 12345)
    assert wire_cut([weird], [weird]) == 1

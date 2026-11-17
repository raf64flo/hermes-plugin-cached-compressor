"""La raison du raté de cache, mesurée par appel et écrite dans la ligne existante.

Sur les routes qui rapportent le cache, 27 des 43 appels de résumé en mode préfixe ont rendu
``cached=0`` alors que le tour principal venait de lire 92 à 99 % de son préfixe, 48 à 428 secondes
plus tôt. Cette ligne ne disait pas *pourquoi* : route muette, préfixe non aligné, entrée expirée ?
Ces tests fixent les quatre étiquettes qui séparent ces causes, et vérifient qu'aucun compteur
parallèle n'est né : le moniteur de ``cron/monitor_cached_compressor.py`` continue de lire
``model=``, ``prompt=``, ``cached=``, ``route=`` dans la même ligne.
"""

from __future__ import annotations

import logging
import re
import time
from types import SimpleNamespace

import pytest
from support import load_plugin

_plugin = load_plugin()
TRACKER = _plugin.state.TRACKER  # type: ignore[attr-defined]
SLOT = _plugin.state.SLOT  # type: ignore[attr-defined]
MIN_PROMPT_TOKENS = _plugin.cacheability.MIN_PROMPT_TOKENS  # type: ignore[attr-defined]

# La forme exacte que lit le moniteur fusionné (MR hermes-optimizer!28).
RE_MONITEUR = re.compile(r"model=(\S+) prompt=(\d+) cached=(\d+) route=(\S+)")
PREFIX = "Compression summary on session prefix:"


class _Compressor:
    _active_compression_telemetry = None


def setup_function() -> None:
    TRACKER.clear()
    SLOT.clear()
    _plugin.cacheability.reset_tuning()


def _wire(tools: list | None = None, age_s: float = 12.0) -> dict:
    messages = [{"role": "system", "content": "S" * 800},
                {"role": "user", "content": "U" * 400}]
    SLOT.record(request={"messages": messages, "tools": tools or []}, request_messages=messages,
                model="deepseek-v4.1-flash", provider="custom",
                base_url="https://opencode.ai/inference/go/openai/v1",
                api_mode="chat_completions", session_id="s-1", conversation_history=messages)
    wire = SLOT.snapshot("s-1")
    wire["recorded_at"] = time.time() - age_s
    return wire


def _reponse(prompt: int, cached: int | None, details: bool = True) -> SimpleNamespace:
    usage: dict = {"prompt_tokens": prompt, "completion_tokens": 10}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached} if details else {}
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="R"))])


def _ligne(caplog, wire: dict, response: SimpleNamespace, sent: list | None = None) -> str:
    caplog.set_level(logging.INFO, logger=_plugin.__name__)
    _plugin._record_cache_read(_Compressor(), response, wire, sent=sent or wire["messages"])
    lignes = [rec.getMessage() for rec in caplog.records if PREFIX in rec.getMessage()]
    assert lignes, "aucune ligne de diagnostic emise"
    return lignes[-1]


def test_le_moniteur_lit_toujours_la_ligne(caplog) -> None:
    """Contrat de non-doublon : les quatre champs du moniteur restent en tete, les autres suivent."""
    ligne = _ligne(caplog, _wire(), _reponse(151_258, 0))
    match = RE_MONITEUR.search(ligne)
    assert match, ligne
    assert match.group(1) == "deepseek-v4.1-flash"
    assert int(match.group(2)) == 151_258
    assert int(match.group(3)) == 0
    assert match.group(4) == "custom"


def test_rate_apres_une_route_chaude_est_nomme(caplog) -> None:
    """La boucle principale vient de lire 99 % : la ligne le dit, et nomme le rate.

    C'est la situation mesurée en production : 48 à 428 s après un appel principal servi à 92-99 %,
    le résumé lit 0. ``prior_main`` porte cette evidence par appel, il ne reste plus d'hypothèse
    « route froide ».
    """
    wire = _wire()
    route = ("custom", "deepseek-v4.1-flash", "https://opencode.ai/inference/go/openai/v1")
    for _ in range(3):
        TRACKER.record(route, MIN_PROMPT_TOKENS + 250_000,
                       int((MIN_PROMPT_TOKENS + 250_000) * 0.99),
                       origin=_plugin.cacheability.ORIGIN_MAIN)
    ligne = _ligne(caplog, wire, _reponse(151_258, 0))
    assert "reason=miss_first" in ligne      # premiere compaction sur cette route
    assert "prior_main=0.99" in ligne
    ligne = _ligne(caplog, wire, _reponse(151_258, 0))
    assert "reason=miss_low" in ligne        # la compaction precedente n'a rien lu non plus
    assert "prior_compaction=0.0" in ligne


def test_rate_intermittent_quand_la_compaction_servait(caplog) -> None:
    """Compactions servies puis un raté : ``miss_warm`` isole le raté intermittent."""
    wire = _wire()
    route = ("custom", "deepseek-v4.1-flash", "https://opencode.ai/inference/go/openai/v1")
    TRACKER.record(route, 156_582, 156_032, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    ligne = _ligne(caplog, wire, _reponse(151_258, 0))
    assert "reason=miss_warm" in ligne
    assert "prior_compaction=0.996" in ligne


def test_tete_servie_est_un_partiel(caplog) -> None:
    """8 064 tokens servis sur 151 258 : seul le debut du prefixe etait en cache."""
    ligne = _ligne(caplog, _wire(), _reponse(151_258, 8_064))
    assert "reason=partial:5" in ligne


def test_prefixe_servi(caplog) -> None:
    ligne = _ligne(caplog, _wire(), _reponse(156_582, 156_032))
    assert "reason=served" in ligne


def test_usage_muet_distingue_du_zero_explicite(caplog) -> None:
    """Champ absent = route muette ; champ present a 0 = le provider a repondu zero. Deux causes."""
    muet = _ligne(caplog, _wire(), _reponse(151_258, None, details=False))
    assert "reason=usage_mute" in muet and "usage=none" in muet
    TRACKER.clear()  # chaque ligne part d'une route inconnue, pour comparer les etiquettes seules
    explicite = _ligne(caplog, _wire(), _reponse(151_258, 0))
    assert "reason=miss_cold" in explicite
    assert "usage=prompt_tokens_details.cached_tokens" in explicite
    racine = _ligne(caplog, _wire(), SimpleNamespace(
        usage={"prompt_tokens": 151_258, "prompt_cache_hit_tokens": 0}, choices=[]))
    assert "usage=usage.prompt_cache_hit_tokens" in racine


def test_l_age_du_payload_est_trace(caplog) -> None:
    """Test du TTL : l'ecart entre la capture du fil et l'appel de resume, par appel."""
    ligne = _ligne(caplog, _wire(age_s=48.0), _reponse(151_258, 0))
    assert re.search(r"wire_age_s=4[89]\.", ligne), ligne


def test_un_doublon_d_outil_est_visible_avant_l_envoi(caplog) -> None:
    """Le client auxiliaire dedoublonne les noms : un doublon casse l'alignement des octets."""
    doublon = {"type": "function", "function": {"name": "read_file", "description": "d",
                                                "parameters": {"type": "object", "properties": {}}}}
    ligne = _ligne(caplog, _wire(tools=[doublon, doublon]), _reponse(151_258, 0))
    assert "tools=2/1" in ligne
    unique = {"type": "function", "function": {"name": "terminal", "description": "d",
                                               "parameters": {"type": "object", "properties": {}}}}
    ligne = _ligne(caplog, _wire(tools=[doublon, unique]), _reponse(151_258, 0))
    assert "tools=2/2" in ligne


def test_les_tailles_envoyees_et_capturees_sont_tracees(caplog) -> None:
    wire = _wire()
    envoye = wire["messages"][:1] + [{"role": "user", "content": "instruction"}]
    ligne = _ligne(caplog, wire, _reponse(151_258, 0), sent=envoye)
    assert f"wire_chars={len('S' * 800) + len('U' * 400)}" in ligne
    assert f"payload_chars={len('S' * 800) + len('instruction')}" in ligne


def test_le_taux_de_rentabilite_voit_enfin_les_compactions() -> None:
    """Le seuil ne pouvait pas se declencher : les hits de la boucle diluaient le taux observe.

    Mesure de production : 99 % sur la boucle, 0 % sur les compactions, et le mode prefixe reste
    arme. Le taux global vaut alors 0,8 a 0,9 et depasse tout seuil realiste.
    """
    route = ("custom", "deepseek-v4.1-flash", "https://opencode.ai/inference/go/openai/v1")
    for _ in range(4):
        TRACKER.record(route, 260_000, 257_000, origin=_plugin.cacheability.ORIGIN_MAIN)
    TRACKER.record(route, 151_258, 0, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    TRACKER.record(route, 156_552, 0, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    assert TRACKER.observed_ratio(route) > 0.6  # le taux melange reste trompeur
    assert TRACKER.observed_ratio(route, origin=_plugin.cacheability.ORIGIN_COMPRESSION) == 0.0
    _plugin.cacheability.set_min_ratio(0.6)
    assert TRACKER.prefix_allowed(route) is False


def test_un_hit_de_la_boucle_ne_remet_pas_les_strikes_de_compaction() -> None:
    route = ("custom", "deepseek-v4.1-flash", "https://opencode.ai/inference/go/openai/v1")
    TRACKER.record(route, 151_258, 0, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    TRACKER.record(route, 151_258, 0, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    assert TRACKER.prefix_strikes(route) == _plugin.cacheability.MAX_STRIKES
    TRACKER.record(route, 260_000, 258_000, origin=_plugin.cacheability.ORIGIN_MAIN)
    assert TRACKER.prefix_strikes(route) == _plugin.cacheability.MAX_STRIKES


def test_sans_seuil_le_rate_de_compaction_ne_desactive_rien() -> None:
    """Comportement par defaut inchange, et la dilution est characterisee noir sur blanc.

    Deux compactions ratées arment bien deux strikes, mais le hit suivant de la boucle principale
    les remet à zéro : sans seuil, la route garde le mode préfixe indéfiniment. C'est pourquoi le
    réglage ``min_cache_ratio`` existe, et pourquoi il doit lire le taux de la compaction.
    """
    route = ("custom", "m", "u")
    for _ in range(2):
        TRACKER.record(route, 151_258, 0, origin=_plugin.cacheability.ORIGIN_COMPRESSION)
    assert TRACKER.prefix_allowed(route) is False
    TRACKER.record(route, 260_000, 258_000, origin=_plugin.cacheability.ORIGIN_MAIN)
    assert TRACKER.prefix_allowed(route) is True  # le hit de la boucle efface les strikes
    assert TRACKER.prefix_strikes(route) == _plugin.cacheability.MAX_STRIKES

    _plugin.cacheability.set_min_ratio(0.6)
    assert TRACKER.prefix_allowed(route) is False  # avec un seuil, la compaction tranche

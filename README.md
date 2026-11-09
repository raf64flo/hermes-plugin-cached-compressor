# cached-compressor

Moteur de contexte Hermes qui compacte la session **au prix du cache provider**, avec le modèle de
la session en cours plutôt qu'un modèle auxiliaire fixé.

## Le problème

Le compresseur intégré sérialise la fenêtre dans un prompt texte neuf et l'envoie au modèle
auxiliaire de configuration. Le provider principal ne voit donc jamais cette requête : il repaie un
prefill complet au tour suivant, et le prompt du résumé est presque aussi gros que la session.

## L'idée

Le préfixe de session est déjà en cache chez le provider. On n'envoie donc pas un résumé du texte :
on renvoie le préfixe réel, tronqué, suivi d'une instruction de synthèse. Seuls les tokens de sortie
sont au plein tarif ; le reste est facturé au tarif de lecture de cache.

## Mesures (2026-11-09, `nous` / `deepseek/deepseek-v4.1-flash-1017`)

Historique de 20 987 tokens, 162 messages, queue de 12 messages protégée, payload construit par le
chemin réel de Hermes (`_iteration_summary_api_messages` puis `_build_api_kwargs_for_mode`).

    requête                          prompt   cached   hit
    W  writer, chemin réel           20 987        0   0,0 %   (froid, normal)
    A  reader, chemin réel           19 690   19 584  99,5 %   conception validée
    B  reader, reconstruction brute  19 803        0   0,0 %   artefact d'une mesure naïve

    effectif A = 106 + 0,1 x 19 584 = 2 064 contre 22 346 au plein tarif, soit un ratio de 0,092

L'enseignement principal est B : une reconstruction artisanale du transcript tombe à 0 %. Le strip
des contenus, la canonisation des arguments de `tool_calls` et le prompt système préfixé à l'envoi
suffisent à invalider le préfixe. D'où la règle du projet : **on ne reconstruit pas le fil, on
capture celui qui vient d'être envoyé.**

## Architecture

Deux points d'entrée dans le même paquet, tous deux par `register(ctx)` :

- moteur de contexte (`context.engine: cached-compressor`) : `CachedCompressor` sous-classe le
  compresseur intégré et ne change que la requête de résumé ;
- plugin ordinaire (`plugins.enabled`) : enregistre le hook `pre_api_request`, qui reçoit le payload
  fil final (`agent/turn_api_request.py:69-113`), après assemblage, sanitizers et plan de cache.

`state.PayloadSlot` garde le dernier payload observé avec sa route : c'est elle qui rend le modèle
dynamique, puisque c'est celui qui vient de répondre dans la session.

## Installation

    ln -s ~/local/git/ia/hermes-plugin-cached-compressor ~/.hermes/plugins/cached-compressor
    # config.yaml
    context:
      engine: cached-compressor
    plugins:
      enabled: [cached-compressor]

`plugins.isolation: host` est incompatible : la sous-classe de `ContextCompressor` ne peut pas être
importée dans ce mode. Le défaut `in_process` (`hermes_cli/config_defaults.py:1730`) est requis.

## État

Phases 0 et 1 terminées. Le cœur (phase 3) est écrit et testé en unité :

- `state.PayloadSlot` capture le payload fil et sa route à chaque requête (`pre_api_request`) ;
- `summary.reader_messages` tronque ce payload à la frontière de fenêtre et y ajoute l'instruction,
  avec deux chemins d'alignement : par index (déterministe, insensible à l'élagage) puis par
  séquence ;
- `CachedCompressor._build_summary_prompt` remplace la recopie des tours par un pointeur, ce qui
  conserve la consigne du parent (budget, gabarit, focus) sans repayer le texte ;
- `CachedCompressor._call_summary_llm` épingle la route de la session, garde
  `aux_interrupt_protection`, et reporte l'échec par `RuntimeError` pour laisser la machinerie du
  parent faire son repli ;
- le hit réel est journalisé et posé dans la télémétrie de compaction (`cache_read_tokens`,
  `cache_read_ratio`).

Reste à faire : la mesure en session réelle (tâche 3.3), les garde-fous de configuration, le test
de contrat des API privées et la documentation d'activation. Voir le plan
`~/Téléchargements/llm-ia/.hermes/plans/2026-11-08_231044-cache-aware-compaction-plugin.md`.

Limite connue : un seul slot, donc une session à la fois ; le clé par `session_id` est à faire.

## Activation

Deux entrées, les deux nécessaires :

    hermes config set context.engine cached-compressor   # le moteur de compaction
    hermes plugins enable cached-compressor              # le hook pre_api_request

Le paquet est alors chargé deux fois, sous deux noms de module. Le slot de payload est donc rangé
dans un module sentinelle de `sys.modules` (`state._shared_slot`), sinon le hook écrirait dans un
slot que le moteur ne lit pas.

## Mesure

`probes/cache_hit_live.py` (vrai provider, vraie assemblée de requête) :

    writer (session)   prompt=20987  cached=20864  hit=99.4%
    reader (plugin)    prompt=21267  cached=21248  hit=99.91%   coût=0.00166 $

Concrètement : sur ~21 000 tokens de contexte, la compaction n'en repaie que ~19. Sans le
`main_runtime` de session, le même code mesurait 0 % — la requête partait hors du contexte de la
session.

## Tests

    python -m pytest tests -q

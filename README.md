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

Phase 1 : squelette, manifeste, contrat de chargement testé. La capture est branchée ; la
construction de la requête de résumé (`wire_cut`, replis) reste à faire, voir le plan
`~/Téléchargements/llm-ia/.hermes/plans/2026-11-08_231044-cache-aware-compaction-plugin.md`.

Limite connue : un seul slot pour l'instant, donc une session à la fois. Le clé par `session_id`
arrive avec la tâche 3.2.

## Tests

    cd ~/.hermes/hermes-agent && python -m pytest ~/local/git/ia/hermes-plugin-cached-compressor/tests

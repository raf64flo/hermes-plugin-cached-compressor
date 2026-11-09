"""Contrat avec les API non publiques de Hermes.

Le plugin ne peut pas rester sur les surfaces publiques : un moteur de contexte doit dériver de
``ContextCompressor`` et poster sa requête de résumé par ``call_llm``
(``__init__.py`` importe ces symboles). Le catalogue des plugins l'interdit d'ailleurs explicitement
(règle 9), ce qui fait de cette dépendance un risque assumé, à contenir plutôt qu'à nier.

Deux garde-fous :

- ``problems()`` énumère ce qui manque ou a changé de signature, et ``register`` refuse de
  s'enregistrer quand la liste n'est pas vide : Hermes garde alors son compresseur intégré, et le
  message dit pourquoi ;
- ``tests/test_contract.py`` échoue dès qu'un symbole requis disparaît, sur la version minimale
  déclarée et sur ``main`` (matrice de la CI).

La signature est vérifiée là où elle compte : un renommage de paramètre dans ``compress`` casserait
l'appel sans qu'aucun import ne bronche.
"""

from __future__ import annotations

import importlib
import inspect

#: (module, attribut, méthode de l'attribut ou None, paramètres exigés ou None)
REQUIRED: tuple[tuple[str, str, str | None, tuple[str, ...] | None], ...] = (
    ("agent.context_compressor", "ContextCompressor", "compress", ("messages",)),
    ("agent.context_compressor", "_is_refusal_response", None, ("content",)),
    ("agent.auxiliary_client", "call_llm", None, None),
    ("agent.auxiliary_client", "extract_content_or_reasoning", None, None),
    ("agent.auxiliary_client", "aux_interrupt_protection", None, None),
    ("agent.auxiliary_client", "AuxiliaryExplicitCancellation", None, None),
    ("agent.chat_completion_helpers", "_iteration_summary_api_messages", None, None),
    ("agent.chat_completion_helpers", "_build_api_kwargs_for_mode", None, None),
)


def problems() -> list[str]:
    """Tout ce qui empêche le plugin de fonctionner sur ce Hermes, en clair et en une passe."""
    found: list[str] = []
    for module_name, attribute, method, params in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            found.append(f"{module_name}: import impossible ({error!r})")
            continue
        target = getattr(module, attribute, None)
        if target is None:
            found.append(f"{module_name}.{attribute} absent")
            continue
        if method is not None:
            target = getattr(target, method, None)
            if target is None:
                found.append(f"{module_name}.{attribute}.{method} absent")
                continue
        if params:
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            missing = [name for name in params if name not in signature.parameters]
            if missing:
                found.append(f"{module_name}.{attribute} n'accepte plus {', '.join(missing)}")
    return found

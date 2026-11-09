import os
import sys
from pathlib import Path

# Le venv de test n'est pas celui de Hermes : le plugin importe ``agent.*``, il faut le dépôt source.
# HERMES_SOURCE permet à la CI de pointer un clone au lieu d'une installation locale.
HERMES = Path(os.environ.get("HERMES_SOURCE", Path.home() / ".hermes" / "hermes-agent"))
if str(HERMES) not in sys.path:
    sys.path.insert(0, str(HERMES))

# En mode importlib, le dossier des tests n'est pas sur sys.path : ``support`` serait introuvable.
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

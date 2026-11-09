"""Chargement du plugin une seule fois, sous un nom de module stable pour tous les tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERMES = Path(os.environ.get("HERMES_SOURCE", Path.home() / ".hermes" / "hermes-agent"))
MODULE_NAME = "cached_compressor_under_test"


def load_plugin() -> object:
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    if str(HERMES) not in sys.path:
        sys.path.insert(0, str(HERMES))
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module

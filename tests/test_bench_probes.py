"""The bench probes' own guards run in CI: the anonymizer canary is a safety property.

The canary must keep failing closed: it plants one sentinel per masking category in a synthetic
transcript and refuses to run a benchmark when any survives. A regression there would publish
identifiers, so it is tested like production code.

The two probes that import the Hermes package are exercised through a subprocess: importing
``hermes_bootstrap`` re-executes the interpreter inside the Hermes virtualenv, which would kill the
pytest process itself. The anonymizer is pure standard library and is imported in-process.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_probe(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("HERMES_SOURCE", os.path.expanduser("~/.hermes/hermes-agent"))
    return subprocess.run([sys.executable, str(REPO / "bench" / args[0]), *args[1:]],
                          capture_output=True, text=True, env=env, timeout=300, cwd=str(REPO))


def test_canary_ne_laisse_passer_aucun_sentinel():
    module = _load("bench_anonymize_ci", "anonymize.py")
    assert module.canary() == 0


def test_canary_detecte_une_regression(monkeypatch):
    """An anonymizer that masks nothing must fail the canary, not pass silently."""
    module = _load("bench_anonymize_ci_neg", "anonymize.py")
    monkeypatch.setattr(module.Anonymizer, "text", lambda self, value: value)
    assert module.canary() != 0


def test_no_content_destroyed_by_masking():
    """The control text survives masking: the anonymizer redacts, it does not blank the corpus."""
    module = _load("bench_anonymize_ci_ctrl", "anonymize.py")
    masked = module.Anonymizer(names=["canaryperson"]).messages(module.CANARY["messages"])
    assert any("calibration sweep" in m["content"] for m in masked)


@pytest.mark.parametrize("script,marker", [
    ("fact_recall.py", "self-test ok"),
    ("hetero_bench.py", "self-test ok"),
])
def test_probe_self_tests(script: str, marker: str):
    result = _run_probe(script, "--self-test")
    assert result.returncode == 0, result.stderr[-2000:]
    assert marker in result.stdout, result.stdout[-2000:]

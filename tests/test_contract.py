"""Le contrat avec les API privées de Hermes, vérifié au lieu d'être supposé.

Le plugin dérive de ``ContextCompressor`` et poste par ``call_llm`` : deux symboles non publics. Le
test échoue à la première version de Hermes qui les bouge, et ``register`` refuse alors de
s'enregistrer plutôt que de casser au milieu d'une compaction.
"""

from __future__ import annotations

import pytest
from support import load_plugin

_plugin = load_plugin()
contract = _plugin.contract  # type: ignore[attr-defined]


def test_contrat_respecte_sur_ce_hermes():
    """Sur la version installée, tout ce que le plugin utilise est présent et signé comme attendu."""
    assert contract.problems() == []


def test_harnais_verifie_a_part():
    """Le harnais de mesure a son propre contrat : il ne conditionne pas l'installation du moteur."""
    pytest.importorskip("agent.chat_completion_helpers",
                        reason="psutil ou une dépendance de mesure absente de cette installation")
    assert contract.harness_problems() == []


def test_un_symbole_absent_est_signale(monkeypatch):
    monkeypatch.setattr(contract, "REQUIRED", contract.REQUIRED + (
        ("agent.module_qui_n_existe_pas", "attendu", None, None),))
    found = contract.problems()
    assert any("agent.module_qui_n_existe_pas" in item for item in found)


def test_register_refuse_un_contrat_rompu(monkeypatch, caplog):
    """Contrat rompu : on ne s'enregistre pas du tout, et le message dit quoi corriger."""
    class Ctx:
        def __init__(self) -> None:
            self.registered: list[str] = []

        def register_context_engine(self, engine: object) -> None:
            self.registered.append("engine")

        def register_hook(self, name: str, callback: object) -> None:
            self.registered.append(name)

    ctx = Ctx()
    monkeypatch.setattr(contract, "problems", lambda: ["agent.context_compressor absent"])
    _plugin.register(ctx)
    assert ctx.registered == []
    assert any("incompatible Hermes build" in record.getMessage() for record in caplog.records)

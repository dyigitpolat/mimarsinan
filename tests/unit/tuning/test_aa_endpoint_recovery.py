"""[PR14b] the AA endpoint-recovery leg: funded by aa_endpoint_recovery_steps."""

import types

from mimarsinan.tuning.tuners import activation_adaptation_tuner as aa_mod
from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)


def _stub(config):
    return types.SimpleNamespace(
        pipeline=types.SimpleNamespace(config=config),
    )


def test_hook_is_inert_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        aa_mod, "run_endpoint_recovery", lambda *a, **k: calls.append((a, k)),
    )
    ActivationAdaptationTuner._post_stabilization_hook(_stub({}))
    assert calls == []


def test_hook_funds_recovery_when_armed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        aa_mod, "run_endpoint_recovery", lambda tuner, **k: calls.append(k),
    )
    ActivationAdaptationTuner._post_stabilization_hook(
        _stub({"aa_endpoint_recovery_steps": 1200})
    )
    assert calls == [{"base_steps": 1200}]

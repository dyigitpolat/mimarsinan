"""SoftCoreMappingStep routes the negative-boundary knob to its two mechanisms."""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


class _StubTrainer:
    def __init__(self, batches):
        self._batches = batches

    def iter_validation_batches(self, n_batches):
        yield from self._batches[:n_batches]


def _step(config, *, batches=((torch.zeros(2, 3), torch.tensor([0, 1])),)):
    base = {
        "simulation_steps": 8,
        "device": "cpu",
        "spiking_mode": "lif",
        "cycle_accurate_lif_forward": True,
    }
    base.update(config)
    step = SoftCoreMappingStep(MockPipeline(config=base))
    step.trainer = _StubTrainer(list(batches))
    return step


@pytest.fixture
def spy(monkeypatch):
    calls = {}

    def _record(name):
        def _fn(*args, **kwargs):
            calls[name] = kwargs
            return {}

        return _fn

    import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as mod

    monkeypatch.setattr(mod, "apply_negative_boundary_policy", _record("policy"))
    return calls


def test_the_knob_selects_the_mechanism(spy):
    _step({"negative_value_shift": True})._apply_negative_boundary_policy(object())
    assert spy["policy"]["shift_enabled"] is True

    spy.clear()
    _step({"negative_value_shift": False})._apply_negative_boundary_policy(object())
    assert spy["policy"]["shift_enabled"] is False


def test_the_policy_always_runs(spy):
    """Neither position is a no-op: OFF must reach the mapper's subsume-forward
    path, never silently skip the boundary (the pre-round-5 behavior)."""
    _step({})._apply_negative_boundary_policy(object())
    assert "policy" in spy


def test_no_validation_batches_skips_the_policy(spy):
    _step({"negative_value_shift": True}, batches=())._apply_negative_boundary_policy(
        object()
    )
    assert spy == {}


def test_the_calibration_forward_follows_the_spiking_mode(spy):
    from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_mode

    _step({"spiking_mode": "ttfs"})._apply_negative_boundary_policy(object())
    assert spy["policy"]["forward_fn"] is calibration_forward_for_mode("ttfs")


# ── The mode gate (folded in from the retired test_negative_shift_gate.py) ──


@pytest.mark.parametrize("mode", ["rate", "bogus"])
@pytest.mark.parametrize("shift", [True, False])
def test_unsupported_spiking_mode_fails_loud(mode, shift):
    """The gate now guards BOTH positions: OFF also calibrates, so an
    unimplemented mode can no longer slip through as a silent no-op."""
    step = _step({"negative_value_shift": shift, "spiking_mode": mode})
    with pytest.raises(NotImplementedError, match="negative_value_shift"):
        step._apply_negative_boundary_policy(model=object())


@pytest.mark.parametrize(
    "mode", ["lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"]
)
@pytest.mark.parametrize("shift", [True, False])
def test_supported_spiking_modes_pass_the_gate(mode, shift, spy):
    step = _step({"negative_value_shift": shift, "spiking_mode": mode})
    step._apply_negative_boundary_policy(model=object())
    assert spy["policy"]["shift_enabled"] is shift

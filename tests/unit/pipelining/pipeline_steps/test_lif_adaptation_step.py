"""Unit tests for LIFAdaptationStep — legacy vs unified value-domain ramp.

LIF Adaptation ramps each chip-targeted perceptron's base activation toward
``LIFActivation`` and installs the chip-aligned cross-layer forward at finalize
(when cycle-accurate). The ramp itself is selectable:

- ``legacy_lif_blend_ramp`` (default, until verified): per-frame
  ``_CycleAccurateForward`` during the ramp (leaks off the continuous teacher
  at rate 0).
- unified (``legacy_lif_blend_ramp=False``): no ramp forward — the golden
  value-domain ``BlendActivation`` ramp.

Either way the deployed chip-aligned forward is installed at finalize.
"""

import pytest
import torch

from conftest import make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import (
    LIFAdaptationStep,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
    _ChipAlignedNFForward,
    _CycleAccurateForward,
)


def _seed_lif_step(mock_pipeline, *, cycle_accurate=True, legacy=True, target=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "lif"
    mock_pipeline.config["cycle_accurate_lif_forward"] = cycle_accurate
    mock_pipeline.config["legacy_lif_blend_ramp"] = legacy
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline.config.setdefault("simulation_steps", 8)
    mock_pipeline._target_metric = target

    mock_pipeline.seed("model", model, step_name="Activation Analysis")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")
    return model, am


def _run_step(mock_pipeline):
    step = LIFAdaptationStep(mock_pipeline)
    step.name = "LIF Adaptation"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


@pytest.mark.parametrize("legacy", [True, False])
def test_lif_active_and_chip_aligned_forward_installed(mock_pipeline, legacy):
    model, am = _seed_lif_step(mock_pipeline, cycle_accurate=True, legacy=legacy)
    _run_step(mock_pipeline)
    assert am.lif_active is True
    for p in model.get_perceptrons():
        assert p.base_activation.rate == pytest.approx(1.0)
        assert isinstance(p.base_activation.target_activation, LIFActivation)
    assert isinstance(model.__dict__.get("forward"), _ChipAlignedNFForward), (
        "cycle-accurate LIF must finalize on the chip-aligned forward "
        "regardless of which ramp ran"
    )


def test_unified_ramp_installs_no_cycle_accurate_ramp_forward(mock_pipeline):
    """The unified value-domain ramp must NOT route training through the
    per-frame ``_CycleAccurateForward`` (the leak); the genuine forward only
    appears at finalize."""
    model, _ = _seed_lif_step(mock_pipeline, cycle_accurate=True, legacy=False)
    step = _run_step(mock_pipeline)
    assert step.tuner._ramp_forward() is None
    assert not isinstance(model.__dict__.get("forward"), _CycleAccurateForward)


def test_non_cycle_accurate_leaves_class_forward(mock_pipeline):
    model, am = _seed_lif_step(mock_pipeline, cycle_accurate=False, legacy=True)
    _run_step(mock_pipeline)
    assert am.lif_active is True
    assert "forward" not in model.__dict__, (
        "non-cycle-accurate LIF must leave the pristine class forward"
    )


def test_default_ramp_is_value_domain(mock_pipeline):
    """With ``legacy_lif_blend_ramp`` unset, the tuner defaults to the golden
    value-domain ramp (no per-frame cascade forward during the ramp)."""
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "lif"
    mock_pipeline.config["cycle_accurate_lif_forward"] = True
    mock_pipeline.config.pop("legacy_lif_blend_ramp", None)
    mock_pipeline.config.setdefault("simulation_steps", 8)
    mock_pipeline._target_metric = 0.5
    mock_pipeline.seed("model", model, step_name="Activation Analysis")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")
    step = _run_step(mock_pipeline)
    assert step.tuner._legacy_blend_ramp is False
    assert step.tuner._ramp_forward() is None


def test_unified_ramp_makes_natural_blend_progress(mock_pipeline):
    """The value-domain ramp is non-destructive and progresses on its own
    (rate 0 == continuous teacher), like the golden LIF/synchronized ramp."""
    torch.manual_seed(7)
    _seed_lif_step(mock_pipeline, cycle_accurate=True, legacy=False)
    step = _run_step(mock_pipeline)
    assert step.tuner._natural_rate > 0.0

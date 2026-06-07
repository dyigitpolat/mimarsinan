"""Unit tests for LIFAdaptationStep — value-domain ramp + chip-aligned finalize.

LIF Adaptation ramps each chip-targeted perceptron's base activation toward
``LIFActivation`` in the value domain (the golden, non-destructive
``BlendActivation`` ramp: rate 0 == continuous teacher), then installs the
deployed chip-aligned cross-layer forward at finalize (when cycle-accurate).
"""

import pytest
import torch

from conftest import make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import (
    LIFAdaptationStep,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import _ChipAlignedNFForward


def _seed_lif_step(mock_pipeline, *, cycle_accurate=True, target=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "lif"
    mock_pipeline.config["cycle_accurate_lif_forward"] = cycle_accurate
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


def test_lif_active_and_chip_aligned_forward_installed(mock_pipeline):
    model, am = _seed_lif_step(mock_pipeline, cycle_accurate=True)
    _run_step(mock_pipeline)
    assert am.lif_active is True
    for p in model.get_perceptrons():
        assert p.base_activation.rate == pytest.approx(1.0)
        assert isinstance(p.base_activation.target_activation, LIFActivation)
    assert isinstance(model.__dict__.get("forward"), _ChipAlignedNFForward), (
        "cycle-accurate LIF must finalize on the chip-aligned forward"
    )


def test_ramp_is_value_domain(mock_pipeline):
    """The ramp runs in the value domain (no instance forward); the chip-aligned
    forward is installed only at finalize."""
    model, _ = _seed_lif_step(mock_pipeline, cycle_accurate=True)
    step = _run_step(mock_pipeline)
    assert step.tuner._ramp_forward() is None


def test_non_cycle_accurate_leaves_class_forward(mock_pipeline):
    model, am = _seed_lif_step(mock_pipeline, cycle_accurate=False)
    _run_step(mock_pipeline)
    assert am.lif_active is True
    assert "forward" not in model.__dict__, (
        "non-cycle-accurate LIF must leave the pristine class forward"
    )


def test_value_domain_ramp_makes_natural_blend_progress(mock_pipeline):
    """The value-domain ramp is non-destructive and progresses on its own
    (rate 0 == continuous teacher)."""
    torch.manual_seed(7)
    _seed_lif_step(mock_pipeline, cycle_accurate=True)
    step = _run_step(mock_pipeline)
    assert step.tuner._natural_rate > 0.0


def test_cycle_accurate_finalize_marks_lr_refind(mock_pipeline):
    """Finalize swaps in the chip-aligned forward, so stabilization must
    re-find the LR on the deployed dynamics (the ramp's cached LR is stale)."""
    _seed_lif_step(mock_pipeline, cycle_accurate=True)
    step = _run_step(mock_pipeline)
    assert step.tuner._stabilization_refinds_lr is True

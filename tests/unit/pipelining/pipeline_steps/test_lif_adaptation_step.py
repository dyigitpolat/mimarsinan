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
    """The value-domain ramp is non-destructive and adapts (rate 0 == continuous
    teacher): it commits at least one positive rate during natural adaptation,
    i.e. the blend rate is NOT pinned at exactly 0. (On a tiny untrained model the
    scheduler's specific committed rate is validation-noise-dependent, so the
    rate-pin guard is the trajectory-robust signal of a working blend.)"""
    torch.manual_seed(7)
    _seed_lif_step(mock_pipeline, cycle_accurate=True)
    step = _run_step(mock_pipeline)
    committed = [r.rate for r in step.tuner._cycle_log.records if r.outcome == "commit"]
    assert committed and max(committed) > 0.0, (
        "LIF blend made no natural progress — the rate is pinned at 0"
    )


def test_cycle_accurate_finalize_marks_lr_refind(mock_pipeline):
    """Finalize swaps in the chip-aligned forward, so stabilization must
    re-find the LR on the deployed dynamics (the ramp's cached LR is stale)."""
    _seed_lif_step(mock_pipeline, cycle_accurate=True)
    step = _run_step(mock_pipeline)
    assert step.tuner._stabilization_refinds_lr is True


class _ResidualStem(torch.nn.Module):
    """A host residual that skips the neural core and re-joins after a host
    Linear — the ViT block shape, minimum size."""

    def __init__(self, d=64, n=4):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, d)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(d, d)
        self.head = torch.nn.Linear(d, n)

    def forward(self, x):
        x = x.flatten(1)
        return self.head(x + self.fc2(self.act(self.fc1(self.norm(x)))))


def _seed_residual_model(mock_pipeline):
    from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = convert_torch_model(
        _ResidualStem(), (1, 8, 8), 4, device="cpu",
        encoding_layer_placement="offload",
    )
    assert heterogeneous_domain_joins(model.get_mapper_repr()), (
        "fixture must carry the mixed-domain seam"
    )
    _seed_lif_step(mock_pipeline, cycle_accurate=True)
    mock_pipeline.seed("model", model, step_name="Activation Analysis")
    return model


def test_mixed_domain_seam_gauge_is_established_before_the_twin(mock_pipeline):
    """A residual crossing the first neural core leaves a host op with two
    currencies; the step establishes the gauge so the chip-aligned LIF twin —
    and the deployed op — decode each source at its producer's gauge."""
    from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins

    model = _seed_residual_model(mock_pipeline)
    _run_step(mock_pipeline)
    repr_ = model.get_mapper_repr()
    assert not heterogeneous_domain_joins(repr_)
    seam = next(n for n in repr_.execution_order() if getattr(n, "name", None) == "add")
    assert seam.per_source_scales is not None and seam.output_scale is not None


def test_classifiable_graph_keeps_every_gauge_slot_empty(mock_pipeline):
    """The repair is scoped: a graph the domain map already classifies is left
    exactly as it was (byte-identity for every covered topology)."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    model, _ = _seed_lif_step(mock_pipeline, cycle_accurate=True)
    _run_step(mock_pipeline)
    for node in model.get_mapper_repr().execution_order():
        if isinstance(node, ComputeOpMapper):
            assert node.output_scale is None and node.per_source_scales is None

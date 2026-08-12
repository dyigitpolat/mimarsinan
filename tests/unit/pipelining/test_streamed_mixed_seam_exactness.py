"""Streamed exactness ACROSS a mixed wire/absolute seam: NF↔SCM window counts
hold at atol=0 when a host residual re-joins the branch that crossed a core.

This is the deployed half of the gauge-establishment contract. The twin decodes
each source of the join at its producer's gauge through the
``ScaleNormalizingWrapper`` its wrap slots describe; IR emission installs the
SAME wrapper and the hybrid executor hands it ``(1, 1)`` outer scales because it
owns its domain. If those two ever drifted apart, the residual would arrive in
the wrong currency and the counts would part — so the atol=0 gate is what turns
"one definition" into a measured fact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_streamed_nf_scm_exact_or_raise,
)
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model

T = 8
INPUT_SHAPE = (1, 1, 8)
NUM_CLASSES = 4


class _ResidualStem(nn.Module):
    """The transformer block shape at minimum size: the stem's value skips the
    neural core and re-joins after a host Linear."""

    def __init__(self, d: int = 8, n: int = NUM_CLASSES):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d, d)
        self.head = nn.Linear(d, n)

    def forward(self, x):
        x = x.flatten(1)
        return self.head(x + self.fc2(self.act(self.fc1(self.norm(x)))))


class _StreamedNFModel(nn.Module):
    """The pipeline's post-adaptation shape: ``forward`` IS the per-segment
    raw-cascade walk ``_ChipAlignedNFForward`` installs."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.repr_ = flow.get_mapper_repr()
        self._perceptrons = nn.ModuleList(list(self.repr_.get_perceptrons()))

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(self.repr_, T, LifSegmentPolicy())(x)


def _streamed_pipeline_stub():
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "spiking_family": "lif",
        "spiking_variant": "streamed",
        "simulation_steps": T,
        "input_shape": INPUT_SHAPE,
        "device": "cpu",
    })
    return SimpleNamespace(config=cfg)


def _seam_model():
    torch.manual_seed(0)
    flow = convert_torch_model(
        _ResidualStem(), INPUT_SHAPE, NUM_CLASSES, device="cpu",
        encoding_layer_placement="offload",
    ).eval()
    repr_ = flow.get_mapper_repr()
    assert heterogeneous_domain_joins(repr_), "fixture must carry the mixed seam"
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(1.0 + 0.4 * (i + 1)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif

    establish_gauge_for_mixed_domain_seams(flow, input_data_scale=1.0)
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    return flow, repr_, ir


def _samples(n=4):
    torch.manual_seed(1)
    return torch.rand(n, *INPUT_SHAPE)


class TestStreamedExactnessAtAMixedSeam:
    def test_deployed_seam_op_is_the_twin_s_own_wrapper(self):
        """One definition, not two: the module IR emission installs at the seam
        IS the composition the twin's host-value forward runs."""
        _, repr_, ir = _seam_model()
        seam = next(
            n for n in repr_.execution_order() if getattr(n, "name", None) == "add"
        )
        assert isinstance(seam._maybe_wrap_for_scales(), ScaleNormalizingWrapper)
        from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
            compute_op_owns_scale_domain,
        )
        from mimarsinan.mapping.ir import ComputeOp

        owning = [
            op for op in ir.nodes
            if isinstance(op, ComputeOp) and compute_op_owns_scale_domain(op)
        ]
        assert owning, "the deployed seam op must own its scale domain"

    def test_window_counts_exact_across_the_mixed_seam(self):
        flow, _, ir = _seam_model()
        assert_streamed_nf_scm_exact_or_raise(
            _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
        )

    def test_gate_has_teeth_on_this_fixture(self):
        """A threshold drift must trip the atol=0 gate here, so the exactness
        above is a measurement and not a vacuous pass."""
        flow, _, ir = _seam_model()
        core = ir.get_neural_cores()[-1]
        core.threshold = float(core.threshold) * 2.0
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), _StreamedNFModel(flow), ir, _samples(),
            )

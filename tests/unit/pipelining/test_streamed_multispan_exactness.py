"""[P6 plan §9] Multi-span streamed exactness: NF↔SCM window counts hold at atol=0
across a hybrid program with an INTERIOR host op (per-segment streaming)."""

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
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.verification.streamed import streamed_span_report_ir
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_streamed_nf_scm_exact_or_raise,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


class _HostRelay(nn.Module):
    """Order- and scale-preserving host op (the MaxPool seam without geometry)."""

    def forward(self, x):
        return x * 1.0


def _lif_perceptron(out_ch, in_features, theta, *, encoding=False):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.is_encoding_layer = encoding
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


def _multispan_model():
    """input(8) -> encode P0 -> P1 -> host relay -> P2 -> P3.

    IR topology: [P0 encode op][P1 core][relay op][P2 core][P3 core] — TWO
    neural segments with an interior host op; the second span carries two
    latency groups, so streaming-within-span is genuinely exercised."""
    torch.manual_seed(0)
    theta_enc, theta_p1 = 2.185, 0.5
    inp = InputMapper((8,))
    p0 = _lif_perceptron(8, 8, theta_enc, encoding=True)
    m0 = PerceptronMapper(inp, p0)
    p1 = _lif_perceptron(6, 8, theta_p1)
    p1.per_input_scales = torch.full((8,), float(theta_enc))
    m1 = PerceptronMapper(m0, p1)
    host = ComputeOpMapper(m1, _HostRelay(), input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(5, 6, 1.0)
    p2.per_input_scales = torch.full((6,), float(theta_p1))
    m2 = PerceptronMapper(host, p2)
    p3 = _lif_perceptron(3, 5, 0.75)
    m3 = PerceptronMapper(m2, p3)

    repr_ = ModelRepresentation(m3)
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    return repr_, ir


class _StreamedNFModel(nn.Module):
    """Shim with the pipeline's post-adaptation shape: ``forward`` IS the
    per-segment raw-cascade walk (what ``_ChipAlignedNFForward`` installs)."""

    def __init__(self, repr_):
        super().__init__()
        self.repr_ = repr_
        self._perceptrons = nn.ModuleList(list(repr_.get_perceptrons()))

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        driver = SegmentForwardDriver(self.repr_, T, LifSegmentPolicy())
        return driver(x)


def _streamed_pipeline_stub():
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "spiking_family": "lif",
        "spiking_variant": "streamed",
        "simulation_steps": T,
        "input_shape": (8,),
        "device": "cpu",
    })
    return SimpleNamespace(config=cfg)


class TestMultiSpanStreamedExactness:
    def test_fixture_is_genuinely_multi_span(self):
        _, ir = _multispan_model()
        report = streamed_span_report_ir(ir)
        assert report.segments == 2
        assert len(report.interior_host_ops) == 1

    def test_window_counts_exact_across_interior_host_op(self):
        repr_, ir = _multispan_model()
        model = _StreamedNFModel(repr_)
        samples = 3.0 * torch.rand(4, 8)
        assert_streamed_nf_scm_exact_or_raise(
            _streamed_pipeline_stub(), model, ir, samples,
        )

    def test_gate_has_teeth_on_the_second_span(self):
        """A threshold drift in the post-host span must trip the atol=0 gate."""
        repr_, ir = _multispan_model()
        model = _StreamedNFModel(repr_)
        samples = 3.0 * torch.rand(4, 8)
        last_core = ir.get_neural_cores()[-1]
        last_core.threshold = float(last_core.threshold) * 2.0
        with pytest.raises(NfScmParityError, match="streamed NF↔SCM"):
            assert_streamed_nf_scm_exact_or_raise(
                _streamed_pipeline_stub(), model, ir, samples,
            )

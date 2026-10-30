"""Bias-row splitting must not cost NF↔SCM exactness.

The bias the NF adds as ONE number is delivered on chip as k always-on rows.
That is only a re-encoding if the k rows reconstruct the same value the NF
used — under `firing_granularity='per_event'` the raster gate admits atol=0,
so a single ulp of reconstruction error is a different computation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.bias_rows import bias_rows_from_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_streamed_nf_scm_exact_or_raise,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

T = 4
BITS = 4


def _lif_perceptron(out_ch, in_features, theta, *, encoding=False):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.is_encoding_layer = encoding
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


def _bias_dominated_model(bias_gain):
    """input(8) -> encode P0 -> P1 -> P2, with P1/P2 biases that dominate their
    weights, so the weight-only projection prices them at several rows."""
    torch.manual_seed(0)
    theta_enc, theta_p1 = 2.0, 0.5
    inp = InputMapper((8,))
    p0 = _lif_perceptron(8, 8, theta_enc, encoding=True)
    m0 = PerceptronMapper(inp, p0)
    p1 = _lif_perceptron(6, 8, theta_p1)
    p1.per_input_scales = torch.full((8,), float(theta_enc))
    m1 = PerceptronMapper(m0, p1)
    p2 = _lif_perceptron(4, 6, 1.0)
    p2.per_input_scales = torch.full((6,), float(theta_p1))
    m2 = PerceptronMapper(m1, p2)

    with torch.no_grad():
        for p in (p1, p2):
            p.layer.weight.data.uniform_(-0.08, 0.08)
            p.layer.bias.data.uniform_(-bias_gain, bias_gain)

    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()
    return repr_


def _install_weight_only_grid(repr_):
    for p in repr_.get_perceptrons():
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)


class _StreamedNFModel(nn.Module):
    def __init__(self, repr_):
        super().__init__()
        self.repr_ = repr_
        self._perceptrons = nn.ModuleList(list(repr_.get_perceptrons()))

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        driver = SegmentForwardDriver(
            self.repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW)
        )
        return driver(x)


def _per_event_pipeline_stub():
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({
        "spiking_family": "lif",
        "spiking_variant": "streamed",
        "simulation_steps": T,
        "weight_bits": BITS,
        "firing_granularity": "per_event",
        "membrane_bits": 8,
        "input_shape": (8,),
        "device": "cpu",
        "has_bias": False,
        "cores": [{"max_axons": 128, "max_neurons": 128, "count": 64,
                   "has_bias": False}],
    })
    return SimpleNamespace(config=cfg)


def _mapped(repr_):
    q_max = (2 ** (BITS - 1)) - 1
    return IRMapping(
        q_max=float(q_max), firing_mode="Default", max_axons=128,
        max_neurons=128, hardware_bias=False,
    ).map(repr_)


def _run_gate(repr_, **soma):
    stub = _per_event_pipeline_stub()
    stub.config.update(soma)
    model = _StreamedNFModel(repr_)
    torch.manual_seed(3)
    samples = 2.0 * torch.rand(4, 8)
    assert_streamed_nf_scm_exact_or_raise(stub, model, _mapped(repr_), samples)


def _split_model(bias_gain):
    repr_ = _bias_dominated_model(bias_gain)
    _install_weight_only_grid(repr_)
    rows = [
        bias_rows_from_scales(p.bias_scale, p.parameter_scale)
        for p in repr_.get_perceptrons()
    ]
    assert max(rows) > 1, f"fixture must actually split; rows={rows}"
    return repr_


class TestWhereTheSplitIsExact:
    """k=1 is exact everywhere; k>1 is exact everywhere the soma does not
    threshold per arriving EVENT against a fixed-width membrane register."""

    def test_one_row_is_exact_at_the_event_serial_point(self):
        repr_ = _bias_dominated_model(0.02)
        _install_weight_only_grid(repr_)
        assert all(
            bias_rows_from_scales(p.bias_scale, p.parameter_scale) == 1
            for p in repr_.get_perceptrons()
        )
        _run_gate(repr_, membrane_bits=8, firing_granularity="per_event")

    def test_split_rows_are_exact_with_an_unbounded_membrane(self):
        _run_gate(_split_model(0.6), membrane_bits=0, firing_granularity="per_event")

    def test_split_rows_are_exact_under_per_cycle_firing(self):
        _run_gate(_split_model(0.6), membrane_bits=8, firing_granularity="per_cycle")


class TestTheEventSerialLimit:
    """MEASURED: the one combination where a k-row bias is not a value-preserving
    re-encoding. The NF adds the bias as ONE number; an event-serial soma with a
    membrane register delivers it as k separately-thresholded contributions.
    The mismatch count is invariant to the register's width and signedness, so
    this is the discipline, not a rail magnitude. WeightQuantizationStep refuses
    the combination up front; this is the evidence behind that refusal."""

    @pytest.mark.parametrize(
        "membrane_bits,membrane_signed", [(8, False), (16, False), (8, True)]
    )
    def test_split_rows_break_raster_exactness(self, membrane_bits, membrane_signed):
        with pytest.raises(NfScmParityError, match="RASTER exactness"):
            _run_gate(
                _split_model(0.6),
                membrane_bits=membrane_bits,
                membrane_signed=membrane_signed,
                firing_granularity="per_event",
            )

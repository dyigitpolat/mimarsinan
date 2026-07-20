"""[calculus §16.6] one-writer source currencies: an ARMED op's emitted gauge
(output_scale) is what consumers must decode by — the source-scale walk must
report it, refresh per_source_scales from it, and the coherence certificate
must fail loud on the (producer output_scale ↔ consumer per_source) pair.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper, ReshapeMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.scale_aware_boundaries import (
    propagate_boundary_input_scales,
    verify_boundary_currency_coherence,
)
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 32


def _prearmed(node: ComputeOpMapper, kappa: float) -> ComputeOpMapper:
    """The §15.8 pre-arm shape: unity per-source slots + table-kappa out."""
    node.per_source_scales = [torch.ones(1)]
    node.output_scale = torch.tensor([float(kappa)])
    return node


def _entry_chain(kappa_a: float = 2.64, kappa_b: float = 1.3):
    """input -> armed A(kappa_a) -> Reshape -> armed B (stale unity per_source)
    -> LIF perceptron: the measured AB3 entry shape."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    a = _prearmed(
        ComputeOpMapper(inp, nn.Identity(), input_shape=(8,), output_shape=(8,)),
        kappa_a,
    )
    reshaped = ReshapeMapper(a, (8,))
    b = _prearmed(
        ComputeOpMapper(reshaped, nn.Identity(), input_shape=(8,), output_shape=(8,)),
        kappa_b,
    )
    p = Perceptron(3, 8, normalization=nn.Identity())
    p.set_activation_scale(0.9)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    repr_ = ModelRepresentation(PerceptronMapper(b, p))
    mark_encoding_layers(repr_, placement="offload")
    return repr_, a, b


def test_consumer_per_source_refreshes_from_producer_armed_gauge():
    repr_, a, b = _entry_chain()
    assert float(b.per_source_scales[0]) == 1.0  # the stale pre-arm state
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(repr_, input_data_scale=1.0)
    assert float(torch.as_tensor(a.per_source_scales[0]).mean()) == pytest.approx(1.0)
    assert float(torch.as_tensor(b.per_source_scales[0]).mean()) == pytest.approx(
        2.64, rel=1e-6
    )
    assert float(torch.as_tensor(a.output_scale).mean()) == pytest.approx(2.64)
    assert float(torch.as_tensor(b.output_scale).mean()) == pytest.approx(1.3)


def test_certificate_fails_loud_on_the_pair_and_passes_after_stamp():
    repr_, a, b = _entry_chain()
    with pytest.raises(RuntimeError, match="per_source"):
        verify_boundary_currency_coherence(repr_, input_data_scale=1.0)
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(repr_, input_data_scale=1.0)
    verify_boundary_currency_coherence(repr_, input_data_scale=1.0)
    b.per_source_scales = [torch.ones(1)]
    with pytest.raises(RuntimeError, match="per_source"):
        verify_boundary_currency_coherence(repr_, input_data_scale=1.0)


def test_unarmed_chains_stay_untouched():
    torch.manual_seed(0)
    inp = InputMapper((8,))
    host = ComputeOpMapper(inp, nn.Identity(), input_shape=(8,), output_shape=(8,))
    p = Perceptron(3, 8, normalization=nn.Identity())
    p.set_activation_scale(1.1)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    repr_ = ModelRepresentation(PerceptronMapper(host, p))
    mark_encoding_layers(repr_, placement="offload")
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(repr_, input_data_scale=1.0)
    verify_boundary_currency_coherence(repr_, input_data_scale=1.0)
    assert host.per_source_scales is None or all(
        float(torch.as_tensor(s).mean()) == pytest.approx(1.0)
        for s in host.per_source_scales
    )


def test_partial_presence_multi_source_refreshes_the_present_entry():
    from mimarsinan.mapping.mappers.structural import ConcatMapper

    torch.manual_seed(0)
    inp = InputMapper((8,))
    a = _prearmed(
        ComputeOpMapper(inp, nn.Identity(), input_shape=(8,), output_shape=(8,)),
        2.64,
    )

    class _ParamSource(InputMapper):
        def propagate_source_scale(self, deps, out_scales):
            return None  # a scale-less producer (parameter/token path)

    tok = _ParamSource((8,))
    cat = ConcatMapper([tok, a], dim=1)
    b = ComputeOpMapper(cat, nn.Identity(), input_shape=(16,), output_shape=(16,))
    b.per_source_scales = [torch.ones(1)]
    b.output_scale = torch.tensor([1.3])
    p = Perceptron(3, 16, normalization=nn.Identity())
    p.set_activation_scale(0.9)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    repr_ = ModelRepresentation(PerceptronMapper(b, p))
    mark_encoding_layers(repr_, placement="offload")
    compute_per_source_scales(repr_)
    assert float(torch.as_tensor(a.per_source_scales[0]).mean()) == pytest.approx(1.0)
    got = float(torch.as_tensor(b.per_source_scales[0]).mean())
    assert got > 1.0  # the cat blends the armed 2.64 through; unity must not survive

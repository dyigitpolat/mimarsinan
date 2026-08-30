"""The encode boundary's straight-through surrogate: hop 0 must train.

The deployed entry boundary is a hard uniform comb built from comparisons, so
it carries NO gradient at all — every parameter upstream of it (the whole
encoding hop, which for a ``subsume`` placement is the conv stem) is frozen for
the entire LIF/WQ adaptation. The fold already carries the same idiom on the
per-event path (``lif_serial.SerialFoldSlot.run_cycle``); this is that idiom at
the one encode site, with the forward left byte-identical.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.spike_trains import (
    straight_through_spike_train,
    uniform_spike_train,
)
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


def _rates(requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(0)
    r = torch.rand(4, 6)
    r.requires_grad_(requires_grad)
    return r


class TestStraightThroughSpikeTrain:
    """The SSOT encode: hard forward, count-derivative backward."""

    @pytest.mark.parametrize("phase_dither", [False, True])
    def test_forward_is_byte_identical_to_the_hard_encode(self, phase_dither):
        rate = _rates()
        ste = straight_through_spike_train(rate, T, phase_dither=phase_dither)
        hard = uniform_spike_train(rate, T, phase_dither=phase_dither)
        assert torch.equal(ste, hard)

    def test_the_hard_encode_carries_no_gradient(self):
        """The defect this surrogate exists for, pinned as a characterization."""
        rate = _rates(requires_grad=True)
        assert uniform_spike_train(rate, T).requires_grad is False

    def test_backward_is_the_window_count_derivative(self):
        """``count = sum_t train_t ~ rate * T``, so d(count)/d(rate) = T."""
        rate = _rates(requires_grad=True)
        straight_through_spike_train(rate, T).sum().backward()
        assert rate.grad is not None
        assert torch.allclose(rate.grad, torch.full_like(rate.grad, float(T)))

    def test_saturated_rates_have_no_gradient(self):
        """Outside [0, 1] the wire is saturated and the surrogate says so."""
        rate = torch.tensor([[-0.5, 0.5, 1.5]], requires_grad=True)
        straight_through_spike_train(rate, T).sum().backward()
        assert rate.grad is not None
        assert torch.equal(rate.grad, torch.tensor([[0.0, float(T), 0.0]]))

    def test_forward_under_no_grad_matches_the_grad_enabled_forward(self):
        rate = _rates(requires_grad=True)
        with torch.no_grad():
            frozen = straight_through_spike_train(rate, T)
        live = straight_through_spike_train(rate, T)
        assert torch.equal(frozen, live.detach())


class _EncodedFlow(nn.Module):
    """input(8) -> ENCODING(6) -> LIF(5) -> LIF(3): one subsumed entry hop."""

    def __init__(self, thresholding: str = "<="):
        super().__init__()
        torch.manual_seed(0)
        inp = InputMapper((8,))
        self.p1 = self._perceptron(6, 8, thresholding, encoding=True)
        self.p2 = self._perceptron(5, 6, thresholding)
        self.p3 = self._perceptron(3, 5, thresholding)
        node = PerceptronMapper(inp, self.p1)
        node = PerceptronMapper(node, self.p2)
        node = PerceptronMapper(node, self.p3)
        self._repr = ModelRepresentation(node)
        mark_encoding_layers(self._repr)

    @staticmethod
    def _perceptron(out_dim, in_dim, thresholding, encoding=False):
        p = Perceptron(out_dim, in_dim, normalization=nn.Identity())
        with torch.no_grad():
            p.layer.weight.data = (
                torch.randint(0, 5, (out_dim, in_dim)).float() / 8.0
            )
            p.layer.bias.data = torch.full((out_dim,), 1.0 / 16.0)
        p.set_activation_scale(1.0)
        p.is_encoding_layer = encoding
        lif = LIFActivation(
            T=T, activation_scale=p.activation_scale,
            thresholding_mode=thresholding,
        )
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
        return p

    def get_perceptrons(self):
        return self._repr.get_perceptrons()

    def get_mapper_repr(self):
        return self._repr

    def forward(self, x):
        return self._repr(x)


def _exact_inputs(n: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 9, (n, 8)).float() / 8.0


def _walk(flow, x, **policy_kwargs):
    return SegmentForwardDriver(
        flow.get_mapper_repr(), T,
        LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW, **policy_kwargs),
    )(x)


class TestEncodingHopTrains:
    """The streamed walk must reach the encoding hop's parameters."""

    @pytest.mark.parametrize("retime", [False, True])
    def test_walk_backprops_to_the_encoding_perceptron(self, retime):
        flow = _EncodedFlow().train()
        _walk(flow, _exact_inputs(), retime=retime).sum().backward()
        grad = flow.p1.layer.weight.grad
        assert grad is not None and bool(torch.any(grad != 0)), (
            "the encoding hop is frozen: the entry encode severed its gradient"
        )

    def test_every_hop_still_trains(self):
        flow = _EncodedFlow().train()
        _walk(flow, _exact_inputs()).sum().backward()
        for p in (flow.p1, flow.p2, flow.p3):
            grad = p.layer.weight.grad
            assert grad is not None and bool(torch.any(grad != 0)), p.name

    def test_forward_is_unchanged_by_the_surrogate(self):
        """Training-only: the walk's output is the deployed one, bit for bit."""
        flow = _EncodedFlow().eval()
        x = _exact_inputs()
        live = _walk(flow, x)
        with torch.no_grad():
            frozen = _walk(flow, x)
        assert torch.equal(live.detach(), frozen)

    def test_encoding_hop_emits_the_hard_uniform_train(self):
        """The emitted entry train IS the deployed comb — values, not a surrogate."""
        flow = _EncodedFlow().eval()
        x = _exact_inputs()
        recorder: dict = {}
        with torch.no_grad():
            SegmentForwardDriver(
                flow.get_mapper_repr(), T,
                LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW),
            )(x, node_value_recorder=recorder)
            flow.p1.activation.set_cycle_accurate(False)
            rate = (flow.p1(x) / flow.p1.activation_scale).clamp(0.0, 1.0)
            expected = uniform_spike_train(rate, T).mean(dim=0)
        assert torch.equal(recorder[id(flow.p1)], expected)

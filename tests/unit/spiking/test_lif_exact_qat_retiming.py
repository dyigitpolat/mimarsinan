"""[R5/C3] LIF per-hop re-timed NF twin + exact-QAT theta promotion seams.

The R5 blocker was twin mismatch: the deployed per-hop re-encode had no NF
counterpart, so arming ``lif_per_hop_retiming`` broke the torch<->deployed-sim
gate. ``LifSegmentPolicy(retime=True)`` IS the twin-side count-preserving
re-encode (round((c/T)*T) = c): under exact-QAT the trained staircase
composition and the re-timed deployment are the same object per hop (Theorem 2
with uniform arrivals; lif_exact_qat_program.md §4.1 (A+R5), measured
train->deploy −0.10 pp / argmax ~1).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


class _Flow(nn.Module):
    """input(8) -> encoding(6) -> LIF(5) -> LIF(3), exact-binary arithmetic."""

    def __init__(self, thresholding="<="):
        super().__init__()
        torch.manual_seed(0)
        inp = InputMapper((8,))
        self.p1 = self._perceptron(6, 8, 1.0, thresholding, encoding=True)
        self.p2 = self._perceptron(5, 6, 1.0, thresholding)
        self.p3 = self._perceptron(3, 5, 1.0, thresholding)
        node = PerceptronMapper(inp, self.p1)
        node = PerceptronMapper(node, self.p2)
        node = PerceptronMapper(node, self.p3)
        self._repr = ModelRepresentation(node)
        mark_encoding_layers(self._repr)

    @staticmethod
    def _perceptron(out_dim, in_dim, theta, thresholding, encoding=False):
        p = Perceptron(out_dim, in_dim, normalization=nn.Identity())
        with torch.no_grad():
            # Exact binary lattice: weights in {0, 1/8, ..., 4/8}, bias 1/16.
            w = torch.randint(0, 5, (out_dim, in_dim)).float() / 8.0
            p.layer.weight.data = w
            p.layer.bias.data = torch.full((out_dim,), 1.0 / 16.0)
        p.set_activation_scale(float(theta))
        p.is_encoding_layer = encoding
        lif = LIFActivation(
            T=T, activation_scale=p.activation_scale, thresholding_mode=thresholding,
        )
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
        if encoding:
            p.input_activation = ChipInputQuantizer(
                T=T, activation_scale=p.input_activation_scale,
            )
        return p

    def get_perceptrons(self):
        return self._repr.get_perceptrons()

    def get_mapper_repr(self):
        return self._repr

    def forward(self, x):
        return self._repr(x)


def _exact_inputs(n=16):
    torch.manual_seed(1)
    return torch.randint(0, 9, (n, 8)).float() / 8.0


@pytest.mark.parametrize("thresholding", ["<=", "<"])
def test_train_forward_equals_retimed_deployed_forward_bit_exact(thresholding):
    """THE parity contract: the trained staircase composition IS the re-timed
    deployment's twin — bit-exact on an exact-binary fixture (§6.4(iii))."""
    flow = _Flow(thresholding).eval()
    x = _exact_inputs()
    with torch.no_grad():
        train_forward = type(flow).forward(flow, x)
        deployed = SegmentForwardDriver(
            flow.get_mapper_repr(), T, LifSegmentPolicy(retime=True),
        )(x)
    assert torch.equal(train_forward, deployed)


def test_retime_preserves_per_hop_counts():
    """The re-encode is count-preserving: round((c/T)*T) = c at the first hop."""
    flow = _Flow().eval()
    x = _exact_inputs()
    counts = {}

    def capture(tag):
        recorder = {}
        driver = SegmentForwardDriver(
            flow.get_mapper_repr(), T,
            LifSegmentPolicy(retime=(tag == "retimed")),
        )
        with torch.no_grad():
            driver(x, node_value_recorder=recorder)
        counts[tag] = recorder[id(flow.p2)]

    capture("raw")
    capture("retimed")
    assert torch.equal(counts["raw"], counts["retimed"])


def test_retimed_walk_passes_gradients_to_every_hop():
    """STE re-encode: a hard uniform re-encode severs every upstream hop's
    gradient (the boundary-grad-severance failure mode — the WQ endpoint would
    silently train only the head); the retimed twin must backprop to hop 1."""
    flow = _Flow()
    flow.train()
    x = _exact_inputs()
    driver = SegmentForwardDriver(
        flow.get_mapper_repr(), T, LifSegmentPolicy(retime=True),
    )
    out = driver(x)
    out.sum().backward()
    for p in (flow.p2, flow.p3):
        grad = p.layer.weight.grad
        assert grad is not None and bool(torch.any(grad != 0)), p.name


def test_retime_default_off_is_byte_identical():
    flow = _Flow().eval()
    x = _exact_inputs()
    policy = LifSegmentPolicy()
    assert policy.retime is False
    with torch.no_grad():
        default_out = SegmentForwardDriver(flow.get_mapper_repr(), T, policy)(x)
        explicit_raw = SegmentForwardDriver(
            flow.get_mapper_repr(), T, LifSegmentPolicy(retime=False),
        )(x)
        chip_aligned = chip_aligned_segment_forward(flow, x, T)
    assert torch.equal(default_out, explicit_raw)
    assert torch.equal(default_out, chip_aligned)


def test_chip_aligned_forward_plumbs_retime():
    flow = _Flow().eval()
    x = _exact_inputs()
    with torch.no_grad():
        retimed = chip_aligned_segment_forward(flow, x, T, retime=True)
        driver_out = SegmentForwardDriver(
            flow.get_mapper_repr(), T, LifSegmentPolicy(retime=True),
        )(x)
    assert torch.equal(retimed, driver_out)


class TestChipAlignedNFForwardRetime:
    def _forward_cls(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import _ChipAlignedNFForward

        return _ChipAlignedNFForward

    def test_retimed_install_runs_the_retimed_walk(self):
        flow = _Flow().eval()
        x = _exact_inputs()
        fwd = self._forward_cls()(flow, T, retime=True)
        with torch.no_grad():
            got = fwd(x)
            expected = chip_aligned_segment_forward(flow, x, T, retime=True)
        assert torch.equal(got, expected)

    def test_legacy_pickle_without_retime_attr_defaults_raw(self):
        flow = _Flow().eval()
        x = _exact_inputs()
        fwd = self._forward_cls()(flow, T)
        assert fwd.retime is False
        del fwd.__dict__["retime"]  # a pre-retime cache artifact
        with torch.no_grad():
            got = fwd(x)
            expected = chip_aligned_segment_forward(flow, x, T, retime=False)
        assert torch.equal(got, expected)


class TestPromoteThetaForExactQAT:
    """[§6.2] per-channel theta ONLY on the R3 matching-axis set; scalar-trainable
    on externally-consumed hops; encoder frozen."""

    def _model(self):
        import sys, os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from conftest import make_tiny_supermodel

        return make_tiny_supermodel(hidden_layers=2)

    def test_promotion_respects_the_seam_constraints(self):
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        model = self._model()
        report = promote_theta_for_exact_qat(model)
        encoder, hidden, head = list(model.get_perceptrons())
        assert not encoder.activation_scale.requires_grad
        assert hidden.activation_scale.requires_grad
        assert hidden.activation_scale.dim() == 1
        assert hidden.activation_scale.numel() == hidden.layer.weight.shape[0]
        assert head.activation_scale.requires_grad
        assert head.activation_scale.dim() == 0
        assert hidden.name in report["per_channel"]
        assert head.name in report["scalar"]

    def test_scale_carrying_nodes_rebind_to_the_promoted_param(self):
        from mimarsinan.spiking.lif_utils import unwrap_lif_activation
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        model = self._model()
        perceptrons = list(model.get_perceptrons())
        for p in perceptrons:
            lif = LIFActivation(T=T, activation_scale=p.activation_scale)
            p.set_activation(lif)
        promote_theta_for_exact_qat(model)
        for p in perceptrons[1:]:
            lif = unwrap_lif_activation(p.activation)
            assert lif is not None
            assert lif.activation_scale is p.activation_scale

    def test_promoted_theta_values_are_preserved(self):
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        model = self._model()
        perceptrons = list(model.get_perceptrons())
        for i, p in enumerate(perceptrons):
            p.set_activation_scale(0.5 + i)
        promote_theta_for_exact_qat(model)
        _, hidden, head = perceptrons
        np.testing.assert_allclose(hidden.activation_scale.detach().numpy(), 1.5)
        np.testing.assert_allclose(float(head.activation_scale.detach()), 2.5)

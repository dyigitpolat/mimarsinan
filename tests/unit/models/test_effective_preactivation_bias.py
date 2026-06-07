"""``effective_preactivation_bias``: the additive constant of ``norm(layer(x))``.

Single source of truth for the bias the deployed chip charges per cycle —
consumed by normalization fusion (the fused bias), the TTFS segment policy
(drive-time bias install), and equal to ``PerceptronTransformer``'s effective
bias up to the ``activation_scale`` division.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


def _bn_perceptron(out_f=6, in_f=8, bias=True):
    torch.manual_seed(0)
    p = Perceptron(out_f, in_f, bias=bias, normalization=nn.BatchNorm1d(out_f))
    p.train()
    with torch.no_grad():
        for _ in range(4):
            p(torch.randn(16, in_f))
        p.normalization.weight.copy_(torch.rand(out_f) + 0.5)
        p.normalization.bias.copy_(torch.randn(out_f))
    p.eval()
    return p


class TestEffectivePreactivationBias:
    def test_identity_norm_returns_layer_bias(self):
        p = Perceptron(6, 8)
        assert p.effective_preactivation_bias() is p.layer.bias

    def test_identity_norm_biasless_returns_none(self):
        p = Perceptron(6, 8, bias=False)
        assert p.effective_preactivation_bias() is None

    def test_bn_matches_constant_term_of_forward(self):
        """eff_b == norm(layer(0)) — the additive constant of the pre-activation."""
        p = _bn_perceptron()
        with torch.no_grad():
            constant = p.normalization(p.layer(torch.zeros(1, 8)))[0]
        torch.testing.assert_close(
            p.effective_preactivation_bias(), constant, atol=1e-6, rtol=0.0,
        )

    def test_bn_biasless_is_nonzero(self):
        p = _bn_perceptron(bias=False)
        eff = p.effective_preactivation_bias()
        assert eff is not None and eff.abs().sum() > 0
        with torch.no_grad():
            constant = p.normalization(p.layer(torch.zeros(1, 8)))[0]
        torch.testing.assert_close(eff, constant, atol=1e-6, rtol=0.0)

    def test_matches_perceptron_transformer_up_to_activation_scale(self):
        p = _bn_perceptron()
        p.set_activation_scale(torch.tensor(2.5))
        eff = p.effective_preactivation_bias().detach()
        pt_eff = PerceptronTransformer().get_effective_bias(p)
        torch.testing.assert_close(eff / 2.5, pt_eff, atol=1e-6, rtol=0.0)

    def test_differentiable_through_norm_and_layer_bias(self):
        p = _bn_perceptron()
        eff = p.effective_preactivation_bias()
        eff.sum().backward()
        assert p.layer.bias.grad is not None
        assert p.normalization.weight.grad is not None
        assert p.normalization.bias.grad is not None

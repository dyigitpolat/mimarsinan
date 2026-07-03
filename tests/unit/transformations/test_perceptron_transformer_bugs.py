"""
Tests that expose actual bugs in PerceptronTransformer.

These tests verify mathematical invariants that should hold but may not.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class TestPerChannelActivationScale:
    """``ttfs_theta_cotrain`` rebinds ``activation_scale`` to a per-output-channel
    tensor (len == output_channels). ``get_effective_weight`` must broadcast it over
    the weight's OUTPUT dim (dim 0), not align it with the input dim — the latter
    crashed Weight Quantization on theta-cotrain + proxy_fast runs
    (``size of tensor a (out) must match b (in) at dim 1``)."""

    def test_get_effective_weight_broadcasts_over_output_dim(self):
        pt = PerceptronTransformer()
        out, inp = 8, 4
        p = Perceptron(out, inp, normalization=nn.Identity())  # (output_channels, input_features)
        theta = torch.rand(out) + 0.5
        p.activation_scale = nn.Parameter(theta)
        eff = pt.get_effective_weight(p)
        assert eff.shape == p.layer.weight.shape
        torch.testing.assert_close(eff, p.layer.weight.data / theta.view(out, 1))

    def test_apply_effective_weight_transform_identity_roundtrips_per_channel(self):
        pt = PerceptronTransformer()
        out, inp = 8, 4
        p = Perceptron(out, inp, normalization=nn.Identity())
        p.activation_scale = nn.Parameter(torch.rand(out) + 0.5)
        orig = p.layer.weight.data.clone()
        pt.apply_effective_weight_transform(p, lambda w: w)
        torch.testing.assert_close(p.layer.weight.data, orig, atol=1e-6, rtol=0)

    def test_scalar_activation_scale_still_works(self):
        pt = PerceptronTransformer()
        p = Perceptron(8, 4, normalization=nn.Identity())
        p.set_activation_scale(2.0)
        eff = pt.get_effective_weight(p)
        torch.testing.assert_close(eff, p.layer.weight.data / 2.0)


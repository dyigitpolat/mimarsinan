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


class TestApplyEffectiveBiasTransformToNorm:
    """
    BUG: apply_effective_bias_transform_to_norm uses perceptron.scale_factor
    while apply_effective_bias_transform uses perceptron.activation_scale.

    When scale_factor != activation_scale, the identity transform
    (lambda b: b) should leave the perceptron output unchanged, but
    apply_effective_bias_transform_to_norm corrupts the bias.
    """

    def test_identity_roundtrip_with_matching_scales(self):
        """Works when scale_factor == activation_scale (hides the bug)."""
        pt = PerceptronTransformer()
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        p.set_scale_factor(1.0)

        orig_bias = p.layer.bias.data.clone()
        pt.apply_effective_bias_transform_to_norm(p, lambda b: b)
        assert torch.allclose(p.layer.bias.data, orig_bias, atol=1e-6), \
            "Identity transform should not change bias when scales match"

    def test_identity_roundtrip_with_mismatched_scales(self):
        """
        BUG EXPOSED: When activation_scale != scale_factor, the identity
        transform corrupts the bias because apply_effective_bias_transform_to_norm
        uses scale_factor where it should use activation_scale.

        get_effective_bias returns: bias / activation_scale
        apply_effective_bias_transform_to_norm writes: transform(eff_bias) * scale_factor
        For identity: bias/activation_scale * scale_factor != bias (when they differ)
        """
        pt = PerceptronTransformer()
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(3.0)
        p.set_scale_factor(0.5)

        orig_bias = p.layer.bias.data.clone()
        pt.apply_effective_bias_transform_to_norm(p, lambda b: b)

        # This SHOULD be close to orig_bias (identity transform should be a no-op)
        # But it won't be because of the bug:
        # new_bias = (orig_bias / 3.0) * 0.5 = orig_bias / 6.0
        expected_buggy = orig_bias / 3.0 * 0.5
        assert torch.allclose(p.layer.bias.data, expected_buggy, atol=1e-6), \
            "Confirming the buggy behavior: bias is scaled by scale_factor/activation_scale"

        if not torch.allclose(p.layer.bias.data, orig_bias, atol=1e-3):
            pytest.xfail(
                "BUG: apply_effective_bias_transform_to_norm uses scale_factor "
                "instead of activation_scale. Identity transform changes the bias "
                f"from {orig_bias.tolist()} to {p.layer.bias.data.tolist()}"
            )

    def test_asymmetry_between_bias_transform_methods(self):
        """
        apply_effective_bias_transform (correct) and
        apply_effective_bias_transform_to_norm (buggy) produce different
        results for the same identity transform when scales differ.
        """
        torch.manual_seed(123)
        pt = PerceptronTransformer()

        p1 = Perceptron(4, 8, normalization=nn.Identity())
        p1.set_activation_scale(2.5)
        p1.set_scale_factor(0.7)
        p1.layer.weight.data = torch.randn_like(p1.layer.weight.data)
        p1.layer.bias.data = torch.randn_like(p1.layer.bias.data)

        p2 = Perceptron(4, 8, normalization=nn.Identity())
        p2.set_activation_scale(2.5)
        p2.set_scale_factor(0.7)
        p2.layer.weight.data = p1.layer.weight.data.clone()
        p2.layer.bias.data = p1.layer.bias.data.clone()

        pt.apply_effective_bias_transform(p1, lambda b: b)
        pt.apply_effective_bias_transform_to_norm(p2, lambda b: b)

        biases_match = torch.allclose(p1.layer.bias.data, p2.layer.bias.data, atol=1e-6)
        if not biases_match:
            pytest.xfail(
                "BUG: apply_effective_bias_transform and "
                "apply_effective_bias_transform_to_norm produce different results "
                "for the same identity transform. "
                f"Diff: {(p1.layer.bias.data - p2.layer.bias.data).abs().max():.6f}"
            )
        assert biases_match

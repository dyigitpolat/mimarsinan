"""
Tests that expose actual bugs in PerceptronTransformer.

These tests verify mathematical invariants that should hold but may not.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


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

"""
Stress tests for TensorQuantization and PerceptronTransformer.

Tests use hand-computed expected values and adversarial inputs.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class TestQuantizationHandComputed:
    """Verify quantization math against hand-computed results."""

    def test_quantize_known_values_8bit(self):
        """
        W = [0.5, -1.0]
        8-bit: q_max = 127
        scale = 127 / max(|W|) = 127 / 1.0 = 127
        W_q = round(W * 127) / 127 = [round(63.5)/127, round(-127)/127]
             = [64/127, -127/127] = [0.50394, -1.0]
        """
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.5, -1.0])
        out = q.quantize(w)
        assert out[0].item() == pytest.approx(64.0 / 127.0, abs=1e-5)
        assert out[1].item() == pytest.approx(-1.0, abs=1e-5)

    def test_scaled_quantize_known_values_8bit(self):
        """
        W = [0.5, -1.0], scale = 127
        scaled_quantize returns round(W * scale) = [64, -127]
        """
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.5, -1.0])
        out = q.scaled_quantize(w)
        assert out[0].item() == 64.0
        assert out[1].item() == -127.0

    def test_quantize_known_values_4bit(self):
        """
        4-bit: q_max = 7, q_min = -8
        W = [1.0, 0.5, -0.5]
        scale = 7 / 1.0 = 7
        q(W) = round(W * 7) / 7 = [7/7, 4/7, -4/7]
             = [1.0, 0.5714, -0.5714]
        """
        q = TensorQuantization(bits=4)
        w = torch.tensor([1.0, 0.5, -0.5])
        out = q.quantize(w)
        assert out[0].item() == pytest.approx(1.0, abs=1e-5)
        assert out[1].item() == pytest.approx(4.0 / 7.0, abs=1e-5)
        assert out[2].item() == pytest.approx(-4.0 / 7.0, abs=1e-5)

    def test_quantization_error_bounded(self):
        """Max quantization error should be <= 0.5 / scale for any element."""
        q = TensorQuantization(bits=8)
        w = torch.randn(100, 100)
        out = q.quantize(w)
        scale = q.q_max / torch.max(torch.abs(w))
        max_err = (w - out).abs().max()
        theoretical_bound = 0.5 / scale
        assert max_err <= theoretical_bound + 1e-6, \
            f"Quantization error {max_err} exceeds bound {theoretical_bound}"

    def test_scaled_quantize_max_element_maps_to_qmax(self):
        """The largest-magnitude element should map to exactly +/-q_max."""
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.3, -0.7, 0.7, 0.1])
        out = q.scaled_quantize(w)
        assert out.abs().max().item() == q.q_max

    def test_quantize_preserves_sign(self):
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.1, -0.2, 0.3, -0.4])
        out = q.quantize(w)
        for i in range(len(w)):
            if w[i] > 0:
                assert out[i] >= 0, f"Positive input w[{i}]={w[i]} mapped to negative {out[i]}"
            elif w[i] < 0:
                assert out[i] <= 0, f"Negative input w[{i}]={w[i]} mapped to positive {out[i]}"


class TestQuantizationAdversarial:
    """Edge cases that may reveal bugs."""

    def test_very_small_weights(self):
        """Weights near machine epsilon — scale becomes huge."""
        q = TensorQuantization(bits=8)
        w = torch.tensor([1e-30, -1e-30])
        out = q.quantize(w)
        assert not torch.isnan(out).any(), "Small weights should not produce NaN"

    def test_mixed_zero_and_nonzero(self):
        """Zero elements mixed with non-zero should quantize cleanly."""
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.0, 1.0, 0.0, -0.5])
        out = q.quantize(w)
        assert not torch.isnan(out).any()
        assert out[0].item() == pytest.approx(0.0)

    def test_single_positive_element(self):
        """Single element should map to itself after quantize round-trip."""
        q = TensorQuantization(bits=8)
        w = torch.tensor([0.42])
        out = q.quantize(w)
        assert out.item() == pytest.approx(0.42, abs=0.5 / 127)

    def test_all_same_value(self):
        q = TensorQuantization(bits=8)
        w = torch.full((4, 4), 0.5)
        out = q.scaled_quantize(w)
        assert (out == out[0, 0]).all(), "Uniform input should produce uniform output"
        assert out[0, 0].item() == 127.0

    def test_large_tensor(self):
        """Stress test with a large tensor."""
        q = TensorQuantization(bits=8)
        w = torch.randn(1000, 1000)
        out = q.scaled_quantize(w)
        assert out.abs().max().item() <= q.q_max
        assert out.abs().min().item() >= 0


class TestPerceptronTransformerStress:
    """
    Tests that check the mathematical correctness of get_effective_* and
    apply_effective_*_transform by computing the fused output and comparing
    to the non-fused output through the actual perceptron forward pass.
    """

    def test_fusion_equivalence_with_batchnorm(self):
        """
        Fused output (effective_weight @ x + effective_bias) should match
        the non-fused perceptron output (linear -> batchnorm) for eval mode.
        """
        torch.manual_seed(42)
        pt = PerceptronTransformer()
        p = Perceptron(5, 10, normalization=nn.BatchNorm1d(5))
        p.set_activation_scale(1.0)

        p.layer.weight.data = torch.randn_like(p.layer.weight.data)
        p.layer.bias.data = torch.randn_like(p.layer.bias.data)
        p.normalization.weight.data = torch.randn_like(p.normalization.weight.data)
        p.normalization.bias.data = torch.randn_like(p.normalization.bias.data)
        p.normalization.running_mean.data = torch.randn_like(p.normalization.running_mean.data)
        p.normalization.running_var.data = torch.abs(torch.randn_like(p.normalization.running_var.data))
        p.normalization.eval()

        x = torch.randn(4, 10)

        with torch.no_grad():
            linear_out = p.layer(x)
            bn_out = p.normalization(linear_out)

        eff_w = pt.get_effective_weight(p)
        eff_b = pt.get_effective_bias(p)

        with torch.no_grad():
            fused_out = x @ eff_w.T + eff_b

        assert torch.allclose(bn_out, fused_out, atol=1e-5), \
            f"Max diff: {(bn_out - fused_out).abs().max()}"

    def test_apply_transform_then_get_roundtrip_with_batchnorm(self):
        """
        Apply a non-trivial transform (scale + offset) to effective params,
        then verify the perceptron output matches doing the transform manually.
        """
        torch.manual_seed(99)
        pt = PerceptronTransformer()
        p = Perceptron(5, 10, normalization=nn.BatchNorm1d(5))
        p.set_activation_scale(1.0)

        p.normalization.weight.data = torch.randn_like(p.normalization.weight.data)
        p.normalization.bias.data = torch.randn_like(p.normalization.bias.data)
        p.normalization.running_mean.data = torch.randn_like(p.normalization.running_mean.data)
        p.normalization.running_var.data = torch.abs(torch.randn_like(p.normalization.running_var.data))
        p.normalization.eval()

        x = torch.randn(4, 10)

        with torch.no_grad():
            out_before = p.normalization(p.layer(x))

        factor = 0.42
        offset = 0.69
        transform = lambda param: param * factor + offset

        eff_w_orig = pt.get_effective_weight(p).clone()
        eff_b_orig = pt.get_effective_bias(p).clone()
        expected_w = eff_w_orig * factor + offset
        expected_b = eff_b_orig * factor + offset

        pt.apply_effective_parameter_transform(p, transform)

        eff_w_after = pt.get_effective_weight(p)
        eff_b_after = pt.get_effective_bias(p)

        assert torch.allclose(eff_w_after, expected_w, atol=1e-5), \
            f"Weight transform mismatch: max diff {(eff_w_after - expected_w).abs().max()}"
        assert torch.allclose(eff_b_after, expected_b, atol=1e-5), \
            f"Bias transform mismatch: max diff {(eff_b_after - expected_b).abs().max()}"

    def test_effective_weight_with_activation_scale(self):
        """
        effective_weight = W / activation_scale (Identity norm).
        With activation_scale = 3.0:
            W = [[1.0, 2.0], [3.0, 4.0]]
            eff_W = [[1/3, 2/3], [1, 4/3]]
        """
        pt = PerceptronTransformer()
        p = Perceptron(2, 2, normalization=nn.Identity())
        p.layer.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p.set_activation_scale(3.0)

        eff_w = pt.get_effective_weight(p)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 3.0
        assert torch.allclose(eff_w, expected, atol=1e-6)

    def test_effective_bias_no_bias_layer(self):
        """Perceptron without bias should return zeros."""
        pt = PerceptronTransformer()
        p = Perceptron(4, 8, bias=False, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        eff_b = pt.get_effective_bias(p)
        assert torch.allclose(eff_b, torch.zeros(4))

    def test_per_input_scales(self):
        """When per_input_scales is set, effective weight should be scaled per input channel."""
        pt = PerceptronTransformer()
        p = Perceptron(2, 4, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        p.per_input_scales = torch.tensor([2.0, 3.0, 0.5, 1.0])

        eff_w = pt.get_effective_weight(p)
        manual_scale = torch.tensor([2.0, 3.0, 0.5, 1.0]).view(1, -1)
        expected = p.layer.weight.data * manual_scale
        assert torch.allclose(eff_w, expected, atol=1e-6)

    def test_quantize_transform_produces_integers(self):
        """
        After applying a quantize transform, effective weights * parameter_scale
        should be near-integer.
        """
        torch.manual_seed(7)
        pt = PerceptronTransformer()
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)

        q_max = 127
        eff_w = pt.get_effective_weight(p)
        scale = q_max / torch.max(torch.abs(eff_w))

        def quantize(param):
            return torch.round(param * scale) / scale

        pt.apply_effective_weight_transform(p, quantize)
        eff_w_q = pt.get_effective_weight(p)
        int_check = eff_w_q * scale
        assert torch.allclose(int_check, torch.round(int_check), atol=1e-3)

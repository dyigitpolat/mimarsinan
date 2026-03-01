"""Tests for TensorQuantization."""

import pytest
import torch
import numpy as np

from mimarsinan.transformations.weight_quantization import TensorQuantization


class TestTensorQuantization:
    def test_quantize_preserves_shape(self):
        q = TensorQuantization(bits=8)
        w = torch.randn(4, 8)
        out = q.quantize(w)
        assert out.shape == w.shape

    def test_quantize_produces_near_integer_scaled(self):
        q = TensorQuantization(bits=8)
        w = torch.randn(4, 8)
        out = q.quantize(w)
        scale = q.q_max / torch.max(torch.abs(w))
        scaled = out * scale
        assert torch.allclose(scaled, torch.round(scaled), atol=1e-4)

    def test_scaled_quantize_produces_integers(self):
        q = TensorQuantization(bits=8)
        w = torch.randn(4, 8)
        out = q.scaled_quantize(w)
        assert torch.allclose(out, torch.round(out), atol=1e-6)

    def test_scaled_quantize_within_range(self):
        q = TensorQuantization(bits=8)
        w = torch.randn(10, 10)
        out = q.scaled_quantize(w)
        assert out.max().item() <= q.q_max
        assert out.min().item() >= q.q_min

    def test_numpy_input(self):
        q = TensorQuantization(bits=8)
        w = np.random.randn(4, 4).astype(np.float32)
        out = q.quantize(w)
        assert isinstance(out, np.ndarray)
        assert out.shape == w.shape

    def test_different_bit_widths(self):
        for bits in [4, 8, 16]:
            q = TensorQuantization(bits=bits)
            w = torch.randn(3, 3)
            out = q.scaled_quantize(w)
            assert out.max().item() <= q.q_max
            assert out.min().item() >= q.q_min

    def test_zero_input_produces_nan(self):
        """Zero weights cause division by zero in scale computation — known behavior."""
        q = TensorQuantization(bits=8)
        w = torch.zeros(2, 2)
        out = q.quantize(w)
        assert torch.isnan(out).all()

    def test_single_element(self):
        q = TensorQuantization(bits=8)
        w = torch.tensor([[0.5]])
        out = q.scaled_quantize(w)
        assert out.item() == pytest.approx(q.q_max, abs=1)


class TestPerceptronTransformer:
    def test_effective_weight_identity_norm(self):
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        import torch.nn as nn

        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        pt = PerceptronTransformer()

        eff_w = pt.get_effective_weight(p)
        assert torch.allclose(eff_w, p.layer.weight.data)

    def test_effective_weight_with_batchnorm(self):
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        import torch.nn as nn

        p = Perceptron(4, 8, normalization=nn.BatchNorm1d(4))
        p.normalization.eval()
        with torch.no_grad():
            p.normalization(torch.randn(10, 4))
        p.set_activation_scale(1.0)

        pt = PerceptronTransformer()
        eff_w = pt.get_effective_weight(p)
        assert eff_w.shape == p.layer.weight.data.shape

    def test_effective_bias_identity_norm(self):
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        import torch.nn as nn

        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(2.0)
        pt = PerceptronTransformer()
        eff_b = pt.get_effective_bias(p)
        expected = p.layer.bias.data / 2.0
        assert torch.allclose(eff_b, expected)

    def test_roundtrip_weight_transform(self):
        """Applying identity transform should preserve effective weights."""
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        import torch.nn as nn

        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        pt = PerceptronTransformer()

        original_w = p.layer.weight.data.clone()
        pt.apply_effective_weight_transform(p, lambda w: w)
        assert torch.allclose(p.layer.weight.data, original_w, atol=1e-6)

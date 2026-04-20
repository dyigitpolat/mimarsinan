"""Tests for Phase C1: LSQ (Learnable Step-Size Quantization) weight quantizer.

LSQ replaces the rate-based mixing approach for weights with:
- Straight-through estimator (STE): forward Q(w), backward identity
- Learnable quantization scale per perceptron (log-space for positivity)

The quantizer is an ``nn.Module`` so its ``scale`` parameter is picked up
by ``BasicTrainer``'s optimizer via ``self.model.parameters()``, and so
the standard PyTorch save/load round-trips preserve the learnt scale.

These tests drive the public contract:
1. LSQ forward produces quantized weights within [-q_max, q_max] * (1/scale)
   (i.e. the integer levels multiplied by scale).
2. LSQ backward propagates gradients through the quantization (STE).
3. The scale parameter is learnable and gets a nonzero gradient under a
   surrogate loss that depends on the quantized weight.
4. When the scale is frozen (``requires_grad=False``), behaviour matches
   legacy ``TensorQuantization.quantize`` for reasonable inputs.
"""

from __future__ import annotations

import math

import pytest
import torch


class TestLSQQuantizerExists:
    def test_importable(self):
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer  # noqa: F401

    def test_build_with_bits(self):
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=8)
        assert q.bits == 8
        assert q.q_max == 127
        assert q.q_min == -128

    def test_log_scale_is_learnable_parameter(self):
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=8)
        assert hasattr(q, "log_scale")
        assert isinstance(q.log_scale, torch.nn.Parameter)
        assert q.log_scale.requires_grad is True

    def test_init_from_tensor_seeds_scale(self):
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=8)
        w = torch.randn(4, 8) * 0.1
        q.init_from_tensor(w)
        p_max = float(torch.max(torch.abs(w)).item())
        expected_step = p_max / float(q.q_max)
        assert math.exp(q.log_scale.item()) == pytest.approx(expected_step, rel=1e-4)


class TestLSQForward:
    def test_output_is_on_quantization_grid(self):
        """Forward output = round_ste(w/step).clamp(q_min, q_max) * step,
        which is exactly integer multiples of the step size."""
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=4)
        w = torch.linspace(-1.0, 1.0, 100).reshape(10, 10)
        q.init_from_tensor(w)
        out = q(w)
        step = torch.exp(q.log_scale.detach())
        ratio = out / step
        assert torch.allclose(ratio, torch.round(ratio), atol=1e-4)

    def test_output_respects_integer_range(self):
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=4)
        w = torch.linspace(-5.0, 5.0, 50)  # far outside init range
        q.init_from_tensor(torch.tensor([1.0]))  # tiny step -> aggressive clipping
        out = q(w)
        step = torch.exp(q.log_scale.detach()).item()
        assert (out / step).max().item() <= q.q_max + 1e-4
        assert (out / step).min().item() >= q.q_min - 1e-4


class TestLSQBackwardSTE:
    def test_gradient_flows_to_input(self):
        """Straight-through: d loss/d w == d loss/d q(w) (sign-identity).
        With surrogate loss = mean(q(w)) the upstream gradient is 1/numel;
        STE means ``w.grad == upstream * clip_mask``, which for in-range
        inputs is just ``upstream``."""
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=8)
        w = torch.randn(4, 8, requires_grad=True)
        q.init_from_tensor(w.detach())

        out = q(w)
        loss = out.sum()
        loss.backward()
        assert w.grad is not None
        # For in-range weights STE gradient should be ~1 (upstream=1 per element).
        # Allow small tolerance for the handful of outlier weights that might
        # be clipped under aggressive random init.
        in_grid = (w.grad.abs() > 1e-6).float().mean().item()
        assert in_grid > 0.5, (
            "Most weights should have nonzero STE gradient; "
            f"only {in_grid:.2f} did."
        )

    def test_log_scale_receives_gradient(self):
        """log_scale must be updatable by the optimizer."""
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        q = LSQQuantizer(bits=4)
        w = torch.randn(4, 8)
        q.init_from_tensor(w)

        out = q(w)
        # Surrogate loss: minimize sum-of-squares between quantized and
        # original weights.  This has a meaningful dependence on scale.
        loss = ((out - w) ** 2).mean()
        loss.backward()
        assert q.log_scale.grad is not None
        assert q.log_scale.grad.abs().item() > 0.0


class TestLSQFrozenMatchesLegacy:
    def test_frozen_forward_matches_legacy_quantize(self):
        """With scale frozen to the legacy 1/max formula the forward pass
        must match ``TensorQuantization.quantize`` up to numerical noise."""
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer
        from mimarsinan.transformations.weight_quantization import TensorQuantization

        bits = 8
        torch.manual_seed(0)
        w = torch.randn(16, 16) * 0.3

        legacy = TensorQuantization(bits=bits)
        legacy_out = legacy.quantize(w)

        q = LSQQuantizer(bits=bits)
        q.init_from_tensor(w)
        q.log_scale.requires_grad_(False)
        lsq_out = q(w)
        assert torch.allclose(lsq_out, legacy_out, atol=1e-4)

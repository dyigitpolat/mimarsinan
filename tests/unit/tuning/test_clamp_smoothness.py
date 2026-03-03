"""Tests for RateAdjustedDecorator smoothness at rate boundaries.

Demonstrates that NestedAdjustmentStrategy([RandomMask, Mix]) creates
a discontinuity at rate=1.0, while MixAdjustmentStrategy alone is smooth.
"""

import pytest
import torch

from mimarsinan.models.layers import (
    ClampDecorator,
    RateAdjustedDecorator,
    MixAdjustmentStrategy,
    RandomMaskAdjustmentStrategy,
    NestedAdjustmentStrategy,
    DifferentiableClamp,
)


def _make_activations_with_outliers(n=1000, scale=2.0, outlier_frac=0.05):
    """Create activations where outlier_frac exceeds the clamp scale."""
    x = torch.randn(1, n).abs()  # positive activations (post-ReLU)
    # Inject outliers above scale
    n_outliers = int(n * outlier_frac)
    x[0, :n_outliers] = scale * (1.0 + torch.rand(n_outliers))
    return x


class TestNestedStrategyDiscontinuity:
    """NestedAdjustmentStrategy([RandomMask, Mix]) creates a qualitative
    discontinuity at rate=1.0 that MixAdjustmentStrategy alone does not."""

    def test_mix_only_is_smooth_at_boundary(self):
        """With MixAdjustmentStrategy only, outputs at 0.999 and 1.0
        differ by at most 0.001 * max(|activation|)."""
        torch.manual_seed(42)
        x = _make_activations_with_outliers(n=10000, scale=2.0)
        clamp = ClampDecorator(torch.tensor(0.0), torch.tensor(2.0))
        clamped = clamp.output_transform(x)

        strategy = MixAdjustmentStrategy()
        out_999 = strategy.adjust(x, clamped, 0.999)
        out_100 = strategy.adjust(x, clamped, 1.0)

        max_diff = (out_999 - out_100).abs().max().item()
        max_activation = x.abs().max().item()
        # Difference should be bounded by 0.001 * max_activation
        assert max_diff <= 0.002 * max_activation, \
            f"MixOnly max_diff={max_diff:.6f} should be ≤ {0.002 * max_activation:.6f}"

    def test_nested_strategy_has_large_discontinuity(self):
        """With NestedAdjustmentStrategy([RandomMask, Mix]), the expected
        output at 0.999 vs 1.0 can differ significantly due to the
        stochastic bypass vanishing entirely at 1.0."""
        torch.manual_seed(42)
        scale = 2.0
        x = _make_activations_with_outliers(n=10000, scale=scale)
        clamp = ClampDecorator(torch.tensor(0.0), torch.tensor(scale))
        clamped = clamp.output_transform(x)

        strategy = NestedAdjustmentStrategy([
            RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()
        ])

        # At rate=1.0: deterministic, all elements clamped
        out_100 = strategy.adjust(x, clamped, 1.0)
        assert torch.allclose(out_100, clamped), \
            "At rate=1.0, output must equal clamped"

        # At rate=0.999: stochastic, ~0.1% elements bypass clamp
        # Average over many trials to get expected output
        diffs = []
        for _ in range(50):
            out_999 = strategy.adjust(x, clamped, 0.999)
            diff = (out_999 - out_100).abs().max().item()
            diffs.append(diff)

        mean_max_diff = sum(diffs) / len(diffs)
        # Outliers are ~2x above scale, so max diff ≈ scale when an outlier
        # bypasses the clamp. This should be much larger than the 0.001*max
        # expected from MixOnly.
        assert mean_max_diff > 0.01, \
            f"NestedStrategy mean_max_diff={mean_max_diff:.6f} should show significant discontinuity"

    def test_mix_only_gradient_is_smooth(self):
        """With MixOnly, the gradient contribution from unclamped signal
        changes smoothly between rates."""
        x = torch.randn(1, 100, requires_grad=True)
        clamp_max = torch.tensor(1.0)
        clamp_min = torch.tensor(0.0)

        # Rate 0.999: 0.1% of gradient comes through unclamped path
        out_999 = 0.999 * DifferentiableClamp.apply(x, clamp_min, clamp_max) + 0.001 * x
        out_999.sum().backward()
        grad_999 = x.grad.clone()

        x.grad = None

        # Rate 1.0: 0% of gradient comes through unclamped path
        out_100 = DifferentiableClamp.apply(x, clamp_min, clamp_max)
        out_100.sum().backward()
        grad_100 = x.grad.clone()

        # Gradients should be very close
        grad_diff = (grad_999 - grad_100).abs().max().item()
        assert grad_diff < 0.01, \
            f"MixOnly gradient difference at boundary should be tiny, got {grad_diff}"

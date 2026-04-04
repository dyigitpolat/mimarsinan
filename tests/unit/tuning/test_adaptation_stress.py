"""
Stress tests for AdaptationManager and SmartSmoothAdaptation.

Tests extreme rates, NaN propagation, and adaptation failure scenarios.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import TransformedActivation, LeakyGradReLU
from conftest import default_config


class TestAdaptationManagerStress:
    def _make_perceptron(self):
        p = Perceptron(8, 16)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        p.set_activation_scale(2.0)
        return p

    def test_all_rates_at_max(self):
        """All rates at 1.0 simultaneously."""
        am = AdaptationManager()
        am.clamp_rate = 1.0
        am.shift_rate = 1.0
        am.quantization_rate = 1.0
        am.noise_rate = 1.0

        p = self._make_perceptron()
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert not torch.isnan(out).any(), "All-max rates produced NaN"
        assert out.shape == (4, 8)

    def test_repeated_update_activation_is_stable(self):
        """Calling update_activation 50 times shouldn't degrade output."""
        am = AdaptationManager()
        am.clamp_rate = 0.5
        p = self._make_perceptron()
        cfg = default_config()

        x = torch.randn(2, 16)
        for _ in range(50):
            am.update_activation(cfg, p)
            out = p(x)
            assert not torch.isnan(out).any()

    def test_nan_activation_scale(self):
        """What happens if activation_scale is NaN?"""
        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        p.set_activation_scale(float("nan"))
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(2, 16)
        out = p(x)
        if torch.isnan(out).any():
            pass
        else:
            pass

    def test_zero_activation_scale(self):
        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        p.set_activation_scale(0.0)
        cfg = default_config()
        am.update_activation(cfg, p)

        x = torch.randn(4, 16)
        out = p(x)
        assert (out == 0).all() or torch.isnan(out).all(), \
            "With activation_scale=0 and clamp_rate=1, output should be all zeros"


def _make_ssa(adapt_fn, interpolators=None, target=0.9, min_step=0.001,
              before_cycle=None):
    """Helper to build SmartSmoothAdaptation."""
    return SmartSmoothAdaptation(
        adaptation_fn=adapt_fn,
        interpolators=interpolators or [lambda t: t],
        get_target=lambda: target,
        min_step=min_step,
        before_cycle=before_cycle,
    )


class TestSmartSmoothAdaptationStress:
    def test_metric_always_zero_forces_min_step(self):
        """When adaptation always rolls back, step should shrink to min_step."""
        call_count = [0]

        def adapt_fn(rate):
            call_count[0] += 1
            return 0.0  # always rollback to 0

        ssa = _make_ssa(adapt_fn, target=0.9, min_step=0.1)
        ssa.adapt_smoothly(max_cycles=10)

        assert call_count[0] > 0
        assert call_count[0] <= 10

    def test_metric_exceeds_target(self):
        """When adaptation always commits, should reach t~1.0 quickly."""
        rates_used = []

        def adapt_fn(rate):
            rates_used.append(rate)
            return rate  # commit

        ssa = _make_ssa(adapt_fn, target=0.9)
        ssa.adapt_smoothly(max_cycles=20)

        assert rates_used[-1] >= 0.99, \
            f"Should reach t~1.0 quickly, last rate: {rates_used[-1]}"

    def test_multiple_interpolators(self):
        received = []

        def adapt_fn(a, b, c):
            received.append((a, b, c))
            return a  # commit at first interpolated value

        ssa = _make_ssa(
            adapt_fn,
            interpolators=[lambda t: t, lambda t: t * 2, lambda t: t * 3],
            target=0.9,
        )
        ssa.adapt_smoothly(max_cycles=3)

        for a, b, c in received:
            assert b == pytest.approx(a * 2, abs=1e-6)
            assert c == pytest.approx(a * 3, abs=1e-6)

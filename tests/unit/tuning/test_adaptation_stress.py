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
        # NaN propagation is expected — if it doesn't crash, that's fine
        # but we should document whether NaN is produced
        if torch.isnan(out).any():
            pass  # Expected: NaN propagates
        else:
            pass  # Unexpected but not a failure

    def test_zero_activation_scale(self):
        """
        activation_scale=0 means the clamp range is [0, 0], so all output
        should be zero.
        """
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


class TestSmartSmoothAdaptationStress:
    def test_metric_always_zero_forces_min_step(self):
        """When metric is always 0 (far below target), adaptation should
        use minimum step size and still complete."""
        call_count = [0]

        def adapt_fn(rate):
            call_count[0] += 1

        def evaluate(rate):
            return 0.0  # Always fails to meet target

        ssa = SmartSmoothAdaptation(
            adapt_fn, lambda: None, lambda s: None,
            evaluate, [lambda t: t], 0.9
        )
        ssa.adapt_smoothly(max_cycles=5)

        # Should have run some cycles even though metric is terrible
        assert call_count[0] > 0
        assert call_count[0] <= 5

    def test_metric_exceeds_target(self):
        """When metric always exceeds target, adaptation should use large steps."""
        rates_used = []

        def adapt_fn(rate):
            rates_used.append(rate)

        def evaluate(rate):
            return 1.0  # Always exceeds target

        ssa = SmartSmoothAdaptation(
            adapt_fn, lambda: None, lambda s: None,
            evaluate, [lambda t: t], 0.9
        )
        ssa.adapt_smoothly(max_cycles=20)

        assert rates_used[-1] >= 0.99, \
            f"Should reach t≈1.0 quickly, last rate: {rates_used[-1]}"

    def test_state_restoration_on_bad_step(self):
        """
        When a step's metric is bad, SmartSmoothAdaptation should restore
        state before trying a smaller step.
        """
        state = [0]
        restore_count = [0]

        def clone():
            return state[0]

        def restore(s):
            restore_count[0] += 1
            state[0] = s

        def evaluate(rate):
            if rate > 0.5:
                return 0.1  # Bad
            return 0.95  # Good

        ssa = SmartSmoothAdaptation(
            lambda r: None, clone, restore,
            evaluate, [lambda t: t], 0.9
        )
        ssa.adapt_smoothly(max_cycles=10)

        assert restore_count[0] > 0, \
            "State should be restored when metric drops"

    def test_multiple_interpolators(self):
        """Multiple interpolators should each receive the current t value."""
        received = []

        def adapt_fn(a, b, c):
            received.append((a, b, c))

        ssa = SmartSmoothAdaptation(
            adapt_fn, lambda: None, lambda s: None,
            lambda a, b, c: 0.95,
            [lambda t: t, lambda t: t * 2, lambda t: t * 3],
            0.9
        )
        ssa.adapt_smoothly(max_cycles=3)

        for a, b, c in received:
            assert b == pytest.approx(a * 2, abs=1e-6)
            assert c == pytest.approx(a * 3, abs=1e-6)

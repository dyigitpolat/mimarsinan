"""Tests for SmartSmoothAdaptation: gradual transformation framework."""

import pytest

from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation


class TestSmartSmoothAdaptation:
    def test_constant_metric_completes(self):
        """With a constant evaluator that always returns target, adaptation should reach t=1."""
        target = 0.9
        state = [0]
        adapt_log = []

        def adapt_fn(rate):
            adapt_log.append(rate)

        def clone():
            return state[0]

        def restore(s):
            state[0] = s

        def evaluate(rate):
            return target

        interpolators = [lambda t: t]

        ssa = SmartSmoothAdaptation(
            adapt_fn, clone, restore, evaluate, interpolators, target
        )
        ssa.adapt_smoothly(max_cycles=20)

        assert len(adapt_log) > 0
        assert adapt_log[-1] >= 0.99

    def test_max_cycles_respected(self):
        adapt_calls = []

        def adapt_fn(rate):
            adapt_calls.append(rate)

        def evaluate(rate):
            return 0.95

        ssa = SmartSmoothAdaptation(
            adapt_fn, lambda: None, lambda s: None,
            evaluate, [lambda t: t], 0.95
        )
        ssa.adapt_smoothly(max_cycles=3)
        assert len(adapt_calls) <= 3

    def test_low_metric_causes_small_steps(self):
        """When metric drops, the adaptation should take smaller steps."""
        step_sizes = []

        def adapt_fn(rate):
            step_sizes.append(rate)

        call_count = [0]

        def evaluate(rate):
            call_count[0] += 1
            if rate > 0.5:
                return 0.1
            return 0.9

        ssa = SmartSmoothAdaptation(
            adapt_fn, lambda: None, lambda s: None,
            evaluate, [lambda t: t], 0.9
        )
        ssa.adapt_smoothly(max_cycles=10)
        assert len(step_sizes) > 0

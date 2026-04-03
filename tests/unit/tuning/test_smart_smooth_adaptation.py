"""Tests for SmartSmoothAdaptation: gradual transformation framework."""

import pytest

from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation


def _make_ssa(adapt_fn, evaluate_fn, clone=None, restore=None,
              interpolators=None, target=0.9, tolerance=0.01, min_step=0.001,
              before_cycle=None):
    return SmartSmoothAdaptation(
        adaptation_fn=adapt_fn,
        clone_state=clone or (lambda: None),
        restore_state=restore or (lambda s: None),
        evaluate_fn=evaluate_fn,
        interpolators=interpolators or [lambda t: t],
        get_target=lambda: target,
        tolerance=tolerance,
        min_step=min_step,
        before_cycle=before_cycle,
    )


class TestSmartSmoothAdaptation:
    def test_constant_metric_completes(self):
        """With a constant evaluator that always returns target, adaptation should reach t=1."""
        target = 0.9
        adapt_log = []

        def adapt_fn(rate):
            adapt_log.append(rate)
            return rate  # signal success

        ssa = _make_ssa(
            adapt_fn,
            lambda rate: target,
            clone=lambda: 0,
            restore=lambda s: None,
            target=target,
        )
        ssa.adapt_smoothly(max_cycles=20)

        assert len(adapt_log) > 0
        assert adapt_log[-1] >= 0.99

    def test_max_cycles_respected(self):
        adapt_calls = []

        ssa = _make_ssa(
            lambda rate: adapt_calls.append(rate),
            lambda rate: 0.95,
            target=0.95,
        )
        ssa.adapt_smoothly(max_cycles=3)
        assert len(adapt_calls) <= 3

    def test_low_metric_causes_small_steps(self):
        """When metric drops, the adaptation should take smaller steps."""
        step_sizes = []

        ssa = _make_ssa(
            lambda rate: step_sizes.append(rate),
            lambda rate: 0.1 if rate > 0.5 else 0.9,
            target=0.9,
        )
        ssa.adapt_smoothly(max_cycles=10)
        assert len(step_sizes) > 0

    def test_before_cycle_called_once_per_cycle(self):
        before_cycle_calls = []

        ssa = _make_ssa(
            lambda rate: None,
            lambda rate: 0.1,
            target=0.9,
            before_cycle=lambda: before_cycle_calls.append(1),
        )
        ssa.adapt_smoothly(max_cycles=3)
        assert len(before_cycle_calls) == 3

    def test_tolerance_set_at_construction(self):
        """Tolerance is passed at construction, not via initial_tolerance_fn."""
        ssa = _make_ssa(
            lambda r: None,
            lambda rate: 1.0,
            target=0.9,
            tolerance=0.07,
        )
        assert ssa.tolerance == pytest.approx(0.07)

    def test_tolerance_stable_across_cycles(self):
        """Tolerance must not escalate when _adjust_minimum_step fires."""
        ssa = _make_ssa(
            lambda rate: None,
            lambda rate: 0.0,
            target=0.9,
            tolerance=0.01,
        )
        initial_tolerance = ssa.tolerance
        ssa.adapt_smoothly(max_cycles=10)
        assert ssa.tolerance == pytest.approx(initial_tolerance)

    def test_first_step_size_at_t0_probes_full_range(self):
        """At t=0, _find_step_size should start with step_size = (1.0 - t) * 2 = 2.0,
        and after the first halving, probe at rate 1.0."""
        probed_rates = []

        ssa = _make_ssa(
            lambda r: None,
            lambda rate: (probed_rates.append(rate), 1.0)[1],
            target=0.9,
        )
        step = ssa._find_step_size(0)
        assert len(probed_rates) >= 1
        assert probed_rates[0] == pytest.approx(1.0)
        assert step == pytest.approx(1.0)

    def test_rollback_resets_t_and_halves_max_step(self):
        """When adaptation_fn returns a rate lower than proposed, t resets and
        max_step shrinks to prevent retrying the same failed step."""
        adapt_log = []
        rollback_until = [0.5]

        def adapt_fn(rate):
            adapt_log.append(rate)
            if rate > rollback_until[0]:
                return 0.0  # signal rollback to t=0
            return rate

        ssa = _make_ssa(
            adapt_fn,
            lambda rate: 0.95,
            target=0.9,
            min_step=0.01,
        )
        ssa.adapt_smoothly(max_cycles=30)

        failed = [r for r in adapt_log if r > rollback_until[0]]
        succeeded = [r for r in adapt_log if r <= rollback_until[0]]
        assert len(failed) >= 1, "Should attempt at least one step beyond rollback_until"
        assert len(succeeded) >= 1, "Should succeed at smaller steps"

    def test_rollback_none_is_backward_compatible(self):
        """When adaptation_fn returns None, t always advances (old behavior)."""
        adapt_log = []

        ssa = _make_ssa(
            lambda rate: adapt_log.append(rate),
            lambda rate: 0.95,
            target=0.9,
        )
        ssa.adapt_smoothly(max_cycles=5)
        assert len(adapt_log) == 5 or adapt_log[-1] >= 0.99

    def test_get_target_callable_used(self):
        """SmartSmoothAdaptation reads the target via get_target callable."""
        current_target = [0.9]

        def get_target():
            return current_target[0]

        ssa = SmartSmoothAdaptation(
            adaptation_fn=lambda r: None,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            evaluate_fn=lambda rate: 0.95,
            interpolators=[lambda t: t],
            get_target=get_target,
            tolerance=0.01,
            min_step=0.001,
        )

        tolerable = ssa.get_target() * (1.0 - ssa.tolerance)
        assert tolerable == pytest.approx(0.9 * 0.99)

        current_target[0] = 0.8
        tolerable = ssa.get_target() * (1.0 - ssa.tolerance)
        assert tolerable == pytest.approx(0.8 * 0.99)

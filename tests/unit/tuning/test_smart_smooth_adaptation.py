"""Tests for SmartSmoothAdaptation: gradual transformation framework."""

import pytest

from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation


def _make_ssa(adapt_fn, interpolators=None, target=0.9, min_step=0.001,
              before_cycle=None):
    return SmartSmoothAdaptation(
        adaptation_fn=adapt_fn,
        interpolators=interpolators or [lambda t: t],
        get_target=lambda: target,
        min_step=min_step,
        before_cycle=before_cycle,
    )


class TestSmartSmoothAdaptation:
    def test_constant_metric_completes(self):
        """With an always-committing adaptation_fn, adaptation should reach t=1."""
        target = 0.9
        adapt_log = []

        def adapt_fn(rate):
            adapt_log.append(rate)
            return rate  # signal success

        ssa = _make_ssa(adapt_fn, target=target)
        ssa.adapt_smoothly(max_cycles=20)

        assert len(adapt_log) > 0
        assert adapt_log[-1] >= 0.99

    def test_max_cycles_respected(self):
        adapt_calls = []

        ssa = _make_ssa(lambda rate: adapt_calls.append(rate))
        ssa.adapt_smoothly(max_cycles=3)
        assert len(adapt_calls) <= 3

    def test_rollback_causes_smaller_steps(self):
        """When adaptation_fn returns a lower rate (rollback), step shrinks."""
        proposed_rates = []

        def adapt_fn(rate):
            proposed_rates.append(rate)
            if rate > 0.5:
                return 0.0  # rollback to 0
            return rate  # commit

        ssa = _make_ssa(adapt_fn, min_step=0.01)
        ssa.adapt_smoothly(max_cycles=30)

        failed = [r for r in proposed_rates if r > 0.5]
        succeeded = [r for r in proposed_rates if r <= 0.5]
        assert len(failed) >= 1, "Should attempt at least one step beyond 0.5"
        assert len(succeeded) >= 1, "Should succeed at smaller steps"

    def test_before_cycle_called_once_per_cycle(self):
        before_cycle_calls = []
        cycle_count = [0]

        def adapt_fn(rate):
            cycle_count[0] += 1
            if cycle_count[0] <= 2:
                return 0.0  # rollback first 2 cycles to force 3 total
            return rate  # commit on 3rd

        ssa = _make_ssa(
            adapt_fn,
            before_cycle=lambda: before_cycle_calls.append(1),
            min_step=0.01,
        )
        ssa.adapt_smoothly(max_cycles=5)
        assert len(before_cycle_calls) >= 3, "before_cycle should be called once per cycle"

    def test_rollback_none_is_backward_compatible(self):
        """When adaptation_fn returns None, t always advances (old behavior)."""
        adapt_log = []

        ssa = _make_ssa(lambda rate: adapt_log.append(rate))
        ssa.adapt_smoothly(max_cycles=5)
        assert len(adapt_log) == 5 or adapt_log[-1] >= 0.99

    def test_get_target_callable_used(self):
        """SmartSmoothAdaptation reads the target via get_target callable."""
        current_target = [0.9]

        def get_target():
            return current_target[0]

        ssa = SmartSmoothAdaptation(
            adaptation_fn=lambda r: None,
            interpolators=[lambda t: t],
            get_target=get_target,
            min_step=0.001,
        )

        assert ssa.get_target() == pytest.approx(0.9)

        current_target[0] = 0.8
        assert ssa.get_target() == pytest.approx(0.8)

    def test_step_grows_on_commit(self):
        """After successful commits, the step should grow."""
        proposed_rates = []

        def adapt_fn(rate):
            proposed_rates.append(rate)
            return rate  # always commit

        ssa = _make_ssa(adapt_fn)
        ssa.adapt_smoothly(max_cycles=10)

        # Should reach 1.0 in few cycles due to step growth
        assert proposed_rates[-1] >= 0.99
        assert len(proposed_rates) <= 10

    def test_step_halves_on_rollback(self):
        """After rollback, the next proposed step should be smaller."""
        proposed_rates = []

        def adapt_fn(rate):
            proposed_rates.append(rate)
            if len(proposed_rates) <= 2:
                return 0.0  # rollback first two attempts
            return rate  # then commit

        ssa = _make_ssa(adapt_fn, min_step=0.01)
        ssa.adapt_smoothly(max_cycles=20)

        # First attempt is at 1.0, rollback → step halves to 0.5
        # Second at 0.5, rollback → step halves to 0.25
        # Third at 0.25, commit
        assert proposed_rates[0] == pytest.approx(1.0)
        assert proposed_rates[1] == pytest.approx(0.5)
        assert proposed_rates[2] == pytest.approx(0.25)

    def test_multiple_interpolators(self):
        received = []

        def adapt_fn(a, b, c):
            received.append((a, b, c))
            return a  # commit at the first interpolated value

        ssa = _make_ssa(
            adapt_fn,
            interpolators=[lambda t: t, lambda t: t * 2, lambda t: t * 3],
            target=0.9,
        )
        ssa.adapt_smoothly(max_cycles=3)

        for a, b, c in received:
            assert b == pytest.approx(a * 2, abs=1e-6)
            assert c == pytest.approx(a * 3, abs=1e-6)

    def test_min_step_terminates_loop(self):
        """When step shrinks below min_step, the loop should exit."""
        call_count = [0]

        def adapt_fn(rate):
            call_count[0] += 1
            return 0.0  # always rollback

        ssa = _make_ssa(adapt_fn, min_step=0.1)
        ssa.adapt_smoothly(max_cycles=100)

        # step starts at 1.0, halves: 0.5, 0.25, 0.125, 0.0625 < 0.1 → stop
        assert call_count[0] <= 4

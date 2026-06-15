"""Unit contract for AcceptanceSensor decision math (P2a, bit-exact)."""

import pytest

from mimarsinan.tuning.orchestration.acceptance_sensor import (
    AcceptanceSensor,
    BaselineRef,
)


class _Budget:
    def __init__(self, se):
        self._se = se

    def accuracy_se(self):
        return self._se


def test_calibrate_baseline_tolerance_and_mean():
    sensor = AcceptanceSensor(_Budget(0.01))
    vals = iter([0.90, 0.86])
    ref = sensor.calibrate_baseline(lambda n: next(vals), eval_n_batches=4)
    assert isinstance(ref, BaselineRef)
    assert ref.se == 0.01
    assert ref.empirical_noise == pytest.approx(0.04)
    # max(min(max(3*0.01, 3*0.04), 0.05), 0.005) = max(min(0.12, 0.05), 0.005) = 0.05
    assert ref.rollback_tolerance == pytest.approx(0.05)
    assert ref.baseline == pytest.approx(0.88)


def test_calibrate_baseline_tolerance_floor_and_se_branch():
    # tiny noise → tolerance from 3*SE, clamped to the 0.005 floor when below it
    sensor = AcceptanceSensor(_Budget(0.001))
    vals = iter([0.9, 0.9])
    ref = sensor.calibrate_baseline(lambda n: next(vals), eval_n_batches=4)
    assert ref.rollback_tolerance == pytest.approx(0.005)  # max(0.003, 0.005)


def test_absolute_floor_combinations():
    assert AcceptanceSensor.absolute_floor(None, None, None) is None
    assert AcceptanceSensor.absolute_floor(0.9, 0.05, None) == pytest.approx(0.855)
    assert AcceptanceSensor.absolute_floor(None, None, 0.8) == pytest.approx(0.8)
    # stricter of baseline-anchored (0.855) and hard floor (0.87) → 0.87
    assert AcceptanceSensor.absolute_floor(0.9, 0.05, 0.87) == pytest.approx(0.87)


def test_absolute_floor_drops_unachievable_hard_floor():
    """A hard floor ABOVE the rate-0 baseline can never be met by any transform, so
    using it as a per-cycle rollback trigger stalls the ramp to rate 0 (every cycle
    rolls back). An unachievable hard floor is dropped — the achievable
    baseline-anchored floor governs the per-cycle gate; the deployment shortfall is
    reported at finalize, not enforced as a ramp-stalling rollback trigger."""
    # baseline 0.5, hard floor 0.855 (unachievable) → only the baseline term 0.475.
    assert AcceptanceSensor.absolute_floor(0.5, 0.05, 0.855) == pytest.approx(0.475)
    # An achievable hard floor (<= baseline) is still honored (the strong-teacher,
    # golden-trace case is unchanged).
    assert AcceptanceSensor.absolute_floor(0.96, 0.05, 0.90) == pytest.approx(0.912)
    assert AcceptanceSensor.absolute_floor(0.9, 0.05, 0.87) == pytest.approx(0.87)
    # No baseline known → achievability can't be judged → hard floor kept.
    assert AcceptanceSensor.absolute_floor(None, None, 0.8) == pytest.approx(0.8)


def test_is_catastrophic():
    assert AcceptanceSensor.is_catastrophic(0.71, 0.9) is True   # < 0.72
    assert AcceptanceSensor.is_catastrophic(0.73, 0.9) is False
    assert AcceptanceSensor.is_catastrophic(None, 0.9) is False


def test_is_catastrophic_factor_is_injectable_and_deliberately_loose():
    # The default margin (0.8) is deliberately coarse: ``instant_acc`` is the
    # PRE-recovery accuracy, where a large drop is expected and reclaimed by
    # recovery training. A 17% instant drop is NOT catastrophic — only an
    # unrecoverable collapse past target*0.8 fast-fails.
    assert AcceptanceSensor.is_catastrophic(0.75, 0.9) is False  # 0.75 > 0.72
    assert AcceptanceSensor.is_catastrophic(0.71, 0.9) is True   # 0.71 < 0.72
    # The factor is an injectable default, not a buried constant: a caller may
    # pass a stricter margin without editing the module.
    assert AcceptanceSensor.is_catastrophic(0.75, 0.9, factor=0.85) is True   # < 0.765
    assert AcceptanceSensor.is_catastrophic(0.80, 0.9, factor=0.85) is False  # > 0.765


def test_rollback_threshold_and_gate():
    # relative 0.85, abs floor 0.86 → stricter 0.86
    assert AcceptanceSensor.rollback_threshold(0.9, 0.05, 0.86) == pytest.approx(0.86)
    # no absolute floor → relative only
    assert AcceptanceSensor.rollback_threshold(0.9, 0.05, None) == pytest.approx(0.85)
    assert AcceptanceSensor.is_rollback(0.84, 0.85) is True
    assert AcceptanceSensor.is_rollback(0.86, 0.85) is False


def test_reached_target():
    assert AcceptanceSensor.reached_target(0.86, 0.9, 0.05) is True   # >= 0.85
    assert AcceptanceSensor.reached_target(0.84, 0.9, 0.05) is False

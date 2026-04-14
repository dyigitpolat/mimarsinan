"""Tests verifying the target relaxation policy in SmoothAdaptationTuner.

The target only decays when the tuner's post-recovery metric repeatedly
**misses the current target** for ``_STUCK_STREAK_REQUIRED`` consecutive
committed cycles. Normal-sized commits that reach the target, and rollbacks,
never trigger target relaxation. The target is always bounded by
``[floor, original_metric]`` and restores to the pre-relaxation value the
moment a committed cycle reaches the original target.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import (
    SmoothAdaptationTuner,
    _STUCK_STREAK_REQUIRED,
)


class _TrackingTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_calls = []
        self._mock_instant_acc = 0.85
        self._mock_post_acc = 0.88
        self._mock_pre_cycle_acc = 0.89

    def _update_and_evaluate(self, rate):
        self.adaptation_calls.append(rate)
        return self._mock_instant_acc

    def _find_lr(self):
        return 0.001

    def _patch_trainer(self):
        tuner = self
        # The trainer is queried twice per cycle:
        #   1) validate_n_batches before the transformation → pre_cycle_acc
        #   2) validate_n_batches after recovery            → post_acc
        # We alternate via a call counter to return the right one.
        tuner._validate_call_idx = 0

        def _mock_validate_n_batches(n):
            idx = tuner._validate_call_idx
            tuner._validate_call_idx = 1 - idx
            if idx == 0:
                return tuner._mock_pre_cycle_acc
            return tuner._mock_post_acc

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.train_steps_until_target = lambda *a, **kw: None
        self.trainer.test = lambda: tuner._mock_post_acc


class TestTargetStableOnReachingTarget:
    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _TrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        tuner._rollback_tolerance = 0.05
        return tuner

    def test_target_unchanged_after_commit_that_reaches_target(self, setup):
        """A committed cycle whose post_acc reaches target does not decay it."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        # post_acc 0.92 reaches target 0.9 → missed streak stays 0.
        tuner._mock_pre_cycle_acc = 0.88
        tuner._mock_post_acc = 0.92
        tuner._patch_trainer()

        tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.5
        assert tuner.target_adjuster.target_metric == original

    def test_target_unchanged_after_many_cycles_reaching_target(self, setup):
        """Multiple commits that reach the target never trigger decay."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_pre_cycle_acc = 0.88
        tuner._mock_post_acc = 0.92
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric == original

    def test_target_unchanged_on_rollback(self, setup):
        """Rollbacks never trigger target decay (they return early)."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_pre_cycle_acc = 0.88
        tuner._mock_post_acc = 0.10  # far below pre_cycle_acc → rollback
        tuner._patch_trainer()

        for _ in range(10):
            tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.0
        assert tuner.target_adjuster.target_metric == original


class TestTargetDecaysOnMissedTargetStreak:
    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        cfg["degradation_tolerance"] = 0.05
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _TrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        tuner._rollback_tolerance = 0.05
        return tuner

    def test_decays_after_streak_of_missed_target_commits(self, setup):
        """Target decays after N consecutive committed cycles that miss the target."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        # post_acc 0.86 is below (target 0.9 - margin 0.05) == 0.85? no, 0.86 >= 0.85.
        # Use 0.84 to definitively miss target - margin.
        tuner._mock_pre_cycle_acc = 0.82
        tuner._mock_post_acc = 0.84
        tuner._patch_trainer()

        for i in range(1, _STUCK_STREAK_REQUIRED + 1):
            tuner._adaptation(0.1 * i)

        assert tuner.target_adjuster.target_metric < original, (
            "Target should decay after missed-target streak"
        )
        assert tuner.target_adjuster.target_metric >= tuner.target_adjuster.floor

    def test_streak_resets_on_cycle_that_reaches_target(self, setup):
        """A committed cycle reaching target resets the missed-target streak
        and restores any pre-relaxation target."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_pre_cycle_acc = 0.82
        tuner._mock_post_acc = 0.84   # misses target
        tuner._patch_trainer()

        for i in range(1, _STUCK_STREAK_REQUIRED):
            tuner._adaptation(0.1 * i)

        assert tuner._missed_target_streak == _STUCK_STREAK_REQUIRED - 1

        # A cycle that reaches the (original) target resets the streak.
        tuner._mock_pre_cycle_acc = 0.88
        tuner._mock_post_acc = 0.92
        tuner._adaptation(0.5)
        assert tuner._missed_target_streak == 0
        assert tuner.target_adjuster.target_metric == original

    def test_target_bounded_by_floor(self, setup):
        """Even after many missed-target streaks, target stays above floor."""
        tuner = setup
        tuner._mock_pre_cycle_acc = 0.82
        tuner._mock_post_acc = 0.84
        tuner._patch_trainer()

        rate = 0.0
        for _ in range(100):
            rate += 0.05
            if rate > 1.0:
                rate = 0.05
                tuner._committed_rate = 0.0
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric >= tuner.target_adjuster.floor

    def test_target_bounded_by_original(self, setup):
        """Target can never exceed original_metric."""
        tuner = setup
        tuner._mock_pre_cycle_acc = 0.90
        tuner._mock_post_acc = 0.95
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric <= tuner.target_adjuster.original_metric


class TestOriginalMetricInvariant:
    def test_original_metric_never_modified(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _TrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        tuner._rollback_tolerance = 0.05
        tuner._mock_pre_cycle_acc = 0.84
        tuner._mock_post_acc = 0.86
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.original_metric == 0.9

    def test_floor_derived_from_degradation_tolerance(self, tmp_path):
        cfg = default_config()
        cfg["degradation_tolerance"] = 0.05
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _TrackingTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        expected_floor = 0.9 * (1.0 - 0.05)
        assert tuner.target_adjuster.floor == pytest.approx(expected_floor, rel=0.01)

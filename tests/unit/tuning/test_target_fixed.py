"""Tests verifying the target relaxation policy in SmoothAdaptationTuner.

The target only decays when the tuner is stuck: 3+ consecutive committed
steps smaller than 1%.  Normal-sized commits and rollbacks never trigger
target relaxation.  The target is always bounded by [floor, original_metric].
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import (
    SmoothAdaptationTuner,
    _SMALL_STEP_THRESHOLD,
    _STUCK_STREAK_REQUIRED,
)


class _TrackingTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_calls = []
        self._mock_instant_acc = 0.85
        self._mock_post_acc = 0.88

    def _update_and_evaluate(self, rate):
        self.adaptation_calls.append(rate)
        return self._mock_instant_acc

    def _find_lr(self):
        return 0.001

    def _patch_trainer(self):
        tuner = self

        def _mock_validate_n_batches(n):
            return tuner._mock_post_acc

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.train_steps_until_target = lambda *a, **kw: None
        self.trainer.test = lambda: tuner._mock_post_acc


class TestTargetStableOnNormalProgress:
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

    def test_target_unchanged_after_large_commit(self, setup):
        """A committed step >= 1% does not decay the target."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.88
        tuner._patch_trainer()

        tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.5
        assert tuner.target_adjuster.target_metric == original

    def test_target_unchanged_after_many_large_commits(self, setup):
        """Multiple large-step commits never trigger decay."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.88
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric == original

    def test_target_unchanged_on_rollback(self, setup):
        """Rollbacks never trigger target decay."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.10  # below threshold → rollback
        tuner._patch_trainer()

        for _ in range(10):
            tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.0
        assert tuner.target_adjuster.target_metric == original


class TestTargetDecaysWhenStuck:
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

    def test_decays_after_streak_of_small_commits(self, setup):
        """Target decays after _STUCK_STREAK_REQUIRED consecutive small steps."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.88
        tuner._patch_trainer()

        small_step = _SMALL_STEP_THRESHOLD / 2
        for i in range(1, _STUCK_STREAK_REQUIRED + 1):
            tuner._adaptation(small_step * i)

        assert tuner.target_adjuster.target_metric < original, (
            "Target should decay after stuck streak"
        )
        assert tuner.target_adjuster.target_metric >= tuner.target_adjuster.floor

    def test_streak_resets_on_large_step(self, setup):
        """A large committed step resets the small-step streak."""
        tuner = setup
        original = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.88
        tuner._patch_trainer()

        small_step = _SMALL_STEP_THRESHOLD / 2
        for i in range(1, _STUCK_STREAK_REQUIRED):
            tuner._adaptation(small_step * i)

        assert tuner._small_step_streak == _STUCK_STREAK_REQUIRED - 1

        tuner._adaptation(0.5)
        assert tuner._small_step_streak == 0
        assert tuner.target_adjuster.target_metric == original

    def test_target_bounded_by_floor(self, setup):
        """Even after many stuck streaks, target stays above floor."""
        tuner = setup
        tuner._mock_post_acc = 0.88
        tuner._patch_trainer()

        rate = 0.0
        small_step = _SMALL_STEP_THRESHOLD / 2
        for _ in range(100):
            rate += small_step
            if rate > 1.0:
                rate = small_step
                tuner._committed_rate = 0.0
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric >= tuner.target_adjuster.floor

    def test_target_bounded_by_original(self, setup):
        """Target can never exceed original_metric."""
        tuner = setup
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

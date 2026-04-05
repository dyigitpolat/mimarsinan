"""Tests verifying that the adaptation target stays fixed (no decay).

After Fix L, update_target() is never called during adaptation loops.
_get_target() always returns the original target_accuracy passed to the
tuner constructor. This prevents progressive degradation that caused
pipeline assertion failures.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class _TrackingTuner(SmoothAdaptationTuner):
    """Concrete tuner that tracks _adaptation calls and returns configurable
    validation accuracy."""

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
        """Replace trainer methods with deterministic stubs."""
        tuner = self

        def _mock_validate_n_batches(n):
            return tuner._mock_post_acc

        def _mock_train_steps(*a, **kw):
            pass

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.train_steps_until_target = _mock_train_steps
        self.trainer.test = lambda: tuner._mock_post_acc


class TestTargetStaysFixed:
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

    def test_target_unchanged_after_successful_commit(self, setup):
        """After a successful adaptation cycle (no rollback), the target must
        remain exactly equal to the original target_accuracy."""
        tuner = setup
        original_target = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.88  # above threshold 0.9*(1-0.05) = 0.855
        tuner._patch_trainer()

        tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.5, "Should commit"
        assert tuner.target_adjuster.target_metric == original_target, (
            f"Target must stay at {original_target}, "
            f"got {tuner.target_adjuster.target_metric}"
        )

    def test_target_unchanged_after_multiple_commits(self, setup):
        """Even after many successful commits, the target never decays."""
        tuner = setup
        original_target = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.86  # above 0.855 threshold
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.target_metric == original_target

    def test_target_unchanged_after_rollback(self, setup):
        """After rollback, the target must remain at the original value."""
        tuner = setup
        original_target = tuner.target_adjuster.target_metric
        tuner._mock_post_acc = 0.10  # far below threshold -> rollback
        tuner._patch_trainer()

        tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.0, "Should rollback"
        assert tuner.target_adjuster.target_metric == original_target

    def test_target_unchanged_after_mixed_commits_and_rollbacks(self, setup):
        """Target stays fixed through any sequence of commits and rollbacks."""
        tuner = setup
        original_target = tuner.target_adjuster.target_metric
        tuner._patch_trainer()

        # Commit
        tuner._mock_post_acc = 0.87
        tuner._adaptation(0.3)
        assert tuner._committed_rate == 0.3

        # Rollback
        tuner._mock_post_acc = 0.10
        tuner._adaptation(0.8)
        assert tuner._committed_rate == 0.3

        # Commit again
        tuner._mock_post_acc = 0.86
        tuner._adaptation(0.6)
        assert tuner._committed_rate == 0.6

        assert tuner.target_adjuster.target_metric == original_target

    def test_get_target_returns_original(self, setup):
        """_get_target() always returns the original target_accuracy."""
        tuner = setup
        tuner._mock_post_acc = 0.86
        tuner._patch_trainer()

        initial = tuner._get_target()
        tuner._adaptation(0.5)
        after_commit = tuner._get_target()

        assert initial == after_commit == 0.9

    def test_rollback_threshold_uses_fixed_target(self, setup):
        """The rollback threshold is always target * (1 - tolerance), not a
        decaying value. This means the same accuracy that passes on the first
        cycle also passes on the tenth cycle."""
        tuner = setup
        tuner._patch_trainer()

        # First cycle: barely above threshold -> commits
        tuner._mock_post_acc = 0.856  # just above 0.9*0.95 = 0.855
        tuner._adaptation(0.3)
        assert tuner._committed_rate == 0.3

        # After several commits with same accuracy, threshold is unchanged
        for rate in [0.5, 0.7, 0.9]:
            tuner._adaptation(rate)
        assert tuner.target_adjuster.target_metric == 0.9

    def test_original_metric_preserved(self, setup):
        """target_adjuster.original_metric is never modified."""
        tuner = setup
        tuner._mock_post_acc = 0.86
        tuner._patch_trainer()

        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tuner._adaptation(rate)

        assert tuner.target_adjuster.original_metric == 0.9

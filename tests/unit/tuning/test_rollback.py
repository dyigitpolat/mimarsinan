"""Tests for best-state rollback in SmoothAdaptationTuner._adaptation()."""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import (
    SmoothAdaptationTuner,
    CATASTROPHIC_DROP_FACTOR,
)


class _DummyTuner(SmoothAdaptationTuner):
    """Concrete subclass that tracks _update_and_evaluate calls and allows
    controlling the accuracy returned by validate_n_batches."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.update_calls = []
        self._validate_sequence = []
        self._validate_idx = 0
        self._committed_rate = 0.0

    def _update_and_evaluate(self, rate):
        self.update_calls.append(rate)
        return 0.8  # non-catastrophic instant accuracy (above 80% of target)

    def _find_lr(self):
        return 0.001

    def set_validate_sequence(self, seq):
        """Set a sequence of values that validate_n_batches will return."""
        self._validate_sequence = list(seq)
        self._validate_idx = 0

    def _patch_validate(self):
        """Replace trainer.validate_n_batches with a deterministic sequence
        and stub out train_steps_until_target so only _adaptation's own
        validate_n_batches call consumes from the sequence."""
        tuner = self

        def _mock_validate_n_batches(n):
            idx = tuner._validate_idx
            tuner._validate_idx += 1
            if idx < len(tuner._validate_sequence):
                return tuner._validate_sequence[idx]
            return tuner._validate_sequence[-1] if tuner._validate_sequence else 0.5

        self.trainer.validate_n_batches = _mock_validate_n_batches
        self.trainer.train_steps_until_target = lambda *a, **kw: None


class TestRollback:
    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _DummyTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        tuner._rollback_tolerance = 0.05
        return tuner

    def test_no_rollback_when_accuracy_holds(self, setup):
        """If post_acc >= pre_cycle_acc - noise_margin, no rollback."""
        tuner = setup
        # validate sequence: [pre_cycle_acc, post_acc]
        tuner.set_validate_sequence([0.87, 0.87])
        tuner._patch_validate()

        result = tuner._adaptation(0.5)

        assert tuner.update_calls == [0.5]
        assert result == 0.5
        assert tuner._committed_rate == 0.5

    def test_rollback_when_post_acc_drops_below_pre_step(self, setup):
        """If post_acc < pre_cycle_acc - noise_margin, state is restored."""
        tuner = setup
        # pre=0.87, post=0.10 → drops far below pre - 0.05 = 0.82 → rollback
        tuner.set_validate_sequence([0.87, 0.10])
        tuner._patch_validate()

        pre_state = {k: v.clone() for k, v in tuner.model.state_dict().items()}

        result = tuner._adaptation(0.5)

        assert result == 0.0, "Should return committed_rate (0.0) on rollback"
        for k, v in tuner.model.state_dict().items():
            assert torch.allclose(v, pre_state[k], atol=1e-6), (
                f"Parameter {k} should be restored after rollback"
            )

    def test_target_unchanged_on_rollback(self, setup):
        """On rollback the target remains unchanged — relaxation only triggers
        after multiple consecutive COMMITTED cycles miss the target."""
        tuner = setup
        # pre=0.87, post=0.01 → huge drop → rollback
        tuner.set_validate_sequence([0.87, 0.01])
        tuner._patch_validate()

        original_target = tuner.target_adjuster.target_metric

        tuner._adaptation(0.5)

        assert tuner._committed_rate == 0.0, "Should rollback"
        assert tuner.target_adjuster.target_metric == original_target, (
            "On rollback, target must remain unchanged. "
            f"Expected {original_target}, got {tuner.target_adjuster.target_metric}"
        )

    def test_rollback_threshold_is_pre_step_based(self, setup):
        """Rollback uses ``pre_cycle_acc - noise_margin``, not target-based."""
        tuner = setup
        tuner._rollback_tolerance = 0.1

        # pre=0.82, post=0.75: drop of 0.07 < 0.10 margin → no rollback.
        tuner.set_validate_sequence([0.82, 0.75])
        tuner._patch_validate()
        result = tuner._adaptation(0.5)
        assert result == 0.5

        # Next cycle: pre=0.75, post=0.60: drop of 0.15 > 0.10 → rollback.
        tuner._validate_idx = 0
        tuner.set_validate_sequence([0.75, 0.60])
        tuner._patch_validate()
        result = tuner._adaptation(0.6)
        assert result == 0.5  # rolled back to last committed rate


class TestDirectToleranceSemantic:
    def test_tolerance_noise_only(self, tmp_path):
        """rollback tolerance = 3 * accuracy_se (noise only, decoupled from dt)."""
        cfg = default_config()
        cfg["degradation_tolerance"] = 0.08
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _DummyTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
        se = tuner._budget.accuracy_se()
        assert tuner._rollback_tolerance == pytest.approx(3 * se)

    def test_catastrophic_drop_factor_value(self):
        """CATASTROPHIC_DROP_FACTOR must be 0.8 (fast-fail below 80% of target)."""
        assert CATASTROPHIC_DROP_FACTOR == 0.8

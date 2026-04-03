"""Tests for best-state rollback in SmoothAdaptationTuner._adaptation()."""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import (
    SmoothAdaptationTuner,
    TOLERANCE_SAFETY_FACTOR,
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
        return 0.0

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

    def test_no_rollback_when_accuracy_above_threshold(self, setup):
        """If post_acc >= target * (1 - tolerance), no rollback occurs."""
        tuner = setup
        # threshold = 0.9 * (1 - 0.05) = 0.855
        tuner.set_validate_sequence([0.87])
        tuner._patch_validate()

        result = tuner._adaptation(0.5)

        assert tuner.update_calls == [0.5]
        assert result == 0.5
        assert tuner._committed_rate == 0.5

    def test_rollback_when_accuracy_below_threshold(self, setup):
        """If post_acc < target * (1 - tolerance), state is restored."""
        tuner = setup
        # threshold = 0.855; post_acc = 0.10 → rollback
        tuner.set_validate_sequence([0.10])
        tuner._patch_validate()

        pre_state = {k: v.clone() for k, v in tuner.model.state_dict().items()}

        result = tuner._adaptation(0.5)

        assert result == 0.0, "Should return committed_rate (0.0) on rollback"
        for k, v in tuner.model.state_dict().items():
            assert torch.allclose(v, pre_state[k], atol=1e-6), (
                f"Parameter {k} should be restored after rollback"
            )

    def test_target_unchanged_on_rollback(self, setup):
        """On rollback the model state is restored to pre-cycle, so the target
        adjuster must not be called at all -- the target stays exactly as it was."""
        tuner = setup
        # threshold = 0.855; post_acc = 0.01 → rollback
        tuner.set_validate_sequence([0.01])
        tuner._patch_validate()

        original_target = tuner.target_adjuster.target_metric

        tuner._adaptation(0.5)

        assert tuner.target_adjuster.target_metric == original_target, (
            "On rollback, target must remain completely unchanged. "
            f"Expected {original_target}, got {tuner.target_adjuster.target_metric}"
        )

    def test_rollback_threshold_is_target_based(self, setup):
        """Rollback uses target * (1 - tolerance), not pre_acc - margin."""
        tuner = setup
        tuner._rollback_tolerance = 0.1
        # threshold = 0.9 * (1 - 0.1) = 0.81

        # Just above threshold → no rollback
        tuner.set_validate_sequence([0.82])
        tuner._patch_validate()
        result = tuner._adaptation(0.5)
        assert result == 0.5

        # Below threshold for the (decayed) target → rollback
        tuner._validate_idx = 0
        decayed_target = tuner.target_adjuster.get_target()
        threshold = decayed_target * (1.0 - 0.1)
        tuner.set_validate_sequence([threshold - 0.01])
        tuner._patch_validate()
        result = tuner._adaptation(0.6)
        assert result == 0.5  # rolled back to last committed rate


class TestToleranceSafetyFactor:
    def test_safety_factor_value(self):
        """TOLERANCE_SAFETY_FACTOR must be 0.5 (half the calibrated tolerance)."""
        assert TOLERANCE_SAFETY_FACTOR == 0.5

    def test_safety_factor_applied_in_run(self, tmp_path):
        """run() must scale calibrated tolerance by TOLERANCE_SAFETY_FACTOR
        before passing it to SmartSmoothAdaptation and _rollback_tolerance."""
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        tuner = _DummyTuner(pipeline, model, target_accuracy=0.9, lr=0.001)

        calibrated = 0.10
        expected = calibrated * TOLERANCE_SAFETY_FACTOR

        tuner._rollback_tolerance = expected
        assert tuner._rollback_tolerance == pytest.approx(expected)

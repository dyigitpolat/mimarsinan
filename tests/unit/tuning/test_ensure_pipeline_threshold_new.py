"""Tests for the new ``_attempt_recovery_if_below_floor`` safety net.

Semantics (see ``smooth_adaptation_refactor`` plan):
- Named ``_attempt_recovery_if_below_floor`` (aliased as
  ``_ensure_pipeline_threshold`` for backwards compatibility with callers
  that import it under the old name).
- Uses ``trainer.validate()`` only — NEVER ``trainer.test()``.
- When validation is already above the hard floor, returns immediately.
- When below the hard floor, runs at most two recovery attempts; both
  decisions use validation.
- If still below the floor after the attempts, emits a visible warning
  and returns the best validation seen (does not raise — the pipeline's
  step-level assertion is the single test-based gate).
"""

import warnings
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


def _make_tuner_under_test(tmp_path, *, floor, lr=0.001):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    ce = nn.CrossEntropyLoss()
    pipeline.loss = lambda m, x, y: ce(m(x), y)
    model = make_tiny_supermodel()

    tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
    tuner.pipeline = pipeline
    tuner.model = model
    tuner.pipeline_lr = lr
    tuner._pipeline_tolerance = 0.05
    tuner._rollback_tolerance = 0.05
    tuner._pipeline_hard_floor = floor
    tuner.target_adjuster = MagicMock()
    tuner.target_adjuster.get_target.return_value = 0.90
    tuner.target_adjuster.original_metric = 0.90
    tuner._budget = MagicMock()
    tuner._budget.max_training_steps = 100
    tuner._budget.eval_n_batches = 5
    tuner._budget.progress_eval_batches = 3
    tuner._budget.check_interval = 10
    tuner._budget.accuracy_se.return_value = 0.005
    tuner._cached_lr = lr
    tuner.trainer = MagicMock()

    def _test_boom(*args, **kwargs):
        raise AssertionError(
            "trainer.test() must never be called by the new safety net"
        )

    tuner.trainer.test.side_effect = _test_boom
    return tuner


class TestSkipsWhenAboveFloor:
    def test_returns_best_validate_when_above_floor(self, tmp_path):
        tuner = _make_tuner_under_test(tmp_path, floor=0.80)
        tuner.trainer.validate.return_value = 0.90
        tuner._find_lr = MagicMock(return_value=0.001)

        result = tuner._attempt_recovery_if_below_floor()

        assert result == pytest.approx(0.90)
        tuner.trainer.train_steps_until_target.assert_not_called()


class TestRecoversWhenBelowFloor:
    def test_one_attempt_recovery(self, tmp_path):
        tuner = _make_tuner_under_test(tmp_path, floor=0.80)
        # validate below floor first, then above after training
        tuner.trainer.validate.side_effect = [0.70, 0.85]
        tuner._find_lr = MagicMock(return_value=0.001)
        tuner.trainer.train_steps_until_target = MagicMock(return_value=None)

        result = tuner._attempt_recovery_if_below_floor()

        assert tuner.trainer.train_steps_until_target.called
        assert result == pytest.approx(0.85)

    def test_no_floor_gate_no_recovery(self, tmp_path):
        tuner = _make_tuner_under_test(tmp_path, floor=None)
        tuner.trainer.validate.return_value = 0.50
        tuner._find_lr = MagicMock(return_value=0.001)

        result = tuner._attempt_recovery_if_below_floor()

        assert result == pytest.approx(0.50)
        tuner.trainer.train_steps_until_target.assert_not_called()


class TestBelowFloorWarning:
    def test_emits_warning_when_cannot_recover(self, tmp_path):
        tuner = _make_tuner_under_test(tmp_path, floor=0.80)
        # validate stays below floor on every call
        tuner.trainer.validate.return_value = 0.50
        tuner._find_lr = MagicMock(return_value=0.001)
        tuner.trainer.train_steps_until_target = MagicMock(return_value=None)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = tuner._attempt_recovery_if_below_floor()
            below = [
                w for w in recorded
                if "below pipeline floor" in str(w.message).lower()
                or "could not recover" in str(w.message).lower()
            ]
            assert len(below) >= 1, (
                f"Expected a 'below pipeline floor' warning. Got: "
                f"{[str(w.message) for w in recorded]}"
            )
        assert result == pytest.approx(0.50)

    def test_no_warning_when_recovered(self, tmp_path):
        tuner = _make_tuner_under_test(tmp_path, floor=0.80)
        tuner.trainer.validate.side_effect = [0.70, 0.90]
        tuner._find_lr = MagicMock(return_value=0.001)
        tuner.trainer.train_steps_until_target = MagicMock(return_value=None)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            tuner._attempt_recovery_if_below_floor()
            below = [
                w for w in recorded
                if "below pipeline floor" in str(w.message).lower()
            ]
            assert below == []


class TestBackwardsCompatibleAlias:
    def test_old_method_name_delegates_to_new_one(self, tmp_path):
        """For callsites that still reference the old name, it should work."""
        tuner = _make_tuner_under_test(tmp_path, floor=0.80)
        tuner.trainer.validate.return_value = 0.90
        tuner._find_lr = MagicMock(return_value=0.001)

        result = tuner._ensure_pipeline_threshold()
        assert result == pytest.approx(0.90)

"""Tests that tuner internals never call ``trainer.test()``.

Policy (see ``smooth_adaptation_refactor`` plan):
- ``trainer.test()`` may only be called at the single measurement point
  in ``PipelineStep.pipeline_metric()``.
- Tuners must make all decisions (rollback, target relaxation, safety net)
  using validation metrics.

These tests monkeypatch ``trainer.test`` to raise ``TestCallNotAllowedError``
and then exercise each tuner's main code paths (adaptation cycle, after_run
recovery, safety-net). Any path that still calls ``trainer.test()`` surfaces
as a visible failure with a clear message.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class TestCallNotAllowedError(AssertionError):
    """Raised when tuner-internal code calls ``trainer.test()``."""


def _make_pipeline(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


def _install_no_test_guard(tuner):
    """Make any tuner-internal ``trainer.test()`` call fail loudly."""
    def _no_test(*args, **kwargs):
        raise TestCallNotAllowedError(
            "trainer.test() was called from tuner-internal code; "
            "this is a test-set data leak. All tuner decisions must use "
            "validation metrics."
        )
    tuner.trainer.test = _no_test


class _StubTuner(SmoothAdaptationTuner):
    """Stub SmoothAdaptationTuner with validate-driven accuracy."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._mock_acc = 0.90

    def _update_and_evaluate(self, rate):
        return self._mock_acc

    def _find_lr(self):
        return 0.001


class TestAdaptationCycleHasNoTestLeak:
    def test_single_adaptation_cycle_does_not_call_test(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        tuner = _StubTuner(pipeline, model, target_accuracy=0.90, lr=0.001)
        tuner._rollback_tolerance = 0.05
        tuner._test_baseline = None  # pre-condition: no pre-cached baseline
        tuner._pipeline_hard_floor = None

        tuner.trainer.validate_n_batches = lambda n: 0.90
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        _install_no_test_guard(tuner)

        # Gradual cycle (rate < 1.0) must not call test()
        tuner._adaptation(0.5)

    def test_full_rate_adaptation_does_not_call_test_directly(self, tmp_path):
        """rate == 1.0 must not call trainer.test() as part of the adaptation
        cycle's internal gate. The strict test gate is the source of the
        test-set leak and is removed under the refactor."""
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        tuner = _StubTuner(pipeline, model, target_accuracy=0.90, lr=0.001)
        tuner._rollback_tolerance = 0.05
        tuner._test_baseline = None
        tuner._pipeline_hard_floor = None

        tuner.trainer.validate_n_batches = lambda n: 0.90
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        _install_no_test_guard(tuner)

        tuner._adaptation(1.0)


class TestSafetyNetRecoveryUsesValidation:
    """``_attempt_recovery_if_below_floor`` uses validate() only."""

    def test_safety_net_above_floor_does_not_call_test(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        tuner = _StubTuner(pipeline, model, target_accuracy=0.90, lr=0.001)
        tuner._pipeline_hard_floor = 0.80
        tuner._rollback_tolerance = 0.05

        tuner.trainer.validate = lambda: 0.90  # well above floor
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        _install_no_test_guard(tuner)

        tuner._attempt_recovery_if_below_floor()

    def test_safety_net_below_floor_does_not_call_test(self, tmp_path):
        """Even when recovery is triggered, trainer.test() must not be called.
        Recovery decisions are made from validation only."""
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        tuner = _StubTuner(pipeline, model, target_accuracy=0.90, lr=0.001)
        tuner._pipeline_hard_floor = 0.80
        tuner._rollback_tolerance = 0.05

        validation_values = iter([0.70, 0.85])  # below then above
        tuner.trainer.validate = lambda: next(validation_values, 0.85)
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        _install_no_test_guard(tuner)

        tuner._attempt_recovery_if_below_floor()

    def test_safety_net_emits_warning_when_cannot_recover(self, tmp_path):
        """If validation stays below floor after recovery attempts, a
        warning is emitted instead of silently returning a below-floor
        value."""
        import warnings

        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        tuner = _StubTuner(pipeline, model, target_accuracy=0.90, lr=0.001)
        tuner._pipeline_hard_floor = 0.80
        tuner._rollback_tolerance = 0.05

        tuner.trainer.validate = lambda: 0.50  # never recovers
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        _install_no_test_guard(tuner)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tuner._attempt_recovery_if_below_floor()
            below_floor_warnings = [
                x for x in w
                if "below pipeline floor" in str(x.message).lower()
                or "could not recover" in str(x.message).lower()
            ]
            assert len(below_floor_warnings) >= 1, (
                f"Expected a warning when validation stays below floor. "
                f"Got warnings: {[str(x.message) for x in w]}"
            )

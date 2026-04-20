"""Tests for the adaptation tuning fixes:

- PruningTuner uses base-class _adaptation (with LR search, not fixed LR)
- PruningTuner._update_and_evaluate is pure evaluation (no training step)
- _recovery_training_hooks protocol injects hooks during recovery
- _ensure_pipeline_threshold retries when test accuracy is marginal
- train_steps_until_target respects min_steps before patience kicks in
- One-shot test gate rejects rate=1.0 commits that fail test() threshold
- Baseline calibration sets tuner target from validate_n_batches at rate 0.0
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from conftest import default_config, MockPipeline, make_tiny_supermodel
from mimarsinan.tuning.adaptation_manager import AdaptationManager


class TestPruningTunerUsesLRSearch:
    """PruningTuner must use _find_lr() (via base-class _adaptation) instead
    of a fixed pretrain LR for recovery training."""

    def test_no_adaptation_override(self):
        """PruningTuner should not define its own _adaptation method — it
        should rely on the base class."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        assert PruningTuner._adaptation is SmoothAdaptationTuner._adaptation

    def test_recovery_hooks_protocol_exists(self):
        """PruningTuner must override _recovery_training_hooks."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        assert PruningTuner._recovery_training_hooks is not SmoothAdaptationTuner._recovery_training_hooks

    def test_base_class_recovery_hooks_returns_empty(self):
        """The default _recovery_training_hooks returns an empty list."""
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner
        obj = object.__new__(SmoothAdaptationTuner)
        assert obj._recovery_training_hooks(0.5) == []


class TestPruningTunerUpdateAndEvaluate:
    """_update_and_evaluate must only apply masks and validate — no training."""

    def test_no_train_one_step_call(self):
        """_update_and_evaluate should not call trainer.train_one_step."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )

        for p in model.get_perceptrons():
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))
        tuner._init_original_weights()

        with patch.object(tuner.trainer, 'train_one_step') as mock_train:
            tuner._update_and_evaluate(0.5)
            mock_train.assert_not_called()


class TestRecoveryHooksProtocol:
    """_recovery_training_hooks returns live PyTorch forward-pre-hooks."""

    def test_pruning_hooks_are_returned_and_removable(self):
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )

        for p in model.get_perceptrons():
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))
        tuner._init_original_weights()

        hooks = tuner._recovery_training_hooks(0.5)
        assert len(hooks) > 0, "Pruning hooks should be returned for non-zero rate"

        for h in hooks:
            assert hasattr(h, 'remove'), "Each hook must be removable"
            h.remove()


class TestEnsurePipelineThreshold:
    """``_ensure_pipeline_threshold`` (new name:
    ``_attempt_recovery_if_below_floor``) uses validation ONLY — never
    ``trainer.test()``. The pipeline's step-level assertion is the only
    place that reads the test set.
    """

    def test_passes_when_above_threshold(self):
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()

        tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
        tuner.pipeline = mock
        tuner.model = model
        tuner.pipeline_lr = 0.001
        tuner._rollback_tolerance = 0.05
        tuner._pipeline_tolerance = 0.05
        tuner._pipeline_hard_floor = 0.80
        tuner.target_adjuster = MagicMock()
        tuner.target_adjuster.get_target.return_value = 0.90
        tuner.target_adjuster.original_metric = 0.90
        tuner._budget = MagicMock()
        tuner._budget.max_training_steps = 100
        tuner._budget.eval_n_batches = 5
        tuner._budget.progress_eval_batches = 3
        tuner._budget.check_interval = 10
        tuner._budget.accuracy_se.return_value = 0.005
        tuner._cached_lr = 0.001
        tuner.trainer = MagicMock()
        tuner.trainer.validate.return_value = 0.90

        result = tuner._ensure_pipeline_threshold()
        assert result == 0.90
        tuner.trainer.train_steps_until_target.assert_not_called()
        tuner.trainer.test.assert_not_called()

    def test_retries_when_below_threshold(self):
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()

        tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
        tuner.pipeline = mock
        tuner.model = model
        tuner.pipeline_lr = 0.001
        tuner._rollback_tolerance = 0.05
        tuner._pipeline_tolerance = 0.05
        tuner._pipeline_hard_floor = 0.85
        tuner.target_adjuster = MagicMock()
        tuner.target_adjuster.get_target.return_value = 0.90
        tuner.target_adjuster.original_metric = 0.90
        tuner._budget = MagicMock()
        tuner._budget.max_training_steps = 100
        tuner._budget.eval_n_batches = 5
        tuner._budget.progress_eval_batches = 3
        tuner._budget.check_interval = 10
        tuner._budget.accuracy_se.return_value = 0.005
        tuner._cached_lr = 0.001
        tuner.trainer = MagicMock()
        tuner.trainer.validate.side_effect = [0.80, 0.88]

        tuner._find_lr = MagicMock(return_value=0.001)

        result = tuner._ensure_pipeline_threshold()
        tuner.trainer.train_steps_until_target.assert_called_once()
        tuner.trainer.test.assert_not_called()
        assert result == 0.88


class TestMinStepsInTraining:
    """train_steps_until_target's min_steps prevents premature patience stopping."""

    def test_min_steps_prevents_early_patience_stop(self):
        from mimarsinan.model_training.basic_trainer import BasicTrainer
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
        from conftest import MockDataProviderFactory

        factory = MockDataProviderFactory()
        dl_factory = DataLoaderFactory(factory, num_workers=0)
        model = make_tiny_supermodel()
        loss = nn.CrossEntropyLoss()

        trainer = BasicTrainer(
            model, "cpu", dl_factory,
            lambda m, x, y: loss(m(x), y),
        )

        steps_executed = [0]
        orig_optimize = trainer._optimize

        def counting_optimize(*args, **kwargs):
            steps_executed[0] += 1
            return orig_optimize(*args, **kwargs)

        trainer._optimize = counting_optimize

        # validate_n_batches always returns 0.5 — never reaches target 1.0,
        # and never improves, so patience would trigger without min_steps.
        trainer.validate_n_batches = lambda n: 0.5
        trainer.test = lambda: 0.5

        trainer.train_steps_until_target(
            lr=0.001,
            max_steps=100,
            target_accuracy=1.0,
            validation_n_batches=1,
            check_interval=1,
            patience=1,
            min_steps=20,
        )

        assert steps_executed[0] >= 20, (
            f"Expected at least 20 steps with min_steps=20, got {steps_executed[0]}"
        )

    def test_without_min_steps_patience_stops_early(self):
        """Without min_steps, patience=1 stops training within a few steps."""
        from mimarsinan.model_training.basic_trainer import BasicTrainer
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
        from conftest import MockDataProviderFactory

        factory = MockDataProviderFactory()
        dl_factory = DataLoaderFactory(factory, num_workers=0)
        model = make_tiny_supermodel()
        loss = nn.CrossEntropyLoss()

        trainer = BasicTrainer(
            model, "cpu", dl_factory,
            lambda m, x, y: loss(m(x), y),
        )

        steps_executed = [0]
        orig_optimize = trainer._optimize

        def counting_optimize(*args, **kwargs):
            steps_executed[0] += 1
            return orig_optimize(*args, **kwargs)

        trainer._optimize = counting_optimize
        trainer.validate_n_batches = lambda n: 0.5
        trainer.test = lambda: 0.5

        trainer.train_steps_until_target(
            lr=0.001,
            max_steps=100,
            target_accuracy=1.0,
            validation_n_batches=1,
            check_interval=1,
            patience=1,
            min_steps=0,
        )

        assert steps_executed[0] < 10, (
            f"Without min_steps, patience=1 should stop early, got {steps_executed[0]}"
        )


class TestPruningTunerAfterRun:
    """PruningTuner._after_run uses LR search and returns test accuracy."""

    def test_after_run_defined(self):
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        assert PruningTuner._after_run is not SmoothAdaptationTuner._after_run

    def test_after_run_calls_ensure_pipeline_threshold(self):
        """_after_run must call _ensure_pipeline_threshold as a safety net."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        import inspect
        source = inspect.getsource(PruningTuner._after_run)
        assert "_ensure_pipeline_threshold" in source

    def test_pruning_tuner_has_no_find_lr_override(self):
        """PruningTuner should use the base-class _find_lr (no override)."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.tuning.unified_tuner import TunerBase

        assert PruningTuner._find_lr is TunerBase._find_lr


class TestOneShotStrictGate:
    """When ``_adaptation`` commits at rate=1.0, it runs a strict
    validation gate against ``_validation_baseline`` (no ``trainer.test()``
    call — that would leak test-set information into the tuning loop).
    """

    def _make_tuner(self):
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
        tuner.name = "Tuning Rate"
        tuner.pipeline = MagicMock()
        tuner.pipeline.reporter = MagicMock()
        tuner.model = MagicMock()
        tuner.pipeline_lr = 0.001
        tuner._committed_rate = 0.0
        tuner._rollback_tolerance = 0.05
        tuner._pipeline_tolerance = 0.01
        tuner._missed_target_streak = 0
        tuner._validation_baseline = 0.90
        tuner._pre_relaxation_target = None
        tuner.target_adjuster = MagicMock()
        tuner.target_adjuster.get_target.return_value = 0.90
        tuner.target_adjuster.original_metric = 0.90
        tuner._budget = MagicMock()
        tuner._budget.max_training_steps = 100
        tuner._budget.eval_n_batches = 5
        tuner._budget.check_interval = 10
        tuner._budget.accuracy_se.return_value = 0.005
        tuner.trainer = MagicMock()
        tuner._find_lr = MagicMock(return_value=0.001)
        tuner._update_and_evaluate = MagicMock(return_value=0.90)
        tuner._recovery_training_hooks = MagicMock(return_value=[])
        tuner._clone_state = MagicMock(return_value=("state", None))
        tuner._restore_state = MagicMock()
        return tuner

    def test_oneshot_rejected_when_validation_fails(self):
        """rate=1.0 commit is rejected when post_acc (validation) falls
        below ``validation_baseline - rollback_tolerance``."""
        tuner = self._make_tuner()
        # pre_cycle_acc = 0.90 (from validate_n_batches), post_acc = 0.80
        # rolls back due to the per-step rollback gate (0.90 - 0.05 = 0.85 > 0.80).
        tuner.trainer.validate_n_batches.side_effect = [0.90, 0.80]
        result = tuner._adaptation(1.0)
        assert result == 0.0
        tuner._restore_state.assert_called()
        tuner.trainer.test.assert_not_called()

    def test_oneshot_accepted_when_validation_passes(self):
        """rate=1.0 commit succeeds when post_acc >= baseline - tolerance."""
        tuner = self._make_tuner()
        tuner.trainer.validate_n_batches.return_value = 0.90
        result = tuner._adaptation(1.0)
        assert result == 1.0
        assert tuner._committed_rate == 1.0
        tuner.trainer.test.assert_not_called()

    def test_sub_one_rate_skips_strict_gate(self):
        """Rates below 1.0 only check the per-step rollback gate."""
        tuner = self._make_tuner()
        tuner.trainer.validate_n_batches.return_value = 0.90
        result = tuner._adaptation(0.5)
        assert result == 0.5
        tuner.trainer.test.assert_not_called()

    def test_oneshot_rejected_when_strict_gate_fails(self):
        """post_acc must stay within rollback_tolerance of the validation
        baseline when rate=1.0 — otherwise rollback."""
        tuner = self._make_tuner()
        # pre_cycle_acc = 0.88; post_acc = 0.84. per-step rollback: 0.88
        # - 0.05 = 0.83 -> 0.84 >= 0.83, so per-step passes. Strict rate=1
        # gate: 0.90 - 0.05 = 0.85 -> 0.84 < 0.85, so rollback.
        tuner.trainer.validate_n_batches.side_effect = [0.88, 0.84]
        result = tuner._adaptation(1.0)
        assert result == 0.0
        tuner._restore_state.assert_called()
        tuner.trainer.test.assert_not_called()


class TestBaselineCalibration:
    """run() sets tuner target from validate_n_batches at rate 0.0."""

    def test_target_set_from_validation(self):
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()

        tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
        tuner.name = "Tuning Rate"
        tuner.pipeline = mock
        tuner.model = model
        tuner.pipeline_lr = 0.001
        tuner._budget = MagicMock()
        tuner._budget.accuracy_se.return_value = 0.005
        tuner._budget.eval_n_batches = 5
        tuner._budget.max_training_steps = 100
        tuner._budget.check_interval = 10
        tuner.target_adjuster = MagicMock()
        tuner.target_adjuster.get_target.return_value = 0.85
        tuner.target_adjuster.original_metric = 0.85
        tuner.trainer = MagicMock()
        tuner.trainer.validate.return_value = 0.87
        tuner.trainer.validate_n_batches.return_value = 0.87

        tuner._update_and_evaluate = MagicMock(return_value=0.87)
        tuner._before_cycle = MagicMock()
        tuner._after_run = MagicMock(return_value=0.87)
        tuner._recovery_training_hooks = MagicMock(return_value=[])
        tuner._clone_state = MagicMock(return_value=("state", None))
        tuner._restore_state = MagicMock()

        tuner._committed_rate = 0.0
        tuner._small_step_streak = 0
        tuner._pipeline_tolerance = 0.05
        tuner._rollback_tolerance = 0.065
        tuner._find_lr = MagicMock(return_value=0.001)

        tuner.run()

        assert tuner.target_adjuster.target_metric == 0.87
        assert tuner.target_adjuster.original_metric == 0.87

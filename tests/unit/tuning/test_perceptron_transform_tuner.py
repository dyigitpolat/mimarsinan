"""Tests for PerceptronTransformTuner and ActivationShiftTuner fixes.

Verifies:
- _adaptation() is NOT overridden: the base SmoothAdaptationTuner._adaptation()
  is used, which includes the one-shot test gate, min_improvement, and hooks.
- _after_run() forces rate=1.0, does recovery training, and calls
  _ensure_pipeline_threshold().
- Rollback correctly saves/restores both model and aux_model state dicts.
- ActivationShiftTuner passes min_improvement to recovery training.
"""

import copy
import inspect
from unittest.mock import patch, MagicMock

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.perceptron_transform_tuner import (
    PerceptronTransformTuner,
)
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)
from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


def _make_pipeline(tmp_path, degradation_tolerance=0.05):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = degradation_tolerance
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


class TestAdaptationNotOverridden:
    """PerceptronTransformTuner must NOT override _adaptation().

    The base class SmoothAdaptationTuner._adaptation() includes the one-shot
    test gate, min_improvement, and recovery hooks. Any divergent override
    silently drops these safeguards.
    """

    def test_adaptation_method_is_base_class(self):
        assert PerceptronTransformTuner._adaptation is SmoothAdaptationTuner._adaptation

    def test_weight_quant_uses_base_adaptation(self):
        assert (
            NormalizationAwarePerceptronQuantizationTuner._adaptation
            is SmoothAdaptationTuner._adaptation
        )


class TestAfterRunForcesFullRate:
    """_after_run() must force perceptron_transformation to rate=1.0,
    do recovery training, and call _ensure_pipeline_threshold()."""

    def test_after_run_overridden(self):
        assert PerceptronTransformTuner._after_run is not SmoothAdaptationTuner._after_run

    def test_after_run_source_calls_ensure_pipeline_threshold(self):
        source = inspect.getsource(PerceptronTransformTuner._after_run)
        assert "_ensure_pipeline_threshold" in source

    def test_after_run_source_forces_full_rate_transform(self):
        source = inspect.getsource(PerceptronTransformTuner._after_run)
        assert "_mixed_transform(1.0)" in source

    def test_after_run_source_calls_continue_to_full_rate(self):
        source = inspect.getsource(PerceptronTransformTuner._after_run)
        assert "_continue_to_full_rate" in source

    def test_after_run_source_calls_update_and_transform_model(self):
        source = inspect.getsource(PerceptronTransformTuner._after_run)
        assert "_update_and_transform_model" in source

    def test_after_run_delegates_recovery_to_ensure_pipeline_threshold(self):
        """_after_run no longer runs unconditional recovery training; that's
        delegated to the validation-only safety net
        (``_attempt_recovery_if_below_floor``, aliased as
        ``_ensure_pipeline_threshold``) which only trains when needed and
        saves/restores state to never make things worse."""
        from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner
        source_after = inspect.getsource(PerceptronTransformTuner._after_run)
        # _after_run does NOT call train_steps_until_target directly
        assert "train_steps_until_target" not in source_after
        # but _attempt_recovery_if_below_floor does, with min_improvement
        source_ept = inspect.getsource(
            SmoothAdaptationTuner._attempt_recovery_if_below_floor
        )
        assert "min_improvement" in source_ept


@pytest.mark.slow
class TestRollbackIncludesAuxModel:
    """clone/restore must round-trip both model and aux_model."""

    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = NormalizationAwarePerceptronQuantizationTuner(
            pipeline, model, quantization_bits=8,
            target_accuracy=0.9, lr=0.001, adaptation_manager=am,
        )
        return tuner

    def test_clone_state_returns_tuple_for_aux_trainer(self, tuner):
        state = tuner._clone_state()
        model_state, extra = state
        assert isinstance(model_state, tuple), (
            "clone_state_for_trainer should return (aux_sd, model_sd) tuple"
        )

    def test_aux_model_restored_on_rollback(self, tuner):
        pre_state = tuner._clone_state()
        aux_before = copy.deepcopy(tuner.trainer.aux_model.state_dict())

        for p in tuner.trainer.aux_model.parameters():
            p.data.fill_(999.0)

        tuner._restore_state(pre_state)

        for key in aux_before:
            assert torch.allclose(
                tuner.trainer.aux_model.state_dict()[key],
                aux_before[key],
                atol=1e-6,
            ), f"aux_model param {key} not restored after rollback"

    def test_model_restored_on_rollback(self, tuner):
        pre_state = tuner._clone_state()
        model_before = copy.deepcopy(tuner.model.state_dict())

        for p in tuner.model.parameters():
            p.data.fill_(999.0)

        tuner._restore_state(pre_state)

        for key in model_before:
            assert torch.allclose(
                tuner.model.state_dict()[key],
                model_before[key],
                atol=1e-6,
            ), f"model param {key} not restored after rollback"


class TestStrictGateActiveForWeightQuantization:
    """The base-class strict validation gate (rate >= 1.0 ->
    ``_validation_baseline`` check) must be active for
    ``NormalizationAwarePerceptronQuantizationTuner``.

    Note: this used to be a ``trainer.test()``-based gate. It has been
    changed to a ``validate()``-based gate to eliminate test-set data
    leakage in the tuner loop. ``trainer.test()`` is only called by
    ``PipelineStep.pipeline_metric()`` at step exit.
    """

    @pytest.fixture
    def tuner(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, degradation_tolerance=0.01)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = NormalizationAwarePerceptronQuantizationTuner(
            pipeline, model, quantization_bits=8,
            target_accuracy=0.9, lr=0.001, adaptation_manager=am,
        )
        return tuner

    def test_strict_gate_rejects_bad_validation_accuracy(self, tuner):
        """If post_acc (validation) drops below ``_validation_baseline -
        rollback_tolerance``, the rate=1.0 one-shot is rolled back."""
        tuner._committed_rate = 0.0

        # pre_cycle_acc baseline high; post_acc below strict gate.
        # ``validate_n_batches`` is called twice per ``_adaptation``:
        # once before ``_update_and_evaluate`` (pre_cycle_acc), once
        # after recovery (post_acc). Use side_effect to return
        # different values.
        call_values = [0.88, 0.50]
        def _validate_n_batches(_n):
            return call_values.pop(0) if call_values else 0.50
        tuner.trainer.validate_n_batches = _validate_n_batches
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None

        tuner.target_adjuster.target_metric = 0.88
        tuner.target_adjuster.original_metric = 0.88
        tuner._validation_baseline = 0.88
        tuner._rollback_tolerance = 0.05

        tuner._adaptation(1.0)

        assert tuner._committed_rate < 1.0 - 1e-6, (
            "One-shot should be rejected when validation is below strict "
            "threshold (validation_baseline 0.88 - tolerance 0.05 = 0.83, "
            "post 0.50)"
        )


class TestActivationShiftTunerMinImprovement:
    """ActivationShiftTuner must pass min_improvement to recovery training."""

    def test_budget_has_nonzero_accuracy_se(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)
        tuner = ActivationShiftTuner(pipeline, model, 0.9, 0.001, am)
        assert tuner._budget.accuracy_se() > 0

    def test_uses_budget_eval_batches_not_validation_steps(self):
        source = inspect.getsource(ActivationShiftTuner.run)
        assert "progress_eval_batches" in source or "eval_n_batches" in source
        assert "validation_steps" not in source

    def test_passes_min_improvement(self):
        source = inspect.getsource(ActivationShiftTuner.run)
        assert "min_improvement" in source

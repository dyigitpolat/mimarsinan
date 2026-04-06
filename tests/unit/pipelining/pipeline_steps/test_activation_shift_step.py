import pytest

from conftest import make_tiny_supermodel

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.pipeline_steps.activation_shift_step import ActivationShiftStep
from mimarsinan.tuning.adaptation_manager import AdaptationManager


def _seed_shift_step(mock_pipeline, *, target_metric=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline._target_metric = target_metric
    mock_pipeline.seed("model", model, step_name="Clamp Adaptation")
    mock_pipeline.seed("adaptation_manager", am, step_name="Clamp Adaptation")
    return model, am


def test_activation_shift_step_uses_step_budgeted_training(monkeypatch, mock_pipeline):
    _seed_shift_step(mock_pipeline)
    calls = {}

    def fail_epoch_path(*_args, **_kwargs):
        raise AssertionError("ActivationShiftStep must not use epoch-based recovery")

    def record_step_path(
        self,
        lr,
        max_steps,
        target_accuracy,
        warmup_steps=0,
        *,
        validation_n_batches=1,
        check_interval=1,
        patience=3,
        min_steps=0,
        min_improvement=1e-3,
    ):
        calls["lr"] = lr
        calls["max_steps"] = max_steps
        calls["target_accuracy"] = target_accuracy
        calls["warmup_steps"] = warmup_steps
        calls["validation_n_batches"] = validation_n_batches
        calls["check_interval"] = check_interval
        calls["patience"] = patience
        calls["min_steps"] = min_steps
        calls["min_improvement"] = min_improvement
        return target_accuracy

    monkeypatch.setattr(BasicTrainer, "train_until_target_accuracy", fail_epoch_path)
    monkeypatch.setattr(BasicTrainer, "train_steps_until_target", record_step_path)
    monkeypatch.setattr(BasicTrainer, "validate_n_batches", lambda self, n_batches: 0.5)

    step = ActivationShiftStep(mock_pipeline)
    step.name = "Activation Shift"
    mock_pipeline.prepare_step(step)
    step.run()

    assert calls["max_steps"] > 0
    assert calls["validation_n_batches"] > 0

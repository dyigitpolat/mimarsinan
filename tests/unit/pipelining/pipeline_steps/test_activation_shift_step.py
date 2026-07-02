import pytest

from conftest import make_tiny_supermodel

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_shift_step import ActivationShiftStep
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager


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


class TestShiftConventionFollowsFloorCeilSSOT:
    """The shift treatment must follow the floor+half-step convention SSOT.

    ttfs_quantized AND the synchronized floor-collapse train the TTFS convention:
    the half-step lives inside the quantize decorator (shift_back), so the shift
    tuner must NOT mutate biases nor raise shift_rate — otherwise the mapping-time
    bias compensation double-shifts and deployment over-fires by a full step
    (the sync_collapse_verify regression: NF 0.9453 -> deployed 0.9289)."""

    @pytest.mark.parametrize(
        "mode,schedule,expect_lif_style_shift",
        [
            ("ttfs_quantized", None, False),
            ("ttfs_cycle_based", "synchronized", False),
            ("ttfs_cycle_based", "cascaded", True),
            ("lif", None, True),
        ],
    )
    def test_shift_branch_matches_convention(
        self, monkeypatch, mock_pipeline, mode, schedule, expect_lif_style_shift
    ):
        import torch

        mock_pipeline.config["spiking_mode"] = mode
        if schedule is not None:
            mock_pipeline.config["ttfs_cycle_schedule"] = schedule
        model, am = _seed_shift_step(mock_pipeline)

        monkeypatch.setattr(
            BasicTrainer,
            "train_steps_until_target",
            lambda self, lr, max_steps, target_accuracy, *a, **kw: target_accuracy,
        )
        monkeypatch.setattr(
            BasicTrainer, "validate_n_batches", lambda self, n_batches: 0.5
        )
        monkeypatch.setattr(BasicTrainer, "validate", lambda self: 0.5)

        before = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        step = ActivationShiftStep(mock_pipeline)
        step.name = "Activation Shift"
        mock_pipeline.prepare_step(step)
        step.run()
        after = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]

        biases_mutated = any(
            not torch.allclose(b, a) for b, a in zip(before, after)
        )
        expected_shift_rate = 1.0 if expect_lif_style_shift else 0.0
        assert am.shift_rate == expected_shift_rate, (
            f"{mode}/{schedule}: shift_rate {am.shift_rate} != {expected_shift_rate}"
        )
        assert biases_mutated == expect_lif_style_shift, (
            f"{mode}/{schedule}: bias mutation {biases_mutated}, "
            f"expected {expect_lif_style_shift}"
        )

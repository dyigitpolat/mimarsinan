"""Config-gated KD against rung-2 (identity-mapped contract) outputs."""

import pytest
import torch

from conftest import make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
    TTFSCycleAdaptationStep,
)
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    _Rung2TeacherFlow,
    TTFSCycleAdaptationTuner,
)


def _seed(mock_pipeline, *, schedule="synchronized", rung2=None):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "ttfs_cycle_based"
    mock_pipeline.config["ttfs_cycle_schedule"] = schedule
    mock_pipeline.config["activation_quantization"] = True
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline.config.setdefault("simulation_steps", 16)
    if rung2 is not None:
        mock_pipeline.config["ttfs_finetune_kd_against_rung2"] = rung2
    mock_pipeline.seed("model", model, step_name="Activation Quantization")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    return model, am


def _run_step(mock_pipeline):
    step = TTFSCycleAdaptationStep(mock_pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


class TestFlagOffMatchesR1:
    def test_default_off_uses_frozen_torch_teacher(self, mock_pipeline):
        _seed(mock_pipeline)
        step = _run_step(mock_pipeline)
        teacher = step.tuner.trainer.loss_function.teacher
        assert not isinstance(teacher, _Rung2TeacherFlow)
        assert hasattr(teacher, "get_perceptrons"), (
            "flag off: teacher must be the frozen torch snapshot"
        )

    def test_flag_ignored_for_cascaded(self, mock_pipeline):
        _seed(mock_pipeline, schedule="cascaded", rung2=True)
        step = _run_step(mock_pipeline)
        teacher = step.tuner.trainer.loss_function.teacher
        assert not isinstance(teacher, _Rung2TeacherFlow), (
            "cascaded trains through the genuine spike forward; rung-2 KD is "
            "a synchronized-only option"
        )


class TestFlagOn:
    def test_kd_teacher_is_identity_mapped_contract_flow(self, mock_pipeline):
        _seed(mock_pipeline, rung2=True)
        step = _run_step(mock_pipeline)
        teacher = step.tuner.trainer.loss_function.teacher
        assert isinstance(teacher, _Rung2TeacherFlow)
        for stage in teacher.flow.hybrid_mapping.stages:
            if stage.kind != "neural":
                continue
            for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
                assert len(placements) == 1, (
                    "rung-2 teacher must evaluate the identity mapping"
                )

    def test_teacher_outputs_are_value_domain_and_finite(self, mock_pipeline):
        _seed(mock_pipeline, rung2=True)
        step = _run_step(mock_pipeline)
        teacher = step.tuner.trainer.loss_function.teacher
        x = torch.rand(4, 1, 8, 8)
        with torch.no_grad():
            logits = teacher(x)
        assert logits.shape == (4, 4)
        assert torch.isfinite(logits).all()
        # Count-scaled flow output is normalized back by 1/T (value domain).
        with torch.no_grad():
            raw = teacher.flow(x)
        torch.testing.assert_close(
            logits.double(), raw.double() / float(teacher.simulation_length),
            rtol=0, atol=0,
        )

    def test_teacher_is_frozen(self, mock_pipeline):
        _seed(mock_pipeline, rung2=True)
        step = _run_step(mock_pipeline)
        teacher = step.tuner.trainer.loss_function.teacher
        assert all(not p.requires_grad for p in teacher.parameters())

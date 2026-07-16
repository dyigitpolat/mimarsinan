"""ActivationAdaptationTuner KD-from-teacher lever (``activation_adaptation_kd``).

Converting a GELU-pretrained backbone onto a ReLU/LIF chip crosses a GELU->ReLU
cliff (full hard-ReLU collapses a ViT-B to chance). Plain-CE recovery from that
near-chance start is a weak signal; distilling against a frozen snapshot of the
pre-blend (GELU) model carries the teacher's dark knowledge across the cliff.
Default-off: byte-identical CE recovery unless opted in.
"""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss
from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)


def _pipeline(tmp_path, **overrides):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg.update(overrides)
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


def _tuner(pipeline):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    return ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, am)


class TestActivationAdaptationKD:
    def test_kd_off_by_default_keeps_ce_loss(self, tmp_path):
        tuner = _tuner(_pipeline(tmp_path))
        assert not isinstance(tuner.trainer.loss_function, KDClassificationLoss)
        assert tuner._kd_teacher is None

    def test_kd_on_snapshots_frozen_teacher_and_sets_kd_loss(self, tmp_path):
        tuner = _tuner(_pipeline(tmp_path, activation_adaptation_kd=True))
        assert isinstance(tuner.trainer.loss_function, KDClassificationLoss)
        teacher = tuner._kd_teacher
        assert teacher is not None
        assert teacher is not tuner.model
        assert all(not p.requires_grad for p in teacher.parameters())

    def test_kd_loss_targets_the_snapshot_teacher(self, tmp_path):
        tuner = _tuner(_pipeline(tmp_path, activation_adaptation_kd=True))
        assert tuner.trainer.loss_function.teacher is tuner._kd_teacher

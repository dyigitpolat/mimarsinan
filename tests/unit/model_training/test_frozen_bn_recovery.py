"""Frozen-BN-stats contract for tuner-internal training (W2 fix B, pillar P2).

Recovery/adaptation training must not mutate BatchNorm running statistics: an
uncommanded stats drift decouples the committed metric from the mapped
artifact (and train-mode BN over a poisoned activation was the shift-crater
kill condition).
"""

import pytest
import torch
import torch.nn as nn

from conftest import MockDataProviderFactory, MockPipeline, make_tiny_supermodel

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.orchestration.tuner_base import TunerBase


def _bn_trainer():
    dpf = MockDataProviderFactory()
    dlf = DataLoaderFactory(dpf, num_workers=0)
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(64, 8), nn.BatchNorm1d(8), nn.Linear(8, 4)
    )

    def loss_fn(mod, x, y):
        return nn.CrossEntropyLoss()(mod(x), y)

    return BasicTrainer(model, "cpu", dlf, loss_fn)


def _bn_module(trainer) -> nn.BatchNorm1d:
    return trainer.model[2]


class TestFreezeFlagOnBasicTrainer:
    def test_default_training_updates_running_stats(self):
        trainer = _bn_trainer()
        bn = _bn_module(trainer)
        before = int(bn.num_batches_tracked.item())
        trainer.train_n_steps(0.01, 3)
        assert int(bn.num_batches_tracked.item()) == before + 3
        trainer.close()

    def test_frozen_flag_holds_stats_while_weights_train(self):
        trainer = _bn_trainer()
        trainer.freeze_bn_stats_in_training = True
        bn = _bn_module(trainer)
        mean_before = bn.running_mean.clone()
        var_before = bn.running_var.clone()
        tracked_before = int(bn.num_batches_tracked.item())
        w_before = trainer.model[1].weight.detach().clone()

        trainer.train_n_steps(0.01, 3)

        assert int(bn.num_batches_tracked.item()) == tracked_before
        assert torch.equal(bn.running_mean, mean_before)
        assert torch.equal(bn.running_var, var_before)
        assert not torch.equal(trainer.model[1].weight.detach(), w_before), (
            "freezing BN stats must not stop weight training"
        )
        trainer.close()

    def test_frozen_flag_covers_step_recovery_path(self):
        trainer = _bn_trainer()
        trainer.freeze_bn_stats_in_training = True
        bn = _bn_module(trainer)
        tracked_before = int(bn.num_batches_tracked.item())
        trainer.train_steps_until_target(
            0.01, 3, 2.0, check_interval=1, patience=10,
        )
        assert int(bn.num_batches_tracked.item()) == tracked_before
        trainer.close()


class TestTunerTrainerContract:
    def test_tuner_base_trainer_freezes_bn_stats(self, tmp_path):
        pipeline = MockPipeline(working_directory=str(tmp_path))
        tuner = TunerBase(pipeline, make_tiny_supermodel(), 0.9, 0.001)
        try:
            assert tuner.trainer.freeze_bn_stats_in_training is True
        finally:
            tuner.close()

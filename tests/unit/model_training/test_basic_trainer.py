"""Tests for BasicTrainer: training loop, validation, test."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from conftest import MockDataProviderFactory


class _WrapperLoss:
    """Matches BasicTrainer's expected signature: loss(model, x, y)."""
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)


def _make_trainer(num_classes=4, input_shape=(1, 8, 8)):
    dp_factory = MockDataProviderFactory(input_shape=input_shape, num_classes=num_classes)
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    in_features = 1
    for d in input_shape:
        in_features *= d
    model = nn.Sequential(nn.Flatten(), nn.Linear(in_features, num_classes))
    loss = _WrapperLoss()
    trainer = BasicTrainer(model, "cpu", dlf, loss)
    return trainer


class TestBasicTrainer:
    def test_validate_returns_float(self):
        trainer = _make_trainer()
        acc = trainer.validate()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_test_returns_float(self):
        trainer = _make_trainer()
        acc = trainer.test()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_train_n_epochs(self):
        trainer = _make_trainer()
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)

    def test_train_until_target(self):
        trainer = _make_trainer()
        trainer.train_until_target_accuracy(
            lr=0.01, max_epochs=2, target_accuracy=0.0, warmup_epochs=0
        )

    def test_report_function(self):
        trainer = _make_trainer()
        reported = []
        trainer.report_function = lambda name, val: reported.append((name, val))
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)
        assert len(reported) > 0

    def test_set_training_batch_size(self):
        trainer = _make_trainer()
        trainer.set_training_batch_size(2)
        assert trainer.training_batch_size == 2

    def test_set_validation_batch_size(self):
        trainer = _make_trainer()
        trainer.set_validation_batch_size(5)
        assert trainer.validation_batch_size == 5

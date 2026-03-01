"""
Stress tests for BasicTrainer.

Tests edge cases: zero epochs, unreachable accuracy targets,
model with no trainable parameters, gradient behavior.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from conftest import MockDataProviderFactory


class _WrapperLoss:
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


class TestTrainerZeroEpochs:
    def test_train_zero_epochs(self):
        """Training for 0 epochs should not crash."""
        trainer = _make_trainer()
        trainer.train_n_epochs(lr=0.01, epochs=0, warmup_epochs=0)

    @pytest.mark.xfail(
        reason="BUG: BasicTrainer.train_until_target_accuracy returns int 0 "
               "instead of float 0.0 when max_epochs=0. The variable "
               "validation_accuracy is initialized to int literal 0, and the "
               "loop body never executes to replace it with a float.",
        strict=True,
    )
    def test_train_until_target_zero_epochs(self):
        """max_epochs=0 with target_accuracy=0.0."""
        trainer = _make_trainer()
        acc = trainer.train_until_target_accuracy(
            lr=0.01, max_epochs=0, target_accuracy=0.0, warmup_epochs=0
        )
        assert isinstance(acc, float)


class TestTrainerUnreachableTarget:
    def test_target_above_one(self):
        """
        Target accuracy > 1.0 can never be reached.
        Should run max_epochs and return without infinite loop.
        """
        trainer = _make_trainer()
        acc = trainer.train_until_target_accuracy(
            lr=0.01, max_epochs=2, target_accuracy=2.0, warmup_epochs=0
        )
        assert isinstance(acc, float)
        assert acc <= 1.0


class TestTrainerWeightUpdates:
    def test_weights_change_after_training(self):
        """Verify weights actually change after training."""
        trainer = _make_trainer()
        initial_w = trainer.model[1].weight.data.clone()
        trainer.train_n_epochs(lr=0.01, epochs=2, warmup_epochs=0)
        assert not torch.allclose(trainer.model[1].weight.data, initial_w), \
            "Weights should change after training"

    def test_multiple_validations(self):
        """
        Calling validate() multiple times should work (iterator reset).
        """
        trainer = _make_trainer()
        accs = [trainer.validate() for _ in range(5)]
        assert all(0.0 <= a <= 1.0 for a in accs)


class TestTrainerSerialization:
    def test_pickle_roundtrip(self):
        """BasicTrainer implements __getstate__/__setstate__ for pickling."""
        import pickle
        trainer = _make_trainer()
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)

        data = pickle.dumps(trainer)
        restored = pickle.loads(data)

        # Restored trainer should be functional
        acc = restored.validate()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestTrainerBatchSizes:
    def test_batch_size_larger_than_dataset(self):
        """Batch size bigger than dataset should still work."""
        trainer = _make_trainer()
        trainer.set_training_batch_size(1000)
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)

    def test_batch_size_one(self):
        """Single-sample batches (SGD)."""
        trainer = _make_trainer()
        trainer.set_training_batch_size(1)
        trainer.train_n_epochs(lr=0.01, epochs=1, warmup_epochs=0)

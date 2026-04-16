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

    def test_train_validation_epochs_does_not_call_test(self):
        trainer = _make_trainer()

        def boom():
            raise AssertionError("test() must not be called")

        trainer.test = boom
        trainer.train_validation_epochs(0.01, 2, 0)

    def test_train_validation_epochs_runs_requested_epochs(self):
        trainer = _make_trainer()
        epoch_count = [0]
        original = trainer._train_one_epoch

        def wrapped(opt, sched, scaler):
            epoch_count[0] += 1
            return original(opt, sched, scaler)

        trainer._train_one_epoch = wrapped  # type: ignore[method-assign]
        trainer.train_validation_epochs(0.01, 3, 0)
        assert epoch_count[0] == 3

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

    def test_train_n_steps_runs_requested_steps(self):
        trainer = _make_trainer()
        steps = [0]
        orig = trainer._optimize

        def counted_opt(x, y, opt, scaler):
            steps[0] += 1
            return orig(x, y, opt, scaler)

        trainer._optimize = counted_opt  # type: ignore[method-assign]
        trainer.train_n_steps(lr=0.01, steps=4, warmup_steps=0)
        assert steps[0] == 4

    def test_validate_n_batches_averages(self):
        trainer = _make_trainer()
        a1 = trainer.validate_n_batches(3)
        assert isinstance(a1, float)
        assert 0.0 <= a1 <= 1.0

    def test_train_steps_until_target_respects_max_steps(self):
        trainer = _make_trainer()
        trainer.train_steps_until_target(
            lr=0.01, max_steps=2, target_accuracy=1.0, warmup_steps=0, validation_n_batches=1
        )

    def test_train_one_step_can_return_post_update_probe_loss(self):
        trainer = _make_trainer()
        batch = trainer.next_training_batch()
        loss = trainer.train_one_step(
            0.01,
            batch=batch,
            eval_batch=batch,
            return_post_update_loss=True,
        )
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_test_honors_max_batches(self):
        """test(max_batches=N) stops at N batches; 0 returns 0.0 without iterating."""
        trainer = _make_trainer()
        trainer.set_test_batch_size(1)
        total_test = len(trainer.test_loader)
        assert total_test >= 2, "test set too small for this invariant"

        # Count forward passes by monkey-patching the model __call__.
        orig_forward = trainer.model.forward
        call_count = [0]

        def counted_forward(x):
            call_count[0] += 1
            return orig_forward(x)

        trainer.model.forward = counted_forward  # type: ignore[method-assign]

        acc = trainer.test(max_batches=1)
        assert call_count[0] == 1, f"expected 1 forward pass, got {call_count[0]}"
        assert isinstance(acc, float)

        call_count[0] = 0
        acc0 = trainer.test(max_batches=0)
        assert call_count[0] == 0
        assert acc0 == 0.0

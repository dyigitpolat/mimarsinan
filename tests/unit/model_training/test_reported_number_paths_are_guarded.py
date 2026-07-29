"""W0.8 — every path that REPORTS or DECIDES on a number is guarded, not just validate().

This program publishes accuracy numbers. The unit's contract is that no path can
report a number measured on corrupted data: each one either repairs the read and
reports the true value, or fails loud. The decision cache has its own file
(``test_decision_path_batch_integrity``); this covers the remaining seams --
``test()``, ``test_on_subsample()``, ``validate_train()`` and the NAS evaluators'
fitness pass -- plus the ``batch_integrity`` primitives they share.
"""

import pytest
import torch

from mimarsinan.data_handling import batch_integrity
from mimarsinan.model_training import basic_trainer_eval, basic_trainer_subsample
from mimarsinan.search.evaluators.loader_accuracy import accuracy_over_loader
from mimarsinan.tuning.learning_rate_explorer import make_loss_slope_signal


class _ConstantClassModel(torch.nn.Module):
    """Finite inputs score perfectly; a non-finite batch collapses to one class."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        if not bool(torch.isfinite(flat).all()):
            logits = torch.zeros(x.shape[0], self.n_classes)
            logits[:, 0] = 1.0
            return logits
        labels = flat[:, 0].round().long().clamp(0, self.n_classes - 1)
        return torch.nn.functional.one_hot(labels, self.n_classes).float()

    def to(self, *args, **kwargs):
        return self


LABELS = [0, 3, 1, 0]


def _clean(labels=LABELS):
    y = torch.tensor(labels, dtype=torch.long)
    x = torch.zeros(len(labels), 1, 2, 2)
    x[:, 0, 0, 0] = y.float()
    return x, y


def _nan(labels=LABELS):
    x, y = _clean(labels)
    return torch.full_like(x, float("nan")), y


class _ReplayLoader:
    """Successive PASSES yield different batch lists: poisoned once, then correct."""

    def __init__(self, passes):
        self._passes = list(passes)
        self.passes_taken = 0

    def __iter__(self):
        batches = self._passes[min(self.passes_taken, len(self._passes) - 1)]
        self.passes_taken += 1
        return iter(batches)


class _EvalTrainer:
    def __init__(self, test_loader=None, train_loader=None):
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_iter = iter(train_loader) if train_loader is not None else None
        self.model = _ConstantClassModel()
        self.device = "cpu"
        self.test_batch_size = 4
        self.reports = []

    def _report(self, name, value):
        self.reports.append((name, value))

    def _validation_metric_name(self, base):
        return base

    def next_training_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)


class TestPublishedTestNumber:
    def test_a_poisoned_test_pass_is_redone_not_reported(self):
        loader = _ReplayLoader([[_nan()], [_clean()]])
        trainer = _EvalTrainer(test_loader=loader)
        assert basic_trainer_eval.test(trainer) == 1.0
        assert loader.passes_taken == 2

    def test_a_restarted_pass_reports_exactly_once(self):
        """A number for an aborted pass must never reach the reporter."""
        trainer = _EvalTrainer(test_loader=_ReplayLoader([[_nan()], [_clean()]]))
        basic_trainer_eval.test(trainer)
        assert [n for n, _ in trainer.reports] == ["Test accuracy"]

    def test_persistent_corruption_fails_loud_instead_of_publishing(self):
        trainer = _EvalTrainer(test_loader=_ReplayLoader([[_nan()]]))
        with pytest.raises(batch_integrity.CorruptedBatchError):
            basic_trainer_eval.test(trainer)
        assert trainer.reports == []

    def test_a_clean_pass_costs_no_extra_read(self):
        loader = _ReplayLoader([[_clean()]])
        trainer = _EvalTrainer(test_loader=loader)
        assert basic_trainer_eval.test(trainer) == 1.0
        assert loader.passes_taken == 1


class TestSubsampleTestNumber:
    """``pipeline_metric`` progresses the pipeline on this number."""

    def _trainer(self, passes):
        trainer = _EvalTrainer(test_loader=_ReplayLoader(passes))

        class _Provider:
            def _get_test_dataset(self):
                raise NotImplementedError

        trainer.data_provider = _Provider()
        return trainer

    def test_a_poisoned_collection_pass_is_redone(self):
        trainer = self._trainer([[_nan()], [_clean()]])
        acc = basic_trainer_subsample.test_on_subsample(trainer, max_samples=2, seed=0)
        assert acc == 1.0

    def test_persistent_corruption_fails_loud(self):
        trainer = self._trainer([[_nan()]])
        with pytest.raises(batch_integrity.CorruptedBatchError):
            basic_trainer_subsample.test_on_subsample(trainer, max_samples=2, seed=0)

    def test_collected_samples_do_not_alias_the_loader_buffer(self):
        """``x[i]`` is a view into a buffer the producer refills; the collected
        subsample must be snapshots or the whole pass measures whatever landed last."""
        x, y = _clean()
        trainer = self._trainer([[(x, y)]])
        xs, _ = basic_trainer_subsample._collect_all(trainer, "loader")
        x.fill_(float("nan"))
        assert all(bool(torch.isfinite(s).all()) for s in xs)


class TestValidateTrainNumber:
    def test_a_poisoned_training_read_is_repaired_before_being_reported(self):
        loader = _ReplayLoader([[_nan()], [_clean()]])
        trainer = _EvalTrainer(train_loader=loader)
        trainer.train_iter = iter(loader)
        assert basic_trainer_eval.validate_train(trainer) == 1.0

    def test_persistent_corruption_fails_loud(self):
        loader = _ReplayLoader([[_nan()]])
        trainer = _EvalTrainer(train_loader=loader)
        trainer.train_iter = iter(loader)
        with pytest.raises(batch_integrity.CorruptedBatchError):
            basic_trainer_eval.validate_train(trainer)


class TestNasFitnessNumber:
    def test_a_poisoned_fitness_pass_is_redone_not_ranked_on(self):
        loader = _ReplayLoader([[_nan()], [_clean()]])
        assert accuracy_over_loader(_ConstantClassModel(), loader, "cpu") == 1.0

    def test_persistent_corruption_fails_loud_instead_of_ranking(self):
        loader = _ReplayLoader([[_nan()]])
        with pytest.raises(batch_integrity.CorruptedBatchError):
            accuracy_over_loader(_ConstantClassModel(), loader, "cpu")


class TestLearningRateProbeSignal:
    """The coarse LR score RANKS candidate learning rates on a training-batch loss."""

    class _LossTrainer(_EvalTrainer):
        def evaluate_loss_on_batch(self, batch):
            x, _ = batch
            return float(x.abs().sum())

    def test_a_poisoned_probe_read_is_repaired_before_it_ranks_an_lr(self):
        loader = _ReplayLoader([[_nan()], [_clean()]])
        trainer = self._LossTrainer(train_loader=loader)
        trainer.train_iter = iter(loader)
        signal = make_loss_slope_signal(trainer)
        assert signal is not None
        value = signal()
        assert value == pytest.approx(sum(LABELS))

    def test_persistent_corruption_fails_loud_instead_of_scoring_nan(self):
        loader = _ReplayLoader([[_nan()]])
        trainer = self._LossTrainer(train_loader=loader)
        trainer.train_iter = iter(loader)
        signal = make_loss_slope_signal(trainer)
        assert signal is not None
        with pytest.raises(batch_integrity.CorruptedBatchError):
            signal()


class TestBatchIntegrityPrimitives:
    def test_non_tensor_inputs_are_not_probed(self):
        assert batch_integrity.is_corrupt_batch([1, 2, 3]) is False

    def test_integer_tensors_are_never_corrupt(self):
        assert batch_integrity.is_corrupt_batch(torch.zeros(4, dtype=torch.long)) is False

    def test_own_batch_passes_non_tensors_through(self):
        x, y = batch_integrity.own_batch(torch.zeros(2), "label")
        assert y == "label"

    def test_verified_pass_reraises_the_last_error_after_exhausting_retries(self):
        attempts = []

        def _pass():
            attempts.append(1)
            raise batch_integrity.CorruptedBatchError("poisoned")

        with pytest.raises(batch_integrity.CorruptedBatchError, match="poisoned"):
            batch_integrity.verified_pass(_pass, source="loader")
        assert len(attempts) == batch_integrity.CORRUPT_BATCH_RETRIES + 1

    def test_verified_pass_does_not_swallow_unrelated_errors(self):
        def _pass():
            raise ValueError("a real bug")

        with pytest.raises(ValueError, match="a real bug"):
            batch_integrity.verified_pass(_pass, source="loader")

    def test_corruption_error_is_not_a_quietly_catchable_value_error(self):
        """A corrupted read must end the measurement, not be swallowed by a
        best-effort fallback that catches the usual data exceptions."""
        assert issubclass(batch_integrity.CorruptedBatchError, RuntimeError)
        assert not issubclass(batch_integrity.CorruptedBatchError, (ValueError, KeyError))

"""W0.7 — a corrupted validation read must never become a reported accuracy.

Observed on GPU (instrumented BC-2 VGG-8 rerun): ``next_validation_batch()``
occasionally returns an image tensor that is entirely NaN, while the identical
batch re-read from a fresh iterator a moment later is bit-correct. Measured on,
the NaN batch makes the model emit one constant class and the run reports a
chance-level accuracy -- e.g. 11/128 = 0.0859375 on the CIFAR-10 validation
batch whose labels put 11 samples in that class, seconds after the same weights
tested 0.9266 on the full test set.
"""

import pytest
import torch

from mimarsinan.model_training.basic_trainer import BasicTrainer


class _FakeTrainer:
    """Only the seam under test: the raw pull plus the re-read hook."""

    next_validation_batch = BasicTrainer.next_validation_batch
    _raw_next_validation_batch = BasicTrainer._raw_next_validation_batch

    def __init__(self, batches):
        self._batches = list(batches)
        self._pulls = 0
        self.reiterations = 0
        self.validation_loader = self

    def __iter__(self):
        self.reiterations += 1
        return self

    def __next__(self):
        batch = self._batches[min(self._pulls, len(self._batches) - 1)]
        self._pulls += 1
        return batch

    @property
    def val_iter(self):
        return self

    @val_iter.setter
    def val_iter(self, value):
        pass


def _clean(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(4, 3, 2, 2, generator=g), torch.zeros(4, dtype=torch.long))


def _nan():
    return (torch.full((4, 3, 2, 2), float("nan")), torch.zeros(4, dtype=torch.long))


def _inf():
    x = torch.zeros(4, 3, 2, 2)
    x[0, 0, 0, 0] = float("inf")
    return (x, torch.zeros(4, dtype=torch.long))


class TestCorruptedValidationBatchIsNeverMeasuredOn:
    def test_clean_batch_passes_through_untouched(self):
        batch = _clean()
        trainer = _FakeTrainer([batch])
        x, y = trainer.next_validation_batch()
        assert x is batch[0] and y is batch[1]
        assert trainer.reiterations == 0

    def test_all_nan_batch_is_discarded_and_re_read(self, capsys):
        clean = _clean()
        trainer = _FakeTrainer([_nan(), clean])
        x, y = trainer.next_validation_batch()
        assert x is clean[0]
        assert trainer.reiterations == 1
        assert "CORRUPTED validation batch discarded" in capsys.readouterr().out

    def test_a_single_inf_value_is_enough_to_reject_the_batch(self):
        clean = _clean()
        trainer = _FakeTrainer([_inf(), clean])
        x, _ = trainer.next_validation_batch()
        assert x is clean[0]

    def test_persistent_corruption_fails_loud_instead_of_reporting_a_number(self):
        trainer = _FakeTrainer([_nan()])
        with pytest.raises(RuntimeError, match="non-finite batches"):
            trainer.next_validation_batch()

    def test_integer_inputs_are_never_rejected(self):
        """Integer batches cannot be non-finite; they must not be probed away."""
        batch = (torch.randint(0, 255, (4, 3, 2, 2), dtype=torch.uint8),
                 torch.zeros(4, dtype=torch.long))
        trainer = _FakeTrainer([batch])
        x, _ = trainer.next_validation_batch()
        assert x is batch[0]

    def test_stop_iteration_still_rewinds_the_iterator(self):
        """The pre-existing wrap contract is unchanged."""
        clean = _clean()

        class _Wrapping(_FakeTrainer):
            def __next__(self):
                self._pulls += 1
                if self._pulls == 1:
                    raise StopIteration
                return clean

        trainer = _Wrapping([clean])
        x, _ = trainer.next_validation_batch()
        assert x is clean[0]
        assert trainer.reiterations == 1

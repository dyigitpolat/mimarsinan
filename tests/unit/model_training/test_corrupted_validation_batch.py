"""W0.7/W0.8 — a corrupted validation read must never become a reported accuracy.

Observed on GPU (instrumented BC-2 VGG-8 rerun): ``next_validation_batch()``
occasionally returns an image tensor that is entirely NaN, while the identical
batch re-read from a fresh iterator a moment later is bit-correct. Measured on,
the NaN batch makes the model emit one constant class and the run reports a
chance-level accuracy -- e.g. 11/128 = 0.0859375 on the CIFAR-10 validation
batch whose labels put 11 samples in that class, seconds after the same weights
tested 0.9266 on the full test set.

W0.8 additions pinned here: the returned batch is an OWNED snapshot (verifying a
live producer view proves nothing about what the forward pass later reads), and
the repair escalates rewind -> positional advance so a corruption correlated with
the epoch-start position cannot trap the retry loop.
"""

import pytest
import torch

from mimarsinan.data_handling import batch_integrity
from mimarsinan.model_training.basic_trainer import BasicTrainer


class _FakeTrainer:
    """Only the seam under test: the raw pull plus the repair hook."""

    next_validation_batch = BasicTrainer.next_validation_batch
    _raw_next_validation_batch = BasicTrainer._raw_next_validation_batch
    _repair_validation_read = BasicTrainer._repair_validation_read
    _validation_source = BasicTrainer._validation_source

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
    def test_clean_batch_passes_through_by_value(self):
        batch = _clean()
        trainer = _FakeTrainer([batch])
        x, y = trainer.next_validation_batch()
        assert torch.equal(x, batch[0]) and torch.equal(y, batch[1])
        assert trainer.reiterations == 0

    def test_the_returned_batch_is_an_owned_snapshot(self):
        """The verdict must be about bytes nothing can rewrite behind it: a loader
        that refills its buffer after the check must not change what was measured."""
        batch = _clean()
        trainer = _FakeTrainer([batch])
        x, _ = trainer.next_validation_batch()
        assert x is not batch[0]
        assert x.data_ptr() != batch[0].data_ptr()
        batch[0].fill_(float("nan"))
        assert bool(torch.isfinite(x).all())

    def test_all_nan_batch_is_discarded_and_re_read(self, capsys):
        clean = _clean()
        trainer = _FakeTrainer([_nan(), clean])
        x, y = trainer.next_validation_batch()
        assert torch.equal(x, clean[0])
        assert trainer.reiterations == 1
        assert "CORRUPTED batch discarded" in capsys.readouterr().out

    def test_a_single_inf_value_is_enough_to_reject_the_batch(self):
        clean = _clean()
        trainer = _FakeTrainer([_inf(), clean])
        x, _ = trainer.next_validation_batch()
        assert torch.equal(x, clean[0])

    def test_persistent_corruption_fails_loud_instead_of_reporting_a_number(self):
        trainer = _FakeTrainer([_nan()])
        with pytest.raises(batch_integrity.CorruptedBatchError, match="non-finite batches"):
            trainer.next_validation_batch()

    def test_integer_inputs_are_never_rejected(self):
        """Integer batches cannot be non-finite; they must not be probed away."""
        batch = (torch.randint(0, 255, (4, 3, 2, 2), dtype=torch.uint8),
                 torch.zeros(4, dtype=torch.long))
        trainer = _FakeTrainer([batch])
        x, _ = trainer.next_validation_batch()
        assert torch.equal(x, batch[0])

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
        assert torch.equal(x, clean[0])
        assert trainer.reiterations == 1


class TestRepairPolicyEscalatesRewindThenAdvance:
    """W0.8 finding 3: rewinding alone can re-read the same poisoned position."""

    class _PositionalTrainer(_FakeTrainer):
        """A loader whose corruption is pinned to POSITION 0 of every epoch --
        the case a pure rewind can never escape."""

        def __init__(self):
            super().__init__([])
            self.position = 0

        def __iter__(self):
            self.reiterations += 1
            self.position = 0
            return self

        def __next__(self):
            batch = _nan() if self.position == 0 else _clean(self.position)
            self.position += 1
            return batch

    def test_a_position_locked_corruption_is_escaped_by_advancing(self):
        trainer = self._PositionalTrainer()
        x, _ = trainer.next_validation_batch()
        assert bool(torch.isfinite(x).all()), (
            "the repair must reach a position the corruption has not already "
            "been observed at; a pure rewind would loop on position 0"
        )

    def test_every_repair_starts_from_a_fresh_iterator(self):
        """Advancing without abandoning the poisoned in-flight queue is not the
        repair the evidence supports; the fresh iterator is kept."""
        trainer = self._PositionalTrainer()
        trainer.next_validation_batch()
        assert trainer.reiterations >= 1

    def test_advance_wraps_instead_of_stalling_on_a_short_loader(self):
        """A loader shorter than the advance distance must wrap, not raise."""
        clean = _clean()

        class _OneBatchThenStop(_FakeTrainer):
            def __init__(self):
                super().__init__([])
                self.served = 0

            def __iter__(self):
                self.reiterations += 1
                self.served = 0
                return self

            def __next__(self):
                if self.served >= 1:
                    raise StopIteration
                self.served += 1
                return _nan() if self.reiterations < 3 else clean

        trainer = _OneBatchThenStop()
        x, _ = trainer.next_validation_batch()
        assert torch.equal(x, clean[0])

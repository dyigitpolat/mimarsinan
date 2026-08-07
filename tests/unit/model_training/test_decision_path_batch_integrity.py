"""W0.8 — the tuner DECISION path must never decide on a corrupted read.

W0.7 guarded ``next_validation_batch`` -> ``validate()``, the single-batch
REPORTING path. The path that actually drives the ladder is a different one:

    validate_n_batches -> iter_validation_batches -> _build_gpu_val_cache

It calibrates the acceptance baseline (``train_steps_until_target``'s ``entry_acc``)
and every subsequent commit/rollback comparison, so a poisoned read there does not
print a wrong number -- it silently makes a wrong decision. Same discipline, same
loud failure, pinned here.

Each test in ``TestPoisonedDecisionRead`` FAILS without the fix: pre-W0.8 the NaN
batch lands in the on-device cache untouched and ``validate_n_batches`` returns the
constant-class share of it (the 11/128 = 0.0859 shape of the original defect).
"""

import pytest
import torch

from mimarsinan.data_handling import batch_integrity
from mimarsinan.model_training import basic_trainer_eval
from mimarsinan.model_training.basic_trainer import BasicTrainer


class _ConstantClassModel(torch.nn.Module):
    """The model the defect produced: finite inputs score perfectly, a NaN batch
    collapses every logit to one constant class (what a real net does when NaNs
    flood the first layer)."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        if not bool(torch.isfinite(flat).all()):
            logits = torch.zeros(x.shape[0], self.n_classes)
            logits[:, 0] = 1.0
            return logits
        # Encode the label in channel 0 so a clean read scores 1.0.
        labels = flat[:, 0].round().long().clamp(0, self.n_classes - 1)
        return torch.nn.functional.one_hot(labels, self.n_classes).float()


def _clean_batch(labels):
    y = torch.tensor(labels, dtype=torch.long)
    x = torch.zeros(len(labels), 1, 2, 2)
    x[:, 0, 0, 0] = y.float()
    return x, y


def _nan_batch(labels):
    x, y = _clean_batch(labels)
    return torch.full_like(x, float("nan")), y


class _ReplayLoader:
    """A loader whose successive PASSES yield different batch lists — the shape of
    a transient corrupted read (poisoned once, bit-correct on re-read)."""

    def __init__(self, passes):
        self._passes = list(passes)
        self.passes_taken = 0

    def __iter__(self):
        batches = self._passes[min(self.passes_taken, len(self._passes) - 1)]
        self.passes_taken += 1
        return iter(batches)


class _FakeTrainer:
    """Only the decision seam: a validation loader, a model, and the cache path."""

    iter_validation_batches = BasicTrainer.iter_validation_batches
    validate_n_batches = BasicTrainer.validate_n_batches
    validate_correctness_on_indices = BasicTrainer.validate_correctness_on_indices

    def __init__(self, loader, model=None):
        self.validation_loader = loader
        self.model = model if model is not None else _ConstantClassModel()
        self.device = "cpu"
        self.validation_batch_size = 4
        self.reports = []

    def _report(self, name, value):
        self.reports.append((name, value))

    def _validation_metric_name(self, base):
        return base


# Two of the four samples are in the collapse class, so an unguarded read of the
# NaN batch reports 2/4 -- the same "share of the constant class" arithmetic that
# produced 11/128 = 0.0859375 on CIFAR-10 validation batch 0 in the field.
LABELS = [0, 3, 1, 0]


class TestPoisonedDecisionRead:
    def test_poisoned_cache_build_is_repaired_not_decided_on(self):
        """The number the ladder decides on is the one measured after repair."""
        loader = _ReplayLoader([
            [_nan_batch(LABELS)],           # first pass: poisoned
            [_clean_batch(LABELS)],         # re-read: bit-correct
        ])
        trainer = _FakeTrainer(loader)
        acc = trainer.validate_n_batches(1)
        assert acc == 1.0, "a repaired decision read must report the true accuracy"
        assert loader.passes_taken == 2, "the poisoned pass must have been redone"

    def test_a_poisoned_batch_mid_pass_discards_the_partial_cache(self):
        """A corrupted read anywhere in the pass aborts it — no partial cache escapes."""
        loader = _ReplayLoader([
            [_clean_batch(LABELS), _nan_batch(LABELS)],
            [_clean_batch(LABELS), _clean_batch(LABELS)],
        ])
        trainer = _FakeTrainer(loader)
        assert trainer.validate_n_batches(2) == 1.0
        assert len(trainer._gpu_val_cache) == 2

    def test_persistent_corruption_fails_loud_instead_of_deciding(self):
        loader = _ReplayLoader([[_nan_batch(LABELS)]])
        trainer = _FakeTrainer(loader)
        with pytest.raises(batch_integrity.CorruptedBatchError):
            trainer.validate_n_batches(1)

    def test_the_defect_shape_is_what_would_be_reported_without_the_guard(self):
        """Documents the number the guard prevents: the constant class's share.

        Measured through the unguarded materializer, the NaN batch yields 2/4 —
        the same arithmetic that produced 11/128 = 0.0859 on CIFAR-10 in the field.
        """
        loader = _ReplayLoader([[_nan_batch(LABELS)]])
        trainer = _FakeTrainer(loader)
        trainer._gpu_val_cache = [batch_integrity.own_batch(*_nan_batch(LABELS))]
        trainer._gpu_val_cursor = 0
        assert trainer.validate_n_batches(1) == pytest.approx(0.5)

    def test_correctness_on_indices_shares_the_guarded_cache(self):
        """The McNemar-style paired path reads the same verified cache."""
        loader = _ReplayLoader([
            [_nan_batch(LABELS)],
            [_clean_batch(LABELS)],
        ])
        trainer = _FakeTrainer(loader)
        assert trainer.validate_correctness_on_indices([0]) == [True] * 4

    def test_a_clean_pass_is_read_exactly_once(self):
        """The guard costs no extra pass on the overwhelmingly common path."""
        loader = _ReplayLoader([[_clean_batch(LABELS)]])
        trainer = _FakeTrainer(loader)
        trainer.validate_n_batches(1)
        assert loader.passes_taken == 1


class TestSubsampledCacheStaysDeterministic:
    def test_restarted_pass_selects_the_identical_reservoir_subsample(self):
        """The reservoir RNG is re-seeded per attempt, so a repair cannot change
        WHICH batches the decision baseline is measured on."""
        many = [_clean_batch([i % 4] * 4) for i in range(8)]
        poisoned = [_nan_batch([0, 0, 0, 0])] + many[1:]

        clean_only = _FakeTrainer(_ReplayLoader([many]))
        clean_only._val_cache_max_batches = 3
        clean_only.validate_n_batches(1)

        repaired = _FakeTrainer(_ReplayLoader([poisoned, many]))
        repaired._val_cache_max_batches = 3
        repaired.validate_n_batches(1)

        selected = [x for x, _ in repaired._gpu_val_cache]
        expected = [x for x, _ in clean_only._gpu_val_cache]
        assert len(selected) == len(expected) == 3
        assert all(torch.equal(a, b) for a, b in zip(selected, expected))


class TestPooledCacheIsVerifiedOnRetrieval:
    def test_a_poisoned_pooled_cache_raises_rather_than_being_decided_on(self):
        """An owned cache cannot be repaired by re-reading, so it must fail loud."""

        class _Factory:
            def __init__(self, cache):
                self._cache = cache

            def owns_loaders(self):
                return True

            def get_eval_cache(self, key):
                return self._cache

            def put_eval_cache(self, key, batches):
                self._cache = batches

        trainer = _FakeTrainer(_ReplayLoader([[_clean_batch(LABELS)]]))
        trainer.data_loader_factory = _Factory([_nan_batch(LABELS)])
        with pytest.raises(batch_integrity.CorruptedBatchError, match="owned snapshot"):
            trainer.validate_n_batches(1)


class TestVerifiedBatchesAreOwnedNotAliased:
    def test_the_cached_tensor_does_not_alias_the_loader_buffer(self):
        """Verifying a live view proves nothing: the producer can refill it between
        the check and the forward pass. The cache must hold snapshots."""
        x, y = _clean_batch(LABELS)
        trainer = _FakeTrainer(_ReplayLoader([[(x, y)]]))
        trainer.validate_n_batches(1)
        cached_x, _ = trainer._gpu_val_cache[0]
        assert cached_x is not x
        assert cached_x.data_ptr() != x.data_ptr()
        x.fill_(float("nan"))
        assert bool(torch.isfinite(cached_x).all())

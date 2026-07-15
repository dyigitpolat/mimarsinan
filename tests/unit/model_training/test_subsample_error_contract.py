"""test_on_subsample: dataset-construction errors propagate; len-unsupported falls back."""

import pytest
import torch
import torch.nn as nn

import mimarsinan.model_training.basic_trainer_subsample as subsample_eval


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _FakeProvider:
    def __init__(self, dataset):
        self._dataset = dataset

    def _get_test_dataset(self):
        if isinstance(self._dataset, Exception):
            raise self._dataset
        return self._dataset


class _FakeTrainer:
    def __init__(self, provider, batches):
        self.data_provider = provider
        self.test_loader = batches
        self.test_batch_size = 4
        self.device = "cpu"
        self.model = _Identity()
        self.reports = []

    def _report(self, name, value):
        self.reports.append((name, value))


def _perfect_batches():
    x = torch.eye(4)
    y = torch.arange(4)
    return [(x, y)]


class _DatasetWithoutLen:
    pass


class TestSubsampleErrorContract:
    def test_dataset_construction_error_propagates(self):
        trainer = _FakeTrainer(
            _FakeProvider(RuntimeError("corrupt dataset")), _perfect_batches())
        with pytest.raises(RuntimeError, match="corrupt dataset"):
            subsample_eval.test_on_subsample(trainer, max_samples=2)

    def test_len_unsupported_falls_back_to_loader_enumeration(self):
        trainer = _FakeTrainer(
            _FakeProvider(_DatasetWithoutLen()), _perfect_batches())
        acc = subsample_eval.test_on_subsample(trainer, max_samples=4)
        assert acc == 1.0
        assert trainer.reports


class TestFullSetDelegation:
    """A cap that covers the whole test set defers to the exact full ``test()``
    path, so a covering cap is byte-identical (the universal eval-cap invariant)."""

    def test_cap_at_or_above_size_defers_to_full_test(self):
        called = {}

        class _T(_FakeTrainer):
            def test(self):
                called["test"] = True
                return 0.99

        trainer = _T(_FakeProvider([0, 1, 2, 3]), _perfect_batches())  # size 4
        assert subsample_eval.test_on_subsample(trainer, max_samples=4) == 0.99
        assert called.get("test") is True
        assert subsample_eval.test_on_subsample(trainer, max_samples=100) == 0.99

    def test_cap_below_size_subsamples_and_never_calls_full_test(self):
        class _T(_FakeTrainer):
            def test(self):
                raise AssertionError("full test() must not run when the cap binds")

        trainer = _T(_FakeProvider([0, 1, 2, 3]), _perfect_batches())  # size 4
        acc = subsample_eval.test_on_subsample(trainer, max_samples=2, seed=0)
        assert 0.0 <= acc <= 1.0

    def test_sized_dataset_uses_len_path(self):
        # max_samples < size exercises the len-based subsample (a covering cap
        # instead defers to full test(); see TestFullSetDelegation).
        trainer = _FakeTrainer(_FakeProvider([0, 1, 2, 3]), _perfect_batches())
        acc = subsample_eval.test_on_subsample(trainer, max_samples=3)
        assert acc == 1.0

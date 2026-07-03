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

    def test_sized_dataset_uses_len_path(self):
        trainer = _FakeTrainer(_FakeProvider([0, 1, 2, 3]), _perfect_batches())
        acc = subsample_eval.test_on_subsample(trainer, max_samples=4)
        assert acc == 1.0

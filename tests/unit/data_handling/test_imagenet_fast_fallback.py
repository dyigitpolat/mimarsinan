"""Fast-recipe torchvision fallback dataloaders on the ImageNet provider.

NO ImageNet on disk: stubs the per-split dataset assembly so we only assert the
throughput knobs (workers / pin_memory / persistent_workers / prefetch) and the
per-split shuffle/drop_last policy. ``IMAGENET_ROOT`` is never touched.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset


class _TinyDS(Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return torch.zeros(3, 8, 8), 0


def _provider_without_imagenet():
    """ImageNet_DataProvider instance whose __init__ (which touches disk) is bypassed."""
    from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
        ImageNet_DataProvider,
    )
    p = ImageNet_DataProvider.__new__(ImageNet_DataProvider)
    p._get_training_dataset = lambda: _TinyDS(64)
    p._get_validation_dataset = lambda: _TinyDS(16)
    p._get_test_dataset = lambda: _TinyDS(16)
    return p


class TestFastFallbackDataloaders:
    def test_returns_three_splits(self):
        p = _provider_without_imagenet()
        loaders = p.fast_fallback_dataloaders(batch_size=8, num_workers=0)
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_throughput_knobs_set_with_workers(self):
        p = _provider_without_imagenet()
        loaders = p.fast_fallback_dataloaders(batch_size=8, num_workers=4)
        train = loaders["train"]
        assert train.num_workers == 4
        assert train.pin_memory is True
        assert train.persistent_workers is True
        assert train.prefetch_factor == 4

    def test_train_shuffles_and_drops_last_eval_does_not(self):
        p = _provider_without_imagenet()
        loaders = p.fast_fallback_dataloaders(batch_size=8, num_workers=0)
        from torch.utils.data import RandomSampler, SequentialSampler

        assert isinstance(loaders["train"].sampler, RandomSampler)
        assert isinstance(loaders["val"].sampler, SequentialSampler)
        assert isinstance(loaders["test"].sampler, SequentialSampler)
        assert loaders["train"].drop_last is True
        assert loaders["val"].drop_last is False
        assert loaders["test"].drop_last is False

    def test_zero_workers_omits_persistent_workers(self):
        p = _provider_without_imagenet()
        loaders = p.fast_fallback_dataloaders(batch_size=8, num_workers=0)
        # With num_workers=0 torch forbids persistent_workers/prefetch tuning.
        assert loaders["train"].num_workers == 0
        assert loaders["train"].persistent_workers is False

    def test_yields_a_batch(self):
        p = _provider_without_imagenet()
        loaders = p.fast_fallback_dataloaders(batch_size=8, num_workers=0)
        x, y = next(iter(loaders["val"]))
        assert x.shape == (8, 3, 8, 8)
        assert y.shape == (8,)

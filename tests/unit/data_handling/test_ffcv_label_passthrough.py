"""``preload_labels`` returns the right labels for every dataset shape we use."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, Subset

from mimarsinan.data_handling.ffcv.label_passthrough import (
    _unwrap,
    preload_labels,
)


class _WithTargetsList(Dataset):
    """Mimics torchvision's CIFAR10/MNIST: ``.targets`` is a Python list."""

    def __init__(self, n: int):
        self.targets = [i % 10 for i in range(n)]
        self.data = list(range(n))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        # __getitem__ deliberately raises so tests fail loudly if the helper
        # falls back to per-sample decode when it shouldn't.
        raise AssertionError("preload_labels must use .targets, not iterate")


class _WithTargetsTensor(Dataset):
    def __init__(self, n: int):
        self.targets = torch.arange(n, dtype=torch.int32) % 7

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int):
        raise AssertionError("preload_labels must use .targets, not iterate")


class _WithLabelsAttr(Dataset):
    def __init__(self, n: int):
        self.labels = [i % 4 for i in range(n)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        raise AssertionError("preload_labels must use .labels, not iterate")


class _AsRGBLike(Dataset):
    """Mimics spec_builder._AsRGB: wraps a base dataset, exposes ``_base``."""

    def __init__(self, base):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        return self._base[idx]


class _IterableOnly(Dataset):
    """No metadata surface — exercises the slow fallback path."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return (f"img-{idx}", idx % 3)


class TestUnwrap:
    def test_unwraps_AsRGB_wrapper(self):
        base = _WithTargetsList(5)
        wrapped = _AsRGBLike(base)
        assert _unwrap(wrapped) is base

    def test_passthrough_when_no_base(self):
        base = _WithTargetsList(5)
        assert _unwrap(base) is base

    def test_does_not_unwrap_past_a_targets_holder(self):
        # If a wrapper *also* carries .targets we should stop there.
        base = _WithTargetsList(5)
        wrapped = _AsRGBLike(base)
        wrapped.targets = [9, 9, 9, 9, 9]  # type: ignore[attr-defined]
        assert _unwrap(wrapped) is wrapped


class TestPreloadLabels:
    def test_torchvision_style_targets_list(self):
        ds = _WithTargetsList(20)
        out = preload_labels(ds)
        assert out.dtype == torch.long
        assert out.tolist() == [i % 10 for i in range(20)]

    def test_torchvision_style_targets_tensor(self):
        ds = _WithTargetsTensor(15)
        out = preload_labels(ds)
        assert out.dtype == torch.long
        assert out.tolist() == [(i % 7) for i in range(15)]

    def test_labels_attribute_fallback(self):
        ds = _WithLabelsAttr(12)
        out = preload_labels(ds)
        assert out.dtype == torch.long
        assert out.tolist() == [i % 4 for i in range(12)]

    def test_unwraps_AsRGB_wrapper(self):
        base = _WithTargetsList(8)
        out = preload_labels(_AsRGBLike(base))
        assert out.tolist() == [i % 10 for i in range(8)]

    def test_subset_with_targets_underlying_list(self):
        base = _WithTargetsList(30)
        sub = Subset(base, indices=[1, 0, 4, 4, 29])
        out = preload_labels(sub)
        assert out.tolist() == [
            base.targets[i] for i in [1, 0, 4, 4, 29]
        ]

    def test_subset_with_targets_underlying_tensor(self):
        base = _WithTargetsTensor(30)
        sub = Subset(base, indices=[5, 10, 12, 0])
        out = preload_labels(sub)
        expected = base.targets[torch.tensor([5, 10, 12, 0])].long().tolist()
        assert out.tolist() == expected

    def test_subset_inside_AsRGB(self):
        base = _WithTargetsList(15)
        sub = Subset(base, indices=[14, 0, 7])
        out = preload_labels(_AsRGBLike(sub))
        assert out.tolist() == [base.targets[i] for i in [14, 0, 7]]

    def test_iterable_only_fallback(self):
        ds = _IterableOnly(6)
        out = preload_labels(ds)
        assert out.dtype == torch.long
        assert out.tolist() == [i % 3 for i in range(6)]

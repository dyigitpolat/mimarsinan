"""ImageNet provider train/val split: seeded permutation must span ALL classes.

torchvision ImageNet ``split="train"`` is sorted BY CLASS, so a contiguous
``range()`` split makes the val holdout a handful of disjoint high-index classes
(a non-representative validation set). The provider must instead use a
deterministic, seed-derived permutation so both splits are representative random
samples spanning all classes, disjoint, and reproducible per seed.

Disk-free: a synthetic class-sorted dataset stand-in replays the exact pathology
(labels == index // per_class), so we can assert representativeness without the
1.2M-image ImageNet. A network/disk-gated test measures the real distinct-class
count when ``IMAGENET_ROOT`` metadata is present.
"""

from __future__ import annotations

import os

import pytest
import torch

from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
    ImageNet_DataProvider,
    _seeded_train_val_indices,
)


# ImageNet-shaped class-sorted stand-in: 1000 classes, a handful of samples each,
# labels strictly ascending with index (the exact property that breaks a
# contiguous range() split).
_N_CLASSES = 1000
_PER_CLASS = 8
_N_TRAIN = _N_CLASSES * _PER_CLASS


def _class_of(idx: int) -> int:
    return idx // _PER_CLASS


def _distinct_classes(indices) -> set[int]:
    return {_class_of(i) for i in indices}


class TestSeededTrainValIndices:
    def test_val_split_spans_all_classes(self):
        """Headline: a seeded permutation val holdout covers ~all 1000 classes.

        A contiguous range() split would cover only ``ceil((1-frac)*N / PER_CLASS)``
        classes (~50). The seeded split must cover essentially every class.
        """
        train_idx, val_idx = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)

        val_classes = _distinct_classes(val_idx)
        # 5% of 8000 = 400 val samples drawn uniformly over 1000 classes:
        # expected distinct ~ 1000*(1-(1-1/1000)^400) ~ 330. Demand >> the
        # ~50-class contiguous-split ceiling; tolerant of permutation variance.
        assert len(val_classes) >= 250, (
            f"val split covers only {len(val_classes)} classes; "
            "expected a representative sample spanning most of the 1000 classes"
        )

        # Contrast: the OLD contiguous split would cover this many classes.
        contiguous_val = range(int(_N_TRAIN * 0.95), _N_TRAIN)
        assert len(_distinct_classes(contiguous_val)) <= 60, (
            "sanity: the contiguous baseline is supposed to be class-starved"
        )
        assert len(val_classes) > 4 * len(_distinct_classes(contiguous_val)), (
            "seeded val must span far more classes than the contiguous baseline"
        )

    def test_train_split_also_spans_all_classes(self):
        train_idx, _ = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        train_classes = _distinct_classes(train_idx)
        # 95% of samples — must cover every single class.
        assert train_classes == set(range(_N_CLASSES)), (
            f"train split missing classes: {set(range(_N_CLASSES)) - train_classes}"
        )

    def test_train_and_val_are_disjoint_and_complete(self):
        train_idx, val_idx = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        train_set, val_set = set(train_idx), set(val_idx)
        assert train_set.isdisjoint(val_set), "train and val share indices"
        assert train_set | val_set == set(range(_N_TRAIN)), (
            "split must partition the full training index range"
        )
        assert len(train_idx) == len(train_set), "train indices contain duplicates"
        assert len(val_idx) == len(val_set), "val indices contain duplicates"

    def test_split_fraction_is_honored(self):
        train_idx, val_idx = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        assert len(train_idx) == int(_N_TRAIN * 0.95)
        assert len(val_idx) == _N_TRAIN - int(_N_TRAIN * 0.95)

    def test_reproducible_for_fixed_seed(self):
        a_train, a_val = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        b_train, b_val = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        assert a_train == b_train
        assert a_val == b_val

    def test_different_seeds_give_different_splits(self):
        _, val0 = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=0)
        _, val1 = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=1)
        assert val0 != val1, "different seeds must yield different val holdouts"

    def test_none_seed_is_still_deterministic_and_representative(self):
        """seed=None (DataProvider allows it) must not crash and must still span classes."""
        train_idx, val_idx = _seeded_train_val_indices(_N_TRAIN, 0.95, seed=None)
        train_set, val_set = set(train_idx), set(val_idx)
        assert train_set.isdisjoint(val_set)
        assert train_set | val_set == set(range(_N_TRAIN))
        assert len(_distinct_classes(val_idx)) >= 250


class TestProviderUsesSeededSplit:
    """The provider __init__ must wire the seeded helper onto class-sorted data.

    A class-sorted in-memory stand-in replaces ``torchvision.datasets.ImageNet``
    so we exercise the real ``__init__`` path (Subset wiring + raw_datasets)
    without disk.
    """

    def _build_provider(self, monkeypatch, *, seed):
        import torchvision

        class _ClassSortedDS(torch.utils.data.Dataset):
            def __init__(self, root, split, transform):
                self.split = split

            def __len__(self):
                return _N_TRAIN if self.split == "train" else 256

            def __getitem__(self, i):
                return torch.zeros(3, 4, 4), _class_of(i)

        monkeypatch.setattr(torchvision.datasets, "ImageNet", _ClassSortedDS)
        monkeypatch.setattr(
            "mimarsinan.data_handling.data_providers.imagenet_data_provider._ensure_imagenet_symlink",
            lambda p: "/tmp",
        )
        monkeypatch.setattr(os.path, "isdir", lambda p: True)
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        return ImageNet_DataProvider("/tmp/datasets", seed=seed)

    def test_provider_val_spans_classes_and_is_disjoint(self, monkeypatch):
        p = self._build_provider(monkeypatch, seed=0)
        raw = p.raw_datasets()
        train_idx = list(raw["train"].indices)
        val_idx = list(raw["val"].indices)

        assert raw["train"].dataset is raw["val"].dataset, (
            "both Subsets must view the SAME full training dataset (.dataset access "
            "used by train_imagenet_fast must stay the full train set)"
        )
        assert set(train_idx).isdisjoint(val_idx)
        val_classes = {raw["val"].dataset[i][1] for i in val_idx}
        assert len(val_classes) >= 250, (
            f"provider val covers only {len(val_classes)} classes"
        )

    def test_provider_split_reproducible_per_seed(self, monkeypatch):
        p0 = self._build_provider(monkeypatch, seed=0)
        p0b = self._build_provider(monkeypatch, seed=0)
        assert list(p0.raw_datasets()["val"].indices) == list(
            p0b.raw_datasets()["val"].indices
        )
        p1 = self._build_provider(monkeypatch, seed=1)
        assert list(p0.raw_datasets()["val"].indices) != list(
            p1.raw_datasets()["val"].indices
        )


def _imagenet_metadata_present() -> bool:
    """True only when real ImageNet train metadata is on disk (opt-in, disk-bound)."""
    if os.environ.get("RUN_IMAGENET_VAL_SPLIT_CHECK") != "1":
        return False
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path("/home/yigit/repos/wt-valgate/.env"))
    except Exception:
        pass
    raw = os.environ.get("IMAGENET_ROOT", "").strip()
    if not raw:
        return False
    root = os.path.abspath(os.path.expanduser(raw))
    meta = os.path.join(root, "meta.bin")
    devkit = os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")
    train = os.path.join(root, "train")
    return os.path.isdir(train) and (os.path.isfile(meta) or os.path.isfile(devkit))


@pytest.mark.integration
@pytest.mark.skipif(
    not _imagenet_metadata_present(),
    reason="Set RUN_IMAGENET_VAL_SPLIT_CHECK=1 and IMAGENET_ROOT with real ImageNet metadata",
)
def test_real_imagenet_val_spans_nearly_all_1000_classes():
    """On real ImageNet: val distinct-class count must be ~1000 (was ~50 contiguous)."""
    p = ImageNet_DataProvider("/tmp/datasets_real", seed=0)
    raw = p.raw_datasets()
    val_subset = raw["val"]
    full = val_subset.dataset
    targets = full.targets  # torchvision ImageNet exposes per-sample targets
    val_classes = {int(targets[i]) for i in val_subset.indices}
    assert len(val_classes) >= 990, (
        f"real ImageNet val covers only {len(val_classes)}/1000 classes"
    )
    train_idx = set(raw["train"].indices)
    assert train_idx.isdisjoint(val_subset.indices)

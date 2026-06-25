"""ImageNet provider val/test split: the TUNER GATE must be genuinely held out.

THE LEAK (now fixed): the tuner gate (``_get_validation_dataset`` -> provider
``val``) used to be a holdout of ``split="train"``. A pretrained / from-scratch
model has SEEN every train image, so that "val" is memorized data — it scores
train accuracy (~0.906), not the genuine held-out number (~0.72). The whole
deployment adaptation gates/calibrates on that loader, so the gate measured
memorized data.

THE FIX (asserted here):
  - train (``_train_raw``)            = the FULL ImageNet ``split="train"``.
  - val   (``_val_raw``, tuner gate)  = a seeded slice of the OFFICIAL ``split="val"``.
  - test  (``_test_raw``, reported)   = the DISJOINT remaining slice of ``split="val"``.

So val and test are disjoint seeded partitions of the official val (NEVER from
train), genuinely held out for BOTH regimes. The headline assertion: a model
that memorized the train indices cannot score on the gate — ``val`` is not a
subset of train.

Disk-free: a synthetic stand-in mimics DISTINCT train vs val sources (train and
val carry disjoint, recognizable sample payloads), so we can prove provenance
without the 1.2M-image ImageNet. A network/disk-gated test runs against real
ImageNet metadata when ``IMAGENET_ROOT`` is present.
"""

from __future__ import annotations

import os

import pytest
import torch

from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
    ImageNet_DataProvider,
    _seeded_val_test_indices,
)


# ImageNet official val is 50 images/class * 1000 classes = 50_000. The stand-in
# uses a smaller class-sorted val so the disk-free path is fast but exercises the
# real partitioning logic.
_N_CLASSES = 1000
_VAL_PER_CLASS = 4
_N_OFFICIAL_VAL = _N_CLASSES * _VAL_PER_CLASS
_N_TRAIN = 8000


def _val_class_of(idx: int) -> int:
    return idx // _VAL_PER_CLASS


def _distinct_val_classes(indices) -> set[int]:
    return {_val_class_of(i) for i in indices}


class TestSeededValTestIndices:
    """The official-val partition helper: seeded, disjoint, complete, sized."""

    def test_val_and_test_partition_the_official_val(self):
        val_idx, test_idx = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        val_set, test_set = set(val_idx), set(test_idx)
        assert val_set.isdisjoint(test_set), "val and test share official-val indices"
        assert val_set | test_set == set(range(_N_OFFICIAL_VAL)), (
            "val+test must partition the FULL official-val index range"
        )
        assert len(val_idx) == len(val_set), "val indices contain duplicates"
        assert len(test_idx) == len(test_set), "test indices contain duplicates"

    def test_val_fraction_is_honored(self):
        val_idx, test_idx = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        assert len(val_idx) == int(_N_OFFICIAL_VAL * 0.2)
        assert len(test_idx) == _N_OFFICIAL_VAL - int(_N_OFFICIAL_VAL * 0.2)

    def test_val_slice_spans_all_classes(self):
        """The gate slice must be representative — a permutation, not a contiguous
        class-starved range (official val is also class-sorted)."""
        val_idx, _ = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        val_classes = _distinct_val_classes(val_idx)
        # 20% of 4000 = 800 drawn over 1000 classes -> expected distinct ~ 550.
        assert len(val_classes) >= 400, (
            f"gate val covers only {len(val_classes)} classes; expected a "
            "representative sample over the 1000 classes"
        )
        contiguous_val = range(int(_N_OFFICIAL_VAL * 0.2))
        assert len(_distinct_val_classes(contiguous_val)) <= 220, (
            "sanity: a contiguous slice of class-sorted official val is class-starved"
        )

    def test_reproducible_for_fixed_seed(self):
        a_val, a_test = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        b_val, b_test = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        assert a_val == b_val
        assert a_test == b_test

    def test_different_seeds_give_different_partitions(self):
        v0, _ = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=0)
        v1, _ = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=1)
        assert v0 != v1, "different seeds must yield different gate slices"

    def test_none_seed_is_still_deterministic_and_disjoint(self):
        """seed=None (DataProvider allows it) must not crash and stay a valid partition."""
        val_idx, test_idx = _seeded_val_test_indices(_N_OFFICIAL_VAL, 0.2, seed=None)
        val_set, test_set = set(val_idx), set(test_idx)
        assert val_set.isdisjoint(test_set)
        assert val_set | test_set == set(range(_N_OFFICIAL_VAL))


# A stand-in with DISTINCT train vs val sources so provenance is provable: a
# train sample's payload starts at 0.0; an official-val sample's payload starts
# at 1.0. A "memorized-train" probe can then assert the gate carries no train
# payloads at all.
_TRAIN_TAG = 0.0
_VAL_TAG = 1.0


class _TaggedSource(torch.utils.data.Dataset):
    """Stand-in for ``torchvision.datasets.ImageNet``: train and val are DISTINCT
    sources. Each sample's first pixel encodes its source tag so the test can
    follow provenance through Subset wiring."""

    def __init__(self, root, split, transform):
        self.split = split
        if split == "train":
            self._n = _N_TRAIN
            self._tag = _TRAIN_TAG
        else:  # official val
            self._n = _N_OFFICIAL_VAL
            self._tag = _VAL_TAG
        # class-sorted, like real ImageNet
        self.targets = [
            (i // (_N_TRAIN // _N_CLASSES)) if split == "train" else _val_class_of(i)
            for i in range(self._n)
        ]

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        x = torch.zeros(3, 4, 4)
        x[0, 0, 0] = self._tag
        return x, int(self.targets[i])


def _build_provider(monkeypatch, *, seed):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "ImageNet", _TaggedSource)
    monkeypatch.setattr(
        "mimarsinan.data_handling.data_providers.imagenet_data_provider._ensure_imagenet_symlink",
        lambda p: "/tmp",
    )
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)
    return ImageNet_DataProvider("/tmp/datasets", seed=seed)


class TestProviderGateIsHeldOut:
    """The provider __init__ must source the gate (val) from the OFFICIAL val."""

    def test_train_raw_is_the_full_train(self, monkeypatch):
        """(d) _train_raw is the full train (every train index, no holdout carved out)."""
        p = _build_provider(monkeypatch, seed=0)
        raw = p.raw_datasets()
        train = raw["train"]
        # The full-train object the from-scratch trainer reads. Either the raw
        # source directly, or a Subset covering EVERY train index.
        full = train.dataset if hasattr(train, "dataset") else train
        assert isinstance(full, _TaggedSource) and full.split == "train"
        assert len(train) == _N_TRAIN, (
            "train must be the FULL ImageNet train (no val carved out of it)"
        )
        if hasattr(train, "indices"):
            assert set(train.indices) == set(range(_N_TRAIN))

    def test_val_is_sourced_from_official_val_not_train(self, monkeypatch):
        """(a) THE LEAK IS GONE: val comes from split='val', not from training_full."""
        p = _build_provider(monkeypatch, seed=0)
        raw = p.raw_datasets()
        val = raw["val"]
        val_source = val.dataset if hasattr(val, "dataset") else val
        assert isinstance(val_source, _TaggedSource) and val_source.split == "val", (
            "the tuner gate must be sourced from the OFFICIAL val (split='val'), "
            "NOT from the training split"
        )

    def test_val_and_test_are_disjoint(self, monkeypatch):
        """(b) val and test are disjoint slices of the SAME official val source."""
        p = _build_provider(monkeypatch, seed=0)
        raw = p.raw_datasets()
        val, test = raw["val"], raw["test"]
        val_source = val.dataset if hasattr(val, "dataset") else val
        test_source = test.dataset if hasattr(test, "dataset") else test
        assert val_source is test_source or (
            isinstance(val_source, _TaggedSource)
            and isinstance(test_source, _TaggedSource)
            and val_source.split == test_source.split == "val"
        ), "val and test must both be the official val source"
        val_idx = set(val.indices) if hasattr(val, "indices") else set(range(len(val)))
        test_idx = (
            set(test.indices) if hasattr(test, "indices") else set(range(len(test)))
        )
        assert val_idx.isdisjoint(test_idx), "val and test slices overlap (leak)"
        assert val_idx | test_idx == set(range(_N_OFFICIAL_VAL)), (
            "val+test must cover the whole official val (no images dropped)"
        )

    def test_reproducible_per_seed(self, monkeypatch):
        """(c) reproducible per seed; different seeds -> different gate."""
        p0 = _build_provider(monkeypatch, seed=0)
        p0b = _build_provider(monkeypatch, seed=0)
        v0 = list(p0.raw_datasets()["val"].indices)
        assert v0 == list(p0b.raw_datasets()["val"].indices)
        p1 = _build_provider(monkeypatch, seed=1)
        assert v0 != list(p1.raw_datasets()["val"].indices)

    def test_memorized_train_model_cannot_score_on_the_gate(self, monkeypatch):
        """HEADLINE: a model that memorized train indices cannot score on the gate.

        Concretely, EVERY sample drawn through the gate (val) carries the
        official-val tag, never the train tag — so a model that only recognizes
        train payloads gets no free accuracy from the gate."""
        p = _build_provider(monkeypatch, seed=0)
        # Read the RAW gate samples (the eval pipeline's transforms are orthogonal
        # to provenance; the source tag rides on the raw payload).
        val_raw = p.raw_datasets()["val"]
        n = len(val_raw)
        assert n > 0
        tags = {float(val_raw[i][0][0, 0, 0]) for i in range(min(n, 256))}
        assert tags == {_VAL_TAG}, (
            "gate samples must ALL come from the official val; a train-memorizing "
            f"model would recognize none of them. Saw tags {tags}"
        )
        # And the test slice is also pure official-val (the reported number stays clean).
        test_raw = p.raw_datasets()["test"]
        test_tags = {
            float(test_raw[i][0][0, 0, 0]) for i in range(min(len(test_raw), 256))
        }
        assert test_tags == {_VAL_TAG}

    def test_gate_carries_no_training_indices(self, monkeypatch):
        """Cross-check (a)+(b): the gate's underlying source is not the train source."""
        p = _build_provider(monkeypatch, seed=0)
        raw = p.raw_datasets()
        train_source = (
            raw["train"].dataset if hasattr(raw["train"], "dataset") else raw["train"]
        )
        val_source = (
            raw["val"].dataset if hasattr(raw["val"], "dataset") else raw["val"]
        )
        assert val_source is not train_source, (
            "the gate must NOT be a view onto the training dataset object"
        )


def _imagenet_metadata_present() -> bool:
    """True only when real ImageNet metadata is on disk (opt-in, disk-bound)."""
    if os.environ.get("RUN_IMAGENET_VAL_SPLIT_CHECK") != "1":
        return False
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path("/home/yigit/repos/wt-leak/.env"))
    except Exception:
        pass
    raw = os.environ.get("IMAGENET_ROOT", "").strip()
    if not raw:
        return False
    root = os.path.abspath(os.path.expanduser(raw))
    meta = os.path.join(root, "meta.bin")
    devkit = os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")
    val = os.path.join(root, "val")
    return os.path.isdir(val) and (os.path.isfile(meta) or os.path.isfile(devkit))


@pytest.mark.integration
@pytest.mark.skipif(
    not _imagenet_metadata_present(),
    reason="Set RUN_IMAGENET_VAL_SPLIT_CHECK=1 and IMAGENET_ROOT with real ImageNet metadata",
)
def test_real_imagenet_gate_is_official_val_and_disjoint_from_test():
    """On real ImageNet: the gate and the test set are disjoint slices of split='val',
    the gate is NOT a subset of split='train', and the train split is the full 1.28M."""
    p = ImageNet_DataProvider("/tmp/datasets_real", seed=0)
    raw = p.raw_datasets()

    val, test = raw["val"], raw["test"]
    val_source = val.dataset if hasattr(val, "dataset") else val
    test_source = test.dataset if hasattr(test, "dataset") else test
    # both views onto the official val (split attr lives on torchvision ImageNet)
    assert getattr(val_source, "split", None) == "val"
    assert getattr(test_source, "split", None) == "val"

    val_idx = set(val.indices) if hasattr(val, "indices") else set(range(len(val)))
    test_idx = set(test.indices) if hasattr(test, "indices") else set(range(len(test)))
    assert val_idx.isdisjoint(test_idx)
    assert len(val_idx) + len(test_idx) == 50_000, "official val is 50k images"

    # train is the full ImageNet train (~1.28M), and the gate is not a slice of it.
    train = raw["train"]
    train_source = train.dataset if hasattr(train, "dataset") else train
    assert getattr(train_source, "split", None) == "train"
    assert len(train) == len(train_source) == 1_281_167
    assert train_source is not val_source

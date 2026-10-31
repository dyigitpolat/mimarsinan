"""[U-HV] The verified sample read takes its split by declaration: the development split, or the test split by name.

``sample_loader.load_pairs_by_index(split=...)`` is the one split-parameterized (sample, label) read a device step
uses. ``split="validation"`` (``DEVELOPMENT_SPLIT``) is the provider's validation carve and neither builds nor reads
the test split; ``split="test"`` reads the test split, and only a run whose declaration names it asks for that. Any
other name is refused by name (``UnknownSampleSplit``, a toolchain refusal) before a provider is built. The private
read refuses it too: before U-HV it served the test loader for every split name other than "validation", so a
misspelt development split read the test split silently. ``sample_read`` is the record of a read (the declaring key,
the split and the per-split indices) that bundles, step entries and the deployment record print.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.data_handling import sample_loader

from unit.data_handling.split_fixtures import ramp_factory


def _same(pairs_a, pairs_b) -> bool:
    return len(pairs_a) == len(pairs_b) and all(
        torch.equal(xa, xb) and torch.equal(ya, yb) for (xa, ya), (xb, yb) in zip(pairs_a, pairs_b))


class TestTheDevelopmentRead:
    def test_it_builds_and_reads_no_test_split(self):
        factory = ramp_factory()
        sample_loader.load_pairs_by_index(factory, [0, 2], split="validation", num_workers=0)
        assert factory.test_events() == {"constructions": 0, "reads": 0}

    def test_it_returns_the_validation_pairs_in_index_order(self):
        factory = ramp_factory()
        pairs = sample_loader.load_pairs_by_index(factory, [3, 1], split="validation", num_workers=0)
        xs, ys = factory.validation
        assert _same(pairs, [(xs[3:4], ys[3:4]), (xs[1:2], ys[1:2])])

    def test_it_equals_the_private_validation_read(self):
        declared = sample_loader.load_pairs_by_index(ramp_factory(), [4, 0, 5], split="validation", num_workers=0)
        private = sample_loader._pairs_by_index(ramp_factory(), [4, 0, 5], num_workers=0, split="validation")
        assert _same(declared, private)

    def test_the_development_split_is_the_validation_carve(self):
        assert sample_loader.DEVELOPMENT_SPLIT == "validation"
        assert sample_loader.SAMPLE_SPLITS == ("validation", "test")


class TestTheTestRead:
    def test_the_test_split_is_read_when_named(self):
        factory = ramp_factory()
        pairs = sample_loader.load_pairs_by_index(factory, [1], split="test", num_workers=0)
        xs, ys = factory.test
        assert _same(pairs, [(xs[1:2], ys[1:2])])
        assert factory.test_events()["constructions"] >= 1 and factory.test_events()["reads"] >= 1

    def test_it_equals_the_explicit_test_read(self):
        declared = sample_loader.load_pairs_by_index(ramp_factory(), [2, 0], split="test", num_workers=0)
        explicit = sample_loader.load_test_pairs_by_index(ramp_factory(), [2, 0], num_workers=0)
        assert _same(declared, explicit)


class TestAnUnknownSplitIsRefusedByName:
    @pytest.mark.parametrize("split", ["valid", "train", "Test", "", None])
    def test_the_declared_read_refuses_before_a_provider_is_built(self, split):
        factory = ramp_factory()
        with pytest.raises(sample_loader.UnknownSampleSplit) as refused:
            sample_loader.load_pairs_by_index(factory, [0], split=split, num_workers=0)
        assert isinstance(refused.value, ToolchainRefusal)
        assert repr(split) in str(refused.value) and "'validation'" in str(refused.value)
        assert factory.built == 0

    def test_the_private_read_no_longer_falls_through_to_the_test_split(self):
        factory = ramp_factory()
        with pytest.raises(sample_loader.UnknownSampleSplit):
            sample_loader._pairs_by_index(factory, [0], num_workers=0, split="training")
        assert factory.test_events() == {"constructions": 0, "reads": 0}

    def test_the_refusal_names_the_declaring_key(self):
        with pytest.raises(sample_loader.UnknownSampleSplit, match="some_declared_key='dev'"):
            sample_loader.require_sample_split("dev", declared_by="some_declared_key")

    def test_a_known_split_passes_through(self):
        assert [sample_loader.require_sample_split(s, declared_by="k") for s in ("validation", "test")] == [
            "validation", "test"]


class TestTheRecordOfARead:
    def test_it_names_the_key_the_split_and_the_indices(self):
        assert sample_loader.sample_read("validation", range(3), declared_by="k") == {
            "declared_by": "k", "split": "validation", "indices": [0, 1, 2]}

    def test_an_unknown_split_cannot_be_recorded(self):
        with pytest.raises(sample_loader.UnknownSampleSplit):
            sample_loader.sample_read("dev", [0], declared_by="k")

"""Tests for PipelineCache: in-memory operations and disk round-trips."""

import pytest
import torch
import torch.nn as nn
import pickle

from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache


class TestCacheInMemory:
    def test_add_and_get(self):
        c = PipelineCache()
        c.add("key", 42)
        assert c.get("key") == 42

    def test_get_missing_returns_none(self):
        c = PipelineCache()
        assert c.get("missing") is None

    def test_remove(self):
        c = PipelineCache()
        c.add("k", 1)
        c.remove("k")
        assert c.get("k") is None

    def test_remove_missing_is_silent(self):
        c = PipelineCache()
        c.remove("nonexistent")

    def test_len(self):
        c = PipelineCache()
        assert len(c) == 0
        c.add("a", 1)
        c.add("b", 2)
        assert len(c) == 2

    def test_contains(self):
        c = PipelineCache()
        c.add("x", 0)
        assert "x" in c
        assert "y" not in c

    def test_iter(self):
        c = PipelineCache()
        c.add("a", 1)
        c.add("b", 2)
        assert set(c) == {"a", "b"}

    def test_getitem(self):
        c = PipelineCache()
        c.add("v", 99)
        assert c["v"] == 99

    def test_setitem(self):
        c = PipelineCache()
        c["val"] = 10
        assert c.get("val") == 10

    def test_delitem(self):
        c = PipelineCache()
        c.add("d", 7)
        del c["d"]
        assert "d" not in c

    def test_overwrite_existing_key(self):
        c = PipelineCache()
        c.add("k", 1)
        c.add("k", 2)
        assert c.get("k") == 2


class TestCacheRoundTrip:
    def test_basic_strategy(self, tmp_path):
        c = PipelineCache()
        c.add("step.scalar", 3.14)
        c.add("step.list", [1, 2, 3])
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        assert c2.get("step.scalar") == pytest.approx(3.14)
        assert c2.get("step.list") == [1, 2, 3]

    def test_torch_model_strategy(self, tmp_path):
        model = nn.Linear(4, 2)
        c = PipelineCache()
        c.add("step.model", model, "torch_model")
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        loaded = c2.get("step.model")
        assert isinstance(loaded, nn.Linear)
        assert loaded.weight.shape == (2, 4)

    def test_pickle_strategy(self, tmp_path):
        obj = {"nested": [1, 2], "tensor": torch.tensor([3.0])}
        c = PipelineCache()
        c.add("step.obj", obj, "pickle")
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        loaded = c2.get("step.obj")
        assert loaded["nested"] == [1, 2]
        assert torch.allclose(loaded["tensor"], torch.tensor([3.0]))

    def test_mixed_strategies(self, tmp_path):
        c = PipelineCache()
        c.add("a.scalar", 1, "basic")
        c.add("b.model", nn.Linear(2, 2), "torch_model")
        c.add("c.blob", {"data": [4, 5]}, "pickle")
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        assert c2.get("a.scalar") == 1
        assert isinstance(c2.get("b.model"), nn.Linear)
        assert c2.get("c.blob")["data"] == [4, 5]

    def test_load_empty_directory(self, tmp_path):
        c = PipelineCache()
        c.load(str(tmp_path))
        assert len(c) == 0

    def test_store_creates_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir")
        c = PipelineCache()
        c.add("x", 1)
        c.store(nested)

        c2 = PipelineCache()
        c2.load(nested)
        assert c2.get("x") == 1

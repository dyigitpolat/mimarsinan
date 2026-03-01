"""
Stress tests for PipelineCache.

Tests corrupted files, strategy mismatches, data loss scenarios.
"""

import pytest
import json
import os
import torch
import torch.nn as nn

from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache


class TestCacheStrategyDowngrade:
    """
    BUG: PipelineCache.__setitem__ uses add(name, object) which defaults
    to strategy="basic". If you previously stored an entry with
    "torch_model" or "pickle" strategy, then use c["key"] = new_model,
    the strategy is silently downgraded to "basic". On store/load the
    entry will be written as JSON, which fails for non-JSON-serializable
    objects.
    """

    def test_setitem_downgrades_strategy(self, tmp_path):
        c = PipelineCache()
        model = nn.Linear(4, 2)
        c.add("step.model", model, "torch_model")

        # Verify strategy is torch_model
        assert c.cache["step.model"][1] == "torch_model"

        # Use __setitem__ to update the model
        new_model = nn.Linear(4, 2)
        c["step.model"] = new_model

        # Strategy is now "basic" — this is the bug
        if c.cache["step.model"][1] == "basic":
            # Try to store — this will fail because nn.Module is not JSON serializable
            with pytest.raises(TypeError):
                c.store(str(tmp_path))
            pytest.xfail(
                "BUG: __setitem__ downgrades storage strategy from torch_model "
                "to basic, causing store to fail with non-JSON-serializable objects"
            )

    def test_overwrite_preserves_strategy_via_add(self, tmp_path):
        """Using add() with explicit strategy works correctly."""
        c = PipelineCache()
        model = nn.Linear(4, 2)
        c.add("step.model", model, "torch_model")
        c.add("step.model", nn.Linear(8, 2), "torch_model")
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        loaded = c2.get("step.model")
        assert isinstance(loaded, nn.Linear)
        assert loaded.weight.shape == (2, 8)


class TestCacheCorruptedFiles:
    def test_corrupted_metadata_json(self, tmp_path):
        """Corrupt the metadata JSON and verify load behavior."""
        c = PipelineCache()
        c.add("test.value", 42)
        c.store(str(tmp_path))

        # Corrupt the metadata
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text("{ broken json }{")

        c2 = PipelineCache()
        with pytest.raises(json.JSONDecodeError):
            c2.load(str(tmp_path))

    def test_missing_data_file(self, tmp_path):
        """Metadata references a file that doesn't exist on disk."""
        c = PipelineCache()
        c.add("test.value", 42)
        c.store(str(tmp_path))

        # Delete the data file but keep metadata
        data_file = tmp_path / "test.value.json"
        if data_file.exists():
            data_file.unlink()

        c2 = PipelineCache()
        with pytest.raises(FileNotFoundError):
            c2.load(str(tmp_path))

    def test_basic_strategy_with_non_json_serializable(self, tmp_path):
        """
        Storing a non-JSON-serializable object with 'basic' strategy should
        raise an error at store time, not silently corrupt data.
        """
        c = PipelineCache()
        c.add("test.tensor", torch.tensor([1.0, 2.0]), "basic")
        with pytest.raises(TypeError):
            c.store(str(tmp_path))


class TestCacheLoadDiscards:
    """
    PipelineCache.load() does self.cache = {} which discards any
    in-memory entries that weren't stored. This is documented behavior
    but can lead to data loss if not understood.
    """

    def test_load_discards_unstored_entries(self, tmp_path):
        c = PipelineCache()
        c.add("persistent", 1)
        c.store(str(tmp_path))

        # Add a new entry but don't store it
        c.add("ephemeral", 2)
        assert c.get("ephemeral") == 2

        # Load overwrites in-memory state
        c.load(str(tmp_path))
        assert c.get("persistent") == 1
        assert c.get("ephemeral") is None, \
            "Load should discard unstored entries"


class TestCacheEdgeCases:
    def test_key_with_dots(self, tmp_path):
        """Keys with multiple dots (common in step.substep.key format)."""
        c = PipelineCache()
        c.add("pipeline.step.sub.value", 99)
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        assert c2.get("pipeline.step.sub.value") == 99

    def test_key_with_special_characters(self, tmp_path):
        """Keys with slashes or other filesystem-unfriendly characters."""
        c = PipelineCache()
        c.add("step/with/slashes", [1, 2])
        # This might fail because the filename would create subdirectories
        try:
            c.store(str(tmp_path))
        except (FileNotFoundError, OSError) as e:
            pytest.xfail(
                f"Cache keys with slashes create filesystem problems: {e}"
            )

    def test_empty_string_key(self, tmp_path):
        c = PipelineCache()
        c.add("", "empty_key_value")
        assert c.get("") == "empty_key_value"
        c.store(str(tmp_path))

        c2 = PipelineCache()
        c2.load(str(tmp_path))
        assert c2.get("") == "empty_key_value"

    def test_store_then_modify_then_load(self, tmp_path):
        """Store, modify in memory, load should restore to disk state."""
        c = PipelineCache()
        c.add("val", 10)
        c.store(str(tmp_path))
        c.add("val", 999)
        assert c.get("val") == 999

        c.load(str(tmp_path))
        assert c.get("val") == 10

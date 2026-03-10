"""Tests for the model registry (ModelRegistry, get_model_types, get_model_config_schema)."""

import pytest

from mimarsinan.pipelining.model_registry import (
    ModelRegistry,
    get_model_types,
    get_model_config_schema,
)


class TestGetModelTypes:
    def test_returns_list(self):
        result = get_model_types()
        assert isinstance(result, list)

    def test_each_entry_has_id_label_category(self):
        result = get_model_types()
        assert len(result) > 0
        for entry in result:
            assert "id" in entry
            assert "label" in entry
            assert "category" in entry
            assert entry["category"] in ("native", "torch")

    def test_contains_expected_ids(self):
        result = get_model_types()
        ids = [e["id"] for e in result]
        assert "mlp_mixer" in ids
        assert "vit" in ids
        assert "torch_sequential_linear" in ids

    def test_sorted_by_id(self):
        result = get_model_types()
        ids = [e["id"] for e in result]
        assert ids == sorted(ids)


class TestGetModelConfigSchema:
    def test_returns_list(self):
        result = get_model_config_schema("mlp_mixer")
        assert isinstance(result, list)

    def test_mlp_mixer_has_expected_fields(self):
        result = get_model_config_schema("mlp_mixer")
        assert len(result) > 0
        keys = [f["key"] for f in result]
        assert "patch_n_1" in keys
        assert "fc_w_1" in keys
        for field in result:
            assert "key" in field
            assert "type" in field
            assert "label" in field
            assert "default" in field
            assert field["type"] in ("number", "text", "select", "toggle")

    def test_unknown_model_returns_empty_list(self):
        result = get_model_config_schema("nonexistent_model_xyz")
        assert result == []

    def test_vit_schema_from_builder(self):
        result = get_model_config_schema("vit")
        assert len(result) > 0
        keys = [f["key"] for f in result]
        assert "patch_size" in keys
        assert "d_model" in keys

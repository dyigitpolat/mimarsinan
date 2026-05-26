"""Unit tests for config_schema.display_view."""

from __future__ import annotations

import pytest

from mimarsinan.config_schema import build_flat_pipeline_config, get_default_deployment_parameters
from mimarsinan.config_schema.display_view import build_config_display_view
from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state


def _minimal_nested_raw(**overrides) -> dict:
    state = {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "test_exp",
        "pipeline_mode": "phased",
        "deployment_parameters": {
            "spiking_mode": "lif",
            "weight_quantization": True,
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
        },
        "platform_constraints": {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 100}],
        },
    }
    state.update(overrides)
    return state


def _minimal_nested(**overrides) -> dict:
    return build_deployment_config_from_state(_minimal_nested_raw(**overrides))


def _field_by_key(view: dict, key: str) -> dict | None:
    for section in view.get("sections", []):
        for field in section.get("fields", []):
            if field.get("key") == key:
                return field
    return None


def _section_ids(view: dict) -> list[str]:
    return [s["id"] for s in view.get("sections", [])]


class TestBuildConfigDisplayView:
    def test_minimal_nested_shows_defaults(self):
        saved = _minimal_nested_raw()
        view = build_config_display_view(saved, saved_config=saved)
        assert view["summary"]["experiment_name"] == "test_exp"
        assert view["summary"]["pipeline_mode"] == "phased"
        assert view["summary"]["spiking_mode"] == "lif"
        lr = _field_by_key(view, "lr")
        assert lr is not None
        assert lr["value"] == 0.001
        assert lr["source"] == "default"

    def test_explicit_override_marked(self):
        saved = _minimal_nested_raw(
            deployment_parameters={
                "spiking_mode": "lif",
                "weight_quantization": True,
                "model_config_mode": "user",
                "hw_config_mode": "fixed",
                "lr": 0.042,
            },
        )
        view = build_config_display_view(saved, saved_config=saved)
        lr = _field_by_key(view, "lr")
        assert lr is not None
        assert lr["value"] == 0.042
        assert lr["source"] == "explicit"

    def test_phased_preset_quant_flags(self):
        saved = _minimal_nested_raw()
        view = build_config_display_view(saved, saved_config=saved)
        wt = _field_by_key(view, "weight_quantization")
        assert wt is not None
        assert wt["value"] is True
        assert wt["source"] in ("preset", "default", "derived", "explicit")

    def test_float_weights_derives_vanilla(self):
        saved = _minimal_nested_raw(
            pipeline_mode="phased",
            deployment_parameters={
                "spiking_mode": "lif",
                "weight_quantization": False,
                "model_config_mode": "user",
                "hw_config_mode": "fixed",
            },
        )
        view = build_config_display_view(saved, saved_config=saved)
        pm = _field_by_key(view, "pipeline_mode")
        assert pm is not None
        assert pm["value"] == "vanilla"
        act = _field_by_key(view, "activation_quantization")
        assert act is not None
        assert act["value"] is False

    def test_flat_runtime_config_does_not_crash(self):
        flat = build_flat_pipeline_config(
            {"spiking_mode": "lif", "weight_quantization": True},
            {"cores": [{"max_axons": 64, "max_neurons": 64, "count": 10}]},
            pipeline_mode="phased",
        )
        flat["device"] = "cpu"
        flat["input_shape"] = [1, 28, 28]
        flat["num_classes"] = 10
        view = build_config_display_view(flat)
        assert view["summary"]["spiking_mode"] == "lif"
        device = _field_by_key(view, "device")
        assert device is not None
        assert device["source"] == "runtime"

    def test_cores_nested_expander(self):
        saved = _minimal_nested_raw()
        view = build_config_display_view(saved, saved_config=saved)
        cores = view.get("nested", {}).get("cores")
        assert cores is not None
        assert cores["type"] == "cores"
        assert len(cores["items"]) >= 1
        assert "max_axons" in cores["items"][0]

    def test_unknown_key_in_other_section(self):
        saved = _minimal_nested_raw()
        saved["deployment_parameters"]["custom_test_key_xyz"] = 99
        view = build_config_display_view(saved, saved_config=saved)
        field = _field_by_key(view, "custom_test_key_xyz")
        assert field is not None
        assert field["source"] == "explicit"

    def test_pipeline_preview_present(self):
        saved = _minimal_nested_raw()
        view = build_config_display_view(saved, saved_config=saved)
        preview = view.get("pipeline_preview")
        assert preview is not None
        assert isinstance(preview.get("steps"), list)
        assert len(preview["steps"]) >= 1

    def test_sections_have_required_groups(self):
        saved = _minimal_nested_raw()
        view = build_config_display_view(saved, saved_config=saved)
        ids = _section_ids(view)
        assert "run" in ids
        assert "pipeline" in ids
        assert "hardware" in ids

"""Serialize the config-key registry + resolve drafts for the wizard frontend."""

from __future__ import annotations

from typing import Any, Dict

from mimarsinan.config_schema.defaults import (
    get_default_training_recipe,
    get_default_tuning_recipe,
)
import mimarsinan.data_handling.data_providers  # noqa: F401  # pyright: ignore[reportUnusedImport] — registers providers + their normalization presets
from mimarsinan.config_schema.registry import serialize_registry
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.data_handling.preprocessing import (
    NORMALIZATION_PRESETS,
    interpolation_mode_names,
)
from mimarsinan.gui.wizard.schema import (
    get_wizard_nas_schema,
    get_wizard_temporal_allocation_schema,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    get_pipeline_semantic_group_by_step_name,
    get_pipeline_step_specs,
)


def _recipe_field_schema() -> Dict[str, Any]:
    """Field schema for RECIPE widgets, derived from the default recipes."""
    fields: Dict[str, Any] = {}
    training = get_default_training_recipe()
    tuning = get_default_tuning_recipe()
    for key, value in training.items():
        if isinstance(value, list):
            field_type = "float_list"
        elif isinstance(value, str):
            field_type = "str"
        else:
            field_type = "float"
        fields[key] = {
            "type": field_type,
            "default_training": value,
            "default_tuning": tuning.get(key),
        }
    return fields


def _preprocessing_field_schema() -> Dict[str, Any]:
    return {
        "resize_to": {"type": "int"},
        "normalize": {"type": "enum", "options": sorted(NORMALIZATION_PRESETS)},
        "interpolation": {"type": "enum", "options": list(interpolation_mode_names())},
    }


# STR keys whose option lists are served by a live endpoint (registries).
_DYNAMIC_OPTION_ENDPOINTS = {
    "model_type": "/api/model_types",
    "data_provider_name": "/api/data_providers",
}


def config_schema_payload() -> Dict[str, Any]:
    """The full wizard rendering payload: registry + sub-schema surfaces."""
    payload = serialize_registry()
    payload["recipe_fields"] = _recipe_field_schema()
    payload["preprocessing_fields"] = _preprocessing_field_schema()
    payload["dynamic_options"] = dict(_DYNAMIC_OPTION_ENDPOINTS)
    payload["nas"] = get_wizard_nas_schema()
    payload["temporal_allocation"] = get_wizard_temporal_allocation_schema()
    return payload


def resolve_payload(draft: Dict[str, Any]) -> Dict[str, Any]:
    """One round-trip for the wizard: resolution + live pipeline-step preview
    + the emitted document (exactly what Launch submits)."""
    resolution = resolve_draft(draft or {})
    out: Dict[str, Any] = {
        "ok": resolution.ok,
        "derived": resolution.derived,
        "errors": resolution.errors,
        "explicit_keys": resolution.explicit_keys,
        "unknown_keys": resolution.unknown_keys,
        "diff_vs_defaults": resolution.diff_vs_defaults,
        "emitted": emit_deployment_config(draft or {}),
        "pipeline": {"steps": [], "semantic_groups": []},
    }
    if resolution.ok:
        config = dict(resolution.resolved)
        specs = get_pipeline_step_specs(config)
        groups = get_pipeline_semantic_group_by_step_name(config)
        steps = [name for name, _ in specs]
        out["pipeline"] = {
            "steps": steps,
            "semantic_groups": [groups.get(name, "other") for name in steps],
        }
    return out

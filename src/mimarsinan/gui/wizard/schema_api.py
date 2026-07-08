"""Serialize the config-key registry + resolve drafts for the wizard frontend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mimarsinan.config_schema.defaults import (
    get_default_training_recipe,
    get_default_tuning_recipe,
)
import mimarsinan.data_handling.data_providers  # noqa: F401  # pyright: ignore[reportUnusedImport] — registers providers + their normalization presets
from mimarsinan.config_schema.registry import (
    Category,
    FieldType,
    REGISTRY,
    parse_deployment_document,
    serialize_registry,
)
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.gui.wizard.starter import load_starter_baseline
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
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
from mimarsinan.search.search_space_description import (
    CORE_DIM_GRANULARITY,
    DEFAULT_CORE_AXONS_BOUNDS,
    DEFAULT_CORE_COUNT_BOUNDS,
    DEFAULT_CORE_NEURONS_BOUNDS,
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


def _hw_search_space_field_schema() -> Dict[str, Any]:
    """Structured-editor schema for ``search_space``, from the search-space SSOT."""
    return {
        "num_core_types": {
            "type": "int", "default": 1, "min": 1,
            "doc": "Distinct core types the hardware co-search may allocate.",
        },
        "core_axons_bounds": {
            "type": "int_range", "default": list(DEFAULT_CORE_AXONS_BOUNDS),
            "step": CORE_DIM_GRANULARITY,
            "doc": "Per-core axon-count bounds the search explores "
                   f"(multiples of {CORE_DIM_GRANULARITY}).",
        },
        "core_neurons_bounds": {
            "type": "int_range", "default": list(DEFAULT_CORE_NEURONS_BOUNDS),
            "step": CORE_DIM_GRANULARITY,
            "doc": "Per-core neuron-count bounds the search explores "
                   f"(multiples of {CORE_DIM_GRANULARITY}).",
        },
        "core_count_bounds": {
            "type": "int_range", "default": list(DEFAULT_CORE_COUNT_BOUNDS),
            "step": 1,
            "doc": "Core-count bounds per core type.",
        },
    }


# STR keys whose option lists are served by a live endpoint (registries).
_DYNAMIC_OPTION_ENDPOINTS = {
    "model_type": "/api/model_types",
    "data_provider_name": "/api/data_providers",
}


def _starter_baseline_values() -> Dict[str, Any]:
    """The wizard's diff baseline: the starter document's pinned values.

    experiment_name is excluded — its fresh derived name has no default and
    always renders user-owned; explicit nulls mean 'unset', never a pin.
    """
    parsed = parse_deployment_document(load_starter_baseline())
    return {
        key: value
        for key, value in parsed.known_flat_keys().items()
        if key != "experiment_name" and value is not None
    }


def config_schema_payload() -> Dict[str, Any]:
    """The full wizard rendering payload: registry + sub-schema surfaces."""
    payload = serialize_registry()
    # The baseline IS the wizard's default: a data-layer overlay next to the
    # workload-neutral framework default (which stays served untouched).
    for key, value in _starter_baseline_values().items():
        payload["keys"][key]["baseline"] = value
    payload["recipe_fields"] = _recipe_field_schema()
    payload["preprocessing_fields"] = _preprocessing_field_schema()
    payload["hw_search_space_fields"] = _hw_search_space_field_schema()
    payload["dynamic_options"] = dict(_DYNAMIC_OPTION_ENDPOINTS)
    payload["nas"] = get_wizard_nas_schema()
    payload["temporal_allocation"] = get_wizard_temporal_allocation_schema()
    return payload


def _apply_baseline_to_diff(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebase the framework diff onto the starter baseline: a knob differs
    only when it diverges from the baseline document (true user deltas)."""
    baseline = _starter_baseline_values()
    for row in rows:
        if row["key"] not in baseline:
            continue
        base = baseline[row["key"]]
        row["default"] = base
        row["differs"] = row["value"] != base
    return rows


def _vehicle_rows(draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-vehicle state, computable for EVERY draft: unrelated errors must
    never remove the rows the on/off toggles render from."""
    dp = parse_deployment_document(draft or {}).dp

    def effective(key: str) -> Any:
        if dp.get(key) is not None:
            return dp[key]
        entry = REGISTRY[key]
        return entry.default if entry.has_default() else None

    mode = str(effective("spiking_mode"))
    schedule = effective("ttfs_cycle_schedule")
    sim_enables: Optional[Dict[str, bool]]
    try:
        sim_enables = dict(ConversionPolicy.derive(mode, schedule).sim_enables)
    except ValueError:
        sim_enables = None

    rows: List[Dict[str, Any]] = []
    for key, entry in REGISTRY.items():
        if (entry.group != "deployment_target"
                or entry.category is not Category.DERIVED
                or entry.type is not FieldType.BOOL):
            continue
        declared = dp.get(key)
        row: Dict[str, Any] = {
            "key": key,
            "label": entry.label,
            "declared": isinstance(declared, bool),
        }
        if sim_enables is None:
            row.update(supported=None, on=None,
                       why=f"unknown spiking_mode {mode!r} — fix the mode to "
                           "see vehicle support")
        else:
            supported = bool(sim_enables.get(key, False))
            on = bool(supported and declared is not False)
            why = None
            if entry.why is not None:
                why = entry.why({
                    "spiking_mode": mode, "ttfs_cycle_schedule": schedule, key: on,
                })
            row.update(supported=supported, on=on, why=why)
        rows.append(row)
    return rows


def _resolved_view(resolution) -> Dict[str, Any]:
    """The concrete resolved value per registry key (the 'derived: <x>' data),
    plus the cheap model-profile registrations; {} while the draft errors —
    hypothetical values never render."""
    if not resolution.ok:
        return {}
    view = {k: v for k, v in resolution.resolved.items() if k in REGISTRY}
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    profile = ModelRegistry.get_workload_profile(str(view.get("model_type") or ""))
    if profile is not None:
        for key, value in profile.config_updates().items():
            view.setdefault(key, value)
    return view


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
        "diff_vs_defaults": _apply_baseline_to_diff(resolution.diff_vs_defaults),
        "emitted": emit_deployment_config(draft or {}),
        "resolved": _resolved_view(resolution),
        "vehicles": _vehicle_rows(draft or {}),
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

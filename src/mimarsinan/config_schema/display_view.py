"""Build structured monitor display payloads from deployment / runtime config."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Set

from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema import display_view_meta as _meta
from mimarsinan.config_schema.display_view_build import (
    build_nested_blocks,
    build_pipeline_preview,
)

def build_config_display_view(
    raw_config: Mapping[str, Any],
    *,
    saved_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured, JSON-serializable config view for the monitor UI."""
    raw_config = raw_config or {}
    saved = saved_config if saved_config is not None else (
        raw_config if _meta._is_nested_config(raw_config) else None
    )

    filled_outer: Dict[str, Any]
    if saved is not None:
        from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state
        filled_outer = build_deployment_config_from_state(dict(saved))
    else:
        filled_outer = {
            "pipeline_mode": raw_config.get("pipeline_mode", "phased"),
            "deployment_parameters": {
                k: v for k, v in raw_config.items()
                if k not in get_default_platform_constraints() and k not in _meta.TOP_LEVEL_RUN_KEYS
                and k not in _meta.RUNTIME_KEYS
            },
            "platform_constraints": {
                k: v for k, v in raw_config.items() if k in get_default_platform_constraints()
            },
        }

    pipeline_mode = str(filled_outer.get("pipeline_mode", "phased"))
    dp = dict(filled_outer.get("deployment_parameters") or {})
    pc_merge = _meta._platform_for_flat_merge(filled_outer.get("platform_constraints"))
    flat = build_flat_pipeline_config(dp, pc_merge, pipeline_mode=pipeline_mode)

    for key in _meta.TOP_LEVEL_RUN_KEYS:
        if key in filled_outer:
            flat[key] = filled_outer[key]

    if not _meta._is_nested_config(raw_config):
        for key, value in raw_config.items():
            flat[key] = value

    explicit_keys = _meta._collect_explicit_keys(saved) if saved is not None else set()
    preset_keys = _meta._collect_preset_keys(saved) if saved is not None else set()
    derived_keys = _meta._collect_derived_keys(saved) if saved is not None else set()

    runtime_keys: Set[str] = set(_meta.RUNTIME_KEYS)
    if not _meta._is_nested_config(raw_config):
        for key in raw_config:
            if key not in explicit_keys and key in flat and key not in get_default_deployment_parameters():
                if key not in get_default_platform_constraints() and key not in _meta.TOP_LEVEL_RUN_KEYS:
                    runtime_keys.add(key)

    nested = build_nested_blocks(flat, flat.get("model_type"))
    skip_in_sections = {"training_recipe", "tuning_recipe", "model_config", "arch_search", "platform_constraints"}

    grouped: Dict[str, List[Dict[str, Any]]] = {g["id"]: [] for g in _meta.CONFIG_DISPLAY_GROUPS}
    from mimarsinan.config_schema.display_view_build import _ordered_keys
    for key in _ordered_keys(flat):
        if key in skip_in_sections:
            continue
        meta = _meta._field_meta(key)
        value = flat.get(key)
        default = _meta._default_for_key(key, filled_outer.get("deployment_parameters", {}))
        field_type = meta.get("type") or _meta._infer_type(value)
        if field_type == "cores" and "cores" in nested:
            field_type = "cores_ref"
        source = _meta._resolve_field_source(
            key,
            explicit_keys=explicit_keys,
            preset_keys=preset_keys,
            derived_keys=derived_keys,
            runtime_keys=runtime_keys,
        )
        if source == "default" and not _meta._values_equal(value, default) and default is not None:
            source = "explicit" if key in explicit_keys else source
        grouped[meta["group"]].append({
            "key": key,
            "label": meta.get("label", _meta._snake_to_label(key)),
            "type": field_type,
            "value": _meta._json_safe(value),
            "default": _meta._json_safe(default),
            "source": source,
            "effect": meta.get("effect"),
            "important": bool(meta.get("important")),
        })

    sections: List[Dict[str, Any]] = []
    for group in _meta.CONFIG_DISPLAY_GROUPS:
        fields = grouped.get(group["id"], [])
        if group["id"] == "search" and not fields and "arch_search" not in nested:
            continue
        if not fields and group["id"] not in ("run", "pipeline", "hardware", "training", "simulation", "runtime"):
            continue
        if not fields:
            continue
        sections.append({
            "id": group["id"],
            "title": group["title"],
            "subtitle": group["subtitle"],
            "accent": group["accent"],
            "fields": fields,
        })

    summary = {
        "experiment_name": flat.get("experiment_name"),
        "pipeline_mode": flat.get("pipeline_mode"),
        "spiking_mode": flat.get("spiking_mode"),
        "model_type": flat.get("model_type"),
        "data_provider_name": flat.get("data_provider_name"),
    }

    return {
        "summary": _meta._json_safe(summary),
        "pipeline_preview": build_pipeline_preview(flat),
        "sections": sections,
        "nested": nested,
        "raw_resolved": _meta._json_safe(dict(flat)),
    }


def load_saved_config_from_run_dir(working_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load raw ``config.json`` from a run working directory, if present."""
    if not working_dir:
        return None
    path = os.path.join(working_dir, "_RUN_CONFIG", "config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def build_pipeline_config_view(
    raw_config: Mapping[str, Any],
    *,
    saved_config: Optional[Mapping[str, Any]] = None,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build config view, loading saved JSON from *working_dir* when needed."""
    saved = saved_config if saved_config is not None else load_saved_config_from_run_dir(working_dir)
    return build_config_display_view(raw_config, saved_config=saved)

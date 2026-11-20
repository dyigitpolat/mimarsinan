"""Build structured monitor display payloads from deployment / runtime config."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Set

from mimarsinan.chip_simulation.stored_point import (
    derived_at_stored_point, document_at_stored_point, read_run_document, stored_run_point,
)
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.registry import REGISTRY, Category
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema import display_view_meta as _meta
import mimarsinan.mapping.platform.target_expansion as target_expansion
from mimarsinan.config_schema.display_view_build import (
    build_nested_blocks,
    build_pipeline_preview,
    _ordered_keys,
)

def _flat_fallback_sections(raw_config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """A flat config split back into the two sections a run lays: each key in its registry section (every
    platform default a platform key), less the derived platform keys whose inputs it holds (the core maxima, the
    target's labels), which the merge derives again; the view then shows the flat config's own values over it."""
    def platform(key: str) -> bool:
        entry = REGISTRY.get(key)
        return key in get_default_platform_constraints() or (entry is not None and entry.section == "platform_constraints")

    def rederived(key: str) -> bool:
        entry = REGISTRY.get(key)
        return (platform(key) and entry is not None and entry.category is Category.DERIVED
                and bool(entry.derived_from) and set(entry.derived_from) <= set(raw_config))

    kept = {k: v for k, v in raw_config.items() if not rederived(k)}
    return {
        "deployment_parameters": {k: v for k, v in kept.items() if not platform(k)
                                  and k not in _meta.TOP_LEVEL_RUN_KEYS and k not in _meta.RUNTIME_KEYS},
        "platform_constraints": {k: v for k, v in kept.items() if platform(k)},
    }


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
        filled_outer = {"pipeline_mode": raw_config.get("pipeline_mode", "phased"),
                        **_flat_fallback_sections(raw_config)}

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

    model_type = flat.get("model_type")
    nested = build_nested_blocks(flat, model_type if isinstance(model_type, str) else None)
    skip_in_sections = {"training_recipe", "tuning_recipe", "model_config", "arch_search", "platform_constraints"}

    grouped: Dict[str, List[Dict[str, Any]]] = {g["id"]: [] for g in _meta.CONFIG_DISPLAY_GROUPS}
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
    """Load raw ``config.json`` from a run working directory, if present (``read_run_document``, the one reader)."""
    if not working_dir:
        return None
    try:
        data = read_run_document(working_dir)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def build_pipeline_config_view(
    raw_config: Mapping[str, Any],
    *,
    saved_config: Optional[Mapping[str, Any]] = None,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a config view. A document handed in (``saved_config``) is derived as the doors derive it. [U-LIF2 r3] A run
    directory's saved document (*working_dir*) is derived at the point its run resolved (``stored_run_point``: the point
    it persisted, else the historical point it states), never under today's entry data."""
    saved = saved_config if saved_config is not None else load_saved_config_from_run_dir(working_dir)
    if saved_config is not None or saved is None or not working_dir:
        return build_config_display_view(raw_config, saved_config=saved)
    point = stored_run_point(working_dir, saved, door=target_expansion.door_read)  # [U-LIF2 r6] as its door read it
    with derived_at_stored_point(point):
        return build_config_display_view(raw_config, saved_config=document_at_stored_point(saved, point))

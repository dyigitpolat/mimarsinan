"""Display view metadata resolved from the config-key registry (no hand tables)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from mimarsinan.config_schema.defaults import (
    PIPELINE_MODE_PRESETS,
    get_default_deployment_parameters,
    get_default_platform_constraints,
    apply_preset,
)
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.registry import REGISTRY, Category
from mimarsinan.config_schema.registry.groups import CONCERN_GROUPS

RUNTIME_KEYS: Set[str] = {
    k for k, e in REGISTRY.items() if e.category is Category.RUNTIME
}

DERIVED_KEYS: Set[str] = {
    k for k, e in REGISTRY.items() if e.category is Category.DERIVED
}

TOP_LEVEL_RUN_KEYS: Tuple[str, ...] = tuple(
    k for k, e in REGISTRY.items()
    if e.section == "top" and e.category in (Category.BASIC, Category.ADVANCED)
)

# Display sections are the concern groups plus a fallback for unregistered
# keys (e.g. recipe-derived internals surfaced in a resolved runtime config).
CONFIG_DISPLAY_GROUPS: Tuple[Dict[str, str], ...] = CONCERN_GROUPS + (
    {"id": "other", "title": "Other",
     "subtitle": "Additional configuration keys", "accent": "107,114,128"},
)

_RECIPE_FIELD_LABELS: Dict[str, str] = {
    "optimizer": "Optimizer",
    "weight_decay": "Weight decay",
    "betas": "Betas",
    "scheduler": "Scheduler",
    "warmup_ratio": "Warmup ratio",
    "grad_clip_norm": "Grad clip norm",
    "layer_wise_lr_decay": "Layer-wise LR decay",
    "label_smoothing": "Label smoothing",
}


def _is_nested_config(config: Mapping[str, Any]) -> bool:
    return isinstance(config, Mapping) and "deployment_parameters" in config


def _snake_to_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        if value and all(isinstance(x, (int, float)) for x in value):
            return "shape"
        return "json"
    if isinstance(value, dict):
        return "json"
    return "str"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _values_equal(a: Any, b: Any) -> bool:
    return _json_safe(a) == _json_safe(b)


def _platform_for_flat_merge(pc: Any) -> Optional[Dict[str, object]]:
    if not isinstance(pc, dict):
        return None
    if pc.get("mode") == "user":
        user = pc.get("user")
        return dict(user) if isinstance(user, dict) else None
    if pc.get("mode") == "auto":
        auto = pc.get("auto")
        if isinstance(auto, dict):
            fixed = auto.get("fixed")
            return dict(fixed) if isinstance(fixed, dict) else None
        return None
    if "mode" not in pc:
        return {k: v for k, v in pc.items() if k != "search_space"}
    return None


def _collect_explicit_keys(saved: Mapping[str, Any]) -> Set[str]:
    explicit: Set[str] = set()
    for key in saved:
        if key.startswith("_") or key in ("deployment_parameters", "platform_constraints"):
            continue
        explicit.add(key)
    dp = saved.get("deployment_parameters")
    if isinstance(dp, Mapping):
        explicit.update(dp.keys())
    pc = saved.get("platform_constraints")
    if isinstance(pc, Mapping):
        mode = pc.get("mode")
        if mode == "user":
            user = pc.get("user")
            if isinstance(user, Mapping):
                explicit.update(user.keys())
        elif mode == "auto":
            auto = pc.get("auto")
            if isinstance(auto, Mapping):
                fixed = auto.get("fixed")
                if isinstance(fixed, Mapping):
                    explicit.update(fixed.keys())
                search_space = auto.get("search_space")
                if isinstance(search_space, Mapping):
                    explicit.add("arch_search")
        elif mode is None:
            explicit.update(k for k in pc if k not in ("mode", "search_space"))
    return explicit


def _collect_preset_keys(saved: Mapping[str, Any]) -> Set[str]:
    pipeline_mode = str(saved.get("pipeline_mode", "phased"))
    preset = PIPELINE_MODE_PRESETS.get(pipeline_mode, {})
    explicit = _collect_explicit_keys(saved)
    return {k for k in preset if k not in explicit}


def _collect_derived_keys(saved: Mapping[str, Any]) -> Set[str]:
    dp_raw = dict(saved.get("deployment_parameters") or {})
    pipeline_mode = str(saved.get("pipeline_mode", "phased"))
    before = dict(get_default_deployment_parameters())
    for key, value in dp_raw.items():
        if key not in DERIVED_KEYS:
            before[key] = value
    apply_preset(pipeline_mode, before)
    after = copy.deepcopy(before)
    derive_deployment_parameters(after)
    derived: Set[str] = set()
    for key in DERIVED_KEYS:
        if key in dp_raw:
            continue
        if before.get(key) != after.get(key):
            derived.add(key)
    return derived


def _default_for_key(key: str, filled_deployment: Mapping[str, Any]) -> Any:
    if key in get_default_deployment_parameters():
        return get_default_deployment_parameters()[key]
    if key in get_default_platform_constraints():
        return get_default_platform_constraints()[key]
    if key == "pipeline_mode":
        return filled_deployment.get("pipeline_mode", "phased")
    return None


def _resolve_field_source(
    key: str,
    *,
    explicit_keys: Set[str],
    preset_keys: Set[str],
    derived_keys: Set[str],
    runtime_keys: Set[str],
) -> str:
    if key in runtime_keys:
        return "runtime"
    if key in explicit_keys:
        return "explicit"
    if key in derived_keys:
        return "derived"
    if key in preset_keys:
        return "preset"
    return "default"


def _field_meta(key: str) -> Dict[str, Any]:
    """Label/group/type/effect/importance for one key, from the registry."""
    entry = REGISTRY.get(key)
    if entry is None:
        return {"label": _snake_to_label(key), "group": "other"}
    meta: Dict[str, Any] = {
        "label": entry.label,
        "group": entry.group,
        "type": entry.type.value,
    }
    if entry.effect:
        meta["effect"] = entry.effect
    if entry.important:
        meta["important"] = True
    if entry.unit:
        meta["unit"] = entry.unit
    return meta

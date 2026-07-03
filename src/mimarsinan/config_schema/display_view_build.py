"""Section and nested block builders for config display views."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.config_schema.display_view_meta import (
    CONFIG_DISPLAY_GROUPS,
    FIELD_DISPLAY_META,
    _RECIPE_FIELD_LABELS,
    _field_meta,
    _infer_type,
    _json_safe,
    _snake_to_label,
)
from mimarsinan.config_schema.defaults import (
    get_default_training_recipe,
    get_default_tuning_recipe,
)

def build_recipe_fields(recipe: Mapping[str, Any], defaults: Mapping[str, Any]) -> List[Dict[str, Any]]:
    keys = list(defaults.keys())
    for sub_key in recipe:
        if sub_key not in keys:
            keys.append(sub_key)
    fields: List[Dict[str, Any]] = []
    for sub_key in keys:
        value = recipe.get(sub_key, defaults.get(sub_key))
        fields.append({
            "key": sub_key,
            "label": _RECIPE_FIELD_LABELS.get(sub_key, _snake_to_label(sub_key)),
            "type": _infer_type(value),
            "value": _json_safe(value),
            "default": _json_safe(defaults.get(sub_key)),
            "source": "explicit" if sub_key in recipe else "default",
        })
    return fields


def build_model_config_fields(model_type: Optional[str], model_config: Any) -> List[Dict[str, Any]]:
    if not isinstance(model_config, Mapping):
        return []
    schema_by_key: Dict[str, Dict[str, Any]] = {}
    if model_type:
        try:
            from mimarsinan.pipelining.core.registry.model_registry import get_model_config_schema
            for entry in get_model_config_schema(model_type):
                schema_by_key[entry["key"]] = entry
        except Exception:
            pass
    fields: List[Dict[str, Any]] = []
    for key, value in model_config.items():
        schema = schema_by_key.get(key, {})
        default = schema.get("default")
        fields.append({
            "key": key,
            "label": schema.get("label", _snake_to_label(key)),
            "type": schema.get("type", _infer_type(value)),
            "value": _json_safe(value),
            "default": _json_safe(default),
            "source": "explicit",
        })
    return fields


def build_arch_search_block(arch_search: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(arch_search, Mapping) or not arch_search:
        return None
    try:
        from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
        nas = get_wizard_nas_schema()
    except Exception:
        nas = {}
    common = nas.get("common_fields", {})
    fields: List[Dict[str, Any]] = []
    for key, value in arch_search.items():
        if key in ("agent_evolve", "compilagent"):
            continue
        meta = common.get(key, {})
        fields.append({
            "key": key,
            "label": meta.get("doc", _snake_to_label(key)),
            "type": meta.get("type", _infer_type(value)),
            "value": _json_safe(value),
            "default": _json_safe(meta.get("default")),
            "source": "explicit",
        })
    return {"type": "arch_search", "fields": fields, "raw": _json_safe(arch_search)}


def build_nested_blocks(flat: Mapping[str, Any], model_type: Optional[str]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    cores = flat.get("cores")
    if isinstance(cores, list):
        nested["cores"] = {
            "type": "cores",
            "items": [_json_safe(item) for item in cores if isinstance(item, Mapping)],
        }
    training_recipe = flat.get("training_recipe")
    if isinstance(training_recipe, Mapping):
        nested["training_recipe"] = {
            "type": "recipe",
            "fields": build_recipe_fields(training_recipe, get_default_training_recipe()),
        }
    tuning_recipe = flat.get("tuning_recipe")
    if isinstance(tuning_recipe, Mapping):
        nested["tuning_recipe"] = {
            "type": "recipe",
            "fields": build_recipe_fields(tuning_recipe, get_default_tuning_recipe()),
        }
    model_config = flat.get("model_config")
    if isinstance(model_config, Mapping) and model_config:
        nested["model_config"] = {
            "type": "model_config",
            "fields": build_model_config_fields(model_type, model_config),
        }
    arch_search = flat.get("arch_search")
    arch_block = build_arch_search_block(arch_search)
    if arch_block:
        nested["arch_search"] = arch_block
    pc = flat.get("platform_constraints")
    if isinstance(pc, Mapping) and pc.get("mode"):
        nested["platform_constraints"] = {
            "type": "platform_mode",
            "mode": pc.get("mode"),
            "raw": _json_safe(pc),
        }
    return nested


def build_pipeline_preview(flat: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
            get_pipeline_semantic_group_by_step_name,
        )
        specs = get_pipeline_step_specs(dict(flat))
        steps = [name for name, _ in specs]
        groups_map = get_pipeline_semantic_group_by_step_name(dict(flat))
        semantic_groups = [groups_map.get(name, "other") for name in steps]
        return {"steps": steps, "semantic_groups": semantic_groups}
    except Exception:
        return {"steps": [], "semantic_groups": []}


def _ordered_keys(flat: Mapping[str, Any]) -> List[str]:
    skip_nested_expanded = {"training_recipe", "tuning_recipe", "model_config", "arch_search"}
    keys = [k for k in flat if k not in skip_nested_expanded]
    group_order = [g["id"] for g in CONFIG_DISPLAY_GROUPS]
    meta_order = list(FIELD_DISPLAY_META.keys())

    def sort_key(key: str) -> Tuple[int, int, str]:
        meta = _field_meta(key)
        group_idx = group_order.index(meta["group"]) if meta["group"] in group_order else len(group_order)
        meta_idx = meta_order.index(key) if key in meta_order else len(meta_order) + 1
        return (group_idx, meta_idx, key)

    return sorted(keys, key=sort_key)



"""Build structured monitor display payloads from deployment / runtime config."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from mimarsinan.config_schema.defaults import (
    PIPELINE_MODE_PRESETS,
    get_default_deployment_parameters,
    get_default_platform_constraints,
    get_default_tuning_recipe,
    get_default_training_recipe,
    apply_preset,
)
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.runtime import build_flat_pipeline_config

RUNTIME_KEYS: Set[str] = {
    "device",
    "input_shape",
    "input_size",
    "num_classes",
}

DERIVED_KEYS: Set[str] = {
    "activation_quantization",
    "weight_quantization",
    "pipeline_mode",
}

TOP_LEVEL_RUN_KEYS: Tuple[str, ...] = (
    "data_provider_name",
    "experiment_name",
    "generated_files_path",
    "seed",
    "start_step",
    "stop_step",
    "target_metric_override",
    "datasets_path",
)

CONFIG_DISPLAY_GROUPS: Tuple[Dict[str, str], ...] = (
    {"id": "run", "title": "Run Identity", "subtitle": "Data source and experiment metadata", "accent": "34,211,238"},
    {"id": "pipeline", "title": "Pipeline & Spiking", "subtitle": "Mode, quantization, and pruning gates", "accent": "168,85,247"},
    {"id": "model", "title": "Model Architecture", "subtitle": "Network type and hyperparameters", "accent": "91,141,245"},
    {"id": "training", "title": "Training", "subtitle": "Learning rates, epochs, and recipes", "accent": "74,222,128"},
    {"id": "hardware", "title": "Hardware", "subtitle": "Cores, precision, and layout constraints", "accent": "249,115,22"},
    {"id": "tuning", "title": "Adaptation & Tuning", "subtitle": "Tuner budget and degradation tolerance", "accent": "251,191,36"},
    {"id": "simulation", "title": "Simulation", "subtitle": "Chip and cycle-accurate backends", "accent": "103,232,249"},
    {"id": "search", "title": "Architecture Search", "subtitle": "NAS optimizer and search space", "accent": "139,92,246"},
    {"id": "runtime", "title": "Runtime Resolved", "subtitle": "Values resolved when the pipeline starts", "accent": "107,114,128"},
    {"id": "other", "title": "Other", "subtitle": "Additional configuration keys", "accent": "107,114,128"},
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

FIELD_DISPLAY_META: Dict[str, Dict[str, Any]] = {
    "data_provider_name": {"group": "run", "type": "str", "label": "Data Provider"},
    "experiment_name": {"group": "run", "type": "str", "label": "Experiment Name", "important": True},
    "generated_files_path": {"group": "run", "type": "path", "label": "Generated Files Path"},
    "seed": {"group": "run", "type": "int", "label": "Random Seed"},
    "start_step": {"group": "run", "type": "str", "label": "Start Step"},
    "stop_step": {"group": "run", "type": "str", "label": "Stop Step"},
    "target_metric_override": {"group": "run", "type": "float", "label": "Target Metric Override"},
    "datasets_path": {"group": "run", "type": "path", "label": "Datasets Path"},
    "pipeline_mode": {"group": "pipeline", "type": "enum", "label": "Pipeline Mode", "important": True,
                      "effect": "Phased enables weight/activation quantization steps"},
    "spiking_mode": {"group": "pipeline", "type": "enum", "label": "Spiking Mode", "important": True,
                     "effect": "Selects LIF/TTFS path and simulation backends"},
    "firing_mode": {"group": "pipeline", "type": "enum", "label": "Firing Mode"},
    "spike_generation_mode": {"group": "pipeline", "type": "enum", "label": "Spike Generation Mode"},
    "thresholding_mode": {"group": "pipeline", "type": "enum", "label": "Thresholding Mode"},
    "activation_quantization": {"group": "pipeline", "type": "bool", "label": "Activation Quantization",
                                "effect": "Gates activation quantization pipeline steps"},
    "weight_quantization": {"group": "pipeline", "type": "bool", "label": "Weight Quantization",
                            "effect": "Gates weight quantization pipeline steps"},
    "pruning": {"group": "pipeline", "type": "bool", "label": "Pruning Enabled"},
    "pruning_fraction": {"group": "pipeline", "type": "float", "label": "Pruning Fraction"},
    "cycle_accurate_lif_forward": {"group": "training", "type": "bool", "label": "Cycle-accurate LIF Forward",
                                   "effect": "Spike-train forward during LIF adaptation training"},
    "enable_training_noise": {"group": "training", "type": "bool", "label": "Training Noise"},
    "model_config_mode": {"group": "model", "type": "enum", "label": "Model Config Mode"},
    "hw_config_mode": {"group": "model", "type": "enum", "label": "Hardware Config Mode"},
    "model_type": {"group": "model", "type": "str", "label": "Model Type", "important": True},
    "model_config": {"group": "model", "type": "json", "label": "Model Config"},
    "model_factory": {"group": "model", "type": "str", "label": "Model Factory"},
    "weight_source": {"group": "training", "type": "str", "label": "Weight Source"},
    "lr": {"group": "training", "type": "float", "label": "Learning Rate"},
    "lr_range_min": {"group": "training", "type": "float", "label": "LR Range Min"},
    "lr_range_max": {"group": "training", "type": "float", "label": "LR Range Max"},
    "training_epochs": {"group": "training", "type": "int", "label": "Training Epochs"},
    "finetune_epochs": {"group": "training", "type": "int", "label": "Fine-tune Epochs"},
    "finetune_lr": {"group": "training", "type": "float", "label": "Fine-tune LR"},
    "batch_size": {"group": "training", "type": "int", "label": "Batch Size"},
    "preprocessing": {"group": "training", "type": "json", "label": "Preprocessing"},
    "training_recipe": {"group": "training", "type": "recipe", "label": "Training Recipe"},
    "tuning_recipe": {"group": "tuning", "type": "recipe", "label": "Tuning Recipe"},
    "tuning_budget_scale": {"group": "tuning", "type": "float", "label": "Tuning Budget Scale"},
    "tuner_target_floor_ratio": {"group": "tuning", "type": "float", "label": "Tuner Target Floor Ratio"},
    "degradation_tolerance": {"group": "tuning", "type": "float", "label": "Degradation Tolerance"},
    "cores": {"group": "hardware", "type": "cores", "label": "Core Types"},
    "target_tq": {"group": "hardware", "type": "int", "label": "Target TQ",
                  "effect": "Activation quantization threshold groups"},
    "simulation_steps": {"group": "hardware", "type": "int", "label": "Simulation Steps"},
    "weight_bits": {"group": "hardware", "type": "int", "label": "Weight Bits"},
    "allow_coalescing": {"group": "hardware", "type": "bool", "label": "Allow Coalescing"},
    "allow_neuron_splitting": {"group": "hardware", "type": "bool", "label": "Allow Neuron Splitting"},
    "allow_scheduling": {"group": "hardware", "type": "bool", "label": "Allow Scheduling",
                         "effect": "Multi-pass layout scheduling when single-pass packing fails"},
    "max_schedule_passes": {"group": "hardware", "type": "int", "label": "Max Schedule Passes"},
    "scheduling_latency_weight": {"group": "hardware", "type": "float", "label": "Scheduling Latency Weight"},
    "arch_search": {"group": "search", "type": "json", "label": "Architecture Search"},
    "enable_nevresim_simulation": {"group": "simulation", "type": "bool", "label": "Nevresim Simulation"},
    "enable_loihi_simulation": {"group": "simulation", "type": "bool", "label": "Loihi Simulation"},
    "enable_sanafe_simulation": {"group": "simulation", "type": "bool", "label": "SANA-FE Simulation"},
    "loihi_parity_sample_index": {"group": "simulation", "type": "int", "label": "Loihi Parity Sample Index"},
    "sanafe_sample_count": {"group": "simulation", "type": "int", "label": "SANA-FE Sample Count"},
    "sanafe_arch_preset": {"group": "simulation", "type": "str", "label": "SANA-FE Arch Preset"},
    "sanafe_custom_arch_path": {"group": "simulation", "type": "path", "label": "SANA-FE Custom Arch Path"},
    "sanafe_parity_check": {"group": "simulation", "type": "bool", "label": "SANA-FE Parity Check"},
    "sanafe_log_potential_trace": {"group": "simulation", "type": "bool", "label": "SANA-FE Log Potential Trace"},
    "sanafe_log_message_trace": {"group": "simulation", "type": "bool", "label": "SANA-FE Log Message Trace"},
    "max_simulation_samples": {"group": "simulation", "type": "int", "label": "Max Simulation Samples"},
    "simulation_batch_count": {"group": "simulation", "type": "int", "label": "Simulation Batch Count"},
    "activation_analysis_batch_size": {"group": "training", "type": "int", "label": "Activation Analysis Batch Size"},
    "generate_visualizations": {"group": "run", "type": "bool", "label": "Generate Visualizations"},
    "store_pre_pruning_heatmap": {"group": "pipeline", "type": "bool", "label": "Store Pre-pruning Heatmap"},
    "device": {"group": "runtime", "type": "str", "label": "Device"},
    "input_shape": {"group": "runtime", "type": "shape", "label": "Input Shape"},
    "input_size": {"group": "runtime", "type": "int", "label": "Input Size"},
    "num_classes": {"group": "runtime", "type": "int", "label": "Num Classes"},
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
    meta = dict(FIELD_DISPLAY_META.get(key, {}))
    meta.setdefault("label", _snake_to_label(key))
    meta.setdefault("group", "other")
    return meta


def _build_recipe_fields(recipe: Mapping[str, Any], defaults: Mapping[str, Any]) -> List[Dict[str, Any]]:
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


def _build_model_config_fields(model_type: Optional[str], model_config: Any) -> List[Dict[str, Any]]:
    if not isinstance(model_config, Mapping):
        return []
    schema_by_key: Dict[str, Dict[str, Any]] = {}
    if model_type:
        try:
            from mimarsinan.pipelining.model_registry import get_model_config_schema
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


def _build_arch_search_block(arch_search: Any) -> Optional[Dict[str, Any]]:
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


def _build_nested_blocks(flat: Mapping[str, Any], model_type: Optional[str]) -> Dict[str, Any]:
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
            "fields": _build_recipe_fields(training_recipe, get_default_training_recipe()),
        }
    tuning_recipe = flat.get("tuning_recipe")
    if isinstance(tuning_recipe, Mapping):
        nested["tuning_recipe"] = {
            "type": "recipe",
            "fields": _build_recipe_fields(tuning_recipe, get_default_tuning_recipe()),
        }
    model_config = flat.get("model_config")
    if isinstance(model_config, Mapping) and model_config:
        nested["model_config"] = {
            "type": "model_config",
            "fields": _build_model_config_fields(model_type, model_config),
        }
    arch_search = flat.get("arch_search")
    arch_block = _build_arch_search_block(arch_search)
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


def _build_pipeline_preview(flat: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from mimarsinan.pipelining.pipelines.deployment_pipeline import (
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


def build_config_display_view(
    raw_config: Mapping[str, Any],
    *,
    saved_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured, JSON-serializable config view for the monitor UI."""
    raw_config = raw_config or {}
    saved = saved_config if saved_config is not None else (
        raw_config if _is_nested_config(raw_config) else None
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
                if k not in get_default_platform_constraints() and k not in TOP_LEVEL_RUN_KEYS
                and k not in RUNTIME_KEYS
            },
            "platform_constraints": {
                k: v for k, v in raw_config.items() if k in get_default_platform_constraints()
            },
        }

    pipeline_mode = str(filled_outer.get("pipeline_mode", "phased"))
    dp = dict(filled_outer.get("deployment_parameters") or {})
    pc_merge = _platform_for_flat_merge(filled_outer.get("platform_constraints"))
    flat = build_flat_pipeline_config(dp, pc_merge, pipeline_mode=pipeline_mode)

    for key in TOP_LEVEL_RUN_KEYS:
        if key in filled_outer:
            flat[key] = filled_outer[key]

    if not _is_nested_config(raw_config):
        for key, value in raw_config.items():
            flat[key] = value

    explicit_keys = _collect_explicit_keys(saved) if saved is not None else set()
    preset_keys = _collect_preset_keys(saved) if saved is not None else set()
    derived_keys = _collect_derived_keys(saved) if saved is not None else set()

    runtime_keys: Set[str] = set(RUNTIME_KEYS)
    if not _is_nested_config(raw_config):
        for key in raw_config:
            if key not in explicit_keys and key in flat and key not in get_default_deployment_parameters():
                if key not in get_default_platform_constraints() and key not in TOP_LEVEL_RUN_KEYS:
                    runtime_keys.add(key)

    nested = _build_nested_blocks(flat, flat.get("model_type"))
    skip_in_sections = {"training_recipe", "tuning_recipe", "model_config", "arch_search", "platform_constraints"}

    grouped: Dict[str, List[Dict[str, Any]]] = {g["id"]: [] for g in CONFIG_DISPLAY_GROUPS}
    for key in _ordered_keys(flat):
        if key in skip_in_sections:
            continue
        meta = _field_meta(key)
        value = flat.get(key)
        default = _default_for_key(key, filled_outer.get("deployment_parameters", {}))
        field_type = meta.get("type") or _infer_type(value)
        if field_type == "cores" and "cores" in nested:
            field_type = "cores_ref"
        source = _resolve_field_source(
            key,
            explicit_keys=explicit_keys,
            preset_keys=preset_keys,
            derived_keys=derived_keys,
            runtime_keys=runtime_keys,
        )
        if source == "default" and not _values_equal(value, default) and default is not None:
            source = "explicit" if key in explicit_keys else source
        grouped[meta["group"]].append({
            "key": key,
            "label": meta.get("label", _snake_to_label(key)),
            "type": field_type,
            "value": _json_safe(value),
            "default": _json_safe(default),
            "source": source,
            "effect": meta.get("effect"),
            "important": bool(meta.get("important")),
        })

    sections: List[Dict[str, Any]] = []
    for group in CONFIG_DISPLAY_GROUPS:
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
        "summary": _json_safe(summary),
        "pipeline_preview": _build_pipeline_preview(flat),
        "sections": sections,
        "nested": nested,
        "raw_resolved": _json_safe(dict(flat)),
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

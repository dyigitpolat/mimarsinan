"""Display view metadata and key-resolution helpers."""

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
    "spike_encoding_seed": {"group": "pipeline", "type": "int", "label": "Spike Encoding Seed"},
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



"""
Default deployment parameters, platform constraints, and pipeline-mode presets.

These mirror DeploymentPipeline.default_deployment_parameters and
default_platform_constraints so that the wizard and any config tooling use
the same values. DeploymentPipeline should import from here to avoid drift.

CONFIG_KEYS_SET is the set of all keys read by pipeline steps, tuners,
builders, and SimulationRunner (see plan "Config keys by consumer").
"""

from __future__ import annotations

from typing import Dict, Set

# ── Defaults (must stay in sync with pipeline behavior) ─────────────────────

# Default training / tuning recipes. These mirror the recipe block used in
# ``templates/cifar_vit_pretrained.json`` so new configs pick up the same
# AdamW + cosine + warmup + LLRD + label-smoothing setup as the reference ViT
# template out of the box.
DEFAULT_TRAINING_RECIPE: Dict[str, object] = {
    "optimizer": "adamw",
    "weight_decay": 0.05,
    "betas": [0.9, 0.999],
    "scheduler": "cosine",
    "warmup_ratio": 0.1,
    "grad_clip_norm": 1.0,
    "layer_wise_lr_decay": 0.75,
    "label_smoothing": 0.1,
}

DEFAULT_TUNING_RECIPE: Dict[str, object] = {
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "scheduler": "cosine",
    "warmup_ratio": 0.0,
    "grad_clip_norm": 1.0,
    "layer_wise_lr_decay": 1.0,
    "label_smoothing": 0.0,
}

DEFAULT_DEPLOYMENT_PARAMETERS: Dict[str, object] = {
    "lr": 0.001,
    "lr_range_min": 1e-5,
    "lr_range_max": 1e-1,
    "training_epochs": 10,
    "tuning_budget_scale": 1.0,
    "tuner_calibrate_smooth_tolerance": True,
    "tuner_smooth_tolerance_residual_threshold": 0.02,
    "tuner_smooth_tolerance_min": 0.01,
    "tuner_smooth_tolerance_max": 0.15,
    "tuner_smooth_tolerance_baseline_epsilon": 1e-9,
    "tuner_smooth_tolerance_lr_scale": 1.0,
    "tuner_target_floor_ratio": 0.90,
    "degradation_tolerance": 0.05,
    "model_config_mode": "user",
    "hw_config_mode": "fixed",
    "spiking_mode": "rate",
    "allow_scheduling": False,
    "training_recipe": dict(DEFAULT_TRAINING_RECIPE),
    "tuning_recipe": dict(DEFAULT_TUNING_RECIPE),
}

DEFAULT_PLATFORM_CONSTRAINTS: Dict[str, object] = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}],
    "target_tq": 32,
    "simulation_steps": 32,
    "weight_bits": 8,
    "allow_coalescing": False,
    "allow_neuron_splitting": False,
    "max_schedule_passes": 8,
    "scheduling_latency_weight": 1.0,
}

PIPELINE_MODE_PRESETS: Dict[str, Dict[str, object]] = {
    "vanilla": {},
    "phased": {
        "activation_quantization": True,
        "weight_quantization": True,
    },
}

# Keys read by deployment_pipeline, steps, tuners, builders, SimulationRunner
CONFIG_KEYS_SET: Set[str] = {
    "degradation_tolerance",
    "spiking_mode",
    "firing_mode",
    "spike_generation_mode",
    "thresholding_mode",
    "model_config_mode",
    "hw_config_mode",
    "activation_quantization",
    "weight_quantization",
    "pruning",
    "pruning_fraction",
    "weight_source",
    "model_type",
    "device",
    "input_shape",
    "input_size",
    "num_classes",
    "model_config",
    "model_factory",
    "lr",
    "lr_range_min",
    "lr_range_max",
    "cores",
    "simulation_steps",
    "arch_search",
    "target_tq",
    "allow_coalescing",
    "allow_neuron_splitting",
    "allow_scheduling",
    "max_schedule_passes",
    "scheduling_latency_weight",
    "weight_bits",
    "tuning_budget_scale",
    "tuner_calibrate_smooth_tolerance",
    "tuner_smooth_tolerance_residual_threshold",
    "tuner_smooth_tolerance_min",
    "tuner_smooth_tolerance_max",
    "tuner_smooth_tolerance_baseline_epsilon",
    "tuner_smooth_tolerance_delta_schedule",
    "tuner_smooth_tolerance_lr",
    "tuner_smooth_tolerance_lr_scale",
    "tuner_target_floor_ratio",
    "finetune_epochs",
    "finetune_lr",
    "batch_size",
    "preprocessing",
    "training_recipe",
    "tuning_recipe",
    "activation_analysis_batch_size",
    "generate_visualizations",
    "max_simulation_samples",
    "seed",
}


def get_default_deployment_parameters() -> Dict[str, object]:
    """Return a copy of default deployment parameters."""
    out = dict(DEFAULT_DEPLOYMENT_PARAMETERS)
    # Deep-copy mutable recipe blocks so callers can mutate freely.
    if "training_recipe" in out:
        out["training_recipe"] = dict(out["training_recipe"])
    if "tuning_recipe" in out:
        out["tuning_recipe"] = dict(out["tuning_recipe"])
    return out


def get_default_training_recipe() -> Dict[str, object]:
    """Return a copy of the default training recipe (AdamW + cosine + LLRD)."""
    return dict(DEFAULT_TRAINING_RECIPE)


def get_default_tuning_recipe() -> Dict[str, object]:
    """Return a copy of the default tuning recipe (AdamW, no LLRD, no warmup)."""
    return dict(DEFAULT_TUNING_RECIPE)


def get_default_platform_constraints() -> Dict[str, object]:
    """Return a copy of default platform constraints."""
    return dict(DEFAULT_PLATFORM_CONSTRAINTS)


def get_pipeline_mode_presets() -> Dict[str, Dict[str, object]]:
    """Return a copy of pipeline-mode presets."""
    return {k: dict(v) for k, v in PIPELINE_MODE_PRESETS.items()}


def get_config_keys_set() -> Set[str]:
    """Return the set of config keys read by the pipeline (read-only)."""
    return CONFIG_KEYS_SET


def apply_preset(pipeline_mode: str, deployment_parameters: Dict[str, object]) -> None:
    """
    Merge pipeline_mode preset into deployment_parameters with setdefault.

    Explicit user values in deployment_parameters are preserved.
    """
    preset = PIPELINE_MODE_PRESETS.get(pipeline_mode, {})
    for key, value in preset.items():
        deployment_parameters.setdefault(key, value)

"""Default deployment parameters, platform constraints, and pipeline-mode presets."""

from __future__ import annotations

from typing import Dict, Set

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
    "tuning_budget_scale_ramp_steps": False,
    "tuner_target_floor_ratio": 0.90,
    "activation_scale_quantile": 0.99,
    "kd_ce_alpha": 0.3,
    "kd_temperature": 3.0,
    "degradation_tolerance": 0.05,
    "paired_confirm_batches": 0,  # 0 -> use eval_n_batches
    "s_allocation": "uniform",
    "s_allocation_explicit": None,
    "s_allocation_budget": None,
    "ttfs_genuine_blend_ce_alpha": 0.3,
    "model_config_mode": "user",
    "hw_config_mode": "fixed",
    # The enable_*_simulation flags carry NO defaults: the ConversionPolicy
    # recipe derives them per mode and overwrites any value (Pure SSOT).
    # cycle_accurate_lif_forward carries NO default: it is recipe-owned (the LIF
    # recipe folds it ON; the runtime derivation writes it) — never a knob.
    "spiking_mode": "lif",
    # Negative-boundary policy: ON = calibrated shift; OFF = subsume-forward.
    "negative_value_shift": True,
    "allow_scheduling": False,
    "nevresim_connectivity_mode": "runtime",
    "enable_training_noise": False,
    "ttfs_cycle_schedule": "cascaded",
    "sanafe_sample_count": 1,
    "sanafe_arch_preset": "loihi",
    "sanafe_custom_arch_path": None,
    "sanafe_log_potential_trace": False,
    "spike_encoding_seed": None,
    "training_recipe": dict(DEFAULT_TRAINING_RECIPE),
    "mirror_training_recipe": False,
    "tuning_recipe": dict(DEFAULT_TUNING_RECIPE),
}

DEFAULT_PLATFORM_CONSTRAINTS: Dict[str, object] = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}],
    "target_tq": 32,
    "simulation_steps": 32,
    "weight_bits": 8,
    "allow_coalescing": False,
    "allow_neuron_splitting": False,
    "allow_per_layer_s": False,
    "max_schedule_passes": 8,
    "scheduling_latency_weight": 1.0,
}

# Presets must not inject AQ/WQ: derivation owns them, and a preset-injected value
# would be indistinguishable from an explicit one under the quantization contract.
PIPELINE_MODE_PRESETS: Dict[str, Dict[str, object]] = {
    "vanilla": {},
    "phased": {},
}

CONFIG_KEYS_SET: Set[str] = {
    "degradation_tolerance",
    "spiking_mode",
    "firing_mode",
    "spike_generation_mode",
    "encoding_layer_placement",
    "negative_value_shift",
    "thresholding_mode",
    "spike_encoding_seed",
    "cycle_accurate_lif_forward",
    "model_config_mode",
    "hw_config_mode",
    "activation_scale_quantile",
    "kd_ce_alpha",
    "kd_temperature",
    "activation_quantization",
    "weight_quantization",
    "pruning",
    "pruning_fraction",
    "prune_sparsity",
    "weight_source",
    "preload_weights",
    "pretrained_weight_set", "pretrained_weight_sets",
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
    "allow_per_layer_s",
    "allow_scheduling",
    "max_schedule_passes",
    "scheduling_latency_weight",
    "weight_bits",
    "tuning_budget_scale",
    "tuning_budget_scale_ramp_steps",
    "tuner_target_floor_ratio",
    "paired_confirm_batches",
    "s_allocation",
    "s_allocation_explicit",
    "s_allocation_budget",
    "ttfs_genuine_blend_ce_alpha",
    "finetune_epochs",
    "finetune_lr",
    "batch_size",
    "tuning_batch_size",
    "preprocessing",
    "training_recipe",
    "tuning_recipe",
    "activation_analysis_batch_size",
    "generate_visualizations",
    "max_simulation_samples",
    "deployment_metric_full_eval",
    "seed",
    "enable_nevresim_simulation",
    "nevresim_connectivity_mode",
    "enable_loihi_simulation",
    "enable_sanafe_simulation",
    "enable_training_noise",
    "ttfs_cycle_schedule",
    "ttfs_finetune_kd_against_rung2",
    "scm_degradation_tolerance",
    "nf_scm_parity_samples",
    "nf_scm_parity_atol", "nf_scm_parity_max_mismatch_fraction",
    "nf_scm_parity_min_agreement",
    "scm_torch_sim_parity_check", "scm_torch_sim_parity_samples",
    "scm_torch_sim_parity_min_agreement",
    "onchip_majority_gate",
    "onchip_min_fraction", "onchip_majority_fraction",
    "pretrain_floor_chance_multiple",
    "capacity_gate",
    "loihi_parity_sample_index",
    "sanafe_sample_count", "sanafe_arch_preset",
    "sanafe_custom_arch_path", "sanafe_log_potential_trace",
    "simulation_batch_count", "simulation_step_timeout_s",
    # Optional per-cell RUN-total STEP budget for the 5u endpoint target floor
    # (the endpoint_steps ledger); read by endpoint_recovery via config.get
    # with the TUNING_POLICY value as fallback. Steps, never wall seconds.
    "endpoint_floor_steps",
    # [MBH-DRAWS] best-of-N conversion draws on the variance-carrying stages
    # (default 1 = single-draw, bit-identical); draws seed torch at seed+k.
    "conversion_draws",
    # WQ knobs, both ConversionPolicy-recipe-defaulted: the endpoint step cap
    # (FAST respec) and the [M2] two-scale projection grids (ttfs_cycle_based).
    "wq_endpoint_recovery_steps", "wq_two_scale_projection",
    # Every-endpoint D-hat target floor (bit-parity-lossless family); read by
    # endpoint_recovery via config.get, the ConversionPolicy recipe may set it.
    "endpoint_target_floor",
    # torch DataLoader worker count; read via config.get with a fallback of 4.
    "num_workers",
    # Workload-profile-injectable keys (common/workload_profile.py): absence
    # is meaningful, so they never get schema defaults here.
    "input_data_scale", "eval_subsample_target", "tuning_step_cap_epochs",
    "calibration_set_policy", "prefix_stage_lr", "endpoint_floor_lr",
    "proven_recovery_depth", "clamp_cuda_assert_prone",
}


def _copy_recipe_blocks(out: Dict[str, object]) -> Dict[str, object]:
    """Shallow-copy nested recipe dicts so callers can mutate them safely."""
    for key in ("training_recipe", "tuning_recipe"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def get_default_deployment_parameters() -> Dict[str, object]:
    """Return a copy of default deployment parameters."""
    return _copy_recipe_blocks(dict(DEFAULT_DEPLOYMENT_PARAMETERS))


def get_default_training_recipe() -> Dict[str, object]:
    """Return a copy of the default training recipe (AdamW + cosine + LLRD)."""
    return dict(DEFAULT_TRAINING_RECIPE)


def get_default_tuning_recipe() -> Dict[str, object]:
    """Return a copy of the default tuning recipe (AdamW, no LLRD, no warmup)."""
    return dict(DEFAULT_TUNING_RECIPE)


def get_default_platform_constraints() -> Dict[str, object]:
    """Return a copy of default platform constraints."""
    return dict(DEFAULT_PLATFORM_CONSTRAINTS)


def get_user_default_deployment_parameters() -> Dict[str, object]:
    """Return user-facing deployment defaults (wizard/template starting points)."""
    from mimarsinan.config_schema.namespaced_schema import keys_with_exposure

    user_keys = set(keys_with_exposure("user")) | {
        "model_type",
        "model_config",
        "arch_search",
        "encoding_layer_placement",
        "negative_value_shift",
        "pruning",
        "pruning_fraction",
        "weight_source",
        "finetune_epochs",
        "finetune_lr",
        "batch_size",
        "preprocessing",
        "max_simulation_samples",
    }
    out = {k: v for k, v in DEFAULT_DEPLOYMENT_PARAMETERS.items() if k in user_keys}
    return _copy_recipe_blocks(out)


def get_system_default_deployment_parameters() -> Dict[str, object]:
    """Return internal/system deployment defaults hidden from saved user config."""
    user_keys = set(get_user_default_deployment_parameters())
    out = {k: v for k, v in DEFAULT_DEPLOYMENT_PARAMETERS.items() if k not in user_keys}
    return _copy_recipe_blocks(out)


def get_user_default_platform_constraints() -> Dict[str, object]:
    """Return user-facing platform defaults."""
    from mimarsinan.config_schema.namespaced_schema import keys_with_exposure

    user_keys = set(keys_with_exposure("user")) | {
        "max_axons",
        "max_neurons",
        "has_bias",
        "search_space",
        "mode",
        "user",
        "auto",
        "fixed",
    }
    return {k: v for k, v in DEFAULT_PLATFORM_CONSTRAINTS.items() if k in user_keys}


def get_system_default_platform_constraints() -> Dict[str, object]:
    """Return internal/system platform defaults."""
    user_keys = set(get_user_default_platform_constraints())
    return {k: v for k, v in DEFAULT_PLATFORM_CONSTRAINTS.items() if k not in user_keys}


def get_pipeline_mode_presets() -> Dict[str, Dict[str, object]]:
    """Return a copy of pipeline-mode presets."""
    return {k: dict(v) for k, v in PIPELINE_MODE_PRESETS.items()}


# Every defaulted key is by definition a key the pipeline reads.
CONFIG_KEYS_SET |= set(DEFAULT_DEPLOYMENT_PARAMETERS)


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

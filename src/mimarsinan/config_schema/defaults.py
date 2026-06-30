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
    # Default off: historical max_training_steps cap remains 4000. When enabled,
    # large tuning_budget_scale values can also lengthen the gradual ramp budget.
    "tuning_budget_scale_ramp_steps": False,
    "tuner_target_floor_ratio": 0.90,
    # Activation-Analysis scale quantile = the per-perceptron LIF/TTFS decode scale
    # AND clamp ceiling. Raising it toward 1.0 reduces the systematic top-percentile
    # clip bias (trades a little rate-resolution); 0.99 = the historical default.
    "activation_scale_quantile": 0.99,
    # Conversion fine-tune objective weighting (LIF + TTFS share the KD-blend loss
    # = kd_ce_alpha*CE + (1-kd_ce_alpha)*KD-to-ANN). 0.3/3.0 = the historical
    # KD-heavy hardcode; raising kd_ce_alpha toward 1.0 re-weights toward hard-label
    # CE (the lever when KD-to-ANN under-fits the spiking student on harder datasets).
    "kd_ce_alpha": 0.3,
    "kd_temperature": 3.0,
    "degradation_tolerance": 0.05,
    # Per-cycle rollback snapshot scope/location (CheckpointGuard, graduated).
    # Defaults ("full"/"device") delegate verbatim to the on-device clone;
    # "tunable" (skip frozen backbone) / "cpu_pinned" (free model VRAM) scale out.
    "checkpoint_scope": "full",
    "checkpoint_location": "device",
    # Paired McNemar rollback gate (opt-in): reference vs candidate on a shared
    # fixed example subsample (a several-fold tighter SE than the marginal gate,
    # which is the default). +~0.5% deployed accuracy but +139% tuning wall — it
    # rolls back ~10x more, bisecting carefully into the quantization cliff.
    # ``global_budget`` is the §8.2 practical-significance floor: a paired drop is
    # rolled back only if it is BOTH statistically significant AND exceeds this
    # budget. 0.0 = no floor (pure significance gate) and is the default because a
    # 0.005 floor was MEASURED to erase the +0.5% gain (one seed fell to 0.9508 —
    # docs/tuning_optimization_flags.md §1), so 0.0 is strictly better here. A
    # positive value is valid (trades accuracy for anti-thrash); negatives are
    # rejected at tuner construction. (Supersedes the "0.5%" of commit 09eef0d.)
    "tuning_use_paired_sensor": False,
    "k_commit": 2.0,
    "paired_confirm_batches": 0,  # 0 → use eval_n_batches
    "global_budget": 0.0,
    # Per-layer-S temporal-allocation axis (EW1, RESERVED): each cascade depth /
    # latency group MAY get its own temporal resolution S_d instead of one global
    # simulation_steps. The Wizard DECLARES the intent here; the per-depth S map
    # derivation is deferred to research (the allocator is not on this branch).
    # "uniform" (default) => the SAME global S for every depth =>
    # byte-identical (no consumer threads a non-uniform map yet). "explicit" reads
    # s_allocation_explicit (a per-depth list, validated against the model depth).
    # "budget" reads s_allocation_budget and is a no-op that returns uniform + a
    # "derivation deferred" marker (the budget allocator is the research keystone).
    # Gated by the allow_per_layer_s chip capability (platform_constraints).
    "s_allocation": "uniform",
    # Reserved per-depth S list, used ONLY when s_allocation == "explicit". None =>
    # not declared; one positive int per cascade depth / latency group otherwise.
    "s_allocation_explicit": None,
    # Reserved budget body, used ONLY when s_allocation == "budget". None => not
    # declared; otherwise a dict with optional {max_energy_proxy, max_latency_steps,
    # target}. Parsed + validated now; its derivation into a map is deferred (research).
    "s_allocation_budget": None,
    # Weight of the extra CE on the PURE genuine logits in the blend-ramp loss
    # (the validated prototype value that pulls the rate-1 endpoint up). Canonical
    # source for the constant; the tuner reads this key once.
    "ttfs_genuine_blend_ce_alpha": 0.3,
    "model_config_mode": "user",
    "hw_config_mode": "fixed",
    "spiking_mode": "lif",
    "allow_scheduling": False,
    "enable_nevresim_simulation": True,
    "nevresim_connectivity_mode": "runtime",
    "enable_loihi_simulation": False,
    "enable_sanafe_simulation": False,
    "cycle_accurate_lif_forward": True,
    "enable_training_noise": False,
    "ttfs_cycle_schedule": "cascaded",
    "ttfs_spike_time_round": "round",
    "sanafe_sample_count": 1,
    "sanafe_arch_preset": "loihi",
    "sanafe_custom_arch_path": None,
    "sanafe_log_potential_trace": False,  # heavy Vm trace; opt-in UI knob
    "spike_encoding_seed": None,
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
    # EW1 RESERVED capability gate: the chip permits per-cascade-depth temporal
    # resolution S_d (per-layer-S). Declared alongside allow_coalescing; no mapping
    # decision consults it yet (the per-depth S map is derived by the ConversionPolicy
    # keystone, research). Default False => uniform global S only => byte-identical.
    "allow_per_layer_s": False,
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
    "checkpoint_scope",
    "checkpoint_location",
    "tuning_use_paired_sensor",
    "k_commit",
    "paired_confirm_batches",
    "global_budget",
    "s_allocation",
    "s_allocation_explicit",
    "s_allocation_budget",
    "ttfs_genuine_blend_ce_alpha",
    "finetune_epochs",
    "finetune_lr",
    "batch_size",
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
    "ttfs_spike_time_round",
    "ttfs_finetune_kd_against_rung2",
    "scm_degradation_tolerance",
    "nf_scm_parity_samples",
    "nf_scm_parity_samples_cascaded",
    "nf_scm_parity_atol",
    "nf_scm_parity_max_mismatch_fraction",
    "nf_scm_parity_min_agreement",
    "scm_torch_sim_parity_check",
    "scm_torch_sim_parity_samples",
    "scm_torch_sim_parity_min_agreement",
    "onchip_majority_gate",
    "onchip_majority_min_fraction",
    "onchip_min_fraction",
    "onchip_majority_fraction",
    "capacity_gate",
    "loihi_parity_sample_index",
    "sanafe_sample_count",
    "sanafe_arch_preset",
    "sanafe_custom_arch_path",
    "sanafe_log_potential_trace",
    "simulation_batch_count",
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
    if "training_recipe" in out:
        out["training_recipe"] = dict(out["training_recipe"])
    if "tuning_recipe" in out:
        out["tuning_recipe"] = dict(out["tuning_recipe"])
    return out


def get_system_default_deployment_parameters() -> Dict[str, object]:
    """Return internal/system deployment defaults hidden from saved user config."""
    user_keys = set(get_user_default_deployment_parameters())
    out = {k: v for k, v in DEFAULT_DEPLOYMENT_PARAMETERS.items() if k not in user_keys}
    if "training_recipe" in out:
        out["training_recipe"] = dict(out["training_recipe"])
    if "tuning_recipe" in out:
        out["tuning_recipe"] = dict(out["tuning_recipe"])
    return out


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

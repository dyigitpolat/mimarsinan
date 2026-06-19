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
    "tuner_target_floor_ratio": 0.90,
    # Activation-Analysis scale quantile = the per-perceptron LIF/TTFS decode scale
    # AND clamp ceiling. Raising it toward 1.0 reduces the systematic top-percentile
    # clip bias (trades a little rate-resolution); 0.99 = the historical default.
    "activation_scale_quantile": 0.99,
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
    # Diagnostic (default off): after each commit, probe the value-domain
    # rate-1.0 accuracy and report whether the gradual ramp's full-transform
    # drop shrinks as the committed rate climbs (is the ramp converging the
    # model toward 1.0-viability, or just inching the rate up?).
    "tuning_full_transform_probe": False,
    # Characterization pre-phase (spec §10 / V9, default off): before the rate
    # search, sweep a coarse α grid to profile the axis — feed the slope-derived
    # epsilon_hint to the scheduler (A3) and, if the drop is non-monotone (A1, e.g.
    # a re-aligning quant grid), downgrade the search to dense_grid safe mode
    # instead of trusting the global monotonicity assumption. Default off keeps the
    # goldens bit-exact; enabling it changes the search trajectory (Tier-B).
    "tuning_enable_characterization": False,
    "tuning_characterization_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    # Recovery-quality knobs (all default-off → byte-identical to current behavior):
    # re-find the LR after a committed cycle MISSES the target (else the LR is
    # cached at cycle 0 forever); the next cycle re-discovers a fresh LR.
    "tuning_refind_lr_on_miss": False,
    # Plateau LR reduction within a recovery: on a plateau, multiply the optimizer
    # LR by ``_factor`` (up to ``_reductions`` times) and continue instead of
    # breaking immediately — a coarse-to-fine recovery ladder.
    "tuning_recovery_lr_plateau": False,
    "tuning_recovery_lr_plateau_factor": 0.3,
    "tuning_recovery_lr_plateau_reductions": 2,
    # Non-stalling rollback ratchet: KEEP the per-step relative gate (so the ramp
    # keeps climbing — each step may give back a little) but cap the CUMULATIVE
    # drift below the best-committed high-water mark; the bound tightens as the
    # best ratchets up (no accumulation, no stall). Only used when the flag is on.
    "tuning_rollback_ratchet": False,
    "tuning_rollback_cumulative_bound": 0.05,
    # Bounded cosine-scheduled stabilization: instead of the open-ended patience/
    # round-based pass, run a SINGLE hard-cutoff pass of ``ratio * gradual_steps``
    # steps with a cosine-decay LR (chosen LR -> ~0 over exactly N steps).
    "tuning_stabilization_bounded": False,
    "tuning_stabilization_ratio": 0.5,
    # Tighter plateau detection (validation is cheap): divide the recovery check
    # interval by ``_divisor`` so the stale-streak patience trips after fewer steps.
    "tuning_tight_plateau": False,
    "tuning_recovery_check_divisor": 1,
    # Recipe-driven STEP recovery (generic: routes tuning_recipe + warmup/cosine
    # into the step recovery instead of the hardcoded Adam(wd=5e-5)/constant-LR path).
    "tuning_recipe_recovery": False,
    # Genuine annealed TTFS-cascade ramp (opt-in): train through the genuine
    # single-spike cascade for the whole ramp with the spike-surrogate sharpness
    # annealed smooth->sharp. Must stay default-off until a full real-model run
    # clears the accuracy-non-regression gate.
    "ttfs_genuine_annealed_ramp": False,
    "ttfs_ramp_alpha_min": 0.5,
    "ttfs_ramp_alpha_max": 2.0,
    # Scale-aware TTFS boundaries (opt-in): before ttfs_cycle fine-tuning, set each
    # block's activation_scale to its Activation-Analysis theta_out and propagate
    # input_activation_scale = upstream theta_out (the LIF scale-aware analog).
    "ttfs_scale_aware_boundaries": False,
    # Teacher->genuine blend ramp + per-neuron DFQ distribution matching (opt-in,
    # experimental): ramp the output from (1-r)*teacher + r*genuine cascade while
    # DFQ-correcting each perceptron's bias to match the ANN activation distribution.
    "ttfs_genuine_blend_ramp": False,
    "ttfs_distmatch_bias_iters": 15,
    "ttfs_distmatch_bias_eta": 0.7,
    "ttfs_distmatch_quantile": 0.99,
    # Weight of the extra CE on the PURE genuine logits in the blend-ramp loss
    # (the validated prototype value that pulls the rate-1 endpoint up). Canonical
    # source for the constant; the tuner reads this key once.
    "ttfs_genuine_blend_ce_alpha": 0.3,
    # Offload-boundary straight-through estimator (opt-in, cascaded only): flow the
    # genuine cascade backward through the round-based re-encode at offload/host-
    # ComputeOp segment boundaries (a soft spike-time STE) so EVERY segment trains
    # on the deployed dynamics, not only the last. Forward stays bit-exact.
    "ttfs_boundary_surrogate": False,
    "ttfs_boundary_surrogate_temp": 1.0,
    # Per-cascade-depth gain correction (opt-in, cascaded only): invert the deployed
    # ramp decode's depth-dependent attenuation (the death cascade) with a per-layer
    # activation_scale trim theta_d *= gamma^d, gamma = 1 - sqrt(S)/(S+1). A pure
    # calibration change (decode untouched -> NF<->SCM parity holds); recovers most of
    # the cold conversion gap and gives the genuine fine-tune a healthy (alive) init.
    "ttfs_gain_correction": False,
    "ttfs_gain_correction_rule": "relative",
    "ttfs_gain_correction_c": 1.9,
    # Rate-gated gain correction (opt-in): instead of applying the gain trim once
    # (cold, where a downstream fine-tune absorbs it at the readout), ramp it as a
    # parameter transformation gated by the SAME rate as the KD blend — theta_d ->
    # base * g_d**rate as rate 0->1 — so the model co-adapts to the calibration and
    # the spiking dynamics together (the gradual non-destructive transformation).
    "ttfs_gain_correction_ramp": False,
    # Per-channel TRAINABLE theta co-training (opt-in, cascaded only): promote each
    # non-encoding perceptron's activation_scale to a per-output-channel requires_grad
    # Parameter so the deployed-cascade fine-tune co-optimises the firing-gain (theta,
    # the death-cascade's root cause) WITH the weights. The near-lossless cascaded
    # recipe's key lever (docs/research_artifacts_for_cascaded_ttfs_tuning/51_*). Unlike
    # the per-DEPTH fixed-geometric gain correction, theta is learned per-neuron; the
    # two are mutually exclusive (gain ramp wins). Encoding/entry theta stays fixed.
    "ttfs_theta_cotrain": False,
    # Staircase-backward STE (opt-in, cascaded only): train the genuine cascade with a
    # straight-through estimator -- forward = genuine fire-once cascade (exact deploy),
    # backward = ttfs_ste_mix * clean complete-sum staircase gradient + (1-mix) * genuine
    # surrogate. Fixes the deep high-S surrogate-gradient plateau -> near-lossless cascaded
    # TTFS in <2 min (docs/.../52_lossless_fast_program.md). mix=0.5 is the robust default
    # across depth (pure staircase scrambles the basin; pure genuine stays on the plateau).
    "ttfs_staircase_ste": False,
    "ttfs_ste_mix": 0.5,
    # Fast fixed-increment genuine-blend ramp (opt-in, requires ttfs_genuine_blend_ramp):
    # runs through the orchestrator with a fixed_ladder RateScheduler policy
    # (schedule-not-search) instead of greedy/bisect — one shared optimizer + spanning
    # warmup/cosine LR over the whole ladder, no per-cycle rollback/recovery/LR-find/
    # stabilization (~30-60s). Inherits the DecisionTrace + finalize observability.
    "ttfs_genuine_blend_fast": False,
    "ttfs_blend_fast_steps_per_rate": 120,
    "ttfs_blend_fast_rates": [0.5, 0.75, 0.9, 0.97, 1.0],
    # Fast PROXY ramp (opt-in, cascaded, NOT a genuine ramp): the value-domain
    # blend ramp via the fixed_ladder policy + a post-finalize bounded stabilization
    # on the genuine cascade (closes the proxy↔genuine cliff) — the LIF pattern for
    # the better-accuracy TTFS path, made fast.
    "ttfs_blend_fast": False,
    "ttfs_blend_fast_stabilize_steps": 0,
    "ttfs_blend_fast_lr_eta_min": 0.1,
    # Fast fixed-ladder LIF ramp (opt-in): the LIF value-domain blend ramp through
    # the orchestrator's fixed_ladder policy (one shared optimizer + spanning cosine,
    # KD recovery, no controller) — the FAST analog of the slow LIF controller ramp.
    "lif_blend_fast": False,
    "lif_blend_fast_steps_per_rate": 120,
    "lif_blend_fast_rates": [0.25, 0.5, 0.75, 1.0],
    "lif_blend_fast_lr_eta_min": 0.1,
    "lif_blend_fast_stabilize_steps": 0,
    # DFQ per-neuron bias correction on the deployed LIF cascade (opt-in): match
    # each perceptron's deployed channel-mean to the teacher ANN's by nudging
    # layer.bias, shrinking the systematic ANN->SNN first-moment conversion gap.
    "lif_distmatch": False,
    "lif_distmatch_bias_iters": 10,
    "lif_distmatch_bias_eta": 0.5,
    "lif_distmatch_cal_batches": 8,
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
    "tuner_target_floor_ratio",
    "checkpoint_scope",
    "checkpoint_location",
    "tuning_use_paired_sensor",
    "k_commit",
    "paired_confirm_batches",
    "global_budget",
    "tuning_full_transform_probe",
    "tuning_enable_characterization",
    "tuning_characterization_grid",
    "tuning_refind_lr_on_miss",
    "tuning_recovery_lr_plateau",
    "tuning_recovery_lr_plateau_factor",
    "tuning_recovery_lr_plateau_reductions",
    "tuning_rollback_ratchet",
    "tuning_rollback_cumulative_bound",
    "tuning_stabilization_bounded",
    "tuning_stabilization_ratio",
    "tuning_tight_plateau",
    "tuning_recovery_check_divisor",
    "tuning_recipe_recovery",
    "ttfs_genuine_annealed_ramp",
    "ttfs_ramp_alpha_min",
    "ttfs_ramp_alpha_max",
    "ttfs_scale_aware_boundaries",
    "ttfs_genuine_blend_ramp",
    "ttfs_distmatch_bias_iters",
    "ttfs_distmatch_bias_eta",
    "ttfs_distmatch_quantile",
    "ttfs_genuine_blend_ce_alpha",
    "ttfs_boundary_surrogate",
    "ttfs_boundary_surrogate_temp",
    "ttfs_gain_correction",
    "ttfs_gain_correction_rule",
    "ttfs_gain_correction_c",
    "ttfs_gain_correction_ramp",
    "ttfs_theta_cotrain",
    "ttfs_staircase_ste",
    "ttfs_ste_mix",
    "ttfs_genuine_blend_fast",
    "ttfs_blend_fast_steps_per_rate",
    "ttfs_blend_fast_rates",
    "ttfs_blend_fast",
    "ttfs_blend_fast_stabilize_steps",
    "ttfs_blend_fast_lr_eta_min",
    "lif_blend_fast",
    "lif_blend_fast_steps_per_rate",
    "lif_blend_fast_rates",
    "lif_blend_fast_lr_eta_min",
    "lif_blend_fast_stabilize_steps",
    "lif_distmatch",
    "lif_distmatch_bias_iters",
    "lif_distmatch_bias_eta",
    "lif_distmatch_cal_batches",
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

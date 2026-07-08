"""Registry entries: spiking semantics, conversion process, and tuning controller keys."""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.spiking_semantics import (
    DEFAULT_THRESHOLDING_MODE,
    derived_firing_mode,
    derived_spike_generation_mode,
    legal_firing_modes,
    legal_spike_generation_modes,
    legal_thresholding_modes,
)
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
    registered_elsewhere as _registered_elsewhere,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_SUPPORTED_MODES,
)
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

SPIKING_MODES = ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based")


def _mode(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("spiking_mode", "lif"))


def _why_activation_quantization(cfg: dict) -> str:
    if cfg.get("activation_quantization"):
        return f"on — derived from spiking_mode={cfg.get('spiking_mode')!r}"
    if cfg.get("pipeline_mode") == "vanilla" or cfg.get("weight_quantization") is False:
        return "off — float-weight (vanilla) deployment"
    return f"off — analytical spiking_mode={cfg.get('spiking_mode')!r}"


ENTRIES = (
    _E("spiking_mode", group="spiking", owner="SpikingDeploymentContract",
       type=T.ENUM, options=SPIKING_MODES, category=Category.BASIC, exposure="user",
       label="Spiking Mode", important=True, effect="Selects LIF/TTFS path and simulation backends",
       doc="Deployable spiking semantics: lif (rate), analytical ttfs, ttfs_quantized, "
           "or ttfs_cycle_based (single-spike, cycle-accurate)."),
    _E("ttfs_cycle_schedule", group="spiking", owner="SpikingDeploymentContract",
       type=T.ENUM, options=("cascaded", "synchronized"), category=Category.BASIC,
       exposure="user", label="TTFS Cycle Schedule",
       doc="cascaded: greedy fire-once cascade (fewer cycles); synchronized: exact "
           "S-windowed schedule (more cycles, SANA-FE only).",
       relevant=R.when("spiking_mode", in_=("ttfs_cycle_based",))),
    _E("firing_mode", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("Default", "Novena", "TTFS"), category=Category.ADVANCED,
       exposure="user", label="Firing Mode",
       doc="Neuron firing semantics: Default (subtractive reset), Novena (zero reset), "
           "or TTFS. The legal set follows spiking_mode: the TTFS family admits only "
           "'TTFS' (the field locks); LIF admits Default/Novena.",
       provenance="derivation rule",
       derived_default=lambda cfg: derived_firing_mode(_mode(cfg)),
       legal_values=lambda cfg: legal_firing_modes(_mode(cfg)),
       empty_means="derived from spiking_mode (TTFS modes force 'TTFS')"),
    _E("spike_generation_mode", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("Uniform", "Deterministic", "Stochastic", "TTFS"),
       category=Category.ADVANCED, exposure="user",
       label="Spike Generation Mode",
       doc="Input spike-train encoding. The legal set follows spiking_mode: the TTFS "
           "family admits only 'TTFS' (the field locks); the rate encoder refuses it.",
       provenance="derivation rule",
       derived_default=lambda cfg: derived_spike_generation_mode(_mode(cfg)),
       legal_values=lambda cfg: legal_spike_generation_modes(_mode(cfg)),
       empty_means="derived from spiking_mode (TTFS modes force 'TTFS')"),
    _E("thresholding_mode", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("<", "<="), category=Category.ADVANCED, exposure="user",
       label="Thresholding Mode",
       doc="Membrane-threshold comparison (strict or inclusive). Both values are legal "
           "in every mode; unset derives '<='.",
       provenance="derivation rule", derived_default=_frozen(DEFAULT_THRESHOLDING_MODE),
       legal_values=lambda cfg: legal_thresholding_modes(_mode(cfg)),
       empty_means="derived from spiking_mode ('<=')"),
    _E("encoding_layer_placement", group="mapping_strategy", owner="mapping/encoding_layer",
       type=T.ENUM, options=("subsume", "offload"), category=Category.BASIC,
       exposure="user", label="Encoding Layer Placement",
       effect="Offload maps the encoding-layer neuralOp on-chip",
       doc="subsume: host-side encoder; offload: encoding layer mapped on-chip "
           "(functionally identical, larger hardware-accelerated surface). "
           "An explicit choice — no schema default; configs pin a value as data.",
       empty_means="no default — choose subsume or offload (the starter pins subsume)"),
    _E("negative_value_shift", group="mapping_strategy", owner="bias_compensation",
       type=T.BOOL, category=Category.BASIC, exposure="user",
       label="Negative-value Shift",
       effect="How a negative ComputeOp→neural boundary stays lossless: "
              "calibrated shift (on) or subsume-forward host mapping (off)",
       doc="Both positions are numerically sound; silent [0,1] clamp corruption "
           "is not authorable. ON (default): a calibrated positive shift moves the "
           "boundary into the encodable domain and the consuming perceptron's bias "
           "is pre-corrected (B − W·s) — mapping structure unchanged; a topology "
           "with no absorbing perceptron (ComputeOp→ComputeOp) fails loud. OFF: the "
           "mapper subsumes consuming perceptrons forward onto the host until a "
           "non-negative-value-generating activation (e.g. ReLU) absorbs the signed "
           "range — exact value-domain math, larger host surface; a graph left with "
           "no on-chip segment fails loud."),
    _E("cycle_accurate_lif_forward", group="spiking", owner="lif_adaptation",
       type=T.BOOL, category=Category.DERIVED, derivation="derived",
       exposure="derived", label="Cycle-accurate LIF Forward",
       effect="Spike-train forward during LIF adaptation training",
       doc="LIF adaptation trains against the cycle-accurate spiking forward "
           "(the deployed forward) — the correctness mechanism that keeps the "
           "QAT train-forward bit-exact to the deployed eval-forward. The LIF "
           "recipe always folds it ON; it is never a knob.",
       derived_from=("spiking_mode",),
       why=lambda cfg: (
           "on — LIF adaptation trains the deployed cycle-accurate forward "
           "(train/eval bit-exactness)"
           if cfg.get("spiking_mode") == "lif"
           else f"inert — no LIF adaptation for spiking_mode={cfg.get('spiking_mode')!r}"
       ),
       declarable=False, provenance="derivation rule"),
    _E("activation_quantization", group="conversion",
       owner="deployment_derivation/activation_analysis", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Activation Quantization",
       effect="Gates activation quantization pipeline steps",
       doc="Derived from the deployment mode: ON for lif/ttfs_quantized/"
           "ttfs_cycle_based, OFF for analytical ttfs and float-weight deployments. "
           "Never pin it in a config; derivation owns it.",
       derived_from=("spiking_mode", "weight_quantization", "pipeline_mode"),
       why=_why_activation_quantization, declarable=False,
       provenance="derivation rule"),
    _E("weight_quantization", group="conversion",
       owner="deployment_derivation/weight_quantization", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="user",
       label="Weight Quantization",
       effect="Gates weight quantization pipeline steps",
       doc="Bits-driven: weight_bits declares a quantized artifact; declare float "
           "weights via pipeline_mode='vanilla' or weight_quantization=false.",
       derived_from=("weight_bits", "pipeline_mode"), declarable=True,
       why=lambda cfg: ("on — weight_bits declares a quantized artifact"
                        if cfg.get("weight_quantization")
                        else "off — float-weight deployment (vanilla mechanism)"),
       provenance="derivation rule"),
    _E("pruning", group="mapping_strategy", owner="pruning_adaptation",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Pruning Enabled",
       effect="Adds the Pruning Adaptation step",
       doc="Enable magnitude pruning adaptation (a deployment-side conversion "
           "step, not an architecture property).",
       provenance="consumer frozen default", derived_default=_frozen(False)),
    _E("pruning_fraction", group="mapping_strategy", owner="pruning_adaptation",
       type=T.FLOAT, category=Category.BASIC, exposure="user", label="Pruning Fraction",
       doc="Fraction of weights pruned by the adaptation.", bounds=(0.0, 1.0),
       relevant=R.when_true("pruning"),
       provenance="consumer frozen default", derived_default=_frozen(0.0),
       empty_means="0 — pruning stays inert (no Pruning Adaptation step)"),
    _E("prune_sparsity", group="mapping_strategy", owner="pruning_adaptation",
       type=T.FLOAT, category=Category.ADVANCED, label="Prune Sparsity",
       doc="Legacy sparsity knob consumed by the pruning tuner mask builder.",
       bounds=(0.0, 1.0), relevant=R.when_true("pruning"),
       provenance="consumer frozen default", derived_default=_frozen(0.0)),
    _E("activation_scale_quantile", group="tuning", owner="activation_analysis",
       type=T.FLOAT, category=Category.ADVANCED, label="Activation Scale Quantile",
       doc="Quantile of observed activations used as the per-layer scale (the mode "
           "recipe may override the schema default).", bounds=(0.0, 1.0),
       provenance="ConversionPolicy recipe"),
    _E("activation_analysis_batch_size", group="tuning", owner="activation_analysis",
       type=T.INT, category=Category.ADVANCED, label="Activation Analysis Batch Size",
       doc="Batch size for the activation-statistics capture pass.", bounds=(1, None),
       provenance="consumer frozen default", derived_default=_registered_elsewhere,
       empty_means="min(validation batch size, the analysis step's frozen cap 16)"),
    _E("ttfs_genuine_blend_ce_alpha", group="tuning", owner="ttfs_adaptation",
       type=T.FLOAT, category=Category.ADVANCED, label="TTFS Blend CE Alpha",
       doc="CE weight in the genuine-forward blend loss of TTFS adaptation.", bounds=(0.0, 1.0)),
    _E("ttfs_finetune_kd_against_rung2", group="tuning", owner="ttfs_adaptation",
       type=T.BOOL, category=Category.ADVANCED, label="TTFS KD Against Rung 2",
       doc="Distill the TTFS fine-tune against the rung-2 teacher snapshot.",
       provenance="consumer frozen default", derived_default=_frozen(False)),
    _E("enable_training_noise", group="tuning", owner="noise_adaptation",
       type=T.BOOL, category=Category.ADVANCED, label="Training Noise",
       effect="Adds the Noise Adaptation step", doc="Enable noise-injection adaptation before quantization."),
    _E("s_allocation", group="mapping_strategy", owner="TemporalAllocation",
       type=T.ENUM, options=("uniform", "explicit", "budget"), category=Category.ADVANCED,
       exposure="user", label="S Allocation",
       effect="Per-cascade-depth temporal resolution (gated by allow_per_layer_s)",
       doc="Temporal-resolution allocation across cascade depths. Only 'uniform' is "
           "wired, so the legal set is a singleton and the field LOCKS; the reserved "
           "explicit/budget modes stay loud-rejected config data.",
       legal_values=lambda cfg: S_ALLOCATION_SUPPORTED_MODES),
    _E("s_allocation_explicit", group="mapping_strategy", owner="TemporalAllocation",
       type=T.INT_LIST, category=Category.ADVANCED, exposure="user",
       label="S Allocation (explicit)", doc="Explicit per-depth S list for s_allocation='explicit'.",
       relevant=R.when("s_allocation", in_=("explicit",))),
    _E("s_allocation_budget", group="mapping_strategy", owner="TemporalAllocation",
       type=T.JSON, category=Category.ADVANCED, exposure="user",
       label="S Allocation (budget)", doc="Budget objective body for s_allocation='budget'.",
       relevant=R.when("s_allocation", in_=("budget",))),
    _E("tuning_budget_scale", group="tuning", owner="AdaptationManager",
       type=T.FLOAT, category=Category.BASIC, exposure="user", label="Tuning Budget Scale",
       doc="Scales every adaptation tuner's step budget (wall-clock lever).",
       bounds=(0.0, None)),
    _E("tuning_budget_scale_ramp_steps", group="tuning", owner="AdaptationManager",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Budget Scale Ramp Steps",
       doc="Also scale ramp step counts by the budget scale (default off)."),
    _E("tuner_target_floor_ratio", group="tuning", owner="AdaptationManager", type=T.FLOAT,
       category=Category.ADVANCED, label="Tuner Target Floor Ratio",
       doc="Per-tuner floor as a fraction of the entry accuracy.", bounds=(0.0, 1.0)),
    _E("degradation_tolerance", group="tuning", owner="AccuracyBudget",
       type=T.FLOAT, category=Category.BASIC, exposure="user", label="Degradation Tolerance",
       doc="Accepted end-to-end accuracy budget from pretrain to deployed.",
       bounds=(0.0, 1.0)),
    _E("paired_confirm_batches", group="tuning", owner="rollback_sensor",
       type=T.INT, category=Category.ADVANCED, label="Paired Confirm Batches",
       doc="Batches for the paired rollback confirmation read (0 = eval_n_batches).",
       bounds=(0, None)),
    _E("pretrain_floor_chance_multiple", group="tuning", owner="engine/pretrain_envelope",
       type=T.FLOAT, category=Category.ADVANCED, label="Pretrain Floor Chance Multiple",
       doc="First seeded metric must exceed this multiple of chance level "
           "(classification); 0 disables the envelope.", bounds=(0.0, None),
       provenance="consumer frozen default", derived_default=_frozen(5.0),
       empty_means="the pretrain envelope's frozen multiple 5.0"),
    _E("endpoint_floor_steps", group="tuning", owner="endpoint_recovery/steps_ledger",
       type=T.INT, category=Category.ADVANCED, unit="steps", label="Endpoint Floor Steps",
       doc="Per-cell RUN-total training-step budget for armed 5u endpoint-floor "
           "stages (one ledger shared by every armed endpoint; steps, never wall "
           "seconds — the reproducibility contract).", bounds=(0, None),
       provenance="TUNING_POLICY",
       derived_default=_frozen(TUNING_POLICY.endpoint_floor_steps),
       empty_means="the frozen TUNING_POLICY run-total budget"),
    _E("endpoint_target_floor", group="tuning", owner="endpoint_recovery",
       type=T.FLOAT, category=Category.ADVANCED, label="Endpoint Target Floor",
       doc="Every-endpoint D-hat target floor. The ConversionPolicy recipe sets it "
           "only for the bit-parity-lossless family (analytical ttfs); every other "
           "mode floors at 0 and takes its floor from the WQ endpoint.", bounds=(0.0, 1.0),
       provenance="ConversionPolicy recipe", derived_default=_frozen(0.0),
       empty_means="the recipe floor where the mode has one, else 0 (no floor)"),
    _E("wq_endpoint_recovery_steps", group="tuning", owner="wq_endpoint_recovery",
       type=T.INT, category=Category.ADVANCED, unit="steps",
       label="WQ Endpoint Recovery Steps",
       doc="Per-cell cap on the WQ endpoint recovery stage (recipe default stays "
           "for families that pass by the floor climb).", bounds=(0, None),
       provenance="ConversionPolicy recipe", derived_default=_frozen(0),
       empty_means="the ConversionPolicy recipe cap for the mode"),
    _E("conversion_draws", group="tuning", owner="conversion_draws",
       type=T.INT, category=Category.ADVANCED, label="Conversion Draws",
       doc="[MBH-DRAWS] best-of-N draws on variance-carrying conversion stages (1 = "
           "single-draw, bit-identical); draw k seeds torch at seed+k.",
       bounds=(1, None), provenance="consumer frozen default", derived_default=_frozen(1),
       empty_means="1 (single-draw, bit-identical)"),
    _E("eval_subsample_target", group="tuning", owner="workload_profile/tuning_budget",
       type=T.INT, category=Category.ADVANCED, unit="samples",
       label="Eval Subsample Target",
       doc="Target evaluation-subset size for tuner accuracy reads. Providers "
           "register it via DataWorkloadProfile; explicit value wins; absent = "
           "the frozen generic 5000.", bounds=(1, None),
       provenance="provider registration", derived_default=_registered_elsewhere,
       empty_means="the provider's registration, else the frozen generic 5000"),
    _E("tuning_step_cap_epochs", group="tuning", owner="workload_profile/tuning_budget",
       type=T.FLOAT, category=Category.ADVANCED, unit="epochs",
       label="Tuning Step Cap (epochs)",
       doc="Cap on per-tuner training steps expressed in dataset epochs. "
           "Providers register it via DataWorkloadProfile; explicit value wins; "
           "absent = the frozen 4000-step cap.", bounds=(0.0, None),
       provenance="provider registration", derived_default=_registered_elsewhere,
       empty_means="the provider's registration, else the frozen 4000-step cap"),
    _E("calibration_set_policy", group="tuning", owner="workload_profile/calibration",
       type=T.JSON, category=Category.ADVANCED, label="Calibration Set Policy",
       doc="Calibration-set extents by purpose (distmatch_bias_iters, "
           "distmatch_cal_batches, gauge_batches, stat_batches, "
           "analysis_batches_max, analysis_batch_size_cap). Field-wise merge: "
           "explicit fields win over provider registrations; absent fields use "
           "the consumers' frozen defaults.",
       provenance="provider registration", derived_default=_registered_elsewhere,
       empty_means="provider registrations, else the consumers' frozen defaults"),
    _E("prefix_stage_lr", group="tuning", owner="workload_profile/tuning_policy",
       type=T.FLOAT, category=Category.ADVANCED, label="Prefix Stage LR Ceiling",
       doc="P4 prefix-stage LR ceiling. Builders register it via "
           "ModelWorkloadProfile; explicit value wins; absent = the frozen "
           "TUNING_POLICY value.", bounds=(0.0, None),
       provenance="builder profile",
       derived_default=_frozen(TUNING_POLICY.prefix_stage_lr),
       empty_means="the builder's registration, else the frozen TUNING_POLICY ceiling"),
    _E("endpoint_floor_lr", group="tuning", owner="workload_profile/tuning_policy",
       type=T.FLOAT, category=Category.ADVANCED, label="Endpoint Floor LR",
       doc="Floor-chasing endpoint LR ceiling. Builders register it via "
           "ModelWorkloadProfile; explicit value wins; absent = the frozen "
           "TUNING_POLICY value.", bounds=(0.0, None),
       provenance="builder profile",
       derived_default=_frozen(TUNING_POLICY.endpoint_floor_lr),
       empty_means="the builder's registration, else the frozen TUNING_POLICY value"),
    _E("proven_recovery_depth", group="tuning", owner="workload_profile/install_resolution",
       type=T.INT, category=Category.ADVANCED, unit="layers",
       label="Proven Recovery Depth",
       doc="A6 chain-depth law override for architectures with different "
           "recovery behavior. Builders register it via ModelWorkloadProfile; "
           "explicit value wins; absent = the corpus-calibrated 6.",
       bounds=(1, None),
       provenance="builder profile", derived_default=_registered_elsewhere,
       empty_means="the builder's registration, else the corpus-calibrated 6"),
)

"""Registry entries: tuning-controller, adaptation/calibration, and budget keys."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
    registered_elsewhere as _registered_elsewhere,
)
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


ENTRIES = (
    _E("activation_scale_quantile", group="tuning", owner="activation_analysis",
       type=T.FLOAT, category=Category.ADVANCED, label="Activation Scale Quantile",
       doc="Quantile of observed activations used as the per-layer scale (the mode "
           "recipe may override the schema default).", bounds=(0.0, 1.0),
       provenance="ConversionPolicy recipe"),
    _E("s_aware_theta_quantile", group="tuning", owner="activation_analysis",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="S-aware Theta Quantile",
       effect="The theta calibration quantile follows the value grid",
       doc="[S1/R4] Quantile descent: the effective calibration quantile is "
           "min(activation_scale_quantile, policy(levels)) with the memo's "
           "measured anchors (S=4 -> 0.95, S=8 -> 0.99, S=16 -> 0.995, "
           "S>=32 -> 1.0; sync_deployment_exactness.md §3.3) — the optimal "
           "theta loading falls as the grid coarsens. Only ever deflates the "
           "base; continuous modes are untouched.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off — the mode's configured quantile stands at every S"),
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
    _E("wq_two_scale_projection", group="tuning",
       owner="normalization_aware_perceptron_quantization",
       type=T.BOOL, category=Category.ADVANCED,
       label="WQ Two-scale Projection",
       doc="Weight-quantization projection grids: ON derives the weight grid "
           "from max|w| alone and quantizes the effective bias on its own "
           "±q_max register range (integer-ratio-snapped to the weight grid, "
           "so chip emission stays an exact integer lattice); OFF is the "
           "legacy shared max(|w|,|b|) grid, which a dominant bias starves "
           "(wq_cascade_crater_repair.md). Platforms without an on-chip bias "
           "register fall back to the shared grid (capability gate).",
       provenance="ConversionPolicy recipe", derived_default=_frozen(False),
       empty_means="the mode recipe (ON for ttfs_cycle_based and lif), else off"),
    _E("sync_first_moment_fold", group="tuning",
       owner="activation_quantization_tuner",
       type=T.BOOL, category=Category.ADVANCED, label="Sync First-moment Fold",
       effect="Sequential closed-form per-hop bias fold at the sync AQ "
              "endpoint, before endpoint recovery",
       doc="[S3/R6] Per hop (input->output), the deployed-vs-float "
           "pre-activation mean gap — measured through the already-folded "
           "prefix, with the hop's OWN +theta/(2S) half-step EXCLUDED — is "
           "folded out of the effective bias. The exclusion is load-bearing: "
           "folding the raw gap cancels the mid-tread compensation "
           "(0.93->0.59 measured, sync_deployment_exactness.md §3.2). "
           "Calibration-only; never spends training budget.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       relevant=R.when("spiking_mode", in_=("ttfs_cycle_based",)),
       empty_means="off — no first-moment fold at the AQ endpoint"),
    _E("lif_exact_qat", group="tuning", owner="lif_exact_qat",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="LIF Exact QAT",
       effect="The AQ stage trains the exact deployed LIF count staircase "
              "(theta in-loop); per-hop re-timed deployment is auto-paired",
       doc="[lif_exact_qat_program §6] Staircase exact-QAT: the AQ stage "
           "(transform-inert for LIF otherwise) installs the deployed count "
           "staircase theta*clamp(F(T*z/theta),0,T)/T under staircase-STE with "
           "the LSQ theta gradient (per-channel on the R3 matching-axis set), "
           "folds the half-step ONCE as trainable entry bias (P-L6; the Shift "
           "bake and the WQ-entry fold are skipped marker-assertedly), and the "
           "LIF Adaptation reduces to finalize+verify. Deployment auto-pairs "
           "lif_per_hop_retiming — the trained staircase IS the per-hop twin "
           "(measured 0.9556 -> 0.9747-0.9751 deployed; raw-cascade staircase "
           "deployment is the -2.5 pp Goodhart hole and fails loud). "
           "Default-reset LIF only (P-L5).",
       provenance="consumer frozen default", derived_default=_frozen(False),
       relevant=R.when("spiking_mode", in_=("lif",)),
       empty_means="off — the shipped T-anneal + one-shot-fold recipe"),
    _E("lif_exact_qat_kd", group="tuning", owner="lif_exact_qat_kd",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="LIF Exact QAT KD teacher",
       effect="Distils the exact-QAT endpoint to the post-structural float "
              "teacher instead of training it with plain CE",
       doc="[lif_exact_qat_program §8, WS-Z] The AQ-hosted exact-QAT trains "
           "with plain CE on-pipeline (the measured WORST KD arm, -1.70 SEd), "
           "so the endpoint saturates at the AQ envelope. This knob distils it "
           "to the POST-STRUCTURAL float teacher (a Reference Teacher Snapshot "
           "step captures the model after Scale Migration, so pruned cells get "
           "the post-prune reference, G8) with the recipe's kd_ce_alpha / "
           "kd_temperature; the exact endpoint then saturates at the teacher's "
           "own accuracy (WS-Z-measured WIN, >1 SE at both S=4 cells). Pairs "
           "with lif_exact_qat: it downgrades with the exact arm (Novena / "
           "explicit opt-out) and an explicit contradiction fails loud. "
           "REFUTED on-pipeline (§9): the isolated WIN inverts on the full "
           "composition (5-mixer mean -0.14pp; t01_01 S=8 -0.51, t01_21 -0.70 "
           "vs t0_01/t01_08 S=4 +0.30) — the KD loss also steers the WQ "
           "endpoint leg and the draws. Config-armable only; NOT recipe-armed.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       relevant=R.when("spiking_mode", in_=("lif",)),
       empty_means="off — the exact-QAT endpoint trains with plain CE"),
    _E("lif_affine_fold", group="tuning", owner="lif_affine_fold",
       type=T.BOOL, category=Category.ADVANCED, label="LIF Affine Fold",
       effect="Adds the pre-WQ LIF Affine Fold calibration step",
       doc="[C4] Calibration-derived per-channel FULL affine (gain + bias) "
           "absorbing the LIF dead-zone/saturation bias: layer-sequential "
           "closed-form LS fits of deployed rates against the float envelope, "
           "folded into consumer weight columns/bias (the negative-shift fold "
           "family) before the weight-quantization QAT. Full affine only — "
           "bias-only folds are refuted (-4.2pp); the Novena arm is gated at "
           "S >= 8. The LIF recipe arms it (lif_deployment_exactness.md).",
       provenance="ConversionPolicy recipe", derived_default=_frozen(False),
       empty_means="the lif recipe arms it; other modes stay off"),
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

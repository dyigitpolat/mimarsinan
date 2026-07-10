"""Registry entries: spiking semantics, conversion process, and mapping-strategy keys."""

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
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_SUPPORTED_MODES,
)

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
)

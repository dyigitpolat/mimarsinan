"""Registry entries: spiking semantics, conversion process, and mapping-strategy keys (pruning keys live in entries_pruning)."""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.activation_semantics import (
    effective_legacy_spiking_mode,
)
from mimarsinan.chip_simulation.soma_axes import (
    firing_mode_for_granularity,
    resolved_firing_granularity,
)
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
from mimarsinan.transformations.channel_scale_equalization import (
    DEFAULT_CLIP_RATIO as DEFAULT_SCALE_MIGRATION_CLIP_RATIO,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_SUPPORTED_MODES,
)


def _mode(cfg: Mapping[str, Any]) -> str:
    return effective_legacy_spiking_mode(cfg)


def _derived_firing_mode(cfg: Mapping[str, Any]) -> str:
    """The mode's derived reset, then the granularity's REQUIREMENT on it.

    A document that declares only ``firing_granularity='per_event'`` therefore
    resolves to the hard-zero reset that law is realizable under, instead of
    silently keeping today's subtractive one (the row-pair lemma).
    """
    return firing_mode_for_granularity(
        resolved_firing_granularity(cfg), derived_firing_mode(_mode(cfg)))


# Relevance vocabulary: trees are evaluated over AUTHORED keys (the frontend
# evaluates them against the draft), so they speak family/variant — never the
# derivation-owned legacy twins.
_LIF_FAMILY_RELEVANT = R.when("spiking_family", in_=("lif",))
_TTFS_CYCLE_RELEVANT = R.all_of(
    R.when("spiking_family", in_=("ttfs",)),
    R.when("spiking_variant", in_=("synchronized", "cascaded")),
)


def _why_activation_quantization(cfg: dict) -> str:
    if cfg.get("activation_quantization"):
        return f"on — derived from spiking mode {_mode(cfg)!r}"
    if cfg.get("pipeline_mode") == "vanilla" or cfg.get("weight_quantization") is False:
        return "off — float-weight (vanilla) deployment"
    return f"off — analytical spiking mode {_mode(cfg)!r}"


ENTRIES = (
    _E("firing_mode", domain="event", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("Default", "Novena", "TTFS"), category=Category.ADVANCED,
       exposure="user", label="Firing Mode",
       doc="Neuron firing semantics: Default (subtractive reset), Novena (zero reset), "
           "or TTFS. The legal set follows spiking_mode: the TTFS family admits only "
           "'TTFS' (the field locks); LIF admits Default/Novena. A per-event "
           "firing_granularity requires 'Novena' — the row-pair realization is a "
           "no-op only while every fire leaves the membrane below theta.",
       provenance="derivation rule",
       derived_default=_derived_firing_mode,
       legal_values=lambda cfg: legal_firing_modes(_mode(cfg)),
       empty_means="derived from spiking_mode (TTFS modes force 'TTFS'; a "
                   "per_event granularity forces 'Novena')"),
    _E("spike_generation_mode", domain="event", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("Uniform", "Deterministic", "Stochastic", "TTFS"),
       category=Category.ADVANCED, exposure="user",
       label="Spike Generation Mode",
       doc="Input spike-train encoding. The legal set follows spiking_mode: the TTFS "
           "family admits only 'TTFS' (the field locks); the rate encoder refuses it.",
       provenance="derivation rule",
       derived_default=lambda cfg: derived_spike_generation_mode(_mode(cfg)),
       legal_values=lambda cfg: legal_spike_generation_modes(_mode(cfg)),
       empty_means="derived from spiking_mode (TTFS modes force 'TTFS')"),
    _E("thresholding_mode", domain="event", group="spiking", owner="DeploymentPipeline", type=T.ENUM,
       options=("<", "<="), category=Category.ADVANCED, exposure="user",
       label="Thresholding Mode",
       doc="Membrane-threshold comparison (strict or inclusive). Both values are legal "
           "in every mode; unset derives '<='.",
       provenance="derivation rule", derived_default=_frozen(DEFAULT_THRESHOLDING_MODE),
       legal_values=lambda cfg: legal_thresholding_modes(_mode(cfg)),
       empty_means="derived from spiking_mode ('<=')"),
    _E("encoding_layer_placement", domain="event", group="mapping_strategy", owner="mapping/encoding_layer",
       type=T.ENUM, options=("subsume", "offload"), category=Category.BASIC,
       exposure="user", label="Encoding Layer Placement",
       effect="Offload maps the encoding-layer neuralOp on-chip",
       doc="subsume: host-side encoder; offload: encoding layer mapped on-chip "
           "(functionally identical, larger hardware-accelerated surface). "
           "An explicit choice — no schema default; configs pin a value as data.",
       empty_means="no default — choose subsume or offload (the starter pins subsume)"),
    _E("negative_value_shift", domain="event", group="mapping_strategy", owner="bias_compensation",
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
    _E("lif_membrane_readout", domain="event", group="spiking", owner="lif_deployment_exactness",
       type=T.BOOL, category=Category.ADVANCED, label="LIF Membrane Readout",
       effect="Torch-side membrane-decode DIAGNOSTIC at the SCM gate; "
              "deployed reads always keep the counts decode",
       doc="[C2/R8] Membrane-augmented readout: by the charge identity "
           "Q_T = theta*c_T + m_T, decoding output cores as counts + m_T/theta "
           "(half-step charge removed when baked) recovers the exact, "
           "sign-carrying, unquantized pre-activation at final-only output "
           "cores. The chip exports spike counts only (nevresim has no "
           "membrane read port), so this decode NEVER reaches a deployed-read "
           "metric: arming it runs the SoftCoreMappingStep engagement "
           "diagnostic ([C2] line + reporter event) and nothing else "
           "(lossless_refinement_ledger.md §2F.1). The LIF recipe arms it.",
       provenance="ConversionPolicy recipe", derived_default=_frozen(False),
       relevant=_LIF_FAMILY_RELEVANT,
       empty_means="the lif recipe arms it; other modes stay off"),
    _E("lif_per_hop_retiming", domain="event", group="mapping_strategy", owner="layout/segmentation",
       type=T.BOOL, category=Category.DERIVED, derivation="derived",
       exposure="derived", hidden=True, declarable=False,
       label="LIF Per-hop Re-timing",
       effect="Splits deep neural segments into per-hop segments",
       doc="[C3] Count-exact re-timing: every hop boundary becomes a "
           "decode/re-encode (round((c/T)*T) = c), resetting arrival timing "
           "and killing the back-loading deficit on deep single-segment "
           "chains at S <= 8. RETIRED as a document key: the exact-QAT "
           "pairing of the windowed-lif recipe owns the arm (the pair "
           "downgrades together on Novena), so re-timing is part of the "
           "lif_sync semantics — never a knob.",
       derived_from=("spiking_family", "spiking_variant", "firing_mode"),
       why=lambda cfg: (
           "armed — the windowed-lif exact-QAT pairing deploys per-hop re-timed"
           if cfg.get("lif_per_hop_retiming")
           else "off — non-lif mode, or the pairing downgraded (Novena)"
       ),
       provenance="ConversionPolicy recipe"),
    _E("lif_depth_balancing_relays", domain="event", group="mapping_strategy",
       owner="latency/depth_balancing",
       type=T.BOOL, category=Category.ADVANCED, label="LIF Depth-balancing Relays",
       effect="Inserts identity relay cores on gap>1 intra-segment edges",
       doc="[C5] Unequal-depth fan-in inside a neural segment both drops the "
           "shallow branch's early spikes and re-reads its stale final buffer "
           "(V6). The exact remedy inserts identity relay chains so every "
           "live intra-segment edge is gap-1; a loud mapping-time guard "
           "rejects dead relays (an exact-theta identity weight never fires "
           "under the strict '<' comparator — the integer-lattice hazard). "
           "No-op on gap-free graphs; the LIF recipe arms it.",
       provenance="ConversionPolicy recipe", derived_default=_frozen(False),
       relevant=_LIF_FAMILY_RELEVANT,
       empty_means="the lif recipe arms it; other modes stay off"),
    _E("comparator_half_step", domain="event", group="spiking", owner="deployment_contract",
       type=T.BOOL, category=Category.ADVANCED, label="Comparator-side Half-step",
       effect="Staircase hops carry the +theta/(2S) mid-tread offset in the "
              "compare ladder instead of the bias",
       doc="[E3/G6/R7] The exact zero-bit-cost half-step placement: every "
           "per-cycle compare level theta*(S-k)/S drops by theta/(2S) "
           "(ceil(S*(1-v/theta) - 1/2)) — a ladder shift, never a theta "
           "rescale — so the two-scale WQ bias lattice cannot erode the "
           "mid-tread compensation (arm when g_b/(1/(2S)) >= ~0.5 at "
           "projection; blanket refolds measured -1.6pp). Both NF and SCM "
           "read the flag from the SpikingDeploymentContract, so the parity "
           "twins shift together (sync_deployment_exactness.md §5/§7).",
       provenance="consumer frozen default", derived_default=_frozen(False),
       relevant=_TTFS_CYCLE_RELEVANT,
       empty_means="off — the half-step stays a bias fold"),
    _E("cycle_accurate_lif_forward", domain="event", group="spiking", owner="lif_adaptation",
       type=T.BOOL, category=Category.DERIVED, derivation="derived",
       exposure="derived", label="Cycle-accurate LIF Forward",
       effect="Spike-train forward during LIF adaptation training",
       doc="LIF adaptation trains against the cycle-accurate spiking forward "
           "(the deployed forward) — the correctness mechanism that keeps the "
           "QAT train-forward bit-exact to the deployed eval-forward. The LIF "
           "recipe always folds it ON; it is never a knob.",
       derived_from=("spiking_family", "spiking_variant"),
       why=lambda cfg: (
           "on — LIF adaptation trains the deployed cycle-accurate forward "
           "(train/eval bit-exactness)"
           if _mode(cfg) == "lif"
           else f"inert — no LIF adaptation for spiking mode {_mode(cfg)!r}"
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
       derived_from=("spiking_family", "spiking_variant", "weight_quantization",
                     "pipeline_mode"),
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
    _E("scale_migration", group="mapping_strategy", owner="scale_migration_step",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Scale Migration",
       effect="Adds the Scale Migration step before Activation Analysis",
       doc="Exact cross-layer per-channel scale migration across ReLU-adjacent "
           "affine pairs (function-preserving by positive homogeneity); repairs "
           "per-hop scalar-theta grid starvation on unnormalized deep chains.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off — weights pass through byte-identical"),
    _E("scale_migration_clip_ratio", group="mapping_strategy", owner="scale_migration_step",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user",
       label="Scale Migration Clip Ratio",
       doc="Per-channel migration scales are clipped to [1/r, r]: unclipped "
           "migration amplifies near-dead channels onto NAPQ's shared weight "
           "grid (measured 5-bit WQ collapse to chance).",
       bounds=(1.0, None), relevant=R.when_true("scale_migration"),
       provenance="consumer frozen default",
       derived_default=_frozen(DEFAULT_SCALE_MIGRATION_CLIP_RATIO),
       empty_means="the mechanism default r=4 (measured to hold 5-bit WQ)"),
    _E("per_channel_theta", domain="event", group="tuning", owner="per_channel_theta",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Per-channel Theta",
       effect="Eligible matching-axis hops calibrate a per-channel decode theta",
       doc="[S2/R3] Exact scale-space level reallocation: Q_S(z/theta) depends "
           "only on z/theta, so a per-channel theta with per-channel decode "
           "refines the grid per channel — the one lever that beats the 1/S "
           "law. Promoted ONLY where the producer's channel axis is consumed "
           "as linear feature axes through structural mappers (the M4 "
           "matching-axis condition); weight-shared / axis-flipped hops and "
           "segment-entry consumers keep the scalar (sync memo §4.2-4.3, "
           "mixer memo §6). Capable modes: lif and synchronized ttfs_cycle.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       relevant=R.any_of(_LIF_FAMILY_RELEVANT, _TTFS_CYCLE_RELEVANT),
       empty_means="off — every hop keeps its pooled scalar theta"),
    # Pruning mapping-strategy keys live in entries_pruning.py (module budget).
    _E("s_allocation", domain="event", group="mapping_strategy", owner="TemporalAllocation",
       type=T.ENUM, options=("uniform", "explicit", "budget"), category=Category.ADVANCED,
       exposure="user", label="S Allocation",
       effect="Per-cascade-depth temporal resolution (gated by allow_per_layer_s)",
       doc="Temporal-resolution allocation across cascade depths. Only 'uniform' is "
           "wired, so the legal set is a singleton and the field LOCKS; the reserved "
           "explicit/budget modes stay loud-rejected config data.",
       legal_values=lambda cfg: S_ALLOCATION_SUPPORTED_MODES),
    _E("s_allocation_explicit", domain="event", group="mapping_strategy", owner="TemporalAllocation",
       type=T.INT_LIST, category=Category.ADVANCED, exposure="user",
       label="S Allocation (explicit)", doc="Explicit per-depth S list for s_allocation='explicit'.",
       relevant=R.when("s_allocation", in_=("explicit",))),
    _E("s_allocation_budget", domain="event", group="mapping_strategy", owner="TemporalAllocation",
       type=T.JSON, category=Category.ADVANCED, exposure="user",
       label="S Allocation (budget)", doc="Budget objective body for s_allocation='budget'.",
       relevant=R.when("s_allocation", in_=("budget",))),
    _E("origin_teacher_kd", group="tuning", owner="origin_teacher_kd",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Origin-teacher KD anchor",
       doc="Conversion tuners distil to the frozen post-structural ORIGIN "
           "teacher instead of per-step snapshots [spiking_deployment_calculus "
           "sec.13.2 L-A]; per-step KD knobs still choose WHERE KD runs.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off"),
    _E("origin_anchored_compact", group="tuning", owner="origin_anchored_compact",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Origin-anchored accuracy compact",
       doc="Step floors/targets anchor to the ORIGIN metric (no multiplicative "
           "per-step licensing, no miss relaxation) [spiking_deployment_calculus "
           "sec.13.2 L-B].",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off"),
)

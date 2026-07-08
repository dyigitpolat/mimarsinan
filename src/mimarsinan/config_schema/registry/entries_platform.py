"""Registry entries: hardware platform constraints and deployment-target keys."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs
from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

_PC = "platform_constraints"

# The NF<->SCM gate runs ONE question ("how many validation inputs?") on two
# statistics: a serial per-neuron sweep whose mass comes from neurons (2 inputs
# suffice) and a batched cascaded decision gate whose mass comes from samples
# (64 Bernoulli trials tolerate exactly one WQ tie-flip at min_agreement 0.98).
NF_SCM_PARITY_SAMPLES = 2
NF_SCM_PARITY_SAMPLES_CASCADED = 64


def _nf_scm_parity_samples(cfg: Mapping[str, Any]) -> int:
    """Mode-aware sample count of the single NF<->SCM parity gate."""
    if is_cascaded_ttfs(cfg.get("spiking_mode", "lif"), cfg.get("ttfs_cycle_schedule")):
        return NF_SCM_PARITY_SAMPLES_CASCADED
    return NF_SCM_PARITY_SAMPLES


def _scm_degradation_tolerance(cfg: Mapping[str, Any]) -> Optional[float]:
    """Absent, the SCM step installs no separate tolerance and the global
    end-to-end degradation_tolerance governs it."""
    return cfg.get("degradation_tolerance")


def _why_core_maximum(dim: str):
    def why(cfg: dict) -> str:
        return f"largest per-core {dim} across the declared core types"
    return why


def _backend_supported(cfg: dict, backend: str) -> bool:
    """Mode capability from the ConversionPolicy SSOT."""
    recipe = ConversionPolicy.derive(
        str(cfg.get("spiking_mode", "lif")), cfg.get("ttfs_cycle_schedule")
    )
    return bool(recipe.sim_enables.get(f"enable_{backend}_simulation", False))


def _why_backend_enable(backend: str, off_reason: str):
    """WHY text for a recipe-defaulted backend enable (user-off aware)."""
    def why(cfg: dict) -> str:
        key = f"enable_{backend}_simulation"
        mode = cfg.get("spiking_mode")
        if cfg.get(key):
            return f"on — ConversionPolicy runs the {backend} gate for {mode!r}"
        if _backend_supported(cfg, backend):
            return f"off — disabled in this config (recipe default for {mode!r}: on)"
        return f"off — {off_reason} (spiking_mode={mode!r})"
    return why


def _meta_backend_enable(backend: str):
    """Machine-readable support flag so the wizard renders toggle vs muted line."""
    def meta(cfg: dict) -> dict:
        return {"supported": _backend_supported(cfg, backend)}
    return meta


ENTRIES = (
    _E("cores", section=_PC, group="hardware", owner="ChipCapabilities/mapping",
       type=T.CORES, category=Category.BASIC, exposure="user", label="Core Types",
       doc="Core-type grid: per type max_axons x max_neurons x count (+ has_bias). "
           "Under hw_config_mode='search' the co-search discovers the grid.",
       relevant=R.when("hw_config_mode", in_=("fixed",)), provided_by="co_search"),
    _E("max_axons", section=_PC, group="hardware", owner="mapping/packing",
       type=T.INT, category=Category.DERIVED, derivation="derived", exposure="user",
       label="Max Axons",
       doc="Largest per-core axon count, derived from the core grid; a "
           "consistent explicit value is accepted, a contradicting one rejected.",
       derived_from=("cores",), why=_why_core_maximum("axon count"),
       provenance="derivation rule"),
    _E("max_neurons", section=_PC, group="hardware", owner="mapping/packing",
       type=T.INT, category=Category.DERIVED, derivation="derived", exposure="user",
       label="Max Neurons",
       doc="Largest per-core neuron count, derived from the core grid; a "
           "consistent explicit value is accepted, a contradicting one rejected.",
       derived_from=("cores",), why=_why_core_maximum("neuron count"),
       provenance="derivation rule"),
    _E("has_bias", section=_PC, group="hardware", owner="mapping/bias",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Hardware Bias",
       doc="Whether cores carry a bias lane; biasless platforms use compensation."),
    _E("target_tq", section=_PC, group="hardware", owner="activation_quantization",
       type=T.INT, category=Category.BASIC, exposure="user", label="Target Tq",
       effect="Activation quantization threshold groups",
       doc="Activation threshold-group count; must divide simulation_steps.",
       bounds=(1, None)),
    _E("simulation_steps", section=_PC, group="hardware", owner="SimulationRunner",
       type=T.INT, category=Category.BASIC, exposure="user", label="Simulation Steps",
       doc="Temporal resolution S: spiking cycles per forward.", bounds=(1, None)),
    _E("weight_bits", section=_PC, group="hardware", owner="weight_quantization",
       type=T.INT, category=Category.BASIC, exposure="user", label="Weight Precision",
       effect="Declares a quantized artifact (bits-driven WQ contract)",
       doc="Weight precision of the deployed artifact: quantized-N-bits "
           "(weight_bits declares a quantized artifact) or float weights "
           "(the vanilla assembly: pipeline_mode='vanilla' + "
           "weight_quantization=false, weight_bits then inert).",
       bounds=(1, 32)),
    _E("allow_coalescing", section=_PC, group="hardware", owner="MappingStrategy",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Allow Coalescing",
       doc="Capability: coalesce compatible soft cores into shared hard cores."),
    _E("allow_neuron_splitting", section=_PC, group="hardware", owner="MappingStrategy",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Allow Neuron Splitting",
       doc="Capability: split oversized layers across cores."),
    _E("allow_per_layer_s", section=_PC, group="hardware",
       owner="ChipCapabilities/TemporalAllocation", type=T.BOOL,
       category=Category.ADVANCED, exposure="user", label="Allow Per-layer S",
       effect="Capability gate for s_allocation explicit/budget",
       doc="Capability: per-cascade-depth temporal resolution."),
    _E("max_schedule_passes", section=_PC, group="mapping_strategy",
       owner="schedule_partitioner",
       type=T.INT, category=Category.ADVANCED, label="Max Schedule Passes",
       doc="Upper bound on multi-pass scheduling when single-pass packing fails.",
       bounds=(1, None), relevant=R.when_true("allow_scheduling")),
    _E("scheduling_latency_weight", section=_PC, group="mapping_strategy",
       owner="schedule_partitioner", type=T.FLOAT, category=Category.ADVANCED,
       label="Scheduling Latency Weight",
       doc="Latency term weight in the schedule-partitioner objective.",
       bounds=(0.0, None), relevant=R.when_true("allow_scheduling")),
    _E("search_space", section=_PC, group="co_search", owner="search/hw_search_space",
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="HW Search Space",
       doc="Hardware co-search bounds (core type counts, axon/neuron bounds, "
           "threshold groups).", relevant=R.when("hw_config_mode", in_=("search",)),
       promote_when=R.when("hw_config_mode", in_=("search",)),
       empty_means="the co-search's default bounds"),
    _E("allow_scheduling", group="mapping_strategy", owner="MappingStrategy/scheduler",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Allow Scheduling",
       effect="Multi-pass layout scheduling when single-pass packing fails",
       doc="Deployment option: time-multiplex core passes when the model "
           "exceeds the grid (how we choose to deploy, not a chip capability)."),
    _E("enable_nevresim_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Nevresim Simulation",
       doc="Whether the nevresim C++ cycle-simulator decision-parity probe "
           "runs. Recipe default: on where the backend supports the mode; "
           "declare false to skip the vehicle (a stored override); an "
           "explicit on for an unsupported mode is rejected.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "nevresim", "nevresim has no synchronized-window backend"),
       meta=_meta_backend_enable("nevresim"), provenance="ConversionPolicy recipe"),
    _E("nevresim_connectivity_mode", group="deployment_target", owner="nevresim_backend",
       type=T.ENUM, options=("runtime", "codegen"), category=Category.ADVANCED,
       label="Nevresim Connectivity Mode",
       doc="How chip connectivity reaches nevresim: runtime JSON or generated code.",
       relevant=R.when_true("enable_nevresim_simulation")),
    _E("enable_loihi_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Loihi Simulation",
       doc="Whether the Lava Loihi spike-parity gate runs. Recipe default: on "
           "where the backend supports the mode; declare false to skip the "
           "vehicle (a stored override); an explicit on for an unsupported "
           "mode is rejected.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "loihi", "Loihi/Lava only implements LIF dynamics"),
       meta=_meta_backend_enable("loihi"), provenance="ConversionPolicy recipe"),
    _E("loihi_parity_sample_index", group="deployment_target", owner="loihi_backend",
       type=T.INT, category=Category.ADVANCED, label="Loihi Parity Sample Index",
       doc="Test-set sample index for the HCM-vs-Lava spike-parity check.",
       bounds=(0, None), relevant=R.when_true("enable_loihi_simulation"),
       provenance="consumer frozen default", derived_default=_frozen(0),
       empty_means="the loihi step's frozen sample index 0"),
    _E("enable_sanafe_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="SANA-FE Simulation",
       doc="Whether the SANA-FE simulator (parity + energy/latency aggregates) "
           "runs. Recipe default: on where the backend supports the mode; "
           "declare false to skip the vehicle (a stored override); an "
           "explicit on for an unsupported mode is rejected.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "sanafe", "SANA-FE does not support this mode"),
       meta=_meta_backend_enable("sanafe"), provenance="ConversionPolicy recipe"),
    _E("sanafe_sample_count", group="deployment_target", owner="sanafe_backend",
       type=T.INT, category=Category.ADVANCED, exposure="user", label="SANA-FE Sample Count",
       doc="Deterministic test samples run through SANA-FE.", bounds=(1, None),
       relevant=R.when_true("enable_sanafe_simulation")),
    _E("sanafe_arch_preset", group="deployment_target", owner="sanafe_backend",
       type=T.ENUM, options=("loihi", "truenorth", "custom"), category=Category.ADVANCED,
       exposure="user", label="SANA-FE Arch Preset",
       doc="SANA-FE architecture YAML preset (custom uses sanafe_custom_arch_path).",
       relevant=R.when_true("enable_sanafe_simulation")),
    _E("sanafe_custom_arch_path", group="deployment_target", owner="sanafe_backend",
       type=T.PATH, category=Category.ADVANCED, exposure="user",
       label="SANA-FE Custom Arch Path",
       doc="Custom architecture YAML for sanafe_arch_preset='custom'.",
       relevant=R.all_of(R.when_true("enable_sanafe_simulation"),
                         R.when("sanafe_arch_preset", in_=("custom",)))),
    _E("sanafe_log_potential_trace", group="deployment_target", owner="sanafe_backend",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="SANA-FE Potential Trace",
       doc="Log membrane-potential traces during SANA-FE runs (large outputs).",
       relevant=R.when_true("enable_sanafe_simulation")),
    _E("max_simulation_samples", group="deployment_target", owner="SimulationRunner",
       type=T.INT, category=Category.BASIC, exposure="user", label="Max Simulation Samples",
       doc="Sample cap for simulator probes (binomial-noise accuracy reads); 0 = no cap.",
       bounds=(0, None), provenance="consumer frozen default", derived_default=_frozen(0),
       empty_means="0 — the runner probes without a sample cap"),
    _E("simulation_batch_count", group="deployment_target", owner="SimulationRunner",
       type=T.INT, category=Category.ADVANCED, label="Simulation Batch Count",
       doc="Batches per simulator probe run.", bounds=(1, None),
       provenance="consumer frozen default", derived_default=_frozen(None),
       empty_means="every test batch (the runner takes no batch cap)"),
    _E("deployment_metric_full_eval", group="deployment_target", owner="SCM identity read",
       type=T.BOOL, category=Category.ADVANCED, label="Deployment Metric Full Eval",
       doc="Use the full test set (not the probe subsample) for the deployed metric.",
       provenance="consumer frozen default", derived_default=_frozen(True)),
    _E("scm_degradation_tolerance", group="deployment_target", owner="soft_core_mapping",
       type=T.FLOAT, category=Category.ADVANCED, label="SCM Degradation Tolerance",
       doc="Retention tolerance of the SCM identity read. Absent, the SCM step "
           "installs no separate tolerance: the global degradation_tolerance governs.",
       bounds=(0.0, 1.0), provenance="derivation rule",
       derived_default=_scm_degradation_tolerance,
       empty_means="the global degradation_tolerance"),
    _E("nf_scm_parity_samples", group="deployment_target", owner="nf_scm_parity",
       type=T.INT, category=Category.ADVANCED, label="NF-SCM Parity Samples",
       doc="Validation inputs the NF<->SCM gate runs on; 0 disables the gate. The "
           "default is mode-aware: the serial per-neuron sweep needs 2, the batched "
           "cascaded decision gate needs 64 Bernoulli trials.",
       bounds=(0, None), provenance="derivation rule",
       derived_default=_nf_scm_parity_samples,
       empty_means="2, or 64 on the cascaded schedule"),
    _E("nf_scm_parity_atol", group="deployment_target", owner="nf_scm_parity",
       type=T.FLOAT, category=Category.ADVANCED, label="NF-SCM Parity Atol",
       doc="Absolute tolerance of the NF<->SCM output comparison.", bounds=(0.0, None),
       provenance="consumer frozen default", derived_default=_frozen(1e-6)),
    _E("nf_scm_parity_max_mismatch_fraction", group="deployment_target",
       owner="nf_scm_parity", type=T.FLOAT, category=Category.ADVANCED,
       label="NF-SCM Max Mismatch Fraction",
       doc="Accepted fraction of mismatching outputs (loose while continuous ttfs "
           "carries an uncalibrated residual; its wrong-dynamics signature is ~40%).",
       bounds=(0.0, 1.0),
       provenance="consumer frozen default", derived_default=_frozen(0.25)),
    _E("nf_scm_parity_min_agreement", group="deployment_target", owner="nf_scm_parity",
       type=T.FLOAT, category=Category.ADVANCED, label="NF-SCM Min Agreement",
       doc="Minimum decision agreement of the NF<->SCM gate (cascaded schedule).",
       bounds=(0.0, 1.0),
       provenance="consumer frozen default", derived_default=_frozen(0.98)),
    _E("scm_torch_sim_parity_check", group="deployment_target", owner="soft_core_mapping",
       type=T.BOOL, category=Category.ADVANCED, label="SCM-Torch Parity Check",
       doc="Enable the torch-vs-deployed-sim agreement check at SCM (a standing "
           "deployment-faithfulness gate: default on).",
       provenance="consumer frozen default", derived_default=_frozen(True)),
    _E("scm_torch_sim_parity_samples", group="deployment_target", owner="soft_core_mapping",
       type=T.INT, category=Category.ADVANCED, label="SCM-Torch Parity Samples",
       doc="Samples for the SCM torch-vs-sim agreement check; 0 disables the check.",
       bounds=(0, None), provenance="consumer frozen default", derived_default=_frozen(256)),
    _E("scm_torch_sim_parity_min_agreement", group="deployment_target",
       owner="soft_core_mapping", type=T.FLOAT, category=Category.ADVANCED,
       label="SCM-Torch Min Agreement",
       doc="Minimum agreement of the SCM torch-vs-sim check.", bounds=(0.0, 1.0),
       provenance="consumer frozen default", derived_default=_frozen(0.98)),
    _E("onchip_majority_gate", group="deployment_target", owner="certification",
       type=T.BOOL, category=Category.ADVANCED, label="On-chip Majority Gate",
       doc="Require the on-chip compute fraction to clear the majority gate.",
       provenance="consumer frozen default", derived_default=_frozen(True)),
    _E("onchip_majority_min_fraction", group="deployment_target", owner="certification",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Majority Min Fraction",
       doc="Minimum on-chip fraction for the majority gate.", bounds=(0.0, 1.0),
       provenance="consumer frozen default", derived_default=_frozen(0.2)),
    _E("onchip_majority_fraction", group="deployment_target", owner="campaign validity gate",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Majority Fraction",
       doc="Declared on-chip fraction the campaign enqueue gate certifies against "
           "(read from the DOCUMENT by the research campaign scheduler, not by the "
           "pipeline).",
       bounds=(0.0, 1.0), provenance="consumer frozen default", derived_default=_frozen(0.5)),
    _E("onchip_min_fraction", group="deployment_target", owner="campaign validity gate",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Min Fraction",
       doc="Validity floor on the on-chip compute fraction (read from the DOCUMENT "
           "by the research campaign scheduler, not by the pipeline).",
       bounds=(0.0, 1.0), provenance="consumer frozen default", derived_default=_frozen(0.2)),
    _E("capacity_gate", group="deployment_target", owner="certification",
       type=T.BOOL, category=Category.ADVANCED, label="Capacity Gate",
       doc="Reject configs whose peak-phase footprint exceeds chip capacity.",
       provenance="consumer frozen default", derived_default=_frozen(True)),
)

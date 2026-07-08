"""Registry entries: hardware platform constraints and deployment-target keys."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_PC = "platform_constraints"


def _why_core_maximum(dim: str):
    def why(cfg: dict) -> str:
        return f"largest per-core {dim} across the declared core types"
    return why


def _why_backend_enable(backend: str, off_reason: str):
    """WHY text for a ConversionPolicy-owned backend enable."""
    def why(cfg: dict) -> str:
        key = f"enable_{backend}_simulation"
        mode = cfg.get("spiking_mode")
        if cfg.get(key):
            return f"on — ConversionPolicy runs the {backend} gate for {mode!r}"
        return f"off — {off_reason} (spiking_mode={mode!r})"
    return why


ENTRIES = (
    _E("cores", section=_PC, group="hardware", owner="ChipCapabilities/mapping",
       type=T.CORES, category=Category.BASIC, exposure="user", label="Core Types",
       doc="Core-type grid: per type max_axons x max_neurons x count (+ has_bias)."),
    _E("max_axons", section=_PC, group="hardware", owner="mapping/packing",
       type=T.INT, category=Category.DERIVED, derivation="derived", exposure="user",
       label="Max Axons",
       doc="Largest per-core axon count, derived from the core grid; a "
           "consistent explicit value is accepted, a contradicting one rejected.",
       derived_from=("cores",), why=_why_core_maximum("axon count")),
    _E("max_neurons", section=_PC, group="hardware", owner="mapping/packing",
       type=T.INT, category=Category.DERIVED, derivation="derived", exposure="user",
       label="Max Neurons",
       doc="Largest per-core neuron count, derived from the core grid; a "
           "consistent explicit value is accepted, a contradicting one rejected.",
       derived_from=("cores",), why=_why_core_maximum("neuron count")),
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
       type=T.INT, category=Category.BASIC, exposure="user", label="Weight Bits",
       effect="Declares a quantized artifact (bits-driven WQ contract)",
       doc="Weight precision of the deployed artifact.", bounds=(1, 32)),
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
    _E("max_schedule_passes", section=_PC, group="hardware", owner="schedule_partitioner",
       type=T.INT, category=Category.ADVANCED, label="Max Schedule Passes",
       doc="Upper bound on multi-pass scheduling when single-pass packing fails.",
       bounds=(1, None), relevant=R.when_true("allow_scheduling")),
    _E("scheduling_latency_weight", section=_PC, group="hardware",
       owner="schedule_partitioner", type=T.FLOAT, category=Category.ADVANCED,
       label="Scheduling Latency Weight",
       doc="Latency term weight in the schedule-partitioner objective.",
       bounds=(0.0, None), relevant=R.when_true("allow_scheduling")),
    _E("search_space", section=_PC, group="hardware", owner="search/hw_search_space",
       type=T.JSON, category=Category.ADVANCED, exposure="user", label="HW Search Space",
       doc="Hardware co-search bounds (core type counts, axon/neuron bounds, "
           "threshold groups).", relevant=R.when("hw_config_mode", in_=("search",)),
       promote_when=R.when("hw_config_mode", in_=("search",)),
       empty_means="the co-search's default bounds"),
    _E("allow_scheduling", group="hardware", owner="MappingStrategy/scheduler",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Allow Scheduling",
       effect="Multi-pass layout scheduling when single-pass packing fails",
       doc="Capability: time-multiplex core passes when the model exceeds the grid."),
    _E("enable_nevresim_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Nevresim Simulation",
       doc="Whether the nevresim C++ cycle-simulator decision-parity probe "
           "runs. Owned by the ConversionPolicy mode recipe (capability-"
           "derived); an explicit value is overwritten, so none is stored.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "nevresim", "nevresim has no synchronized-window backend"),
       declarable=False),
    _E("nevresim_connectivity_mode", group="deployment_target", owner="nevresim_backend",
       type=T.ENUM, options=("runtime", "codegen"), category=Category.ADVANCED,
       label="Nevresim Connectivity Mode",
       doc="How chip connectivity reaches nevresim: runtime JSON or generated code.",
       relevant=R.when_true("enable_nevresim_simulation")),
    _E("enable_loihi_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Loihi Simulation",
       doc="Whether the Lava Loihi spike-parity gate runs. Owned by the "
           "ConversionPolicy mode recipe (capability-derived); an explicit "
           "value is overwritten, so none is stored.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "loihi", "Loihi/Lava only implements LIF dynamics"),
       declarable=False),
    _E("loihi_parity_sample_index", group="deployment_target", owner="loihi_backend",
       type=T.INT, category=Category.ADVANCED, label="Loihi Parity Sample Index",
       doc="Test-set sample index for the HCM-vs-Lava spike-parity check.",
       bounds=(0, None), relevant=R.when_true("enable_loihi_simulation")),
    _E("enable_sanafe_simulation", group="deployment_target",
       owner="ConversionPolicy/backend_registry", type=T.BOOL,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="SANA-FE Simulation",
       doc="Whether the SANA-FE simulator (parity + energy/latency aggregates) "
           "runs. Owned by the ConversionPolicy mode recipe (capability-"
           "derived); an explicit value is overwritten, so none is stored.",
       derived_from=("spiking_mode", "ttfs_cycle_schedule"),
       why=_why_backend_enable(
           "sanafe", "SANA-FE does not support this mode"),
       declarable=False),
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
       relevant=R.when_true("enable_sanafe_simulation")),
    _E("sanafe_log_potential_trace", group="deployment_target", owner="sanafe_backend",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="SANA-FE Potential Trace",
       doc="Log membrane-potential traces during SANA-FE runs (large outputs).",
       relevant=R.when_true("enable_sanafe_simulation")),
    _E("max_simulation_samples", group="deployment_target", owner="SimulationRunner",
       type=T.INT, category=Category.BASIC, exposure="user", label="Max Simulation Samples",
       doc="Sample cap for simulator probes (binomial-noise accuracy reads).",
       bounds=(1, None)),
    _E("simulation_batch_count", group="deployment_target", owner="SimulationRunner",
       type=T.INT, category=Category.ADVANCED, label="Simulation Batch Count",
       doc="Batches per simulator probe run.", bounds=(1, None)),
    _E("deployment_metric_full_eval", group="deployment_target", owner="SCM identity read",
       type=T.BOOL, category=Category.ADVANCED, label="Deployment Metric Full Eval",
       doc="Use the full test set (not the probe subsample) for the deployed metric."),
    _E("scm_degradation_tolerance", group="deployment_target", owner="soft_core_mapping",
       type=T.FLOAT, category=Category.ADVANCED, label="SCM Degradation Tolerance",
       doc="Retention tolerance of the SCM identity read.", bounds=(0.0, 1.0)),
    _E("nf_scm_parity_samples", group="deployment_target", owner="nf_scm_parity",
       type=T.INT, category=Category.ADVANCED, label="NF-SCM Parity Samples",
       doc="Samples for the NF<->SCM bit-exactness gate.", bounds=(1, None)),
    _E("nf_scm_parity_samples_cascaded", group="deployment_target", owner="nf_scm_parity",
       type=T.INT, category=Category.ADVANCED, label="NF-SCM Parity Samples (cascaded)",
       doc="Sample count override for the cascaded schedule.", bounds=(1, None)),
    _E("nf_scm_parity_atol", group="deployment_target", owner="nf_scm_parity",
       type=T.FLOAT, category=Category.ADVANCED, label="NF-SCM Parity Atol",
       doc="Absolute tolerance of the NF<->SCM output comparison.", bounds=(0.0, None)),
    _E("nf_scm_parity_max_mismatch_fraction", group="deployment_target",
       owner="nf_scm_parity", type=T.FLOAT, category=Category.ADVANCED,
       label="NF-SCM Max Mismatch Fraction",
       doc="Accepted fraction of mismatching outputs.", bounds=(0.0, 1.0)),
    _E("nf_scm_parity_min_agreement", group="deployment_target", owner="nf_scm_parity",
       type=T.FLOAT, category=Category.ADVANCED, label="NF-SCM Min Agreement",
       doc="Minimum decision agreement of the NF<->SCM gate.", bounds=(0.0, 1.0)),
    _E("scm_torch_sim_parity_check", group="deployment_target", owner="soft_core_mapping",
       type=T.BOOL, category=Category.ADVANCED, label="SCM-Torch Parity Check",
       doc="Enable the torch-vs-deployed-sim agreement check at SCM."),
    _E("scm_torch_sim_parity_samples", group="deployment_target", owner="soft_core_mapping",
       type=T.INT, category=Category.ADVANCED, label="SCM-Torch Parity Samples",
       doc="Samples for the SCM torch-vs-sim agreement check.", bounds=(1, None)),
    _E("scm_torch_sim_parity_min_agreement", group="deployment_target",
       owner="soft_core_mapping", type=T.FLOAT, category=Category.ADVANCED,
       label="SCM-Torch Min Agreement",
       doc="Minimum agreement of the SCM torch-vs-sim check.", bounds=(0.0, 1.0)),
    _E("onchip_majority_gate", group="deployment_target", owner="certification",
       type=T.BOOL, category=Category.ADVANCED, label="On-chip Majority Gate",
       doc="Require the on-chip compute fraction to clear the majority gate."),
    _E("onchip_majority_min_fraction", group="deployment_target", owner="certification",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Majority Min Fraction",
       doc="Minimum on-chip fraction for the majority gate.", bounds=(0.0, 1.0)),
    _E("onchip_majority_fraction", group="deployment_target", owner="certification",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Majority Fraction",
       doc="Declared on-chip fraction the majority gate certifies against.",
       bounds=(0.0, 1.0)),
    _E("onchip_min_fraction", group="deployment_target", owner="certification",
       type=T.FLOAT, category=Category.ADVANCED, label="On-chip Min Fraction",
       doc="Validity floor on the on-chip compute fraction.", bounds=(0.0, 1.0)),
    _E("capacity_gate", group="deployment_target", owner="certification",
       type=T.BOOL, category=Category.ADVANCED, label="Capacity Gate",
       doc="Reject configs whose peak-phase footprint exceeds chip capacity."),
)

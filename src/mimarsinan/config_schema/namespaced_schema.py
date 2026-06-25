"""Namespaced config schema: §2 concern groups + flat<->namespaced translation shim.

V8 (docs/DESIGN_GOALS_and_refactoring_vectors.md): the runtime config stays a FLAT
dict (the byte-identical SSOT every pipeline step reads), but each flat deployment
key is given an owning concern (the §2 input-axis groups) and recorded provenance
(owner / derivation). A single ``LEGACY_KEY_TABLE`` is the one translation table:
``to_namespaced`` projects a flat dict into the nested concern groups and ``to_flat``
inverts it. The pair round-trips byte-identically for every registered key, so
existing flat configs resolve identically while new code can read the concern view.

Strangler-fig status: the translation shim + provenance registry are the new seam;
the ``hardware`` group is migrated end-to-end (its keys carry namespaced paths and a
byte-identical accessor); the remaining groups are registered with provenance but
still consumed flat by their owners. **Consumer-side resolution is owned by V1's
``DeploymentPlan``, not this module** — this is the REGISTRY/PROVENANCE layer (it
records each key's owner + derivation and gives the concern view); ``DeploymentPlan``
is where steps read resolved decisions. Provenance is real: derived keys (written by
``deployment_derivation`` / ``DeploymentPipeline``) and runtime keys (device / data
provider) are tagged ``derived`` / ``runtime``; only declared defaults are
``default``. See ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)

# ── Concern groups (the §2 input-axis groups) ───────────────────────────────
# Each id is the top-level namespace a key projects under; the title mirrors the
# §2 group label so the schema reads like the spec.
CONCERN_GROUPS: Tuple[Dict[str, str], ...] = (
    {"id": "workload", "title": "Workload",
     "subtitle": "Model, dataset, weight init, pruning"},
    {"id": "spiking", "title": "Spiking semantics",
     "subtitle": "Firing/sync/encoding/thresholding/spike-gen/bias modes"},
    {"id": "hardware", "title": "Hardware platform / capabilities",
     "subtitle": "Core grid, weight quantization, capability gates"},
    {"id": "conversion", "title": "Conversion process",
     "subtitle": "Activation quant, calibration health, optimization driver"},
    {"id": "tuning", "title": "Adaptation & tuning controller",
     "subtitle": "Rate-search budget, rollback, recovery, stabilization knobs"},
    {"id": "training", "title": "Training",
     "subtitle": "Learning rate, epochs, recipes"},
    {"id": "deployment_target", "title": "Deployment target",
     "subtitle": "Simulation backends and acceptance/parity gates"},
    {"id": "run", "title": "Run / runtime",
     "subtitle": "Run identity and pipeline-resolved values"},
)

_VALID_GROUP_IDS = frozenset(g["id"] for g in CONCERN_GROUPS)

# Derivation provenance: where a key's resolved value comes from.
#   "default"  — DEFAULT_DEPLOYMENT_PARAMETERS / DEFAULT_PLATFORM_CONSTRAINTS
#   "preset"   — injected by a PIPELINE_MODE_PRESETS preset (setdefault)
#   "derived"  — computed by derive_deployment_parameters from other keys
#   "runtime"  — resolved by the pipeline at start (device / input shape / …)
_VALID_DERIVATIONS = frozenset({"default", "preset", "derived", "runtime"})


@dataclass(frozen=True)
class KeySpec:
    """Provenance + namespacing for one flat deployment/platform key.

    ``flat_key`` is the legacy runtime key (the SSOT every step reads today).
    ``group`` / ``name`` give the namespaced path ``group.name``. ``owner`` records
    the subsystem that consumes the resolved value; ``derivation`` records where the
    value comes from (see _VALID_DERIVATIONS).
    """

    flat_key: str
    group: str
    name: str
    owner: str
    derivation: str

    def __post_init__(self) -> None:
        if self.group not in _VALID_GROUP_IDS:
            raise ValueError(f"KeySpec {self.flat_key!r}: unknown group {self.group!r}")
        if self.derivation not in _VALID_DERIVATIONS:
            raise ValueError(
                f"KeySpec {self.flat_key!r}: unknown derivation {self.derivation!r}"
            )

    @property
    def namespaced_path(self) -> str:
        return f"{self.group}.{self.name}"


def _spec(flat_key: str, group: str, owner: str, derivation: str = "default",
          name: Optional[str] = None) -> KeySpec:
    """KeySpec builder; ``name`` defaults to the flat key (identity rename)."""
    return KeySpec(
        flat_key=flat_key,
        group=group,
        name=name if name is not None else flat_key,
        owner=owner,
        derivation=derivation,
    )


# ── The provenance registry: one KeySpec per flat key ───────────────────────
# Owner strings name the consuming subsystem (grep target for the next reviewer).
_KEY_SPECS: Tuple[KeySpec, ...] = (
    # Workload ──────────────────────────────────────────────────────────────
    _spec("model_config_mode", "workload", "deployment_specs/search_mode"),
    _spec("hw_config_mode", "workload", "deployment_specs/search_mode"),
    _spec("spike_encoding_seed", "workload", "spike_generation"),
    # Spiking semantics ──────────────────────────────────────────────────────
    _spec("spiking_mode", "spiking", "SpikingDeploymentContract"),
    _spec("ttfs_cycle_schedule", "spiking", "SpikingDeploymentContract"),
    _spec("cycle_accurate_lif_forward", "spiking", "lif_adaptation"),
    # Hardware platform / capabilities (MIGRATED group) ──────────────────────
    _spec("cores", "hardware", "ChipCapabilities/mapping"),
    _spec("target_tq", "hardware", "activation_quantization"),
    _spec("simulation_steps", "hardware", "SimulationRunner"),
    _spec("weight_bits", "hardware", "weight_quantization"),
    _spec("allow_coalescing", "hardware", "MappingStrategy"),
    _spec("allow_neuron_splitting", "hardware", "MappingStrategy"),
    _spec("allow_per_layer_s", "hardware", "ChipCapabilities/TemporalAllocation"),
    _spec("allow_scheduling", "hardware", "MappingStrategy/scheduler"),
    _spec("max_schedule_passes", "hardware", "schedule_partitioner"),
    _spec("scheduling_latency_weight", "hardware", "schedule_partitioner"),
    # Conversion process — activation quant + calibration health + driver ─────
    _spec("activation_scale_quantile", "conversion", "activation_analysis"),
    _spec("ttfs_genuine_annealed_ramp", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ramp_alpha_min", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ramp_alpha_max", "conversion", "ttfs_adaptation"),
    _spec("ttfs_scale_aware_boundaries", "conversion", "ttfs_adaptation"),
    _spec("ttfs_genuine_blend_ramp", "conversion", "ttfs_adaptation"),
    _spec("ttfs_distmatch_bias_iters", "conversion", "ttfs_adaptation"),
    _spec("ttfs_distmatch_bias_eta", "conversion", "ttfs_adaptation"),
    _spec("ttfs_distmatch_quantile", "conversion", "ttfs_adaptation"),
    _spec("ttfs_genuine_blend_ce_alpha", "conversion", "ttfs_adaptation"),
    _spec("ttfs_boundary_surrogate", "conversion", "ttfs_adaptation"),
    _spec("ttfs_boundary_surrogate_temp", "conversion", "ttfs_adaptation"),
    _spec("ttfs_gain_correction", "conversion", "ttfs_adaptation"),
    _spec("ttfs_gain_correction_rule", "conversion", "ttfs_adaptation"),
    _spec("ttfs_gain_correction_c", "conversion", "ttfs_adaptation"),
    _spec("ttfs_gain_correction_ramp", "conversion", "ttfs_adaptation"),
    _spec("ttfs_theta_cotrain", "conversion", "ttfs_adaptation"),
    _spec("ttfs_staircase_ste", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ste_mix", "conversion", "ttfs_adaptation"),
    _spec("ttfs_staircase_ste_fast", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ste_steps", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ste_w_lr", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ste_theta_lr", "conversion", "ttfs_adaptation"),
    _spec("ttfs_ste_init_frac", "conversion", "ttfs_adaptation"),
    _spec("ttfs_genuine_blend_fast", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast_steps_per_rate", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast_rates", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast_stabilize_steps", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast_lr_eta_min", "conversion", "ttfs_adaptation"),
    _spec("ttfs_blend_fast_ste_refine", "conversion", "ttfs_adaptation"),
    _spec("lif_blend_fast", "conversion", "lif_adaptation"),
    _spec("lif_blend_fast_steps_per_rate", "conversion", "lif_adaptation"),
    _spec("lif_blend_fast_rates", "conversion", "lif_adaptation"),
    _spec("lif_blend_fast_lr_eta_min", "conversion", "lif_adaptation"),
    _spec("lif_blend_fast_stabilize_steps", "conversion", "lif_adaptation"),
    _spec("lif_distmatch", "conversion", "lif_adaptation"),
    _spec("lif_distmatch_bias_iters", "conversion", "lif_adaptation"),
    _spec("lif_distmatch_bias_eta", "conversion", "lif_adaptation"),
    _spec("lif_distmatch_cal_batches", "conversion", "lif_adaptation"),
    _spec("enable_training_noise", "conversion", "noise_adaptation"),
    # Adaptation & tuning controller ──────────────────────────────────────────
    _spec("tuning_budget_scale", "tuning", "AdaptationManager"),
    _spec("tuner_target_floor_ratio", "tuning", "AdaptationManager"),
    _spec("degradation_tolerance", "tuning", "AccuracyBudget"),
    _spec("checkpoint_scope", "tuning", "CheckpointGuard"),
    _spec("checkpoint_location", "tuning", "CheckpointGuard"),
    _spec("tuning_use_paired_sensor", "tuning", "rollback_sensor"),
    _spec("k_commit", "tuning", "rollback_sensor"),
    _spec("paired_confirm_batches", "tuning", "rollback_sensor"),
    _spec("global_budget", "tuning", "rollback_sensor"),
    _spec("tuning_full_transform_probe", "tuning", "AdaptationManager"),
    _spec("tuning_enable_characterization", "tuning", "AdaptationManager"),
    _spec("tuning_characterization_grid", "tuning", "AdaptationManager"),
    _spec("tuning_refind_lr_on_miss", "tuning", "AdaptationManager"),
    _spec("tuning_recovery_lr_plateau", "tuning", "AdaptationManager"),
    _spec("tuning_recovery_lr_plateau_factor", "tuning", "AdaptationManager"),
    _spec("tuning_recovery_lr_plateau_reductions", "tuning", "AdaptationManager"),
    _spec("tuning_rollback_ratchet", "tuning", "rollback_sensor"),
    _spec("tuning_rollback_cumulative_bound", "tuning", "rollback_sensor"),
    _spec("tuning_stabilization_bounded", "tuning", "AdaptationManager"),
    _spec("tuning_stabilization_ratio", "tuning", "AdaptationManager"),
    _spec("tuning_tight_plateau", "tuning", "AdaptationManager"),
    _spec("tuning_recovery_check_divisor", "tuning", "AdaptationManager"),
    _spec("tuning_keepbest_certified", "tuning", "rollback_sensor"),
    _spec("tuning_recipe_recovery", "tuning", "AdaptationManager"),
    _spec("optimization_driver", "tuning", "OptimizationDriver"),
    # Per-layer-S temporal allocation (EW1 RESERVED): the Wizard declares the intent;
    # the per-depth S map is derived by the ConversionPolicy keystone (TemporalAllocation).
    _spec("s_allocation", "conversion", "TemporalAllocation"),
    _spec("s_allocation_explicit", "conversion", "TemporalAllocation"),
    _spec("s_allocation_budget", "conversion", "TemporalAllocation"),
    # Training ────────────────────────────────────────────────────────────────
    _spec("lr", "training", "training_loop"),
    _spec("lr_range_min", "training", "lr_finder"),
    _spec("lr_range_max", "training", "lr_finder"),
    _spec("training_epochs", "training", "training_loop"),
    _spec("training_recipe", "training", "training_loop"),
    _spec("tuning_recipe", "training", "AdaptationManager"),
    # Deployment target — backends + acceptance gates ─────────────────────────
    _spec("enable_nevresim_simulation", "deployment_target", "backend_registry"),
    _spec("nevresim_connectivity_mode", "deployment_target", "nevresim_backend"),
    _spec("enable_loihi_simulation", "deployment_target", "backend_registry"),
    _spec("enable_sanafe_simulation", "deployment_target", "backend_registry"),
    _spec("sanafe_sample_count", "deployment_target", "sanafe_backend"),
    _spec("sanafe_arch_preset", "deployment_target", "sanafe_backend"),
    _spec("sanafe_custom_arch_path", "deployment_target", "sanafe_backend"),
    _spec("sanafe_log_potential_trace", "deployment_target", "sanafe_backend"),
)

# ── Derived keys: written by a derivation pass, NOT in the defaults dict ──────
# These have no default value of their own — their value is *computed* from other
# keys. ``deployment_derivation.derive_deployment_parameters`` (wizard parity)
# writes ``pipeline_mode`` / ``activation_quantization`` / ``weight_quantization``
# from ``spiking_mode`` × ``weight_quantization``; ``DeploymentPipeline.__init__``
# resolves ``firing_mode`` / ``spike_generation_mode`` / ``thresholding_mode`` from
# the spiking mode via ``setdefault``. Tagged ``derived`` so provenance is real.
# (The SSOT for the conversion-trio matches ``display_view_meta.DERIVED_KEYS``.)
_DERIVED_KEY_SPECS: Tuple[KeySpec, ...] = (
    _spec("pipeline_mode", "run", "deployment_derivation", derivation="derived"),
    _spec("activation_quantization", "conversion",
          "deployment_derivation/activation_analysis", derivation="derived"),
    _spec("weight_quantization", "conversion",
          "deployment_derivation/weight_quantization", derivation="derived"),
    _spec("firing_mode", "spiking", "DeploymentPipeline", derivation="derived"),
    _spec("spike_generation_mode", "spiking", "DeploymentPipeline",
          derivation="derived"),
    _spec("thresholding_mode", "spiking", "DeploymentPipeline", derivation="derived"),
)

# ── Runtime keys: resolved by the pipeline at start (device / data-provider) ──
# No default and not derivable from config alone — they need the live environment
# (device probe) or the data provider (input shape / class count). Tagged
# ``runtime`` (mirrors ``display_view_meta.RUNTIME_KEYS``).
_RUNTIME_KEY_SPECS: Tuple[KeySpec, ...] = (
    _spec("device", "run", "DeploymentPipeline/select_device", derivation="runtime"),
    _spec("input_shape", "run", "DeploymentPipeline/data_provider",
          derivation="runtime"),
    _spec("input_size", "run", "DeploymentPipeline/data_provider",
          derivation="runtime"),
    _spec("num_classes", "run", "DeploymentPipeline/data_provider",
          derivation="runtime"),
)

_ALL_KEY_SPECS: Tuple[KeySpec, ...] = (
    _KEY_SPECS + _DERIVED_KEY_SPECS + _RUNTIME_KEY_SPECS
)

# flat_key -> KeySpec
KEY_SPECS: Dict[str, KeySpec] = {s.flat_key: s for s in _ALL_KEY_SPECS}

# ── The one translation table: legacy flat key <-> namespaced (group, name) ──
LEGACY_KEY_TABLE: Dict[str, str] = {
    s.flat_key: s.namespaced_path for s in _ALL_KEY_SPECS
}

# Inverse, validated for uniqueness at import (the shim must be a bijection).
NAMESPACED_KEY_TABLE: Dict[str, str] = {}
for _flat, _path in LEGACY_KEY_TABLE.items():
    if _path in NAMESPACED_KEY_TABLE:
        raise ValueError(
            f"Namespaced path collision: {_path!r} from {_flat!r} and "
            f"{NAMESPACED_KEY_TABLE[_path]!r}"
        )
    NAMESPACED_KEY_TABLE[_path] = _flat


def to_namespaced(flat: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Project a flat config into nested concern groups.

    Registered keys land under ``group -> name``; unregistered keys are passed
    through unchanged under the ``run`` group so no data is dropped (round-trips
    via ``to_flat``).
    """
    nested: Dict[str, Dict[str, Any]] = {}
    for key, value in flat.items():
        spec = KEY_SPECS.get(key)
        if spec is None:
            nested.setdefault("run", {})[key] = value
        else:
            nested.setdefault(spec.group, {})[spec.name] = value
    return nested


def to_flat(nested: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Invert ``to_namespaced``: flatten concern groups back to legacy flat keys."""
    flat: Dict[str, Any] = {}
    for group, members in nested.items():
        if not isinstance(members, Mapping):
            raise ValueError(f"Group {group!r} must map to a dict, got {type(members)}")
        for name, value in members.items():
            path = f"{group}.{name}"
            flat_key = NAMESPACED_KEY_TABLE.get(path)
            if flat_key is None:
                # Pass-through (unregistered) key — restore under its bare name.
                if name in flat:
                    raise ValueError(f"Flatten collision on pass-through key {name!r}")
                flat[name] = value
            else:
                flat[flat_key] = value
    return flat


def provenance_table() -> Dict[str, Dict[str, str]]:
    """Return {flat_key: {group, name, owner, derivation, namespaced_path}}."""
    return {
        s.flat_key: {
            "group": s.group,
            "name": s.name,
            "owner": s.owner,
            "derivation": s.derivation,
            "namespaced_path": s.namespaced_path,
        }
        for s in _ALL_KEY_SPECS
    }


def registered_flat_keys() -> frozenset:
    """All flat keys carrying a KeySpec (registered in the provenance table)."""
    return frozenset(KEY_SPECS)


def keys_with_derivation(derivation: str) -> frozenset:
    """Flat keys whose recorded provenance is ``derivation`` (see _VALID_DERIVATIONS)."""
    if derivation not in _VALID_DERIVATIONS:
        raise ValueError(f"unknown derivation {derivation!r}")
    return frozenset(k for k, s in KEY_SPECS.items() if s.derivation == derivation)


def unregistered_default_keys() -> frozenset:
    """Default flat keys (deployment + platform) that have no KeySpec yet.

    These are still consumed flat; they are the strangler-fig remainder to register.
    """
    defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
    return frozenset(defaults - set(KEY_SPECS))

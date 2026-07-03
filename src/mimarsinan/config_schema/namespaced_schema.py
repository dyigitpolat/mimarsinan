"""Namespaced config schema: concern groups + flat<->namespaced translation shim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)

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

_VALID_DERIVATIONS = frozenset({"default", "preset", "derived", "runtime"})

_VALID_EXPOSURES = frozenset({"user", "derived", "system", "runtime"})


@dataclass(frozen=True)
class KeySpec:
    """Provenance + namespacing for one flat deployment/platform key.

    ``flat_key`` is the runtime SSOT key; ``group.name`` is the namespaced
    path; ``owner``/``derivation`` record the consumer and value source.
    """

    flat_key: str
    group: str
    name: str
    owner: str
    derivation: str
    exposure: str = "system"

    def __post_init__(self) -> None:
        if self.group not in _VALID_GROUP_IDS:
            raise ValueError(f"KeySpec {self.flat_key!r}: unknown group {self.group!r}")
        if self.derivation not in _VALID_DERIVATIONS:
            raise ValueError(
                f"KeySpec {self.flat_key!r}: unknown derivation {self.derivation!r}"
            )
        if self.exposure not in _VALID_EXPOSURES:
            raise ValueError(
                f"KeySpec {self.flat_key!r}: unknown exposure {self.exposure!r}"
            )

    @property
    def namespaced_path(self) -> str:
        return f"{self.group}.{self.name}"


def _spec(flat_key: str, group: str, owner: str, derivation: str = "default",
          name: Optional[str] = None, exposure: str = "system") -> KeySpec:
    """KeySpec builder; ``name`` defaults to the flat key (identity rename)."""
    return KeySpec(
        flat_key=flat_key,
        group=group,
        name=name if name is not None else flat_key,
        owner=owner,
        derivation=derivation,
        exposure=exposure,
    )


_KEY_SPECS: Tuple[KeySpec, ...] = (
    _spec("model_config_mode", "workload", "deployment_specs/search_mode",
          exposure="user"),
    _spec("hw_config_mode", "workload", "deployment_specs/search_mode",
          exposure="user"),
    _spec("spike_encoding_seed", "workload", "spike_generation"),
    _spec("spiking_mode", "spiking", "SpikingDeploymentContract", exposure="user"),
    _spec("ttfs_cycle_schedule", "spiking", "SpikingDeploymentContract",
          exposure="user"),
    _spec("cycle_accurate_lif_forward", "spiking", "lif_adaptation",
          exposure="user"),
    _spec("cores", "hardware", "ChipCapabilities/mapping", exposure="user"),
    _spec("target_tq", "hardware", "activation_quantization", exposure="user"),
    _spec("simulation_steps", "hardware", "SimulationRunner", exposure="user"),
    _spec("weight_bits", "hardware", "weight_quantization", exposure="user"),
    _spec("allow_coalescing", "hardware", "MappingStrategy", exposure="user"),
    _spec("allow_neuron_splitting", "hardware", "MappingStrategy",
          exposure="user"),
    _spec("allow_per_layer_s", "hardware", "ChipCapabilities/TemporalAllocation",
          exposure="user"),
    _spec("allow_scheduling", "hardware", "MappingStrategy/scheduler",
          exposure="user"),
    _spec("max_schedule_passes", "hardware", "schedule_partitioner"),
    _spec("scheduling_latency_weight", "hardware", "schedule_partitioner"),
    _spec("activation_scale_quantile", "conversion", "activation_analysis"),
    _spec("ttfs_genuine_blend_ce_alpha", "conversion", "ttfs_adaptation"),
    _spec("enable_training_noise", "conversion", "noise_adaptation"),
    _spec("tuning_budget_scale", "tuning", "AdaptationManager", exposure="user"),
    _spec("tuning_budget_scale_ramp_steps", "tuning", "AdaptationManager",
          exposure="user"),
    _spec("tuner_target_floor_ratio", "tuning", "AdaptationManager"),
    _spec("degradation_tolerance", "tuning", "AccuracyBudget", exposure="user"),
    _spec("checkpoint_scope", "tuning", "CheckpointGuard"),
    _spec("checkpoint_location", "tuning", "CheckpointGuard"),
    _spec("tuning_use_paired_sensor", "tuning", "rollback_sensor"),
    _spec("k_commit", "tuning", "rollback_sensor"),
    _spec("paired_confirm_batches", "tuning", "rollback_sensor"),
    _spec("global_budget", "tuning", "rollback_sensor"),
    _spec("s_allocation", "conversion", "TemporalAllocation", exposure="user"),
    _spec("s_allocation_explicit", "conversion", "TemporalAllocation",
          exposure="user"),
    _spec("s_allocation_budget", "conversion", "TemporalAllocation",
          exposure="user"),
    _spec("lr", "training", "training_loop", exposure="user"),
    _spec("lr_range_min", "training", "lr_finder"),
    _spec("lr_range_max", "training", "lr_finder"),
    _spec("kd_ce_alpha", "training", "training_loop/distillation"),
    _spec("kd_temperature", "training", "training_loop/distillation"),
    _spec("training_epochs", "training", "training_loop", exposure="user"),
    _spec("training_recipe", "training", "training_loop", exposure="user"),
    _spec("tuning_recipe", "training", "AdaptationManager", exposure="user"),
    _spec("enable_nevresim_simulation", "deployment_target", "backend_registry",
          exposure="user"),
    _spec("nevresim_connectivity_mode", "deployment_target", "nevresim_backend"),
    _spec("enable_loihi_simulation", "deployment_target", "backend_registry",
          exposure="user"),
    _spec("enable_sanafe_simulation", "deployment_target", "backend_registry",
          exposure="user"),
    _spec("sanafe_sample_count", "deployment_target", "sanafe_backend",
          exposure="user"),
    _spec("sanafe_arch_preset", "deployment_target", "sanafe_backend",
          exposure="user"),
    _spec("sanafe_custom_arch_path", "deployment_target", "sanafe_backend",
          exposure="user"),
    _spec("sanafe_log_potential_trace", "deployment_target", "sanafe_backend",
          exposure="user"),
)

_DERIVED_KEY_SPECS: Tuple[KeySpec, ...] = (
    _spec("pipeline_mode", "run", "deployment_derivation", derivation="derived",
          exposure="derived"),
    _spec("activation_quantization", "conversion",
          "deployment_derivation/activation_analysis", derivation="derived",
          exposure="derived"),
    _spec("weight_quantization", "conversion",
          "deployment_derivation/weight_quantization", derivation="derived",
          exposure="user"),
    _spec("firing_mode", "spiking", "DeploymentPipeline", derivation="derived",
          exposure="derived"),
    _spec("spike_generation_mode", "spiking", "DeploymentPipeline",
          derivation="derived", exposure="derived"),
    _spec("thresholding_mode", "spiking", "DeploymentPipeline", derivation="derived",
          exposure="derived"),
)

_RUNTIME_KEY_SPECS: Tuple[KeySpec, ...] = (
    _spec("device", "run", "DeploymentPipeline/select_device", derivation="runtime",
          exposure="runtime"),
    _spec("input_shape", "run", "DeploymentPipeline/data_provider",
          derivation="runtime", exposure="runtime"),
    _spec("input_size", "run", "DeploymentPipeline/data_provider",
          derivation="runtime", exposure="runtime"),
    _spec("num_classes", "run", "DeploymentPipeline/data_provider",
          derivation="runtime", exposure="runtime"),
)

_ALL_KEY_SPECS: Tuple[KeySpec, ...] = (
    _KEY_SPECS + _DERIVED_KEY_SPECS + _RUNTIME_KEY_SPECS
)

KEY_SPECS: Dict[str, KeySpec] = {s.flat_key: s for s in _ALL_KEY_SPECS}

LEGACY_KEY_TABLE: Dict[str, str] = {
    s.flat_key: s.namespaced_path for s in _ALL_KEY_SPECS
}

# The flat<->namespaced shim must be a bijection; the inverse is validated for uniqueness at import.
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

    Registered keys land under ``group -> name``; unregistered keys pass
    through under ``run`` so nothing is dropped (round-trips via ``to_flat``).
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
            "exposure": s.exposure,
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


def keys_with_exposure(exposure: str) -> frozenset:
    """Flat keys with the requested persistence exposure."""
    if exposure not in _VALID_EXPOSURES:
        raise ValueError(f"unknown exposure {exposure!r}")
    return frozenset(k for k, s in KEY_SPECS.items() if s.exposure == exposure)


def unregistered_default_keys() -> frozenset:
    """Default flat keys (deployment + platform) that have no KeySpec yet.

    These are still consumed flat; they are the strangler-fig remainder to register.
    """
    defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
    return frozenset(defaults - set(KEY_SPECS))

"""Build ``platform_constraints_resolved`` from flat pipeline config."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

from mimarsinan.chip_simulation.core_semantics import resolve_core_semantics
from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import resolve_floorplan
from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS
from mimarsinan.config_schema.registry.build import passthrough_keys
from mimarsinan.config_schema.registry.consumer_reading import (
    consumer_refusal, read_like_consumers, workload_reading_errors,
)
# The one judge both doors call lives in the registry (S0 amendment B); the resolver is the run door's caller and
# re-exports the two names of its frozen interface.
from mimarsinan.config_schema.registry.declaration_judge import (
    declaration_errors as declaration_errors,
    domain_of,
    run_door_errors as run_door_errors,
)
from mimarsinan.config_schema.registry.entries_write import platform_bias_mode
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.deployment_record.platform_physics.resolve import (
    compute_physics_profile, resolve_platform_physics, write_physics_profile,
)
from mimarsinan.deployment_record.quantities.compute_technology import COMPUTE_PHYSICS_RESOLVED
from mimarsinan.deployment_record.quantities.write_technology import WRITE_PHYSICS_RESOLVED, core_technologies
from mimarsinan.mapping.lanes.declaration import LANE_KEY, declared_lane_block
from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY, normalize_coalescing_config
from mimarsinan.mapping.platform.core_residency import RESIDENCY_KEY, resolve_residency_policy
from mimarsinan.mapping.platform.platform_constraints import bias_mode_for_cores
from mimarsinan.mapping.platform.weight_cells import check_conversion_agreement
from mimarsinan.mapping.support.bias_rows import (
    BIAS_ROW_SPLITTING_FIXED,
    BIAS_ROW_SPLITTING_MODES,
    BIAS_ROW_SPLITTING_OFF,
    BiasRowSplitting,
    MODE_KEY,
    ROWS_KEY,
)

_Read = TypeVar("_Read")


def _read_by_consumer(pipeline_config: Mapping[str, Any], keys: Iterable[str], read: Callable[[], _Read]) -> _Read:
    """A consumer's reading of declarations its value schemas do not judge whole (the residency granularities, the
    physics profile and overrides, the floorplan's capacity against the declared cores), whose refusal is a
    ``DeclarationValueError`` naming each declared key and value with the consumer's reason, where the run would
    raise a plain TypeError, ValueError, KeyError or AttributeError."""
    try:
        return read()
    except (TypeError, ValueError, KeyError, AttributeError) as refusal:
        declared = {key: pipeline_config[key] for key in keys if pipeline_config.get(key) is not None}
        raise DeclarationValueError(consumer_refusal(declared, refusal)) from refusal


def _residency_declaration(pipeline_config: Mapping[str, Any]) -> dict[str, Any]:
    """The declared granularities, read by their consumer (``resolve_residency_policy``) before they ride the
    surface, so a quantity or granularity it does not know is refused here."""
    declared = pipeline_config.get(RESIDENCY_KEY, {}) or {}
    resolve_residency_policy({RESIDENCY_KEY: declared}, value_domain=domain_of(pipeline_config) == "value")
    return dict(declared)


def _sealed_write_physics(cores: list) -> dict[str, Any]:
    """N2b: each write physics a core type names, as the write law will price it (``write_physics_resolved``); the
    core types judged as the write law reads them (``core_technologies``); a name no profile ships is refused."""
    core_technologies({"cores": cores})
    names = sorted({str(ct["write_physics"]) for ct in cores if ct.get("write_physics")})
    return {name: write_physics_profile(name).to_dict() for name in names}


def _sealed_compute_physics(cores: list) -> dict[str, Any]:
    """[E-6] Each compute physics a core type names, as the energy pricer will price its MAC events
    (``compute_physics_resolved``); a name no profile ships is refused."""
    names = sorted({str(ct["compute_physics"]) for ct in cores if ct.get("compute_physics")})
    return {name: compute_physics_profile(name).to_dict() for name in names}


def build_platform_constraints_resolved(
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    """Single source for resolved platform constraints dict.

    ONE function of the config, with no mode switch: a searched chip and the
    same chip declared by hand must resolve identically, and an omitted
    permission reads as DENIED downstream (``ChipCapabilities``). The config is
    read as its consumers read it (``read_like_consumers``) before anything is
    judged, as the document door reads a document; the caller's dict is not
    changed.
    """
    pipeline_config = read_like_consumers(pipeline_config)
    errors = run_door_errors(pipeline_config) or workload_reading_errors(pipeline_config, pipeline_config)
    if errors:
        raise DeclarationValueError("; ".join(errors))
    cores = pipeline_config.get("cores")
    if cores is None:
        cores = list(cast("list[dict[str, Any]]", DEFAULT_PLATFORM_CONSTRAINTS["cores"]))

    cores = [dict(ct) for ct in cores]
    bias_mode = platform_bias_mode(pipeline_config)
    for ct in cores:
        ct.setdefault("has_bias", bias_mode)

    pcfg: dict[str, Any] = {"cores": cores}
    pcfg["allow_neuron_splitting"] = bool(
        pipeline_config.get("allow_neuron_splitting", False)
    )
    pcfg["allow_scheduling"] = bool(pipeline_config.get("allow_scheduling", False))
    pcfg[RESIDENCY_KEY] = _read_by_consumer(pipeline_config, (RESIDENCY_KEY,),
                                            lambda: _residency_declaration(pipeline_config))
    # The scheduled-build pass budget: dropping it here silently pins the
    # builder to its default and disarms residency streaming (t0_44 measured).
    pcfg["max_schedule_passes"] = int(
        pipeline_config.get("max_schedule_passes", 8) or 8
    )
    # SANA-FE NoC floorplan declaration (0 = derived): rides the resolved
    # surface so the SANA-FE step reads floorplan + capacity from ONE place.
    for key in ("cores_per_tile", "tile_grid_rows", "tile_grid_cols"):
        pcfg[key] = int(pipeline_config.get(key, 0) or 0)
    # The CONCRETE floorplan derivation is part of the resolved surface too:
    # declared keys stay verbatim above, and the *_resolved keys expose the
    # floorplan the SANA-FE step will build — computed by the SAME pure
    # function of the declared platform (capacity = sum of declared core
    # counts, preset = the flat-config sanafe_arch_preset, defaulting like
    # the SANA-FE step does), so both seams agree by construction. An
    # invalid declaration (an explicit grid too small for the declared
    # capacity) fails loud HERE, at resolution time. With
    # sanafe_arch_preset='custom' the loaded arch YAML remains the floorplan
    # SSOT at the SANA-FE step (declared keys are validated against the
    # file); these keys then carry the declared-platform derivation only.
    # Missing "count" means one core of that type (the imc_platforms
    # convention) — minimal declarations (e.g. bias-mode queries) stay valid.
    floorplan = _read_by_consumer(pipeline_config, ("cores_per_tile", "tile_grid_rows", "tile_grid_cols"),
                                  lambda: resolve_floorplan(
                                      sum(int(core_type.get("count", 1)) for core_type in cores),
                                      str(pipeline_config.get("sanafe_arch_preset", "loihi")),
                                      pcfg["cores_per_tile"], pcfg["tile_grid_rows"], pcfg["tile_grid_cols"]))
    pcfg["cores_per_tile_resolved"] = int(floorplan.cores_per_tile)
    pcfg["tile_grid_rows_resolved"] = int(floorplan.rows)
    pcfg["tile_grid_cols_resolved"] = int(floorplan.cols)

    # The TARGET'S PHYSICS, resolved the same way and for the same reason as the
    # floorplan: profile + overrides is a declaration, and the CONCRETE constants
    # the cost model will price with belong on the surface the record carries
    # verbatim, so a run states exactly what physics produced its numbers even if
    # the profile file changes later. None means none declared — the absolute
    # objectives are then unavailable rather than defaulted. An unknown profile or
    # an unknown constant fails loud HERE, at resolution time.
    physics = _read_by_consumer(
        pipeline_config, ("platform_physics_profile", "platform_physics_overrides", LANE_KEY),
        lambda: resolve_platform_physics(str(pipeline_config.get("platform_physics_profile", "") or ""),
                                         pipeline_config.get("platform_physics_overrides") or {},
                                         lane_block=declared_lane_block(pipeline_config)))
    pcfg["platform_physics_resolved"] = physics.to_dict() if physics else None
    # [U-C4] The conversion model's cells_per_weight is the core types' own count, stated twice: they agree.
    _read_by_consumer(pipeline_config, ("cores", "platform_physics_profile"),
                      lambda: check_conversion_agreement(cores, physics.conversion_model if physics else None))
    # N2b: each core type's named write physics, sealed the same way (only when a core type names one).
    sealed = _read_by_consumer(pipeline_config, ("cores",), lambda: _sealed_write_physics(cores))
    if sealed:
        pcfg[WRITE_PHYSICS_RESOLVED] = sealed
    # [E-6] each core type's named compute physics, sealed the same way (only when a core type names one).
    computed = _read_by_consumer(pipeline_config, ("cores",), lambda: _sealed_compute_physics(cores))
    if computed:
        pcfg[COMPUTE_PHYSICS_RESOLVED] = computed

    if "target_tq" in pipeline_config:
        pcfg["target_tq"] = pipeline_config["target_tq"]
    if "weight_bits" in pipeline_config:
        pcfg["weight_bits"] = pipeline_config["weight_bits"]
    # The deployment spike window and the declared switching-activity
    # assumption ride the resolved surface: candidate-time quantities and the
    # sealed record's twin read the SAME declarations (0 activity = undeclared).
    pcfg["simulation_steps"] = int(
        pipeline_config.get(
            "simulation_steps", DEFAULT_PLATFORM_CONSTRAINTS["simulation_steps"],
        ) or 0
    )
    pcfg["activity_factor"] = float(
        pipeline_config.get("activity_factor", 0.0) or 0.0
    )
    # [B] The declared pass-carry buffer ceiling (0 = undeclared, metric only).
    pcfg["pass_buffer_capacity_bytes"] = int(
        pipeline_config.get("pass_buffer_capacity_bytes", 0) or 0
    )

    if CANONICAL_KEY in pipeline_config:
        pcfg[CANONICAL_KEY] = bool(pipeline_config[CANONICAL_KEY])
    else:
        pcfg[CANONICAL_KEY] = False
    normalize_coalescing_config(pcfg)
    pcfg.update(declared_surface(pipeline_config))
    return pcfg


def judged_passthrough(pipeline_config: dict[str, Any], run_domain: str) -> dict[str, Any]:
    """The passthrough keys verbatim (an absent key does not appear), after ``declaration_errors`` judges every
    present key (``DeclarationValueError`` names each). No ``src/`` caller since S0 amendment B (the resolver
    judges through ``run_door_errors``); kept for the frozen interface groups code against."""
    errors = declaration_errors(pipeline_config, run_domain)
    if errors:
        raise DeclarationValueError("; ".join(errors))
    return _passthrough_surface(pipeline_config)


def _passthrough_surface(pipeline_config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(pipeline_config[key]) for key in passthrough_keys()
            if pipeline_config.get(key) is not None}


def declared_surface(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """The resolved surface's declaration part, over the config as its consumers read it, which
    ``run_door_errors`` judged first: the passthrough keys verbatim, and the value domain named, with its value
    grid when declared, exactly for value-domain runs, so every spiking surface is unchanged.
    """
    value_domain = domain_of(pipeline_config) == "value"
    surface = _passthrough_surface(pipeline_config)
    if value_domain:
        surface["core_semantics"] = resolve_core_semantics(pipeline_config)
        if pipeline_config.get("activation_bits") is not None:
            surface["activation_bits"] = pipeline_config["activation_bits"]
    return surface


def resolve_bias_mode(pipeline_config: dict[str, Any]) -> str:
    """Deployment bias delivery (``"on_chip"`` / ``"param_encoded"``) for this config.

    Single source shared by the tuners and the mapping step: normalizes the
    declared grid the way ``SoftCoreMappingStep`` does, then asks the
    bias-mode SSOT, so training-time nodes and the deployed mapping agree.
    """
    return bias_mode_for_cores(
        build_platform_constraints_resolved(pipeline_config)["cores"]
    )


def resolve_bias_row_splitting(config: dict[str, Any]) -> BiasRowSplitting:
    """THE bias-row splitting decision for this config.

    One function of the declared document, shared by the WQ projection, the
    mapping backends and the advisories, so training-time grids and the
    deployed core matrices never disagree about how many always-on rows a
    bias occupies. A platform whose cores all carry an on-chip bias register
    has no row to split, so the mode resolves inactive there.
    """
    mode = str(config.get(MODE_KEY, BIAS_ROW_SPLITTING_OFF) or BIAS_ROW_SPLITTING_OFF)
    if mode not in BIAS_ROW_SPLITTING_MODES:
        raise ValueError(
            f"{MODE_KEY}={mode!r} is not one of {BIAS_ROW_SPLITTING_MODES}"
        )
    rows = int(config.get(ROWS_KEY, 0) or 0)
    if mode == BIAS_ROW_SPLITTING_FIXED and rows < 1:
        raise ValueError(
            f"{MODE_KEY}='fixed' requires {ROWS_KEY} >= 1 (declare the row "
            f"count, or use 'auto' for the computed bound)"
        )
    active = mode != BIAS_ROW_SPLITTING_OFF and resolve_bias_mode(config) == "param_encoded"
    return BiasRowSplitting(mode=mode, fixed_rows=rows, active=active)


def resolve_wq_weight_only_grid(config: dict[str, Any]) -> bool:
    """Whether the WQ projection derives the weight grid from ``max|w|`` alone.

    TWO routes to the same projection: an on-chip bias register takes the
    two-scale flag, and a param-encoded bias takes bias-row splitting (the
    coarser bias grid becomes k always-on rows instead of one register).
    """
    return (
        resolve_wq_two_scale_projection(config)
        or resolve_bias_row_splitting(config).active
    )


def resolve_wq_two_scale_projection(config: dict[str, Any]) -> bool:
    """Effective two-scale WQ flag: the ``wq_two_scale_projection`` key AND the
    platform's on-chip bias capability — a parameter-encoded bias rides the
    core matrix as an always-on axon row and must obey the ±q_max
    weight-register contract on the weight grid, so two-scale is not mappable
    there (``wq_cascade_crater_repair.md`` §5, backend audit)."""
    if not bool(config.get("wq_two_scale_projection", False)):
        return False
    return resolve_bias_mode(config) == "on_chip"

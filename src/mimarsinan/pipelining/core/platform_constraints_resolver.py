"""Build ``platform_constraints_resolved`` from flat pipeline config."""

from __future__ import annotations

from typing import Any, cast

from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import resolve_floorplan
from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY, normalize_coalescing_config
from mimarsinan.mapping.platform.platform_constraints import bias_mode_for_cores
from mimarsinan.mapping.platform.core_residency import RESIDENCY_KEY


def build_platform_constraints_resolved(
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    """Single source for resolved platform constraints dict.

    ONE function of the config, with no mode switch: a searched chip and the
    same chip declared by hand must resolve identically, and an omitted
    permission reads as DENIED downstream (``ChipCapabilities``).
    """
    cores = pipeline_config.get("cores")
    if cores is None:
        cores = list(cast("list[dict[str, Any]]", DEFAULT_PLATFORM_CONSTRAINTS["cores"]))

    global_has_bias = pipeline_config.get("platform_constraints", {}).get(
        "has_bias", True
    )
    cores = [dict(ct) for ct in cores]
    for ct in cores:
        ct.setdefault("has_bias", global_has_bias)

    pcfg: dict[str, Any] = {"cores": cores}
    pcfg["allow_neuron_splitting"] = bool(
        pipeline_config.get("allow_neuron_splitting", False)
    )
    pcfg["allow_scheduling"] = bool(pipeline_config.get("allow_scheduling", False))
    pcfg[RESIDENCY_KEY] = dict(pipeline_config.get(RESIDENCY_KEY, {}) or {})
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
    floorplan = resolve_floorplan(
        sum(int(core_type.get("count", 1)) for core_type in cores),
        str(pipeline_config.get("sanafe_arch_preset", "loihi")),
        pcfg["cores_per_tile"],
        pcfg["tile_grid_rows"],
        pcfg["tile_grid_cols"],
    )
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
    physics = resolve_platform_physics(
        str(pipeline_config.get("platform_physics_profile", "") or ""),
        pipeline_config.get("platform_physics_overrides") or {},
    )
    pcfg["platform_physics_resolved"] = physics.to_dict() if physics else None

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
    return pcfg


def resolve_bias_mode(pipeline_config: dict[str, Any]) -> str:
    """Deployment bias delivery (``"on_chip"`` / ``"param_encoded"``) for this config.

    Single source shared by the tuners and the mapping step: normalizes the
    declared grid the way ``SoftCoreMappingStep`` does, then asks the
    bias-mode SSOT, so training-time nodes and the deployed mapping agree.
    """
    return bias_mode_for_cores(
        build_platform_constraints_resolved(pipeline_config)["cores"]
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

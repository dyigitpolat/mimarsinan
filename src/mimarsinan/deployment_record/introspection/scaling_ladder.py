"""The shape-only scaling ladder: one network on one declared platform, priced as the toolchain prices it.

Free of workloads: the caller hands in a built mapper representation and resolved platform dicts (cores,
capabilities, ``platform_physics_resolved``, declarations), plus the toolchain's candidate-plane program census as a
callable, so this module never re-implements the census the search and the record build share. Per network and core
geometry, one shape-only layout (``shape_census``); per resolved platform (one core budget), one packed and scheduled
program (``schedule_program``), priced per physics declaration and batch (``price_scheduled``; ``price_program``
composes both); per declared batch K, rate and lifetime, one row (``ladder_row``).

Cells: logical weight cells are the network's stationary weights (``stationary_weight_cells``: per on-chip perceptron
its weight tensor once plus a bias row, the same on every target [fix_b]); as-mapped cells count every replica (every
conv output position or token is its own softcore), so replication is their ratio. Only x_fluid is free of resolution.

The value types and the refusals by name live in ``scaling_ladder_types``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.build.payload_sizes import pass_program_bytes
from mimarsinan.deployment_record.cost.absolute import price_absolute
import mimarsinan.deployment_record.introspection.scaling_ladder_law as law
import mimarsinan.deployment_record.introspection.scaling_ladder_reading as point_reading
from mimarsinan.deployment_record.introspection.scaling_ladder_types import (
    LadderDeclarationError, LadderGeometryError, LadderNetwork, LadderScheduleRefusal, PricedProgram,
    ScheduledProgram, ShapeCensus, certificate_record,
)
from mimarsinan.deployment_record.objectives import CandidateStaticView, declared_core_capacity
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, core_types_from_declaration
from mimarsinan.mapping.noc import census_of_walk, collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.residency_rules import ProgramGraph, spec_runs
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats

#: (softcores, NoC fragments, resolved platform) -> (fragments, program facts with ``latency_steps``,
#: ``programming`` and ``carry``): the candidate plane's census, injected by the caller.
ProgramFacts = Callable[[Sequence[Any], Any, Mapping[str, Any]], Tuple[Any, Any]]
#: Declarations a row reads that leave the priced program unchanged.
ROW_KEYS = ("inference_batch_K", "inference_rate_hz", "min_lifetime_hours")
#: [CL5] Declarations a priced platform may change against its scheduled program: the row keys and the physics.
PRICE_KEYS = ROW_KEYS + ("platform_physics_profile", "platform_physics_overrides", "platform_physics_resolved")
_CARRY = ("carried_raster_bytes", "carry_peak_live_bytes", "carry_out_bytes", "carry_in_bytes")
_PROGRAMMING = ("reprogrammed_bytes", "reprogrammed_cores", "reprogram_passes")
#: axis -> (compute-only term, headline (whose refusal names an absence), programming term, host term); [consts] the
#: programming term is the write law's tile writes (``scaling_ladder_reading.tile_writes``)
_AXES = {
    "energy": ("energy_dynamic_mj", "energy_per_inference_mj", "energy_write_mj", "energy_host_mj"),
    "time": ("e2e_latency_s", "e2e_latency_s", "latency_write_exposed_s", "latency_host_s"),
}
_HEADLINES = ("e2e_latency_s", "energy_per_inference_mj")
__all__ = ["LadderDeclarationError", "LadderGeometryError", "LadderNetwork", "LadderScheduleRefusal",
           "PricedProgram", "ScheduledProgram", "ShapeCensus", "certificate_record", "ladder_row", "packed_tiles",
           "price_program", "price_scheduled", "schedule_program", "shape_census"]


def shape_census(network: LadderNetwork, platform: Mapping[str, Any], *, wire_census: bool = True) -> ShapeCensus:
    """Lay the network out on the platform's core geometry; count logical and as-mapped cells. A program with no
    on-chip softcore (every op on the host) has no cell to count and is refused by name (``LadderGeometryError``)."""
    start = time.time()
    geometry = resolve_platform_mapping_params(platform["cores"],
                                               allow_coalescing=bool(platform.get("allow_coalescing", False)))
    layout = LayoutIRMapping(
        max_axons=geometry.effective_max_axons, max_neurons=geometry.effective_max_neurons,
        allow_coalescing=geometry.allow_coalescing, hardware_bias=geometry.hardware_bias,
        collect_wire_census=wire_census, rows_per_input=geometry.rows_per_input,
    )
    if hasattr(network.mapper_repr, "assign_perceptron_indices"):
        network.mapper_repr.assign_perceptron_indices()
    softcores = tuple(layout.collect_layout_softcores(network.mapper_repr))
    runs = spec_runs(layout)
    logical = point_reading.stationary_weight_cells(network.mapper_repr, softcores)
    if logical == 0:
        raise LadderGeometryError(
            f"the program holds no on-chip softcore ({len(softcores)} softcores, 0 logical cells)")
    return ShapeCensus(
        softcores=softcores, wire_census=census_of_walk(layout) if wire_census else None,
        host_segments=int(getattr(layout, "host_side_segment_count", 0)),
        segments=len(set(runs)),
        logical_cells=logical,
        as_mapped_cells=sum(int(s.input_count) * int(s.output_count) for s in softcores),
        wall_s=time.time() - start, program=ProgramGraph.of_walk(layout) if wire_census else None, runs=runs,
    )


def packed_tiles(census: ShapeCensus, platform: Mapping[str, Any]) -> int:
    """Physical cores the whole program packs into with an unbounded count of each declared core type.

    With coalescing declared, the layout packer keeps a softcore wider than one core whole on a
    fused instance of ceil(a / A) cores, as the hard-core packer does, and reports the instance
    as one used core, so a fused instance counts its member cores here (``LayoutCoreSnapshot.tiles``).
    Without it, nothing fuses and such a softcore does not pack (refused by name).
    """
    kwargs = ChipCapabilities.from_platform_constraints(platform).layout_kwargs()
    declared = core_types_from_declaration(platform["cores"])
    widths = {(t.max_axons, t.max_neurons) for t in declared}
    if len(widths) != 1:
        raise LadderGeometryError(f"x_packed counts fused cores by one declared core shape, got {sorted(widths)}")
    width, neurons = widths.pop()
    bound = sum(-(-int(s.input_count) // width) * -(-int(s.output_count) // neurons) for s in census.softcores)
    unbounded = [LayoutHardCoreType(t.max_axons, t.max_neurons, bound) for t in declared]
    pack = pack_layout(softcores=census.softcores, core_types=unbounded,
                       allow_neuron_splitting=kwargs["allow_neuron_splitting"],
                       allow_coalescing=kwargs["allow_coalescing"])
    if not pack.feasible:
        raise LadderGeometryError(f"the program does not pack even on unbounded cores: {pack.error}")
    return sum(-(-int(s.axons_per_core) // width) for s in pack.used_core_snapshots or ())


def _variant(q: Quantities, drop: Sequence[str] = (), put: Optional[Mapping[str, float]] = None) -> Quantities:
    values = {k: q.get(k) for k in q.keys() if k not in drop}
    values.update({k: QuantityValue(value=float(v), provenance="static") for k, v in (put or {}).items()})
    return Quantities(values)


def schedule_program(
    network: LadderNetwork, census: ShapeCensus, platform: Mapping[str, Any], program_facts: ProgramFacts,
    *, tiles: Optional[int] = None, on_stage: Optional[Callable[[str], None]] = None,
) -> ScheduledProgram:
    """Pack and schedule with the builder's planner: U1 on the builder's segments (the walk's runs [fix_b]), a
    declared ``residency_plan`` on the walk's program graph [W03]; ``on_stage`` hears "pack" (``tiles`` not given)
    and "schedule"."""
    stage = on_stage or (lambda name: None)
    start = time.time()
    if tiles is None:
        stage("pack")
        tiles = packed_tiles(census, platform)
    stage("schedule")
    core_types = core_types_from_declaration(platform["cores"])
    kwargs = ChipCapabilities.from_platform_constraints(platform).layout_kwargs()
    planned = {} if kwargs.get("residency_plan") is None else {"program": census.program}
    stats, error = compute_mapping_stats(census.softcores, core_types, census=census.wire_census, **planned,
                                         runs=census.runs, **kwargs)
    if not stats.feasible:
        refusal = {"name": LadderScheduleRefusal.__name__,
                   "reason": error or "the planner reports the program infeasible"}
        return ScheduledProgram(platform=platform, packed_tiles=tiles, refusal=refusal, wall_s=time.time() - start)
    noc = collect_noc_fragments(softcores=census.softcores, core_types=core_types,
                                census=census.wire_census, declared_cores=platform["cores"], **planned,
                                runs=census.runs, **kwargs)
    noc, facts = program_facts(census.softcores, noc, platform)
    return ScheduledProgram(platform=platform, packed_tiles=tiles, stats=stats, noc=noc, facts=facts,
                            wall_s=time.time() - start)


def price_program(
    network: LadderNetwork, census: ShapeCensus, platform: Mapping[str, Any], program_facts: ProgramFacts,
    *, tiles: Optional[int] = None, on_stage: Optional[Callable[[str], None]] = None,
) -> PricedProgram:
    """``schedule_program`` then ``price_scheduled`` on the same platform; stages "pack", "schedule", "price"."""
    scheduled = schedule_program(network, census, platform, program_facts, tiles=tiles, on_stage=on_stage)
    return price_scheduled(network, census, scheduled, platform, on_stage=on_stage)


def price_scheduled(
    network: LadderNetwork, census: ShapeCensus, scheduled: ScheduledProgram, platform: Mapping[str, Any],
    *, on_stage: Optional[Callable[[str], None]] = None,
) -> PricedProgram:
    """Price a scheduled program on ``platform``, which may differ from the scheduled one only in its physics and
    the row keys (refused by name otherwise): three variants (the program, its compute-only cost, the program
    without host), the write law's tile writes at K = 1 (writing every tile once, and the program's period writes,
    [consts] ``tile_writes``) and, at a declared ``inference_batch_K``, the write law at that batch."""
    if _base(platform, PRICE_KEYS) != _base(scheduled.platform, PRICE_KEYS):
        raise LadderDeclarationError("a priced platform may differ from its scheduled program only in "
                                     + ", ".join(PRICE_KEYS))
    start = time.time()
    tiles = scheduled.packed_tiles
    if scheduled.refusal is not None:
        return PricedProgram(platform=platform, packed_tiles=tiles, refusal=scheduled.refusal, wall_s=scheduled.wall_s)
    (on_stage or (lambda name: None))("price")
    stats, noc, facts = scheduled.stats, scheduled.noc, scheduled.facts
    host: Dict[str, Any] = dict(network.host_census or {})
    context = candidate_context_from_platform(
        platform, latency_steps=facts.latency_steps, programming=facts.programming, carry=facts.carry,
        passes=facts.passes, **{k: host.get(k) for k in ("host_macs", "onchip_macs", "host_params",
                                                          "onchip_params", "compute_op_count")},
    )
    payload = platform.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    q = CandidateStaticView(
        layout=stats, chip_param_capacity=declared_core_capacity(dict(platform)),
        total_params=network.total_params, host_side_segment_count=census.host_segments,
        physics=physics, quantity_context=context, noc_fragments=noc,
    ).quantities
    bits = platform.get("weight_bits")
    once_tiles = sum(p.tiles for p in noc.pass_programs)  # physical cores: a fused core counts its members
    once_bytes = None if bits is None else sum(b for p in noc.pass_programs for b in pass_program_bytes(p, int(bits)))
    written, period_bytes, max_writes = point_reading.period_writes(facts.passes, platform, bits)
    terms: Dict[str, Dict[str, Band]] = {}
    refusals: Dict[str, Dict[str, str]] = {}
    write_law: Dict[str, Any] = {}
    if physics is not None:
        compute = _variant(q, _CARRY + _PROGRAMMING + ("sync_count", "compute_op_count"), {"host_macs": 0})
        for name, variant in (("program", q), ("compute_only", compute),
                              ("no_host", _variant(q, ("compute_op_count",), {"host_macs": 0}))):
            pricing = price_absolute(variant, physics)
            terms[name] = {t.name: t.band for t in pricing.terms if t.band is not None}
            refusals[name] = {r.name: r.reason for r in pricing.refusals}
        for name, load in (("once", True), ("period", False)):
            terms[name], refusals[name] = point_reading.tile_writes(facts.passes, platform, bits, physics, load=load)
        if context.writes is not None:
            at_k = price_absolute(q, physics, writes=WriteInputs.of(context.writes))
            write_law = point_reading.write_law_columns(at_k, platform.get("inference_batch_K"))
    return PricedProgram(
        platform=platform, packed_tiles=tiles, passes=len(noc.pass_placements), latency_steps=facts.latency_steps,
        census_passes=None if facts.passes is None else len(facts.passes.rows),
        reprogram_passes=int(facts.programming.reprogram_passes), written_passes=written, carry=facts.carry,
        period_bytes=period_bytes, once_bytes=once_bytes, once_cores=once_tiles,
        max_writes=max_writes, terms=terms, refusals=refusals, wall_s=scheduled.wall_s + time.time() - start,
        write_law=write_law, **point_reading.endurance_fields(physics),
    )


def _base(platform: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: v for k, v in platform.items() if k not in keys}


def _digest(network: LadderNetwork, platform: Mapping[str, Any]) -> str:
    blob = json.dumps({"network": network.name, "resolution": list(network.resolution),
                       "platform": platform}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def ladder_row(
    network: LadderNetwork, census: ShapeCensus, priced: PricedProgram,
    point_platform: Mapping[str, Any], reading: law.RegimeReading,
) -> Dict[str, Any]:
    """One row: the priced program read at the point's declared K, rate and lifetime. The capacity reading
    (x_packed) and the schedule reading (the period census) side by side; the regime, M's period writes and the
    write census follow the schedule; M for every program of two or more passes; the CL5 reading columns
    (``scaling_ladder_reading.row_readings``)."""
    if _base(point_platform, ROW_KEYS) != _base(priced.platform, ROW_KEYS):
        raise LadderDeclarationError("a row's platform may differ from its priced program only in "
                                     + ", ".join(ROW_KEYS))
    cores = sum(int(c.get("count", 1)) for c in point_platform["cores"])
    capacity = sum(int(c.get("count", 1)) * int(c["max_axons"]) * int(c["max_neurons"])
                   for c in point_platform["cores"])
    x_fluid, x_packed = census.logical_cells / capacity, priced.packed_tiles / cores
    k = point_platform.get("inference_batch_K")
    row: Dict[str, Any] = {
        "network": network.name, "resolution": list(network.resolution), "cores": cores,
        "weight_bits": point_platform.get("weight_bits"),
        "inference_batch_K": k, "softcores": len(census.softcores), "segments": census.segments,
        "logical_cells": census.logical_cells, "as_mapped_cells": census.as_mapped_cells,
        "capacity_cells": capacity, "replication": census.as_mapped_cells / census.logical_cells,
        "packed_tiles": priced.packed_tiles, "x_fluid": x_fluid, "x_packed": x_packed,
        "capacity_reading": law.capacity_reading(x_packed), "config_digest": _digest(network, point_platform),
        "conversion": dict(network.conversion or certificate_record(None)),
        "layout_wall_s": census.wall_s, "point_wall_s": priced.wall_s,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    if priced.refusal is not None:
        row["refusal"] = dict(priced.refusal)
        return row
    multiplexed = priced.written_passes > 0
    row.update(passes=priced.passes, census_passes=priced.census_passes, latency_steps=priced.latency_steps,
               reprogram_passes=priced.reprogram_passes,
               schedule_reading=law.schedule_reading(priced.written_passes, census.segments),
               carry=dict(priced.carry or {}), programmed_bytes_once=priced.once_bytes,
               programmed_cores_once=priced.once_cores, reprogrammed_bytes_per_period=priced.period_bytes,
               headlines={h: law.triple(priced.terms.get("program", {}).get(h)) for h in _HEADLINES},
               headline_refusals={h: priced.refusals.get("program", {}).get(h) for h in _HEADLINES})
    for axis, names in _AXES.items():
        row[axis] = law.axis_columns(priced.terms, priced.refusals, names, k=k, reading=reading,
                                     multiplexed=multiplexed, x=x_packed, passes=priced.passes, axis=axis)
    row["axes_priced"] = [axis for axis in _AXES if row[axis]["pi"] is not None]
    row["k_star"] = law.k_star(
        x_fluid=x_fluid, x_packed=x_packed, max_writes=priced.max_writes[0], max_writes_source=priced.max_writes[1],
        rate_hz=point_platform.get("inference_rate_hz"), lifetime_hours=point_platform.get("min_lifetime_hours"),
        endurance=priced.endurance, pulses_per_write=None if priced.pulses_band is None else priced.pulses_band[1],
    )
    row.update(point_reading.row_readings(
        priced, x_fluid=x_fluid, x_packed=x_packed, k=k, logical_cells=census.logical_cells,
        onchip_macs=(network.host_census or {}).get("onchip_macs"), rate_hz=point_platform.get("inference_rate_hz"),
        lifetime_hours=point_platform.get("min_lifetime_hours"), axes={a: row[a] for a in _AXES},
        multiplexed=multiplexed))
    if multiplexed and priced.passes >= 2 and priced.packed_tiles > cores:
        f, c, p = priced.packed_tiles, cores, priced.passes
        row["bc4_saving_closed_form"] = (p - 1) * f / (p * (f - c))
    row["regime_reading"] = {"capacity_free_max": reading.capacity_free_max,
                             "resident_dominant_min": reading.resident_dominant_min, "source": reading.source}
    return row

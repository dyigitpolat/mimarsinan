"""The programming census of a program on both planes: per-core payload records, the candidate's
programming and carry census, the executed residency marks a build seals (declared-only), and each
plane's passes as the period census reads them (``write_census.PassRow``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.stage_timesteps import program_latency_steps
from mimarsinan.deployment_record.build.pass_carry import (  # candidate_carry_census moved there: re-export
    boundary_widths_of, candidate_carry_census as candidate_carry_census,
)
from mimarsinan.deployment_record.build.payload_sizes import (
    core_connectivity_entries, core_payload_bytes, core_sign_rows, pass_program_bytes,
)
from mimarsinan.deployment_record.build.producer_census import (
    candidate_operand_cells, operand_cells_by_core, operand_cells_per_tile,
)
from mimarsinan.deployment_record.build.write_census import (
    BoundaryWidths, PassRow, ProgramPasses, stage_latency,
)
from mimarsinan.deployment_record.schema import SegmentCoreRecord, SegmentRecord
from mimarsinan.deployment_record.schema.schedule import ResidencyPlanRecord
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.noc.fragments import delayed_carry_wires, execution_stage_latencies, spec_segments
from mimarsinan.mapping.packing.residency_tiles import declared_tiles, expand_to_tiles, tile_rows
from mimarsinan.mapping.support.period_census import (
    SCHEDULE_RESIDENCY_PLAN, SCHEDULE_RESIDENT_CORES, resident_core_indices,
)
from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_softcore_spans, carried_wire_spans, delayed_input_slices,
)
from mimarsinan.mapping.weight_programming import weight_programming_report


class _SingleStageProgram:
    """One-stage view so per-segment programming calls the report verbatim."""

    def __init__(self, stage: Any) -> None:
        self.stages = [stage]


def core_record(core_index: int, core: Any, *, weight_bits: int, resident: bool) -> SegmentCoreRecord:
    occupancy = CoreOccupancy.from_hard_core(core)
    latency = getattr(core, "latency", None)
    signs = core_sign_rows(core, occupancy)
    return SegmentCoreRecord(
        core_index=int(core_index),
        axons=int(occupancy.axons_physical),
        neurons=int(occupancy.neurons_physical),
        axons_used=int(occupancy.axons_used),
        neurons_used=int(occupancy.neurons_used),
        cells_used=int(occupancy.cells_used),
        # A resident pass sends no weight payload (schema §2.1: 0 when resident); a per-axon core's payload
        # carries its sign register (plan section 21 item 11).
        params_bytes=0 if resident else core_payload_bytes(occupancy.cells_used, weight_bits, signs),
        connectivity_entries=core_connectivity_entries(core),
        static_delay_levels=None if latency is None else int(latency),
        sign_register_rows=signs or None,
    )


def segment_record(stage_index: int, segment_index: int, pass_index: int, stage: Any, *,
                   weight_bits: int) -> SegmentRecord:
    """One (segment, pass) of the built program; an executed residency plan's marks ride along."""
    segment = stage.hard_core_mapping
    resident = bool(getattr(stage, "schedule_weights_resident", False))
    programming = weight_programming_report(_SingleStageProgram(stage))
    cores = tuple(core_record(index, core, weight_bits=weight_bits, resident=resident)
                  for index, core in enumerate(segment.cores))
    bank_ids = sorted({
        int(placement["weight_bank_id"])
        for placements in segment.soft_core_placements_per_hard_core
        for placement in placements
        if placement.get("weight_bank_id") is not None
    })
    marks = None
    if getattr(stage, SCHEDULE_RESIDENT_CORES, None) is not None:
        marks = tuple(sorted(resident_core_indices(stage, len(cores))))
    return SegmentRecord(
        stage_index=int(stage_index),
        segment_index=int(segment_index),
        pass_index=int(pass_index),
        pass_reason="initial" if pass_index == 0 else "capacity_overflow",
        programming="resident" if resident else "reprogram",
        bank_ids=tuple(bank_ids),
        cores=cores,
        params_programmed=int(programming.params_programmed),
        params_unique=int(programming.params_unique),
        params_bytes=sum(core.params_bytes for core in cores),
        connectivity_entries=sum(core.connectivity_entries for core in cores),
        # The dependence-level currency: distinct latency tiers in the segment.
        static_latency_levels=len({int(core.latency) for core in segment.cores
                                   if getattr(core, "latency", None) is not None}),
        resident_core_indices=marks,
    )


def residency_plan_record(hybrid_mapping: Any) -> Optional[ResidencyPlanRecord]:
    """The executed residency plan the build marked on the mapping, or None (declared-only)."""
    plan = getattr(hybrid_mapping, SCHEDULE_RESIDENCY_PLAN, None)
    return None if plan is None else ResidencyPlanRecord.from_plan(plan)


@dataclass(frozen=True)
class CandidateProgramming:
    """The programming census as the quantity surface consumes it."""

    segment_cores: int
    #: [R1] Σ over EVERY pass of the occupied used-rows x used-columns cells — the AS-MAPPED
    #: figure (replicas count once per replica): ``macs``/``cells_used`` at candidate completeness.
    committed_cells: int
    reprogrammed_cores: int
    #: None when the platform declares no weight width — absent, never zero.
    reprogrammed_bytes: Optional[int]
    reprogram_passes: int
    pass_cores: Tuple[int, ...]
    reprogrammed_cells: Tuple[int, ...]


def candidate_programming_census(noc: Any, *, weight_bits: Any) -> CandidateProgramming:
    """[E2] The programming multiplicands of a candidate's pass structure.

    A resident pass contributes its cores to core INIT and nothing else. An undeclared
    ``weight_bits`` cannot size a payload, so the BYTES stay absent while the core counts count.
    Every core count is of physical tiles (a fused core counts its members, ``PassProgram.tiles``); the payload is
    sized per allocated crossbar (``pass_program_bytes``), as the record sizes ``SegmentCoreRecord.params_bytes``.
    """
    programs = () if noc is None else noc.pass_programs
    reprogrammed = tuple(cells for program in programs if not program.resident
                         for cells in program.core_cells)
    payloads = None if weight_bits is None else tuple(
        b for program in programs if not program.resident for b in pass_program_bytes(program, weight_bits))
    return CandidateProgramming(
        segment_cores=sum(_tiles(program) for program in programs),
        committed_cells=sum(sum(program.core_cells) for program in programs),
        reprogrammed_cores=sum(_tiles(program) for program in programs if not program.resident),
        reprogrammed_bytes=None if payloads is None else sum(payloads),
        reprogram_passes=sum(1 for p in programs if not p.resident),
        pass_cores=tuple(_tiles(program) for program in programs),
        reprogrammed_cells=reprogrammed,
    )


def _tiles(program: Any) -> int:
    """A pass program's physical tiles (``PassProgram.tiles``); a program that states none counts its cores."""
    return int(getattr(program, "tiles", program.cores))


#: Why each plane that cannot see operand regions states none (N2): its write terms refuse under an operand core type.
CANDIDATE_OPERANDS_UNSTATED = ("the candidate plane does not lower products onto operand cores (they stay host ops "
                               "there), so it states no operand regions")
RECORD_OPERANDS_UNSTATED = "the sealed record's schema states no operand regions per pass"


def candidate_program_passes(softcores: Sequence[Any], noc: Any, *, semantics: Any,
                             timesteps: Optional[int], dynamic_banks=None) -> Optional[ProgramPasses]:
    """A candidate's passes, one row per (planned pass, segment), with the spiking rule's steps; ``dynamic_banks``
    (the program's bound banks) states its operand regions, None that the plane does not lower products.

    A hard core a shape-only pack shares between segments belongs to its lowest segment, so no core
    is counted twice (the builder never shares one). ``semantics`` duck-types the executed-window
    branch (``retimed``, ``is_cycle``, ``is_cascade``, ``latency_group_count``, ``chip_latency``).
    A pass program that states each core's ``core_geometry`` (axons, neurons) types its cores;
    otherwise the census types them only on a chip of one core type. One entry per physical tile (its cells, sign
    rows and operand cells), a fused core's members per ``PassProgram.core_tile_cells``.
    """
    if noc is None:
        return None
    runs = getattr(noc, "runs", None)
    segment_of = spec_segments(softcores, runs)
    rows: List[PassRow] = []
    position: Dict[Tuple[int, int], int] = {}
    for index, (placements, program) in enumerate(zip(noc.pass_placements, noc.pass_programs)):
        kept = resident_core_indices(program, program.cores)
        geometry = getattr(program, "core_geometry", None)
        signs = tuple(getattr(program, "core_sign_rows", ()) or ())
        split = getattr(program, "core_tile_cells", ())  # one row entry per physical tile (a fused core's members)
        split_signs = getattr(program, "core_tile_sign_rows", ())
        owner: Dict[int, int] = {}
        for softcore, hardcore in placements:
            previous = owner.get(hardcore)
            owner[hardcore] = segment_of[softcore] if previous is None else min(previous, segment_of[softcore])
        for segment in sorted({segment_of[softcore] for softcore, _ in placements}):
            cores = sorted(h for h, s in owner.items() if s == segment)
            latencies = execution_stage_latencies(
                softcores, [[pl for pl in placements if segment_of[pl[0]] == segment]],
                retimed=bool(semantics.retimed), runs=runs)
            position[(index, segment)] = len(rows)
            tiles = [split[h] if split else (program.core_cells[h],) for h in cores]
            first = [sum(len(t) for t in tiles[:i]) for i in range(len(cores))]
            operand = operand_cells_per_tile(
                candidate_operand_cells(softcores, placements, cores, dynamic_banks or ()), tiles)
            rows.append(PassRow(
                segment_index=segment, resident=bool(program.resident),
                core_cells=tuple(int(c) - o for c, o in zip((c for t in tiles for c in t), operand)),
                producer_cells=sum(operand), producer_cores=frozenset(i for i, o in enumerate(operand) if o),
                producer_core_cells=tuple(operand) if any(operand) else None,
                marks=frozenset(first[i] + j for i, h in enumerate(cores) if h in kept for j in range(len(tiles[i]))),
                core_extents=None,
                compute_steps=None if timesteps is None else program_latency_steps(
                    stage_max_latencies=latencies, timesteps=int(timesteps), is_cycle=semantics.is_cycle,
                    is_cascade=semantics.is_cascade, latency_group_count=semantics.latency_group_count,
                    chip_latency=semantics.chip_latency),
                stage_latencies=tuple(latencies),
                core_types=None if geometry is None else tuple(  # a fused core's members are of one type
                    (int(geometry[h][0]) // len(t), int(geometry[h][1])) for h, t in zip(cores, tiles) for _ in t),
                core_sign_rows=tuple(r for h, t in zip(cores, tiles) for r in (
                    split_signs[h] if split_signs else (int(signs[h]),) + (0,) * (len(t) - 1))) if signs else (),
            ))
    widths: Tuple[BoundaryWidths, ...] = ()
    if noc.census is not None:
        widths = boundary_widths_of(
            ((w, position[(made, segment_of[node])], position[(last, segment_of[node])])
             for node, w, made, last in carried_softcore_spans(softcores, noc.pass_placements,
                                                                noc.census.pair_wires, segment_of)),
            ((w, position[(reader, segment_of[node])]) for node, w, reader in delayed_carry_wires(softcores, noc)))
    return ProgramPasses(rows=tuple(rows), boundary_widths=widths,
                         operands_unstated=None if dynamic_banks is not None else CANDIDATE_OPERANDS_UNSTATED)


def mapping_program_passes(hybrid_mapping: Any) -> ProgramPasses:
    """A built program's passes (record plane at build time) with its per-boundary carried widths.

    A build runs no simulation, so compute steps are unstated (the value rule's window reads the
    stage latencies instead).
    """
    stages = hybrid_mapping.stages
    banks = getattr(hybrid_mapping, "dynamic_bank_bindings", None) or {}
    rows: List[PassRow] = []
    position: Dict[int, int] = {}
    for index, stage in enumerate(stages):
        if stage.kind != "neural":
            continue
        cores = list(stage.hard_core_mapping.cores)
        occupancy = [CoreOccupancy.from_hard_core(c) for c in cores]
        operand = operand_cells_by_core(stage.hard_core_mapping, banks) if banks else [0] * len(cores)
        span = stage_latency([getattr(c, "latency", None) for c in cores])
        latencies = None if span is None else (span,)
        position[index] = len(rows)
        tiles = [tuple((int(a), o.neurons_physical) for a in getattr(c, "fused_component_axons", None) or ())
                 or ((o.axons_physical, o.neurons_physical),) for c, o in zip(cores, occupancy)]
        cells, marks, extents, types = expand_to_tiles(  # one entry per physical tile (a fused core's members)
            [(o.cells_used, (o.axons_used, o.neurons_used), t) for o, t in zip(occupancy, tiles)],
            resident_core_indices(stage, len(cores)) if getattr(stage, SCHEDULE_RESIDENT_CORES, None) is not None
            else ())
        operand = operand_cells_per_tile(operand, tiles)
        rows.append(PassRow(
            segment_index=int(stage.schedule_segment_index if stage.schedule_segment_index is not None
                              else len(rows)),
            resident=bool(getattr(stage, "schedule_weights_resident", False)),
            core_cells=tuple(c - p for c, p in zip(cells, operand)),
            producer_cells=sum(operand), producer_cores=frozenset(i for i, c in enumerate(operand) if c),
            producer_core_cells=tuple(operand) if any(operand) else None,
            marks=marks, core_extents=extents, compute_steps=None, stage_latencies=latencies, core_types=types,
            core_sign_rows=tuple(r for c, o, t in zip(cores, occupancy, tiles)
                                 for r in tile_rows(core_sign_rows(c, o), t)),
        ))
    widths = boundary_widths_of(((w, position[made], position[last])
                                 for _node, w, made, last in carried_wire_spans(stages)),
                                ((w, position[reader]) for _node, w, reader in delayed_input_slices(stages)))
    return ProgramPasses(rows=tuple(rows), boundary_widths=widths)


def record_program_passes(record: Any) -> ProgramPasses:
    """A sealed record's passes; compute steps from the measured stage timing where sealed.

    A sealed record keeps no per-wire spans, so it states no per-boundary carry (its carry totals
    ride ``PassCarryRecord``).
    """
    measured: Dict[int, int] = {}
    for row in record.timing.per_segment:
        measured[int(row.stage_index)] = measured.get(int(row.stage_index), 0) + int(row.timesteps_executed)
    declared = declared_core_geometries(record.identity.platform)
    rows = []
    for segment in record.schedule.segments():
        span = stage_latency([c.static_delay_levels for c in segment.cores])
        latencies = None if span is None else (span,)
        steps = measured.get(int(segment.stage_index))
        # One entry per physical tile; a geometry the declared types do not determine stays one entry of its own
        # geometry, which the census refuses by name (a core type the chip does not declare).
        tiles = [declared_tiles(c.axons, c.neurons, declared) or ((c.axons, c.neurons),) for c in segment.cores]
        cells, marks, extents, types = expand_to_tiles(
            [(c.cells_used, (c.axons_used, c.neurons_used), t) for c, t in zip(segment.cores, tiles)],
            segment.resident_core_indices or ())
        rows.append(PassRow(
            segment_index=segment.segment_index, resident=segment.programming == "resident",
            core_cells=cells, marks=marks, core_extents=extents,
            compute_steps=steps, stage_latencies=latencies, core_types=types,
            core_sign_rows=tuple(r for c, t in zip(segment.cores, tiles)
                                 for r in tile_rows(c.sign_register_rows or 0, t)),
        ))
    return ProgramPasses(rows=tuple(rows), operands_unstated=RECORD_OPERANDS_UNSTATED)


def declared_core_geometries(platform: Any) -> Tuple[Tuple[int, int], ...]:
    """The (axons, neurons) of every core type a record's platform declares completely."""
    return tuple(sorted({(int(c["max_axons"]), int(c["max_neurons"])) for c in (platform or {}).get("cores") or ()
                         if "max_axons" in c and "max_neurons" in c}))

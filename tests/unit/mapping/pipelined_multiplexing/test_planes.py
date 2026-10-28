"""[U-F4, P-2] The candidate plane prices the pipelined schedule the builder builds: on lane programs (U-E2's lane-aware
shape-only walk, whose wires through absorbed ops are core-to-core) both planes state the same passes, physical cores
and cells per pass, the cores each pass writes per period, the carry census, one evaluation per wavefront pass, and
the same MAC events and core initializations (a head's dark cores loaded, never evaluated).

[U-N] One programming census (``payload_sizes.candidate_programming_census``, which ``program_census`` re-exports): on
any pass programs it is the every-core figures less the dark cores' physical tiles and cells, and on the lane programs
it is the sealed record of the program the builder builds, field by field, through a full ``from_record`` call."""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build import payload_sizes as PS
from mimarsinan.deployment_record.build import program_census as PCN
from mimarsinan.deployment_record.build.program_census import candidate_programming_census, declared_core_geometries
from mimarsinan.deployment_record.design_point.pricing import executed_record
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.noc.pass_program import PassProgram
from mimarsinan.mapping.packing.residency_tiles import declared_tiles
from mimarsinan.mapping.support.wavefront_stages import dark_cores
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.mapping.lanes.lane_planes import STEPS, TRANSFER, builder_plane, candidate_plane
from unit.mapping.lanes.test_lane_shape_walk import FIXTURES, _assert_same, _fixture

#: Lane chips the residual and LeNet fixtures run pipelined on: copies of every level together (8 cores), and in
#: two or more groups (3 cores; LeNet's 784-axon classifier needs more cores in one pass, so LeNet takes 12).
GEOMETRIES = {
    "together": dict(axons=256, neurons=64, count=8, passes=1 << 16, coalescing=True, hardware_bias=True),
    "groups": dict(axons=256, neurons=64, count=3, passes=1 << 16, coalescing=True, hardware_bias=True),
}
COUNTS = {("lenet", "groups"): 12}
#: The carry census both programs share: every carried value crosses once each way (the peak live buffer differs).
SHARED_CARRY = ("transfer", "carried_wires", "carried_bytes", "boundary_out_bytes", "boundary_in_bytes")


def _pipelined(fixture: str, geometry: str):
    shape = dict(GEOMETRIES[geometry], count=COUNTS.get((fixture, geometry), GEOMETRIES[geometry]["count"]))
    platform, program = _fixture(fixture, ("L", "E"), multiplexing=True, **shape)
    platform["pipelined_multiplexing"] = True
    return platform, program


def _ladder_noc(program, platform, num_classes):
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD
    from unit.mapping.lanes.lane_planes import ladder, network_of

    network = network_of(program, num_classes)
    census = LD.shape_census(network, platform)
    scheduled = LD.schedule_program(network, census, platform, ladder().program_facts,
                                    tiles=LD.packed_tiles(census, platform))
    assert scheduled.refusal is None, scheduled.refusal
    return scheduled, list(census.softcores)


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("fixture", ["residual", "lenet"])
def test_both_planes_state_the_pipelined_program(fixture, geometry):
    platform, program = _pipelined(fixture, geometry)
    candidate, builder = candidate_plane(program, platform, FIXTURES[fixture][2]), builder_plane(program, platform)
    _assert_same(candidate, builder)
    per_pass = builder_plane(program, {k: v for k, v in platform.items() if k != "pipelined_multiplexing"})
    assert builder["passes"] <= per_pass["passes"]
    if fixture == "residual":  # LeNet's pools are host ops: each of its segments holds one level, so nothing shares
        assert builder["passes"] < per_pass["passes"]
    carry, base = candidate["carry"], per_pass["carry"]
    assert {k: carry[k] for k in carry if k in SHARED_CARRY} == {k: base[k] for k in base if k in SHARED_CARRY}


@pytest.mark.parametrize("fixture", ["residual", "lenet"])
def test_both_planes_price_one_evaluation_per_wavefront_pass_and_the_evaluated_cores(fixture):
    platform, program = _pipelined(fixture, "together")
    scheduled, _specs = _ladder_noc(program, platform, FIXTURES[fixture][2])
    mapping = build_hybrid_mapping_for_pipeline(program.ir, platform)
    neural = [s for s in mapping.stages if s.kind == "neural"]
    rows = scheduled.facts.passes.rows
    assert [r.stage_latencies for r in rows] == [(0,)] * len(neural)
    programming = candidate_programming_census(scheduled.noc, weight_bits=4)
    evaluated = [c for s in neural for i, c in enumerate(s.hard_core_mapping.cores) if i not in dark_cores(s)]
    assert programming.committed_cells == sum(CoreOccupancy.from_hard_core(c).cells_used for c in evaluated)
    tiles = sum(CoreOccupancy.from_hard_core(c).tiles for c in evaluated)
    loaded = sum(CoreOccupancy.from_hard_core(c).tiles for s in neural for c in s.hard_core_mapping.cores)
    assert programming.segment_cores == tiles <= loaded
    if fixture == "residual":  # its wavefront heads load cores their first pass does not use
        assert tiles < loaded


@pytest.mark.parametrize("value", [None, False])
def test_absent_or_false_is_position_multiplexing_on_both_planes(value):
    platform, program = _fixture("residual", ("L", "E"), multiplexing=True, **GEOMETRIES["together"])
    base = candidate_plane(program, platform, FIXTURES["residual"][2])
    if value is not None:
        platform = {**platform, "pipelined_multiplexing": value}
    assert candidate_plane(program, platform, FIXTURES["residual"][2]) == base


@pytest.mark.slow
@pytest.mark.parametrize("geometry", [dict(axons=256, neurons=64, count=96, passes=1 << 16, coalescing=True,
                                           hardware_bias=True),
                                      dict(axons=256, neurons=64, count=24, passes=1 << 16, coalescing=True,
                                           hardware_bias=True)], ids=["96_cores", "24_cores"])
def test_the_quarter_width_resnet18_with_fo_star_s_pass_shape(geometry):
    platform, program = _fixture("resnet18_quarter", ("L", "E"), multiplexing=True, **geometry)
    platform["pipelined_multiplexing"] = True
    candidate, builder = candidate_plane(program, platform, 10), builder_plane(program, platform)
    _assert_same(candidate, builder)


# --- [U-N] one programming census (R-1 review, minors 1 and 2) --------------------------------------------------------

FIELDS = tuple(field.name for field in dataclasses.fields(PS.CandidateProgramming))
#: The lane programs: each fixture at each geometry, pipelined and position-multiplexed one level per pass.
CASES = [(fixture, geometry, pipelined) for fixture in ("residual", "lenet") for geometry in sorted(GEOMETRIES)
         for pipelined in (True, False)]


def _row(census):
    return {name: getattr(census, name) for name in FIELDS}


def _every_core_less_dark(noc, weight_bits):
    """The seven fields by the arithmetic of every core a pass loads (``PassProgram.tiles``, every core's cells), then
    each wavefront head's dark cores out: their physical tiles (``core_tile_cells``) out of the segment cores and their
    cells out of the committed cells. The written cores, bytes, passes and cells and the per-pass cores keep them (the
    head loads its dark cores). Returns the fields and the number of dark cores."""
    programs = () if noc is None else noc.pass_programs
    written = [p for p in programs if not p.resident]
    dark = [(p, i) for p in programs for i in sorted(p.dark_cores or ())]
    return {
        "segment_cores": sum(p.tiles for p in programs)
        - sum(len(p.core_tile_cells[i]) if p.core_tile_cells else 1 for p, i in dark),
        "committed_cells": sum(sum(p.core_cells) for p in programs) - sum(p.core_cells[i] for p, i in dark),
        "reprogrammed_cores": sum(p.tiles for p in written),
        "reprogrammed_bytes": None if weight_bits is None else sum(
            b for p in written for b in PS.pass_program_bytes(p, weight_bits)),
        "reprogram_passes": len(written),
        "pass_cores": tuple(p.tiles for p in programs),
        "reprogrammed_cells": tuple(cells for p in written for cells in p.core_cells),
    }, len(dark)


def _random_program(rng):
    """A pass program of 0 to 6 cores: fused tiles, sign rows, residency and a dark subset each drawn at random."""
    n = rng.randint(0, 6)
    cells = tuple(rng.randint(0, 4096) for _ in range(n))
    fused = n and rng.random() < 0.5
    tile_cells = tuple(tuple(rng.randint(0, 2048) for _ in range(rng.randint(1, 3))) for _ in range(n)) if fused else ()
    signs = tuple(rng.randint(0, 256) for _ in range(n)) if rng.random() < 0.5 else ()
    subset = frozenset(i for i in range(n) if rng.random() < 0.35)
    dark = rng.choice([None, frozenset(), subset, subset])
    return PassProgram(core_cells=cells, resident=rng.random() < 0.4, core_sign_rows=signs, core_tile_cells=tile_cells,
                       wavefront=bool(dark), dark_cores=dark)


def test_one_programming_census_on_every_import_path():
    """One definition, in ``payload_sizes``: ``program_census`` and the search's candidate facts hand out that one
    function and type, and the carry module states no census of its own."""
    from mimarsinan.deployment_record.build import pass_carry
    from mimarsinan.search.problems.joint import candidate_fragments

    assert PCN.candidate_programming_census is PS.candidate_programming_census
    assert candidate_fragments.candidate_programming_census is PS.candidate_programming_census
    assert PCN.CandidateProgramming is PS.CandidateProgramming is candidate_fragments.CandidateProgramming
    assert not hasattr(pass_carry, "candidate_programming_census")


def test_the_census_is_every_core_less_the_dark_ones_on_random_pass_programs():
    rng = random.Random(20261029)
    seen = {True: 0, False: 0}
    for _ in range(2000):
        noc = None if rng.random() < 0.02 else SimpleNamespace(
            pass_programs=tuple(_random_program(rng) for _ in range(rng.randint(0, 7))))
        bits = rng.choice([None, 1, 3, 4, 8])
        expected, dark = _every_core_less_dark(noc, bits)
        assert _row(PS.candidate_programming_census(noc, weight_bits=bits)) == expected
        seen[dark > 0] += 1
    assert min(seen.values()) > 500, seen


def test_a_program_stating_no_tiles_or_dark_cores_counts_each_core_once_and_evaluates_all():
    """A ``PassProgram`` twin that states no physical tiles and no dark cores (``g3_fixtures.HandPassProgram``) reads
    one tile per core, every core evaluated."""
    from unit.deployment_record.g3_fixtures import HandPassProgram

    noc = SimpleNamespace(pass_programs=(HandPassProgram(core_cells=(6, 4, 9)), HandPassProgram((5,), resident=True)))
    census = PS.candidate_programming_census(noc, weight_bits=None)
    assert (census.segment_cores, census.committed_cells, census.pass_cores) == (4, 24, (3, 1))
    assert (census.reprogrammed_cores, census.reprogram_passes, census.reprogrammed_cells) == (3, 1, (6, 4, 9))


def _lane_program(fixture, geometry, pipelined):
    platform, program = _pipelined(fixture, geometry)
    if not pipelined:
        platform = {k: v for k, v in platform.items() if k != "pipelined_multiplexing"}
    scheduled, _specs = _ladder_noc(program, platform, FIXTURES[fixture][2])
    return platform, program, scheduled


@pytest.mark.parametrize("fixture,geometry,pipelined", CASES)
def test_the_census_is_every_core_less_the_dark_ones_on_the_lane_programs(fixture, geometry, pipelined):
    _platform, _program, scheduled = _lane_program(fixture, geometry, pipelined)
    for bits in (None, 4, 8):
        expected, dark = _every_core_less_dark(scheduled.noc, bits)
        assert _row(PS.candidate_programming_census(scheduled.noc, weight_bits=bits)) == expected
    assert (dark > 0) == (fixture == "residual" and pipelined)  # LeNet's heads load no unused core


@pytest.mark.parametrize("fixture,geometry,pipelined", CASES)
def test_the_candidate_census_is_the_sealed_record_field_by_field(fixture, geometry, pipelined):
    """The candidate's census of a lane program against the record of the program the builder builds, sealed from its
    mapping as the deployment-record step seals it (``executed_record``) and read by a full ``from_record`` call: the
    five scalar fields, the pass count and the allocation (``pass_cores`` summed); the per-pass physical cores and the
    written cells off the sealed segments by ``from_record``'s tile rule (``declared_tiles``)."""
    platform, program, scheduled = _lane_program(fixture, geometry, pipelined)
    census = candidate_programming_census(scheduled.noc, weight_bits=4)
    record = executed_record(build_hybrid_mapping_for_pipeline(program.ir, platform), platform, weight_bits=4,
                             label="pipelined lane program", timesteps=STEPS, pass_transfer=TRANSFER)
    segments = list(record.schedule.segments())
    assert any(s.dark_core_indices for s in segments) == (fixture == "residual" and pipelined)
    sealed = from_record(record)
    assert not sealed.withheld, sealed.withheld
    declared = declared_core_geometries(platform)

    def tiles(core):
        return len(declared_tiles(core.axons, core.neurons, declared) or ()) if declared else 1

    stated = {**_row(census), "pass_count": len(census.pass_cores), "cores_allocated": sum(census.pass_cores),
              "macs": census.committed_cells}
    read = {key: sealed.get(key).value for key in ("segment_cores", "reprogrammed_cores", "reprogrammed_bytes",
                                                   "reprogram_passes", "pass_count", "cores_allocated", "macs")}
    read.update(committed_cells=sealed.get("cells_used").value,
                pass_cores=tuple(sum(tiles(c) for c in s.cores) for s in segments),
                reprogrammed_cells=tuple(c.cells_used for s in segments if s.programming == "reprogram"
                                         for c in s.cores))
    assert stated == read

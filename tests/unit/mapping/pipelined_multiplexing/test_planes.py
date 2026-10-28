"""[U-F4, P-2] The candidate plane prices the pipelined schedule the builder builds: on lane programs (U-E2's lane-aware
shape-only walk, whose wires through absorbed ops are core-to-core) both planes state the same passes, physical cores
and cells per pass, the cores each pass writes per period, the carry census, one evaluation per wavefront pass, and
the same MAC events and core initializations (a head's dark cores loaded, never evaluated)."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.pass_carry import candidate_programming_census
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.support.wavefront_stages import dark_cores
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.mapping.lanes.lane_planes import builder_plane, candidate_plane
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

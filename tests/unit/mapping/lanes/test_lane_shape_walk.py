"""[U-E2 H-1] The exploration's shape-only plane applies the declared lane plan as the build applies it.

The builder fuses the planned lanes into the IR at Soft Core Mapping (``mapping/lanes/install.py``): absorbed host ops
leave the program, fused adds give their accumulators operand rows, pool lanes add reduction cores, reductions
replace their producers, and the wires through an absorbed op are core-to-core. The shape-only plane walked the
lane-free program on the same chip (``objectives/context.py``'s note), so its segments, host census, passes and
carry contradicted the records it prices. Here both planes state the same facts on lane programs built the
pipeline's way: neural segments, host ops, the planner's specs (operand rows included), passes, physical cores and
cells per pass (an operand row is no cell, as the record counts it), and the carry census; on chips that run the
lanes, at every class subset, on one pool, scheduled, coalesced, with on-core bias and under position multiplexing.
"""

from __future__ import annotations

import functools

import pytest

from unit.chip_simulation.value_run.integer_programs import lenet5, mlp, residual_block_net, resnet18_quarter
from unit.mapping.lanes.lane_planes import (
    builder_depth_specs,
    builder_plane,
    candidate_plane,
    ir_geometry,
    plane_platform,
)
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform
from unit.mapping.lanes.test_lane_reduction import transition_net

ALL = ("L", "E", "P")
#: name -> (model, input shape, classes)
FIXTURES = {
    "residual": (residual_block_net, (3, 16, 16), 10),
    "lenet": (lenet5, (1, 28, 28), 10),
    "transition": (transition_net, (3, 16, 16), 10),
    "mlp": (mlp, (8,), 4),
    "resnet18_quarter": (resnet18_quarter, (3, 32, 32), 10),
}
#: Chips wide enough for every fixture's fan-in (or coalescing it), on one pool or scheduled over a few cores.
GEOMETRIES = {
    "one_pool": dict(axons=1024, neurons=64, count=4096),
    "scheduled": dict(axons=1024, neurons=32, count=8, passes=512),
    "coalesced": dict(axons=64, neurons=32, count=64, passes=2048, coalescing=True),
    "on_core_bias": dict(axons=1024, neurons=32, count=8, passes=512, hardware_bias=True),
}


@functools.lru_cache(maxsize=None)
def _program(fixture: str, classes, axons: int, neurons: int, hardware_bias: bool, coalescing: bool,
             declared: tuple = ()):
    """The fixture's lane program, mapped at the chip's tiler limits and fused on its declared cores (the geometry a
    reduction core stays within), as Soft Core Mapping maps and fuses it."""
    model, shape, num_classes = FIXTURES[fixture]
    extra = {"cores": [dict(core) for core in declared]} if declared else {}
    return build_lane_program(model(), shape, num_classes=num_classes, platform=lane_platform(classes, **extra),
                              max_axons=axons, max_neurons=neurons, hardware_bias=hardware_bias,
                              allow_coalescing=coalescing)


def _fixture(fixture: str, classes, plan=None, **geometry):
    platform = plane_platform(classes, **geometry)
    if plan is not None:
        platform["residency_plan"] = dict(plan)
    ir = ir_geometry(platform)
    declared = tuple(tuple(sorted(core.items())) for core in platform["cores"])
    return platform, _program(fixture, tuple(classes), ir["max_axons"], ir["max_neurons"], ir["hardware_bias"],
                              ir["allow_coalescing"], declared)


def _planes(fixture: str, classes, plan=None, **geometry):
    platform, program = _fixture(fixture, classes, plan, **geometry)
    return candidate_plane(program, platform, FIXTURES[fixture][2]), builder_plane(program, platform)


def _assert_same(candidate, builder):
    for key in ("segments", "host_ops", "specs", "passes", "tiles", "cells", "written", "carry"):
        assert candidate[key] == builder[key], f"{key}: shape-only plane {candidate[key]} != builder {builder[key]}"


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("fixture", sorted(set(FIXTURES) - {"resnet18_quarter"}))
def test_both_planes_state_one_lane_program(fixture, geometry):
    _assert_same(*_planes(fixture, ALL, **GEOMETRIES[geometry]))


@pytest.mark.parametrize("classes", [("L",), ("L", "E"), ("L", "P")])
def test_every_class_subset_is_applied_as_the_build_applies_it(classes):
    _assert_same(*_planes("residual", classes, **GEOMETRIES["scheduled"]))


@pytest.mark.parametrize("fixture", ["residual", "lenet"])
def test_position_multiplexing_runs_the_same_passes_on_both_planes(fixture):
    # E without P: position multiplexing is illegal beside executed pool lanes (entries_residency)
    _assert_same(*_planes(fixture, ("L", "E"), axons=1024, neurons=32, count=8, passes=4096, multiplexing=True))


def test_absorbed_ops_leave_the_host_census_and_the_segmentation():
    candidate, builder = _planes("residual", ALL, **GEOMETRIES["scheduled"])
    assert candidate["host_ops"] == builder["host_ops"] == 0
    assert candidate["segments"] == builder["segments"] == 1
    assert any(row[2] for row in candidate["specs"]), "fused adds bring their operand rows"
    assert candidate["carry"]["carried_wires"] > 0, "wires through an absorbed op are carried between passes"


def test_a_lane_chip_off_the_integer_path_walks_the_lane_free_program():
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD
    from unit.mapping.lanes.lane_planes import network_of

    geometry = GEOMETRIES["scheduled"]
    lanes = plane_platform(ALL, readout_law=None, **geometry)
    plain = plane_platform(None, readout_law=None, **geometry)
    program = _program("residual", ALL, *(ir_geometry(lanes)[k] for k in ("max_axons", "max_neurons",
                                                                          "hardware_bias", "allow_coalescing")))
    network = network_of(program, 10)
    a, b = LD.shape_census(network, lanes), LD.shape_census(network, plain)
    assert (a.softcores, a.runs, a.segments, a.host_segments) == (b.softcores, b.runs, b.segments, b.host_segments)
    assert a.wire_census == b.wire_census


@pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "index", "index": 0}, {"step": "index", "index": 1}])
def test_a_declared_residency_plan_composes_one_step_function_on_both_planes(plan):
    """The lane-free walk saw no core-to-core wire, so its step function had one step and a declared index above it
    refused on the shape-only plane alone (B5i); the lane program's cut census is the builder's."""
    geometry = GEOMETRIES["scheduled"]
    candidate, builder = _planes("residual", ALL, plan, **geometry)
    for key in ("segments", "host_ops", "passes", "tiles", "cells", "written", "carry"):
        assert candidate[key] == builder[key], f"{key}: shape-only plane {candidate[key]} != builder {builder[key]}"
    platform, program = _fixture("residual", ALL, plan, **geometry)
    expected = builder_depth_specs(program, platform)
    order = {s: i for i, s in enumerate(dict.fromkeys(row[4] for row in candidate["specs"]))}
    assert [row[:4] + (order[row[4]],) + row[5:] for row in candidate["specs"]] == expected


def test_an_index_beyond_the_step_function_is_refused_by_one_name_on_both_planes():
    from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError

    platform, program = _fixture("residual", ALL, {"step": "index", "index": 3}, **GEOMETRIES["scheduled"])
    with pytest.raises(ResidencyPlanError) as candidate:
        candidate_plane(program, platform, 10)
    with pytest.raises(ResidencyPlanError) as builder:
        builder_plane(program, platform)
    assert str(candidate.value) == str(builder.value), "the shape-only plane refuses by the builder's name"


def test_the_lane_work_is_the_fusions_count_and_the_stage_says_the_walk_applied_it():
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD
    from mimarsinan.deployment_record.objectives.context import (
        CANDIDATE_LANE_WALK_NOTE,
        candidate_context_from_platform,
    )
    from mimarsinan.mapping.lanes.install import LANE_REPORT
    from unit.mapping.lanes.lane_planes import network_of

    platform, program = _fixture("transition", ALL, **GEOMETRIES["scheduled"])
    census = LD.shape_census(network_of(program, 10), platform)
    assert census.lane_ops == getattr(program.ir, LANE_REPORT)["lane_ops"]
    stage = candidate_context_from_platform(platform, lane_ops=census.lane_ops, lane_walked=True).output_stage
    assert stage is not None and stage["notes"] == [CANDIDATE_LANE_WALK_NOTE] and stage["lane_ops"] == census.lane_ops



@pytest.mark.parametrize("site_geometry", [0, 1], ids=["one_pool", "scheduled"])
@pytest.mark.parametrize("classes", [("L",), ("L", "E"), ALL], ids=lambda c: ",".join(c))
def test_a_module_called_at_several_sites_takes_one_role_per_site_on_both_planes(classes, site_geometry):
    """[U-E2 over U-K2] The walk takes an op's role by its call site, from the fusion's own site table
    (``call_sites.planned_call_sites``, matched as ``call_site_roles`` matches it on the IR): one ReLU module called
    after each sum of a block is a lane where the plan absorbs that call and a host op where it does not."""
    from unit.mapping.lanes.test_lane_call_sites import GEOMETRIES as SITE_GEOMETRIES, SHAPE, _net

    platform = plane_platform(classes, **SITE_GEOMETRIES[site_geometry])
    ir = ir_geometry(platform)
    program = build_lane_program(_net(True), SHAPE, num_classes=10,
                                 platform=lane_platform(classes, cores=[dict(c) for c in platform["cores"]]),
                                 max_axons=ir["max_axons"], max_neurons=ir["max_neurons"],
                                 hardware_bias=ir["hardware_bias"], allow_coalescing=ir["allow_coalescing"])
    _assert_same(candidate_plane(program, platform, 10), builder_plane(program, platform))


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("classes,declared", [
    (ALL, dict(passes=4096)),
    (("L", "E"), dict(passes=8192, multiplexing=True)),
])
def test_a_residual_network_at_the_route_b_geometry_runs_the_same_program_on_both_planes(classes, declared):
    """CIFAR ResNet-18 at width 1/4 on 256 x 64 cores with on-core bias and coalescing, 64 cores: U1 under the
    lanes L, E, P, and one weight copy per bank (position multiplexing) under L and E, every pass on both planes."""
    _assert_same(*_planes("resnet18_quarter", classes, axons=256, neurons=64, count=64, coalescing=True,
                          hardware_bias=True, **declared))

"""[U-L, L-1] Plane parity with the pool lane on the one-copy arm: the candidate plane (the exploration's shape-only
plane with the lane plan applied, U-E2) and the builder's plane state the same neural segments, host ops, planner
specs, passes, physical cores and cells per pass, cores written per period and carry census, on W02 and in its
pipelined form (U-F4); the sealed record plane states the builder's passes, cells, residency and writes; and the
pool lane's one host op leaves both planes' host census."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import mapping_program_passes, record_program_passes

from unit.deployment_record.test_wavefront_record import _record
from unit.mapping.lanes.lane_planes import builder_plane, candidate_plane
from unit.mapping.lanes.test_lane_shape_walk import _assert_same

from . import vehicles as V

ARMS = {"one_copy": None, "pipelined": True}


def _planes(network: str, arm, **shape):
    platform = V.chip(V.POOL, pipelined=arm, **shape)
    prog = V.program(network, V.POOL, platform)
    return prog, platform, candidate_plane(prog, platform, V.NETWORKS[network][2]), builder_plane(prog, platform)


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", ["residual", "shared", "transition"])
def test_both_planes_state_the_pool_lane_program(network, geometry, arm):
    _prog, _platform, candidate, builder = _planes(network, ARMS[arm], **V.GEOMETRIES[geometry])
    _assert_same(candidate, builder)


@pytest.mark.parametrize("network", ["residual", "transition"])
def test_the_pool_lane_leaves_the_host_census_on_both_planes(network):
    shape = V.GEOMETRIES["together"]
    _p, _q, pool, _b = _planes(network, None, **shape)
    platform = V.chip(V.HOST_POOL, **shape)
    host = candidate_plane(V.program(network, V.HOST_POOL, platform), platform, V.NETWORKS[network][2])
    assert pool["host_ops"] < host["host_ops"]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_record_plane_states_the_builders_passes_and_writes(arm):
    shape = V.GEOMETRIES["groups"]
    platform = V.chip(V.POOL, pipelined=ARMS[arm], **shape)
    hybrid = V.build(V.program("residual", V.POOL, platform), platform)
    schedule = schedule_record_from_mapping(hybrid, weight_bits=4, params_reloaded=0, timesteps=1,
                                            pass_transfer="collapse")
    sealed, built = record_program_passes(_record(schedule)), mapping_program_passes(hybrid)
    assert len(sealed.rows) == len(built.rows) == len(V.neural(hybrid))
    assert [r.core_cells for r in sealed.rows] == [r.core_cells for r in built.rows]
    assert [r.resident for r in sealed.rows] == [r.resident for r in built.rows]
    assert [r.marks for r in sealed.rows] == [r.marks for r in built.rows]
    assert [r.dark for r in sealed.rows] == [r.dark for r in built.rows]


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_both_planes_state_the_quarter_width_resnet18_at_the_route_b_tile(arm):
    """The route (b) one-copy rows' shape (ResNet-18 at width 1/4 on 256 x 64 tiles, 64 cores) with the pool on its
    P lane: every pass on both planes, W02 and pipelined."""
    _prog, _platform, candidate, builder = _planes("resnet18_quarter", ARMS[arm], count=64, **V.ROUTE_B)
    _assert_same(candidate, builder)

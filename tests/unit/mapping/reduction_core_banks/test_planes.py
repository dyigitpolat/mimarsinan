"""[U-L2, B-1] Plane parity with banked reduction cores: the candidate plane (the exploration's shape-only plane with
the lane plan applied, U-E2) and the builder's plane state the same neural segments, host ops, planner specs (each
instance's bank included), passes, physical cores and cells per pass, cores written per period and carry census, on
the one-copy arm (W02) and its pipelined form (U-F4); the sealed record plane states the builder's passes, cells,
residency, marks and dark cores."""

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
    assert V.lane_banks(prog)
    return candidate_plane(prog, platform, V.NETWORKS[network][2]), builder_plane(prog, platform)


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", ["residual", "shared", "transition"])
def test_both_planes_state_the_banked_program(network, geometry, arm):
    candidate, builder = _planes(network, ARMS[arm], **V.GEOMETRIES[geometry])
    _assert_same(candidate, builder)
    assert any(row[5] is not None and row[6] is None for row in builder["specs"]), "a pool bank's instance spec"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_record_plane_states_the_builders_passes_and_writes(arm):
    shape = V.GEOMETRIES["groups"]
    platform = V.chip(V.POOL, pipelined=ARMS[arm], **shape)
    prog = V.program("residual", V.POOL, platform)
    assert V.lane_banks(prog), "the vehicle banks its pool lane"
    hybrid = V.build(prog, platform)
    schedule = schedule_record_from_mapping(hybrid, weight_bits=4, params_reloaded=0, timesteps=1,
                                            pass_transfer="collapse")
    sealed, built = record_program_passes(_record(schedule)), mapping_program_passes(hybrid)
    assert len(sealed.rows) == len(built.rows) == len(V.neural(hybrid))
    for field in ("core_cells", "resident", "marks", "dark"):
        assert [getattr(r, field) for r in sealed.rows] == [getattr(r, field) for r in built.rows], field


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_both_planes_state_the_quarter_width_resnet18_banked_at_the_route_b_tile(arm):
    """The route (b) one-copy rows' shape (ResNet-18 at width 1/4 on 256 x 64 tiles, 64 cores) with the pool on its
    P lane, banked: every pass on both planes, W02 and pipelined."""
    candidate, builder = _planes("resnet18_quarter", ARMS[arm], count=64, **V.ROUTE_B)
    _assert_same(candidate, builder)

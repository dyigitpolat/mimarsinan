"""[U-L2, B-1] W15's static check establishes Theorem 1's program conditions on programs that run a pool lane's banked
reduction cores on the one-copy arm (W02) and in its pipelined form (U-F4): every condition established, H10
non-vacuous over the resident passes (a bank's later instances executing on the copy its run's first pass wrote, the
pool banks included), and no unknown or unread program form (a lane bank is a weight bank of the IR, each instance a
bank-backed core)."""

from __future__ import annotations

import functools

import pytest

from mimarsinan.certification.program_check.report import ESTABLISHED

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.mapping.reduction_core_banks import vehicles as V

ARMS = {"one_copy": None, "pipelined": True}


@functools.lru_cache(maxsize=None)
def _built(network: str, geometry: str, arm: str):
    shape = V.GEOMETRIES[geometry]
    platform = V.chip(V.POOL, pipelined=ARMS[arm], **shape)
    prog = V.program(network, V.POOL, platform)
    return prog, V.build(prog, platform), platform


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", ["residual", "transition"])
def test_a_banked_pool_lane_program_on_the_one_copy_arm_is_established(network, geometry, arm):
    prog, hybrid, platform = _built(network, geometry, arm)
    banks = V.lane_banks(prog)
    assert banks, "the vehicle banks its pool lane"
    report = P.run(hybrid, prog.ir, UV.value_decl(platform["cores"]))
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
    assert not report.unknown_forms and not report.unread_forms, (report.unknown_forms, report.unread_forms)
    h10 = report.results["H10"]
    assert h10.status == ESTABLISHED and h10.checked > 0, h10.note
    placed = {int(p["ir_node_id"]) for s in V.neural(hybrid)
              for ps in s.hard_core_mapping.soft_core_placements_per_hard_core for p in ps}
    assert {int(c.id) for instances in banks.values() for c in instances} <= placed, "every instance is placed"

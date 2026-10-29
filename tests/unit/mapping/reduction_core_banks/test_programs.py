"""[U-L2, B-1 and B-2] Banked reduction cores on the one-copy arm (W02) and in its pipelined form (U-F4): the builder
places each bank's instances on one copy in turn, the planner orders them by their records (U-L), and both programs
are exact at atol 0, on the value executor and its int64 golden, against U-L's unbanked P-lane program (the same
integer function, core for core) and the identity program, on U-L's test networks (U-F4's lane-fused residual block,
U-K2's network whose ReLU module runs at two call sites, the transition network's reductions beside a pool lane) at
two geometries, and on the quarter-width ResNet-18 at the route (b) rows' tile.

The host-pool program. U-L found that the P-lane program and the program with the pool on the host (L, E) are two
integer functions (the lane rounds once onto its own grid; ``pool_lane_multiplexing.test_programs``'
``TestTheFunctionAgainstTheHostPool``), so no executor makes either equal to the host-pool program. The closest
correct check, as U-L's: the banked program's difference to the host-pool program is the unbanked one's, value for
value, on every arm (the banks add none)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.weight_programming import weight_programming_report

from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.lanes.lane_programs import real_inputs

from . import vehicles as V

SMALL = ["residual", "shared", "transition"]
ARMS = (False, True)  # the one-copy arm (W02), its pipelined form


def _built(prog, shape, *, arm=None, banks=True):
    """``prog`` built on U1 (``arm`` None), the one-copy arm (False) or its pipelined form (True)."""
    return V.build(prog, V.chip(V.POOL, banks=banks if arm is not None else None, multiplexing=arm is not None,
                                pipelined=arm or None, **shape))


def _inputs(network, prog, n=8):
    return real_inputs(prog, n) if network == "shared" else integer_inputs(prog, n)


@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", SMALL)
def test_the_banked_programs_equal_u_l_s_unbanked_program_and_the_golden_at_atol_0(network, geometry):
    shape = V.GEOMETRIES[geometry]
    platform = V.chip(V.POOL, **shape)
    prog, before = V.program(network, V.POOL, platform), V.unbanked(network, V.POOL, platform)
    assert V.lane_banks(prog), "the vehicle banks its pool lane"
    x = _inputs(network, prog)
    want = V.value(_built(before, shape, arm=False, banks=None), x)
    assert_exact(V.value(before.identity().hybrid_mapping, x), want)
    assert_exact(V.value(prog.identity().hybrid_mapping, x), want)
    assert_exact(V.value(_built(prog, shape), x), want)  # U1 places one copy per instance: the same function
    for arm in ARMS:
        hybrid = _built(prog, shape, arm=arm)
        got = V.value(hybrid, x)
        assert_exact(got, want)
        codes, steps = run_golden(hybrid, x.numpy())
        assert np.array_equal(codes * steps, got.reshape(codes.shape).numpy())
    with torch.no_grad():
        assert_exact(prog.nf()(x.double()), want)


@pytest.mark.parametrize("network", SMALL)
def test_one_copy_of_each_bank_serves_its_instances_in_turn(network):
    """On the one-copy arm every pass holding a bank's instance holds exactly one instance of that bank, each placed on
    the copy its run's first pass programmed (later passes weight-resident), after every pass placing a producer."""
    shape = V.GEOMETRIES["together"]
    prog = V.program(network, V.POOL, V.chip(V.POOL, **shape))
    hybrid = _built(prog, shape, arm=False)
    by_id = {int(n.id): n for n in prog.ir.get_neural_cores()}
    assert V.lane_banks(prog), "the vehicle banks its pool lane"
    for bank, instances in V.lane_banks(prog).items():
        ids = {int(c.id) for c in instances}
        holding = [k for k, s in enumerate(V.neural(hybrid)) if any(
            int(p["ir_node_id"]) in ids for ps in s.hard_core_mapping.soft_core_placements_per_hard_core for p in ps)]
        assert len(holding) == len(instances), (bank, holding)
        for k in holding:
            stage = V.neural(hybrid)[k]
            placed = [int(p["ir_node_id"]) for ps in stage.hard_core_mapping.soft_core_placements_per_hard_core
                      for p in ps if int(p["ir_node_id"]) in ids]
            assert len(placed) == 1
        assert [bool(V.neural(hybrid)[k].schedule_weights_resident) for k in holding] == [False] + [True] * (
            len(holding) - 1)
        for core in instances:
            producers = {int(s.node_id) for s in core.input_sources.flatten() if int(s.node_id) in by_id}
            assert max(V.passes_of(hybrid, producers)) < min(V.passes_of(hybrid, [core.id]))


@pytest.mark.parametrize("network", SMALL)
def test_every_bank_is_programmed_once_per_run_and_the_pool_s_cells_shrink(network):
    """Each bank's cells are programmed once per run on both arms, so the one-copy program writes one copy of a pool
    lane's matrix per distinct matrix where U-L's wrote one per window group: exactly (instances - 1) copies fewer
    per bank."""
    shape = V.GEOMETRIES["together"]
    platform = V.chip(V.POOL, **shape)
    prog, before = V.program(network, V.POOL, platform), V.unbanked(network, V.POOL, platform)
    banks = V.lane_banks(prog)
    assert banks, "the vehicle banks its pool lane"
    for arm in ARMS:
        report = weight_programming_report(_built(prog, shape, arm=arm))
        assert report.params_programmed == report.params_unique
    report = weight_programming_report(_built(prog, shape, arm=False))
    old = weight_programming_report(_built(before, shape, arm=False, banks=None))
    saved = sum((len(instances) - 1) * int(prog.ir.weight_banks[bank].core_matrix.size)
                for bank, instances in banks.items())
    assert old.params_unique - report.params_unique == saved


@pytest.mark.parametrize("network", SMALL)
def test_control_declared_false_the_programs_are_u_l_s(network):
    """Control (green before the change): declared false, the one-copy and pipelined programs are U-L's (decision
    absent), pass for pass and placement for placement."""
    shape = V.GEOMETRIES["together"]
    platform = V.chip(V.POOL, banks=False, **shape)
    off, before = V.program(network, V.POOL, platform, banks=False), V.unbanked(network, V.POOL, platform)
    for arm in ARMS:
        assert _placements(_built(off, shape, arm=arm, banks=False)) == _placements(
            _built(before, shape, arm=arm, banks=None))


def _placements(hybrid) -> list:
    return [(bool(s.schedule_weights_resident), tuple(
        tuple(sorted((int(p["ir_node_id"]), int(p["axon_offset"]), int(p["neuron_offset"]), int(p["axons"]),
                      int(p["neurons"])) for p in ps)) for ps in s.hard_core_mapping.soft_core_placements_per_hard_core))
            for s in V.neural(hybrid)]


class TestTheFunctionAgainstTheHostPool:
    def test_the_banked_and_unbanked_programs_differ_from_the_host_pool_program_alike_on_every_arm(self):
        shape = V.GEOMETRIES["together"]
        platform = V.chip(V.POOL, **shape)
        prog, before = V.program("residual", V.POOL, platform), V.unbanked("residual", V.POOL, platform)
        host = V.unbanked("residual", V.HOST_POOL, V.chip(V.HOST_POOL, banks=None, **shape))
        assert V.lane_banks(prog), "the vehicle banks its pool lane"
        x = integer_inputs(prog, 16)
        want_host = V.value(host.identity().hybrid_mapping, x)
        want = V.value(before.identity().hybrid_mapping, x)
        assert float((want - want_host).abs().max()) > 0, "the pool lane rounds onto its own grid"
        for arm in ARMS:
            got = V.value(_built(prog, shape, arm=arm), x)
            assert_exact(got - want_host, want - want_host)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("count", [96, 24], ids=["96_cores", "24_cores"])
def test_the_quarter_width_resnet18_at_the_route_b_tile_is_exact_banked(count):
    """The quarter-width ResNet-18 at 32 x 32 on 256 x 64 tiles (on-core bias, coalescing) with the pool on its P lane,
    banked: the one-copy and pipelined programs equal U-L's unbanked program, the identity program and the golden at
    atol 0. Its head pool (128 channels at 4 x 4, 16 windows per 256-axon group) is one bank of 8 instances, so the
    one-copy arm runs the pool level as 8 passes on one copy where U-L's ran its 8 owned cores in one pass."""
    platform = V.chip(V.POOL, count=count, **V.ROUTE_B)
    prog, before = V.program("resnet18_quarter", V.POOL, platform), V.unbanked("resnet18_quarter", V.POOL, platform)
    banks = V.lane_banks(prog)
    assert [len(v) for v in banks.values()] == [8], {k: len(v) for k, v in banks.items()}
    x = integer_inputs(prog, 4)
    want = V.value(prog.identity().hybrid_mapping, x)
    assert_exact(V.value(before.identity().hybrid_mapping, x), want)
    passes = {}
    for arm in ARMS:
        hybrid = V.build(prog, V.chip(V.POOL, count=count, pipelined=arm or None, **V.ROUTE_B))
        assert_exact(V.value(hybrid, x), want)
        codes, steps = run_golden(hybrid, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), want)
        old = V.build(before, V.chip(V.POOL, banks=None, count=count, pipelined=arm or None, **V.ROUTE_B))
        passes[arm] = (len(V.neural(hybrid)), len(V.neural(old)))
    assert passes[False][0] == passes[False][1] + 7, passes  # the pool level: 8 passes on one copy, not 1
    assert passes[True][0] <= passes[True][1] + 7, passes
    if count == 96:
        assert passes[False] == (6473, 6466), passes

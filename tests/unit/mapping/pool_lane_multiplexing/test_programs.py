"""[U-L, L-1 and L-2] Pool lanes under position multiplexing: the builder places and schedules the pool-lane cores on
the one-copy arm (W02: one copy per bank, positions in turn) and in its pipelined form (U-F4), and both programs are
exact at atol 0, on the value executor and its int64 golden, on the lane-fused residual networks (U-F4's block, U-K2's
network whose ReLU module runs at two call sites, the transition network's reductions) and on the quarter-width
ResNet-18 at the route (b) rows' tile.

The reference. A pool lane requantizes the average onto its own grid with one rounding (``pool_cores``), so the P-lane
program is another integer function than the program with the pool on the host (L, E), which rounds the host's
average at the next segment's entry grid: the lane changes the function, as every lane does (``test_lane_programs``'s
``test_lanes_change_the_function``), and no executor makes the two equal. So each P-lane program is checked against its
own identity program (one core per IR core), its U1 program, its NF model and the golden; the host-pool program is
checked against its own the same way; and the difference between the two, which the multiplexing does not touch, is
pinned as a difference (``TestTheFunctionAgainstTheHostPool``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.wavefront_stages import wavefront_head
from mimarsinan.mapping.weight_programming import weight_programming_report

from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.lanes.lane_programs import real_inputs

from . import vehicles as V

SMALL = ["residual", "shared", "transition"]


def _programs(network: str, geometry: str):
    shape = V.GEOMETRIES[geometry]
    platform = V.chip(V.POOL, **shape)
    prog = V.program(network, V.POOL, platform)
    built = {arm: V.build(prog, V.chip(V.POOL, multiplexing=arm is not None, pipelined=arm or None, **shape))
             for arm in (None, False, True)}  # U1, the one-copy arm (W02), its pipelined form
    return prog, built


@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", SMALL)
def test_the_one_copy_and_pipelined_programs_equal_the_identity_program_and_the_golden_at_atol_0(network, geometry):
    prog, built = _programs(network, geometry)
    x = real_inputs(prog, 8) if network == "shared" else integer_inputs(prog, 8)
    want = V.value(prog.identity().hybrid_mapping, x)
    assert_exact(V.value(built[None], x), want)
    for arm in (False, True):
        got = V.value(built[arm], x)
        assert_exact(got, want)
        codes, steps = run_golden(built[arm], x.numpy())
        assert np.array_equal(codes * steps, got.reshape(codes.shape).numpy())
    with torch.no_grad():
        assert_exact(prog.nf()(x.double()), want)


@pytest.mark.parametrize("network", SMALL)
def test_the_pool_lane_cores_run_on_chip_after_their_producers(network):
    """The builder places every inserted core, each in a pass after every pass that places one of its producers."""
    prog, built = _programs(network, "together")
    inserted = V.lane_cores(prog)
    assert inserted
    by_id = {int(n.id): n for n in prog.ir.get_neural_cores()}
    for arm in (False, True):
        hybrid = built[arm]
        for core in inserted:
            placed = V.passes_of(hybrid, [core.id])
            assert placed, f"{core.name} is placed on the {'pipelined' if arm else 'one-copy'} program"
            producers = {int(s.node_id) for s in core.input_sources.flatten() if int(s.node_id) in by_id}
            assert max(V.passes_of(hybrid, producers)) < min(placed)
        assert len(V.neural(hybrid)) > len(V.neural(built[None]))
    assert len(V.neural(built[True])) <= len(V.neural(built[False]))


@pytest.mark.parametrize("network", SMALL)
def test_every_bank_is_programmed_once_per_run(network):
    _prog, built = _programs(network, "together")
    for arm in (False, True):
        report = weight_programming_report(built[arm])
        assert report.params_programmed == report.params_unique


def test_the_pool_lane_level_keeps_its_own_passes_between_the_wavefront_groups():
    """A pool lane's (and a reduction's) cores are owned cores, each window group with its own weights, so no copy
    serves several of them: on the one-copy arm their level takes the capacity split, and in the pipelined form it
    keeps those passes between wavefront groups, as U-F4 keeps every level holding owned cores. Every pass holding an
    inserted core holds its level alone on both arms; the pipelined program has no more passes (fewer where bank
    levels share a wavefront)."""
    for network in ("residual", "transition"):
        prog, built = _programs(network, "together")
        level = {int(n.id): int(n.latency) for n in prog.ir.get_neural_cores()}
        lanes = {int(c.id) for c in V.lane_cores(prog)}
        for arm in (False, True):
            holding = [{level[n] for n in placed} for placed in (
                {int(p["ir_node_id"]) for ps in s.hard_core_mapping.soft_core_placements_per_hard_core for p in ps}
                for s in V.neural(built[arm])) if placed & lanes]
            assert holding and all(len(levels) == 1 for levels in holding), (network, arm)
        assert any(wavefront_head(s) is not None for s in V.neural(built[True]))
        assert len(V.neural(built[True])) <= len(V.neural(built[False]))
        if network == "residual":  # its bank levels share wavefront passes (the transition's reductions own theirs)
            assert len(V.neural(built[True])) < len(V.neural(built[False]))


class TestTheFunctionAgainstTheHostPool:
    """The departure from "exactness against the program with the pool on the host": the two are two functions (the
    lane rounds once onto its own grid). Each is exact against itself on every arm, and the difference between them
    is the same on U1 and on the one-copy arm: the multiplexing adds none."""

    def test_the_p_lane_program_and_the_host_pool_program_differ_by_the_same_amount_on_every_arm(self):
        shape = V.GEOMETRIES["together"]
        x = integer_inputs(V.program("residual", V.POOL, V.chip(V.POOL, **shape)), 16)
        outputs = {}
        for classes in (V.POOL, V.HOST_POOL):
            prog = V.program("residual", classes, V.chip(classes, **shape))
            want = V.value(prog.identity().hybrid_mapping, x)
            for arm in (None, False, True):
                got = V.value(V.build(prog, V.chip(classes, multiplexing=arm is not None, pipelined=arm or None,
                                                   **shape)), x)
                assert_exact(got, want)
            outputs[classes] = want
        delta = (outputs[V.POOL] - outputs[V.HOST_POOL]).abs()
        assert float(delta.max()) > 0, "the pool lane rounds onto its own grid: another integer function"


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("count", [96, 24], ids=["96_cores", "24_cores"])
def test_the_quarter_width_resnet18_at_the_route_b_tile_is_exact_on_both_arms(count):
    """The quarter-width ResNet-18 at 32 x 32 on 256 x 64 tiles (on-core bias, coalescing) with the pool on its P lane:
    the one-copy and pipelined programs equal the identity program and the golden at atol 0. The pool lane adds one
    level after the last block (one pass per run of it on the one-copy arm), and the pipelined program shares it."""
    platform = V.chip(V.POOL, count=count, **V.ROUTE_B)
    prog = V.program("resnet18_quarter", V.POOL, platform)
    host = V.program("resnet18_quarter", V.HOST_POOL, V.chip(V.HOST_POOL, count=count, **V.ROUTE_B))
    x = integer_inputs(prog, 4)
    want = V.value(prog.identity().hybrid_mapping, x)
    passes = {}
    for arm in (False, True):
        hybrid = V.build(prog, V.chip(V.POOL, count=count, pipelined=arm or None, **V.ROUTE_B))
        assert_exact(V.value(hybrid, x), want)
        codes, steps = run_golden(hybrid, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), want)
        host_passes = len(V.neural(V.build(host, V.chip(V.HOST_POOL, count=count, pipelined=arm or None,
                                                        **V.ROUTE_B))))
        passes[arm] = (len(V.neural(hybrid)), host_passes)
    assert passes[True][0] < passes[False][0]
    assert passes[False][0] == passes[False][1] + 1, passes  # one pass for the pool lane's level on W02
    if count == 96:
        assert passes == {False: (6466, 6465), True: (1267, 1266)}, passes


class TestRefusalsByName:
    def test_a_pass_budget_below_the_positions_is_refused_by_name(self):
        shape = dict(V.GEOMETRIES["together"], passes=16)
        platform = V.chip(V.POOL, **shape)
        prog = V.program("residual", V.POOL, platform)
        with pytest.raises(PositionMultiplexingError, match="above max_schedule_passes 16"):
            V.build(prog, platform)

    def test_an_inserted_core_without_its_record_is_refused_by_name(self):
        """The planner still orders passes by the layout record: a fused IR whose inserted record was dropped (as
        before this unit) is refused by name, never planned on a guessed level."""
        import copy

        from mimarsinan.mapping.ir.lane_law import INSERTED_RECORD

        shape = V.GEOMETRIES["together"]
        platform = V.chip(V.POOL, **shape)
        prog = copy.deepcopy(V.program("residual", V.POOL, platform))
        core = V.lane_cores(prog)[0]
        core.layout_softcore_index = None
        delattr(core, INSERTED_RECORD)
        with pytest.raises(PositionMultiplexingError, match="no layout record"):
            V.build(prog, platform)

    def test_the_pipelined_form_under_per_hop_retiming_is_refused_by_name(self):
        from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
        from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

        shape = V.GEOMETRIES["together"]
        prog = V.program("residual", V.POOL, V.chip(V.POOL, **shape))
        caps = ChipCapabilities(allow_scheduling=True, max_schedule_passes=V.PASSES, position_multiplexing=True,
                                pipelined_multiplexing=True, allow_coalescing=True, allow_neuron_splitting=True)
        with pytest.raises(PositionMultiplexingError, match="per-hop re-timing"):
            build_hybrid_hard_core_mapping(ir_graph=prog.ir, cores_config=V.chip(V.POOL, **shape)["cores"],
                                           strategy=MappingStrategy.resolve(caps), max_schedule_passes=V.PASSES,
                                           retimed_level_stages=True)

"""[U-CG] Plane parity where a coalescing group alone exceeds one pass: a group whose members each hold their whole
fan-in is planned between its members on both planes, the executed program runs the passes the shape-only plane
plans, and its outputs are the bits of the program that holds the group in one pass.

The vehicle (``planes.vehicle("wide_group")``): Flatten, Linear(300, 300), ReLU, Linear(300, 10) on 128 x 128
crossbars, where the first layer is three output tiles of three fused crossbars (one group of nine crossbars) and the
second one fused core of three. Below nine crossbars and from three up, the group cannot share one pass while every
tile fits one; before the rule both planes refused these budgets by one name. The capacity estimate (U-PP's
``estimate_cores_needed``, which binds one fused core) and the planner now agree there. A point the one-pass rule
admitted keeps its plan on every plane-parity vehicle."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.verification.capacity.estimate import estimate_cores_needed

from . import planes as P
from .test_planes import PAIRS, _budgets

A = N = 128
SPLIT = (3, 4, 6, 8)        # at least one tile (3 crossbars), below the group (9)
WHOLE = (9, 12)             # the group fits one pass


@pytest.fixture(scope="module")
def rep():
    return P.representation("wide_group")


def _cores(count: int):
    return [{"max_axons": A, "max_neurons": N, "count": int(count), "has_bias": True}]


def _build(rep, count: int):
    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=A, max_neurons=N, allow_coalescing=True,
                   hardware_bias=True).map(rep)
    rep.clear_ir_caches()
    IRLatency(ir).calculate()
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=True,
                            max_schedule_passes=P.PASSES)
    hybrid = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(count),
                                            strategy=MappingStrategy.resolve(caps), max_schedule_passes=P.PASSES)
    return ir, hybrid


def _groups(ir):
    groups = {}
    for node in ir.get_neural_cores():
        if node.coalescing_group_id is not None:
            groups.setdefault(int(node.coalescing_group_id), []).append(node)
    return groups


def test_the_vehicle_holds_one_group_of_whole_fan_in_members(rep):
    ir, _ = _build(rep, 9)
    groups = _groups(ir)
    members = max(groups.values(), key=len)
    assert len(members) == 3 and {m.coalescing_role for m in members} == {"master"}
    assert sum(math.ceil(m.core_matrix.shape[0] / A) for m in members) == 9


@pytest.mark.parametrize("count", SPLIT + WHOLE)
def test_both_planes_plan_and_build_the_same_passes(rep, count):
    layout, specs = P.walk(rep, A, N)
    shape = P.shape_answer(layout, specs, A, N, count, scheduled=True)
    executed = P.executed_answer(rep, A, N, count, scheduled=True)
    assert shape.ok and executed.ok, (shape, executed)
    assert shape.passes == executed.passes and shape.census == executed.census, (shape, executed)


@pytest.mark.parametrize("count", SPLIT)
def test_the_executed_program_splits_the_group_between_whole_members(rep, count):
    ir, hybrid = _build(rep, count)
    members = {int(m.id) for m in max(_groups(ir).values(), key=len)}
    units = {}
    for index, stage in enumerate(s for s in hybrid.stages if s.kind == "neural"):
        for records in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            for r in records:
                if int(r["ir_node_id"]) in members:
                    units.setdefault(int(r["ir_node_id"]), set()).add(index)
    assert set(units) == members
    assert all(len(u) == 1 for u in units.values())                    # never inside a member
    assert len(set().union(*units.values())) > 1                        # the group spans passes


def test_the_outputs_are_the_one_pass_programs_bits(rep):
    torch.manual_seed(1)
    x = torch.rand(6, 3, 10, 10, dtype=torch.float64) * 2.0 - 1.0
    outputs = {}
    for count in SPLIT + WHOLE:
        _, hybrid = _build(rep, count)
        with torch.no_grad():
            fp64 = ValueHybridCoreFlow(hybrid, device="cpu", dtype=torch.float64).eval()(x)
            fp32 = ValueHybridCoreFlow(hybrid, device="cpu", dtype=torch.float32).eval()(x.to(torch.float32))
        outputs[count] = (fp64, fp32)
    want64, want32 = outputs[9]
    for count, (fp64, fp32) in outputs.items():
        assert torch.equal(fp64, want64) and torch.equal(fp32, want32), count


@pytest.mark.parametrize("count", SPLIT)
def test_the_capacity_estimate_and_the_planner_agree(rep, count):
    ir, _ = _build(rep, 9)
    estimate = estimate_cores_needed(ir, {"cores": _cores(count), "allow_coalescing": True}, allow_scheduling=True)
    assert estimate.feasible and estimate.largest_atomic_unit == 3
    _build(rep, count)  # the planner builds it


def test_below_one_member_both_refuse_by_one_name(rep):
    layout, specs = P.walk(rep, A, N)
    shape = P.shape_answer(layout, specs, A, N, 2, scheduled=True)
    executed = P.executed_answer(rep, A, N, 2, scheduled=True)
    assert (shape.ok, executed.ok) == (False, False) and shape.refusal == executed.refusal == "CoreCapacityRefusal"
    ir, _ = _build(rep, 9)
    assert not estimate_cores_needed(ir, {"cores": _cores(2), "allow_coalescing": True},
                                     allow_scheduling=True).feasible


@pytest.fixture(scope="module")
def reps():
    return {name: P.representation(name) for name in {p[0] for p in PAIRS}}


@pytest.mark.parametrize("name,a,n", PAIRS, ids=[f"{p[0]}-{p[1]}x{p[2]}" for p in PAIRS])
def test_every_point_the_one_pass_rule_admits_keeps_its_plan(reps, name, a, n):
    """On the shape-only plane, the walk's specs with their roles against the same specs with the roles cleared (the
    one-pass rule): wherever the latter plans, the plan, its passes and its census are the same."""
    rep = reps[name]
    layout, specs = P.walk(rep, a, n)
    plain = [replace(s, coalescing_role=None) for s in specs]   # the walk's runs read only the spec count
    moved = []
    for count in _budgets(P.footprint(specs, a, n)):
        before = P.shape_answer(layout, plain, a, n, count, scheduled=True)
        now = P.shape_answer(layout, specs, a, n, count, scheduled=True)
        if before.ok:
            assert now == before, (count, before, now)
        elif now != before:
            moved.append(count)
    if name == "wide_fc":   # one group of five whole-fan-in tiles of thirteen fused crossbars on 16 x 16
        assert moved, "the wide vehicle's group binds below its bound"

"""[F2] Order identity on programs that planned before the unit order read the layer graph: the units, the executed
plan's rows and marks, the record plane's census and the plan plane's census equal a record frozen on the head of
record c913118818b83b2225562556427a397321fa6aaf (``tests/fixtures/residency_front/order_identity_c9131188.json``,
written by the residency unit's ``freeze_order_identity.py`` on an export of that head).

The vehicles: the seeded LIF chain at three budgets and two branchy LIF graphs whose builds replay a carried producer
(late_skip, deep_skip), the lane residual net with clamp lanes only (class L, its adds on the host: it planned on the
head), and the tier-0 cells t0_22 and t0_03 at the budgets of the spiking-cell certificate. On the two replaying graphs
the plan plane now states the builder's replays, so their plan census is the record's carry instead of the head's
(``test_plan_carry.TestThePlanesAgreeWhereTheBuilderReplays``); every other field is the head's.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.schedule.residency_units import facts_period_census

from . import certified as CF
from . import vehicles as V

FIXTURE = Path(__file__).resolve().parents[4] / "fixtures" / "residency_front" / "order_identity_c9131188.json"
STEPS = ("floor", "stateless")
#: (vehicle, budget) of the default suite; the slow ones build the lane net and the tier-0 cells.
FAST = [("chain", 5), ("chain", 6), ("chain", 8), ("late_skip", 2), ("deep_skip", 5)]
SLOW = [("lane_L", None), ("t0_22", 0.7), ("t0_03", 1.0)]
#: The cases whose builds replay a carried producer: their plan census moved to the record's carry by design.
REPLAYING = {"late_skip@2:floor", "late_skip@2:stateless", "deep_skip@5:floor"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _units(units) -> List[Any]:
    """The fields the head's units carry (members, tiles, live sets, forced cuts, layers, segments, cores, spans)."""
    return [list(map(list, units.members)), list(units.tiles), list(units.live), list(units.forced),
            [str(x) for x in units.layers], list(units.segments), [list(map(list, c)) for c in units.cores],
            [list(s) for s in (units.spans or ())]]


def _observed(hybrid, comp, *, bits: int, capacity: Dict[Tuple[int, int], int]) -> Dict[str, Any]:
    record = period_census(mapping_program_passes(hybrid), weight_bits=bits, capacity=capacity,
                           bytes_of=lambda w: w).census
    plan = facts_period_census(comp.facts, forced=comp.units.forced, bytes_of_cells=lambda c: params_bytes(c, bits),
                               tile_cells=0, bytes_per_element=1)
    return {"units": _digest(_units(comp.units)), "unit_count": len(comp.units.members),
            "plan_rows": CF.executed_plan(hybrid), "marks": _digest([list(m) for m in CF.executed_marks(hybrid)]),
            "passes": len(V.neural_passes(hybrid)), "record_census": _digest(record.to_dict()),
            "plan_census": _digest(plan.to_dict()),
            "record_carry": {str(k): list(v) for k, v in CF.carried(record).items()},
            "plan_carry": {str(k): list(v) for k, v in CF.carried(plan).items()}}


def _lif(name: str, count: int, step: str) -> Dict[str, Any]:
    rep = V.chain_repr(seed=0) if name == "chain" else V.branchy_repr(name)
    ir = V.ir_of(rep)
    plan = {"step": step}
    hybrid = V.build(ir, count, plan, max_schedule_passes=64)
    comp = CF.builder_composition(ir, {"cores": V.cores(count), "allow_scheduling": True, "max_schedule_passes": 64},
                                  plan)
    return _observed(hybrid, comp, bits=8, capacity={(V.MAX_AXONS, V.MAX_NEURONS): count})


def _lane(step: str) -> Dict[str, Any]:
    from unit.chip_simulation.value_run.integer_programs import residual_block_net
    from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform

    with contextlib.redirect_stdout(io.StringIO()):
        platform = lane_platform(["L"])
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=platform)
    ir = program.ir
    a = max(int(n.get_core_matrix(ir).shape[0]) for n in ir.nodes if isinstance(n, NeuralCore))

    def build(cores_config, caps):
        return build_hybrid_hard_core_mapping(ir_graph=copy.deepcopy(ir), cores_config=copy.deepcopy(cores_config),
                                              strategy=MappingStrategy.resolve(ChipCapabilities(**caps)),
                                              max_schedule_passes=caps.get("max_schedule_passes", 8))

    with contextlib.redirect_stdout(io.StringIO()):
        flat = build([{"max_axons": a + 2, "max_neurons": 3, "count": 4096}], dict(allow_neuron_splitting=True))
    F = sum(len(s.hard_core_mapping.cores) for s in V.neural_passes(flat))
    cores_config = [{"max_axons": a + 2, "max_neurons": 3, "count": math.ceil(F / 3)}]
    caps = dict(allow_neuron_splitting=True, allow_scheduling=True, max_schedule_passes=4096,
                residency_plan={"step": step})
    with contextlib.redirect_stdout(io.StringIO()):
        hybrid = build(cores_config, caps)
    surface = {"cores": cores_config, "allow_neuron_splitting": True, "allow_scheduling": True,
               "max_schedule_passes": 4096}
    comp = CF.builder_composition(ir, surface, {"step": step})
    return _observed(hybrid, comp, bits=4, capacity={(a + 2, 3): cores_config[0]["count"]})


def _template(name: str, scale: float, step: str) -> Dict[str, Any]:
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

    from . import templates as TV

    vehicle = TV.template_vehicle(name)
    platform = TV.residency_platform(vehicle, scale, {"step": step})
    with contextlib.redirect_stdout(io.StringIO()):
        hybrid = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    comp = CF.builder_composition(vehicle.ir, {k: v for k, v in platform.items() if k != "residency_plan"},
                                  {"step": step})
    capacity = {(int(c["max_axons"]), int(c["max_neurons"])): int(c["count"]) for c in platform["cores"]}
    return _observed(hybrid, comp, bits=int(vehicle.config.get("weight_bits", 8)), capacity=capacity)


def observe(name: str, budget: Any, step: str) -> Dict[str, Any]:
    """The frozen fields of one vehicle at one step (the freeze script calls this on the head's export)."""
    if name == "lane_L":
        return _lane(step)
    if name.startswith("t0_"):
        return _template(name, float(budget), step)
    return _lif(name, int(budget), step)


def case_id(name: str, budget: Any, step: str) -> str:
    return f"{name}@{budget}:{step}"


def _check(name, budget, step):
    case = case_id(name, budget, step)
    frozen = json.loads(FIXTURE.read_text())["cases"][case]
    now = observe(name, budget, step)
    fields = [k for k in frozen if not (case in REPLAYING and k in ("plan_census", "plan_carry"))]
    assert {k: now[k] for k in fields} == {k: frozen[k] for k in fields}
    if case in REPLAYING:                       # the plan census states the replays: the record's carry
        assert now["plan_carry"] == now["record_carry"] != frozen["plan_carry"]
    else:
        assert now["plan_carry"] == now["record_carry"]


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("name,budget", FAST)
def test_a_program_that_planned_on_the_head_keeps_its_units_plan_and_census(name, budget, step):
    _check(name, budget, step)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("name,budget", SLOW)
def test_a_lane_or_tier0_program_that_planned_on_the_head_keeps_its_executed_plan(name, budget, step):
    _check(name, budget, step)

"""[F3] A program with dynamic operand banks (N2) plans and executes a declared residency step.

The vehicle is the two-head attention block of the operand certificates (``operand_programs.attention_program``,
softmax), whose score and value products run on operand cores: their payload is written every inference. The chip is
the operand arm's form (route b-op): one core type of 64 x 64 that declares the operand, at half of the program's
tiles (C = ceil(F / 2), F the tiles of the resident program on unbounded cores), neuron splitting, scheduling. Before
F3 the builder refused every step (the planner did not state which cores hold operand banks).

An operand bank's instance reads the host op that writes its payload: the program graph and the depth segments
count it among the instance's producers (``dynamic_bank.payload_sources`` on the IR, the walk's bindings on the layout
walk), so the planner's segments put the instances after it, as the IR's runs and the walk's spec segments do. Before F3 they counted only the input sources and segmented every instance
beside its activation's producer, before its payload exists (the segmentation-parity rule refused the program).

Per step (floor, stateless, index 1):
- the build executes more than one pass and its plan rows state the declared step; its outputs equal the identity
  program's and the resident program's at atol 0 on 16 inputs, and the operand certificate passes on them (every
  edge at atol 0, every payload cell, a nonzero census);
- no executed pass keeps a core holding an operand bank resident, each executed pass packs and pins the cores of
  the composition's packing of it, and the floor pins cores;
- its record plane equals its plan plane field by field per pass, the producer cells included, and the carry per
  boundary (K12's census, CL3 (d));
- W15 establishes Theorem 1 and Proposition 2 on the executed program, with no unknown program form.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from typing import Any, Dict, Tuple

import pytest
import torch

from mimarsinan.certification.operand_certificate import certify_operand_program
from mimarsinan.certification.program_check.check import check_deployment
from mimarsinan.certification.program_check.report import P2, T1
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.quantities.write_law import plane_write_census
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import dynamic_bank_ids
from mimarsinan.mapping.layout.segmentation import NeuralSegment
from mimarsinan.mapping.packing.schedule_residency_plan import depth_partition
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing.schedule_residency_plan import SCHEDULE_RESIDENCY_PASS
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENCY_PLAN, SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.schedule.residency_units import facts_period_census
from unit.certification.program_check.unit_vehicles import value_decl
from unit.chip_simulation.value_run.operand_programs import attention_program, image_inputs

from . import certified as CF

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

STEPS = ({"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1})
AXONS = NEURONS = 64
WEIGHT_BITS = 8
_CACHE: Dict[Any, Any] = {}


def _build(ir, cores_config, plan=None):
    caps: Dict[str, Any] = dict(allow_neuron_splitting=True, allow_scheduling=plan is not False,
                                max_schedule_passes=4096)
    if plan:
        caps["residency_plan"] = dict(plan)
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_hard_core_mapping(ir_graph=copy.deepcopy(ir), cores_config=copy.deepcopy(cores_config),
                                              strategy=MappingStrategy.resolve(ChipCapabilities(**caps)),
                                              max_schedule_passes=4096)


def _program() -> Tuple[Any, list, Any]:
    """(program, the chip's cores at half of the program's tiles, the resident program on unbounded cores)."""
    if "program" not in _CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            program = attention_program("softmax")
        operand = {"operand": {"payload_bits": 8, "write_source": "host"}}
        flat = _build(program.ir, [{"max_axons": AXONS, "max_neurons": NEURONS, "count": 4096, **operand}], False)
        F = sum(len(s.hard_core_mapping.cores) for s in flat.stages if s.kind == "neural")
        _CACHE["program"] = (program, [{"max_axons": AXONS, "max_neurons": NEURONS, "count": math.ceil(F / 2),
                                        **operand}], flat)
    return _CACHE["program"]


def _executed(plan):
    key = tuple(sorted(plan.items()))
    if key not in _CACHE:
        program, cores_config, _flat = _program()
        _CACHE[key] = _build(program.ir, cores_config, plan)
    return _CACHE[key]


def _neural(hybrid):
    return [s for s in hybrid.stages if s.kind == "neural"]


def _surface(program, cores_config):
    return {**program.platform, "cores": cores_config, "weight_bits": WEIGHT_BITS, "core_semantics": "mvm",
            "allow_neuron_splitting": True, "allow_scheduling": True, "max_schedule_passes": 4096}


def _step_id(plan):
    return plan["step"] if plan["step"] != "index" else f"index{plan['index']}"


@pytest.mark.parametrize("plan", STEPS, ids=_step_id)
def test_an_operand_program_plans_and_executes_a_residency_step(plan):
    program, cores_config, flat = _program()
    hybrid = _executed(plan)
    passes = _neural(hybrid)
    assert len(passes) > 1
    rows = getattr(hybrid, SCHEDULE_RESIDENCY_PLAN)
    assert {p.step_rule for p in rows} == {plan["step"]} and sum(p.passes for p in rows) == len(passes)
    assert all(p.core_budget == cores_config[0]["count"] for p in rows)
    x = image_inputs(16)
    with torch.no_grad():
        want = program.identity()(x)
        got = ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x).reshape(want.shape)
        resident = ValueHybridCoreFlow(flat, dtype=torch.float64)(x).reshape(want.shape)
    assert torch.equal(got, want) and torch.equal(got, resident) and float(want.abs().sum()) > 0
    certificate = certify_operand_program(program.nf(), program.identity_mapping(), hybrid, x)
    assert certificate.passed, certificate.summary()
    assert certificate.payload.cells_compared > 0 and certificate.nonzero_census["operand_core_outputs"] > 0


@pytest.mark.parametrize("plan", STEPS, ids=_step_id)
def test_no_core_holding_an_operand_bank_is_kept_and_each_pass_pins_the_compositions_cores(plan):
    program, _cores_config, _flat = _program()
    hybrid = _executed(plan)
    banks = dynamic_bank_ids(program.ir)
    held, pinned = 0, 0
    for stage in _neural(hybrid):
        placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
        operand = {i for i, ps in enumerate(placements)
                   if any(p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in banks for p in ps)}
        kept = getattr(stage, SCHEDULE_RESIDENT_CORES)
        held, pinned = held + len(operand), pinned + len(kept)
        assert not operand & set(kept), (operand, sorted(kept))
        census = getattr(stage, SCHEDULE_RESIDENCY_PASS)
        assert sorted(census.planned_cores) == sorted(census.executed_cores)
        assert sorted(census.planned_cores[i] for i in census.planned_resident) == sorted(
            census.executed_cores[i] for i in census.executed_resident)
    assert held > 0
    if plan["step"] == "floor":
        assert pinned > 0


@pytest.mark.parametrize("plan", STEPS, ids=_step_id)
def test_its_record_plane_equals_its_plan_plane_with_the_producer_cells(plan):
    program, cores_config, _flat = _program()
    hybrid = _executed(plan)
    surface = _surface(program, cores_config)
    comp = CF.builder_composition(program.ir, surface, plan)
    census = facts_period_census(comp.facts, forced=comp.units.forced,
                                 bytes_of_cells=lambda cells: params_bytes(cells, WEIGHT_BITS), tile_cells=0,
                                 bytes_per_element=1, label="plan")
    written, reason = plane_write_census(mapping_program_passes(hybrid), surface, weight_bits=WEIGHT_BITS,
                                         timesteps=None, label="record")
    assert written is not None, reason
    assert CF.differences(written.census, census) == []
    assert sum(p.written_cells_producer for p in census.passes) > 0
    assert len(comp.facts.phases) == len(_neural(hybrid))


@pytest.mark.parametrize("plan", STEPS[:2], ids=_step_id)
def test_w15_establishes_the_operand_residency_program(plan):
    program, cores_config, _flat = _program()
    hybrid = _executed(plan)
    with contextlib.redirect_stdout(io.StringIO()):
        report = check_deployment(hybrid, ir_graph=program.ir, declaration=value_decl(cores_config),
                                  unbounded_ir=program.ir)
    assert report.established(T1) and report.established(P2), (report.theorem1, report.proposition2,
                                                                report.failures_by_condition())
    assert not report.unknown_forms


def test_an_operand_instance_reads_the_host_op_writing_its_payload_and_follows_it():
    program, _cores_config, _flat = _program()
    ir = program.ir
    bindings = ir.dynamic_bank_bindings
    cores = [n for n in ir.nodes if isinstance(n, NeuralCore)]
    graph = ProgramGraph.of_ir(ir, {int(n.id): i for i, n in enumerate(cores)})
    producers = dict(zip(graph.ids, graph.producers))
    instances = [n for n in cores if getattr(n, "weight_bank_id", None) is not None
                 and int(n.weight_bank_id) in bindings]
    assert instances
    position = {int(n.id): i for i, n in enumerate(ir.nodes)}
    segment_of = {int(n.id): k for k, seg in enumerate(depth_partition(ir)) for n in
                  (seg.nodes if isinstance(seg, NeuralSegment) else [seg.compute_op])}
    for node in instances:
        writer = int(bindings[int(node.weight_bank_id)].source_node_id)
        assert writer in producers[int(node.id)]
        assert segment_of[writer] < segment_of[int(node.id)] and position[writer] < position[int(node.id)]


def test_a_program_whose_depth_segments_differ_from_its_ir_runs_still_refuses_by_name():
    """The relu form writes both products' payloads before either product, so the second product streams after the
    first in the IR's runs and starts a depth segment of its own (its payload's writer is a host op): the
    segmentation-parity rule refuses it by name, as it refuses any program whose depth segments are not its runs."""
    with contextlib.redirect_stdout(io.StringIO()):
        program = attention_program("relu", block={"depth": 1})
    operand = {"operand": {"payload_bits": 8, "write_source": "host"}}
    with pytest.raises(ResidencyPlanError, match="change class"):
        _build(program.ir, [{"max_axons": AXONS, "max_neurons": NEURONS, "count": 8, **operand}], {"step": "floor"})


def test_the_shape_only_plane_plans_the_operand_program_on_its_layout_walk():
    """The layout walk's specs segment an instance after its payload's writer; the walk's program graph counts the
    writer among the instance's producers (``ProgramGraph.of_walk``), so the shape-only plane's composition (what the
    candidate plane and K12's census compose) plans the declared step instead of refusing its own segments."""
    from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
    from mimarsinan.mapping.noc import collect_noc_fragments

    program, cores_config, _flat = _program()
    walk = CF.deferred_walk(program.model.get_mapper_repr(), cores_config, allow_coalescing=False)
    caps = ChipCapabilities(allow_neuron_splitting=True, allow_scheduling=True, max_schedule_passes=4096,
                            residency_plan={"step": "floor"})
    with contextlib.redirect_stdout(io.StringIO()):
        fragments = collect_noc_fragments(softcores=walk.specs, core_types=core_types_from_declaration(cores_config),
                                          census=walk.census, program=walk.graph, **caps.layout_kwargs())
    assert len(fragments.pass_programs) > 1
    writers = {int(b.source_node_id) for b in program.ir.dynamic_bank_bindings.values()}
    assert any(writers & set(p) for p in walk.graph.producers)

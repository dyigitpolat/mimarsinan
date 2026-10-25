"""Value-domain programs under the executed plan: CIFAR ResNet-18 w1/4 and a t0_44-class lenet5 (MVM).

Acceptance 9 (ResNet-18 part): both planes execute the step's phases with the same pins. Acceptance 12: the executed
floor equals the identity value program within the value C edge (fp64, ``VALUE_TWIN_FP64_ATOL``). Acceptance 13:
programmed bytes, defined once as the occupied-cell bytes (``params_bytes``) of every core a pass writes, are the
same number on the planner (the shape-only composition's facts) and in the record of the executed program, at
relative error 0 (both packers pack every ResNet-18 w1/4 phase alike; where they do not, the record's bytes are the
marks', ``test_packing_planes``); here W_exec is at most the step's W, each per-pass difference printed; the
full-tile product W_exec x TB is printed as a bound only.

Both value programs are segment-level: under the MVM conversion every layer is followed by a host op (ResNet-18
w1/4; a plain Linear chain too: a 5-layer MLP gives 5 segments), and the t0_44-class program at C 76 cuts after every
unit, so each pass is one neural segment and the plan's carry is 0. This is a boundary of the conversion, not of the
test: under MVM (and under the spiking conversions measured on ResNet-18, review round 3) residency is segment-level,
so the value domain certifies partial residency as pins across segments, and claim C2's carry term arises only on
chains of neural layers with no host op between them (the spiking CUT cells, ``test_spiking_cells``).
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.certification.value_certificate import VALUE_TWIN_FP64_ATOL
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage
from mimarsinan.mapping.support.schedule.residency_composition import ShapeOnlyResidency, compose_shape_only_program
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from .. import resnet18 as R
from . import templates as TV

AXONS, NEURONS, COUNT = 1153, 128, 400          # F = 1439 tiles on ResNet-18 w1/4: Q >= 2, every unit within C
WEIGHT_BITS = 8


@pytest.fixture(scope="module")
def resnet():
    _model, _flow, rep = R.mvm_mapper_repr(0.25, dtype=torch.float64)
    mapping = IRMapping(q_max=1.0, firing_mode="Default", max_axons=AXONS, max_neurons=NEURONS)
    mapping.collect_wire_census = True
    ir = mapping.map(rep)
    return ir, list(ir.layout_softcores), census_of_walk(mapping), ProgramGraph.of_walk(mapping)


def _caps(plan):
    return ChipCapabilities(allow_scheduling=True, residency_plan=plan, max_schedule_passes=8)


def _shape_only(specs, census, program, plan):
    return compose_shape_only_program(specs, [LayoutHardCoreType(AXONS, NEURONS, COUNT)],
                                      ShapeOnlyResidency(plan, census, program), allow_coalescing=False,
                                      allow_splitting=False, max_schedule_passes=8)


def _build(ir, plan):
    return build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": AXONS, "max_neurons": NEURONS, "count": COUNT}],
        strategy=MappingStrategy.resolve(_caps(plan)))


def _layers_of_passes(hybrid, ir):
    layer_of = {int(n.id): n.perceptron_index for n in ir.get_neural_cores()}
    return [tuple(sorted({layer_of[int(p["ir_node_id"])] for ps in s.hard_core_mapping.soft_core_placements_per_hard_core
                          for p in ps})) for s in hybrid.stages if s.kind == "neural"]


@pytest.mark.slow
@pytest.mark.timeout(900)
class TestResNet18Quarter:
    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
    def test_both_planes_execute_the_step(self, resnet, plan):
        ir, specs, census, program = resnet
        comp = _shape_only(specs, census, program, plan)
        hybrid = _build(ir, plan)
        frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(AXONS, NEURONS, COUNT)],
                                     census=census, program=program, **_caps(plan).layout_kwargs())
        # a layer's tile groups may span phases [W03]: each phase's layers as a set
        step_layers = [tuple(sorted({comp.units.layers[u] for u in range(a, b)})) for a, b in comp.facts.phases]
        frag_layers = [tuple(sorted({specs[i].perceptron_index for i, _ in pl})) for pl in frag.pass_placements]
        assert _layers_of_passes(hybrid, ir) == frag_layers == step_layers
        assert len(step_layers) == comp.step[2] >= 2
        built = [s.schedule_resident_cores for s in hybrid.stages if s.kind == "neural"]
        assert built == [p.schedule_resident_cores for p in frag.pass_programs]
        assert sum(len(p) for p in built) == (0 if plan["step"] == "stateless" else COUNT - comp.r)

    def test_the_executed_floor_equals_the_identity_value_program(self, resnet):
        ir, _specs, _census, _program = resnet
        identity = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=ir), device="cpu",
                                       dtype=torch.float64)
        executed = ValueHybridCoreFlow(_build(ir, {"step": "floor"}), device="cpu", dtype=torch.float64)
        for seed in range(4):
            x = torch.randn(2, *R.INPUT_SHAPE, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                want, got = identity(x), executed(x)
            assert float((got - want).abs().max()) <= VALUE_TWIN_FP64_ATOL
            assert float(want.std(0).max()) > 0

    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
    def test_the_planned_programmed_bytes_are_the_records(self, resnet, plan):
        ir, specs, census, program = resnet
        hybrid = _build(ir, plan)
        comp = _shape_only(specs, census, program, plan)
        record = schedule_record_from_mapping(hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0)
        passes = [s for s in hybrid.stages if s.kind == "neural"]
        segments = [r for r in record.stages if hasattr(r, "cores")]
        recorded, w_exec = 0, 0
        for q, (stage, segment) in enumerate(zip(passes, segments)):
            pins = stage.schedule_resident_cores
            recorded += sum(core.params_bytes for core in segment.cores if core.core_index not in pins)
            w_exec += len(stage.hard_core_mapping.cores) - len(pins)
            print(f"pass {q}: planned {comp.facts.pass_tiles[q]} tiles, packed {len(stage.hard_core_mapping.cores)}"
                  f" (difference {comp.facts.pass_tiles[q] - len(stage.hard_core_mapping.cores)}: joint packing),"
                  f" pinned {len(pins)}")
        planned = comp.facts.programmed_bytes(lambda cells: params_bytes(cells, WEIGHT_BITS))
        assert recorded == planned                                       # relative error 0
        assert w_exec == comp.facts.written_core_tiles <= comp.facts.written_tiles   # W_exec <= the step's W
        tile_bytes = params_bytes(AXONS * NEURONS, WEIGHT_BITS)
        print(f"{plan['step']}: programmed bytes {recorded} (planner {planned}); W_exec {w_exec} x TB {tile_bytes} "
              f"= {w_exec * tile_bytes} (full-tile bound); step W {comp.facts.written_tiles}")
        plans = hybrid.schedule_residency_plan
        assert [(p.rotating_tiles, p.core_budget, p.step_rule, p.total_tiles, p.written_tiles) for p in plans] == [
            (comp.r, COUNT, plan["step"], comp.plan.F, comp.facts.written_tiles)] * len(plans)
        by_segment = comp.segment_passes()
        assert [p.passes for p in plans] == [len(by_segment[s]) for s in sorted(by_segment)]
        assert sum(p.carried_elements for p in plans) == comp.step[1] == 0     # segment-level passes (the boundary)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_t0_44_class_program_executed_equals_the_identity_value_program():
    vehicle = TV.template_vehicle("t0_44")
    # [fix_c] at x4 (48 + 48 cores) the program's unscheduled pack holds (its footprint is 42 + 42), so the fit rule
    # makes the floor that single pool, programmed once; the executed multi-pass floor is checked below the footprint
    platform = TV.residency_platform(vehicle, 3.4, {"step": "floor"})        # 40 + 40 cores
    mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    assert sum(p.passes for p in mapping.schedule_residency_plan) >= 2
    assert not carried_outputs_by_stage(mapping.stages)                     # segment-level passes (the boundary)
    identity = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=vehicle.ir), device="cpu",
                                   dtype=torch.float64)
    executed = ValueHybridCoreFlow(mapping, device="cpu", dtype=torch.float64)
    for seed in range(4):
        x = torch.rand(4, *vehicle.input_shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            want, got = identity(x), executed(x)
        assert float((got - want).abs().max()) <= 1e-9
        assert float(want.std(0).max()) > 0

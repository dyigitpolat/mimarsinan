"""N2 design §2.7: operand cores are rewritten every inference, so G5's pinned rule never pins one and the census
never treats an operand bank as resident across periods; with a separate operand core type operand banks pack only
onto it and static tiles only onto the others, while one shared core type packs both."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.schedule.residency_units import pinned_cores
from unit.chip_simulation.value_run.integer_programs import assert_exact
from unit.chip_simulation.value_run.operand_programs import attention_program, image_inputs
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

import torch

STATIC = {"max_axons": 64, "max_neurons": 64, "count": 6}
OPERAND = {"max_axons": 48, "max_neurons": 48, "count": 4, "operand": {"payload_bits": 8}}


@pytest.fixture(scope="module")
def program():
    return attention_program("softmax")


def test_the_pinned_rule_never_pins_an_operand_core():
    payloads = [90, 50, 80, 10]
    assert pinned_cores(payloads, 2) == frozenset({0, 2})
    assert pinned_cores(payloads, 2, operand_cores=frozenset({0})) == frozenset({1, 2})
    assert pinned_cores(payloads, 4, operand_cores=frozenset({0, 2})) == frozenset({1, 3})


def test_a_resident_pass_still_writes_its_producer_cells_every_inference():
    row = PassRow(segment_index=0, resident=True, core_cells=(12,), marks=frozenset(), core_extents=((12, 1),),
                  compute_steps=1, stage_latencies=(0,), core_types=((64, 64),), producer_cells=20,
                  producer_cores=frozenset({0}))
    census = period_census(ProgramPasses(rows=(row,)), weight_bits=8, capacity={(64, 64): 4}).census
    assert census.passes[0].written_cells_backing_store == 0
    assert census.passes[0].written_cells_producer == 20


def test_every_executed_pass_holding_an_operand_bank_states_its_producer_cells(program):
    mapping = program.packed_mapping(axons=64, neurons=64, count=3, passes=128)
    banks = set(mapping.dynamic_bank_bindings)
    stages = [s for s in mapping.stages if s.kind == "neural"]
    for stage, row in zip(stages, mapping_program_passes(mapping).rows):
        held = any(p.get("weight_bank_id") in banks
                   for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core for p in placements)
        assert (row.producer_cells > 0) == held


def _typed_mapping(program, cores):
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=True, max_schedule_passes=128))
    return build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=cores, strategy=strategy,
                                          max_schedule_passes=128)


def _placements(mapping):
    banks = set(mapping.dynamic_bank_bindings)
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        segment = stage.hard_core_mapping
        for core, placements in zip(segment.cores, segment.soft_core_placements_per_hard_core):
            for placement in placements:
                yield (int(core.axons_per_core), int(core.neurons_per_core)), placement.get("weight_bank_id") in banks


def test_a_separate_operand_core_type_takes_the_operand_banks_and_only_them(program):
    mapping = _typed_mapping(program, [STATIC, OPERAND])
    seen = list(_placements(mapping))
    assert any(operand for _kind, operand in seen) and any(not operand for _kind, operand in seen)
    for kind, operand in seen:
        assert (kind == (48, 48)) == operand, (kind, operand)
    x = image_inputs(8)
    assert_exact(ValueHybridCoreFlow(mapping, dtype=torch.float64)(x), program.identity()(x))


def test_one_shared_core_type_packs_both_roles(program):
    shared = {"max_axons": 64, "max_neurons": 64, "count": 6, "operand": {"payload_bits": 8}}
    seen = list(_placements(_typed_mapping(program, [shared])))
    assert {operand for _kind, operand in seen} == {True, False}


def test_the_shape_only_packer_routes_roles_and_keeps_them_through_its_copies():
    from mimarsinan.mapping.layout.layout_packer import pack_layout
    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec, core_pool_types
    from mimarsinan.mapping.packing.canonical import OPERAND_ROLE, STATIC_ROLE
    types = [LayoutHardCoreType(768, 768, 2, frozenset({STATIC_ROLE})),
             LayoutHardCoreType(256, 256, 2, frozenset({OPERAND_ROLE}))]
    unnamed = [LayoutSoftCoreSpec(64, 65, residency_class_id=1, latency_tag=0, core_role=OPERAND_ROLE)
               for _ in range(6)]
    fits = pack_layout(softcores=unnamed, core_types=types, allow_neuron_splitting=False, allow_coalescing=False)
    assert fits.feasible and {s.axons_per_core for s in fits.used_core_snapshots} == {256}
    over = pack_layout(softcores=unnamed + unnamed[:3], core_types=types, allow_neuron_splitting=True,
                       allow_coalescing=False)
    assert not over.feasible  # nine banks need three operand cores; the static cores and their fusions refuse them
    import pytest as _pytest
    from mimarsinan.mapping.layout.layout_types import CoreRoleGeometryError
    with _pytest.raises(CoreRoleGeometryError):
        core_pool_types([LayoutHardCoreType(256, 256, 1, frozenset({STATIC_ROLE})),
                         LayoutHardCoreType(256, 256, 1, frozenset({OPERAND_ROLE}))])


def test_a_fused_core_never_accepts_the_operand_role():
    from mimarsinan.mapping.packing.canonical import OPERAND_ROLE, STATIC_ROLE, fused_core_roles
    assert fused_core_roles(None) == frozenset({STATIC_ROLE})
    assert fused_core_roles(frozenset({OPERAND_ROLE})) == frozenset()
    assert fused_core_roles(frozenset({STATIC_ROLE})) == frozenset({STATIC_ROLE})


def test_a_segment_holding_operand_instances_takes_the_capacity_split():
    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
    from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
    from mimarsinan.mapping.support.schedule.pass_planner import plan_segment_passes
    specs = [LayoutSoftCoreSpec(16, 16, residency_class_id=1, latency_tag=0, segment_id=0, bank_id=0,
                                core_role=OPERAND_ROLE, name=f"v{i}") for i in range(8)]
    plan = plan_segment_passes(specs, 2, core_types=[LayoutHardCoreType(32, 32, 2)], max_schedule_passes=64)
    assert plan.feasible and not plan.residency_applied
    static = [LayoutSoftCoreSpec(16, 16, residency_class_id=1, latency_tag=0, segment_id=0, bank_id=0, name=f"s{i}")
              for i in range(8)]
    assert plan_segment_passes(static, 2, core_types=[LayoutHardCoreType(32, 32, 2)],
                               max_schedule_passes=64).residency_applied


def test_a_runtime_split_fragment_keeps_its_operand_role():
    import numpy as np
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments
    from mimarsinan.mapping.packing.softcore.soft_core import SoftCore
    parent = SoftCore(np.zeros((4, 6)), [SpikeSource(-1, 0, is_input=True) for _ in range(4)], 0)
    parent.core_role = "operand"
    assert {f.core_role for f in make_real_fragments(softcore=parent, first_neurons=2, remaining_neurons=4,
                                                     group_id=0)} == {"operand"}

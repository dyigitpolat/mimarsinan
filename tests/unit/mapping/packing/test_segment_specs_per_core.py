"""[F2] The planner's specs of a segment (``_segment_specs``) choose a spec per core: the layout record where the core
has one, the reconstruction from the IR core only for a core without one (a core inserted after the layout walk:
a lane fusion's pool cores, a depth-balancing relay).

The segment's all-or-nothing choice rebuilt every spec of a segment from its IR cores when one core had no layout
record. A lane program's accumulator (WI-5e class E) holds its fused add's operand rows on axons (W08): the builder
places them, and the record plane counts them as the lane's port, no programmed cell. The planner now packs the placed
axons (the record's extent plus the operand rows) and states the operand rows (``operand_axons``), which the plan
plane's cells leave out, so the planes count one program's cells (the head's rebuilt specs over-counted them under
L, E, P: packer divergence) and the plan packs what the builder places (the record's extent alone under-states it).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.lane_law import inserted_record, operand_axons
from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs
from mimarsinan.mapping.packing.residency_tiles import pack_specs, tile_groups
from unit.chip_simulation.value_run.integer_programs import residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform
from unit.mapping.residency.executed import vehicles as V


def _without_record(core: NeuralCore, new_id: int, **changes) -> NeuralCore:
    return dataclasses.replace(core, id=new_id, name=f"{core.name}_inserted", layout_softcore_index=None, **changes)


def test_a_core_without_a_layout_record_leaves_the_other_specs_on_theirs():
    """An inserted core (no layout record, no perceptron, as a pool-lane core) beside recorded ones: every recorded core
    keeps its record (extent, name, classes), and the order facts the capacity split reads (segment id, latency tag)
    are its IR core's, as the inserted core's, one convention across the segment."""
    ir = V.ir_of(V.chain_repr(seed=0))
    cores = list(ir.get_neural_cores())
    recorded = _segment_specs(cores, ir_graph=ir)[0]
    inserted = _without_record(cores[2], 1 + max(int(c.id) for c in cores), perceptron_index=None)
    mixed_cores = cores[:3] + [inserted] + cores[3:]
    mixed, spec_to_core, groups = _segment_specs(mixed_cores, ir_graph=ir)
    assert len(mixed) == len(mixed_cores) and len(groups) == len(mixed_cores)
    rebuilt = [_segment_specs([_without_record(c, int(c.id))], ir_graph=ir)[0][0] for c in cores]
    kept = mixed[:3] + mixed[4:]
    order_facts_as_recorded = [dataclasses.replace(m, segment_id=r.segment_id, latency_tag=r.latency_tag)
                               for m, r in zip(kept, recorded)]
    assert order_facts_as_recorded == recorded                                 # every recorded core on its record
    assert [(m.segment_id, m.latency_tag) for m in kept] == [(r.segment_id, r.latency_tag) for r in rebuilt]
    assert [spec_to_core[id(s)] for s in mixed] == mixed_cores
    alone = _segment_specs([inserted], ir_graph=ir)[0][0]                     # the reconstruction of the inserted
    assert dataclasses.replace(mixed[3], residency_class_id=alone.residency_class_id) == alone
    others = {s.residency_class_id for i, s in enumerate(mixed) if i != 3}
    assert mixed[3].residency_class_id not in others                          # an ungrouped id of its own


def test_a_segment_whose_cores_all_have_records_keeps_the_records_order_facts():
    ir = V.ir_of(V.chain_repr(seed=0))
    cores = list(ir.get_neural_cores())
    specs = _segment_specs(cores, ir_graph=ir)[0]
    assert [s.latency_tag for s in specs] == [ir.layout_softcores[c.layout_softcore_index].latency_tag for c in cores]


def test_a_segment_without_layout_records_is_rebuilt_as_before():
    """No core of the segment has a record: every spec is the reconstruction, its ungrouped ids by segment position."""
    ir = V.ir_of(V.chain_repr(seed=0))
    cores = [_without_record(c, int(c.id)) for c in ir.get_neural_cores()]
    cores[1] = dataclasses.replace(cores[1], perceptron_index=None)
    specs = _segment_specs(cores, ir_graph=ir)[0]
    assert [s.residency_class_id for s in specs] == [-2 if i == 1 else s.perceptron_index for i, s in enumerate(specs)]
    assert [s.name for s in specs] == [c.name for c in cores]


def _placed_axons(mapping) -> dict:
    """Per IR core, the axons the builder's hard-core packer places for it (every neuron-split piece holds them all)."""
    out: dict = {}
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            for p in placements:
                out[int(p["ir_node_id"])] = max(out.get(int(p["ir_node_id"]), 0), int(p["axons"]))
    return out


@pytest.mark.parametrize("classes", (["L", "E"], ["L", "E", "P"]), ids=("L_E", "L_E_P"))
def test_an_accumulator_is_planned_at_the_axons_the_builder_places_and_states_its_operand_rows(classes):
    """The lane residual net: every accumulator (a fused add's core) is planned at the axons the builder's packer
    places for it (its layout record's extent, which predates the fusion, plus its operand rows), and states those
    operand rows (``operand_axons``), which the plan plane counts as no programmed cell, as the record plane does
    (``CoreOccupancy``, W08). Under L, E, P the segment also holds the pool lane's inserted cores ([U-L] each with its
    own record, marked as written after the walk): the head rebuilt every spec of it; under L, E the head planned the
    record's extent, below the placed axons."""
    with torch.no_grad():
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(classes))
    ir = program.ir
    cores = [n for n in ir.nodes if isinstance(n, NeuralCore)]
    assert any(inserted_record(c) for c in cores) == ("P" in classes)   # [U-L] the pool lane's inserted cores
    assert all(c.layout_softcore_index is not None for c in cores)   # [U-L] each with a layout record
    accumulators = [c for c in cores if getattr(c, OPERAND_ROWS, None) is not None]
    assert accumulators
    specs = _segment_specs(cores, ir_graph=ir)[0]
    by_core = {int(c.id): s for c, s in zip(cores, specs)}
    a = max(int(n.get_core_matrix(ir).shape[0]) for n in cores)
    placed = _placed_axons(program.packed_mapping(axons=a + 2, neurons=3, count=4096))
    for core in accumulators:
        spec, rows = by_core[int(core.id)], operand_axons(core)
        assert rows == len(getattr(core, OPERAND_ROWS)) > 0
        assert spec.input_count == placed[int(core.id)] and spec.operand_axons == rows
        assert spec.input_count - rows == ir.layout_softcores[core.layout_softcore_index].input_count
    assert all(s.operand_axons == 0 for c, s in zip(cores, specs) if getattr(c, OPERAND_ROWS, None) is None)


def test_a_specs_operand_axons_are_packed_and_count_as_no_programmed_cell():
    """Two 10-axon specs of 4 neurons, 2 of whose axons are operand rows, on one 32 x 8 core (diagonally): 20 used
    axons, 16 of them programmed, so 16 x 8 cells, the record's count of such a built core."""
    core = [LayoutHardCoreType(max_axons=32, max_neurons=8, count=4)]
    lane = [LayoutSoftCoreSpec(input_count=10, output_count=4, perceptron_index=0, latency_tag=0, name=f"a{i}",
                               operand_axons=2) for i in range(2)]
    plain = [dataclasses.replace(s, operand_axons=0) for s in lane]
    assert pack_specs(plain, core, allow_coalescing=False, allow_splitting=False)[0] == ((20 * 8, 1),)
    assert pack_specs(lane, core, allow_coalescing=False, allow_splitting=False)[0] == ((16 * 8, 1),)
    assert tile_groups(lane, core, allow_coalescing=False, allow_splitting=False) == (((0, 1), ((16 * 8, 1),)),)

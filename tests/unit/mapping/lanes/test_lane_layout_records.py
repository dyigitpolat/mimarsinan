"""[U-L, L-1] Every core a lane fusion inserts carries a layout record of its own.

The layout walk records every core it emits (``IRGraph.layout_softcores``, ``NeuralCore.layout_softcore_index``) and
the planes plan from those records; the fusion (WI-5e) inserts a pool lane's 0/1 reduction cores and a reduction's
reduced cores after the walk, so they had none, and the position-multiplexing planner, which orders passes by the
records' dependency levels, refused every program holding one ("this IR carries no layout record for its cores").
The fusion now appends, per inserted core, the record the walk would state for it in the fused program: its extent,
its dependency level there (the fused IR's latency), no bank, its producer's perceptron for a reduction, and marks
the core (``INSERTED_RECORD``) because the walk's own records predate the fusion. The shape-only walk that applies
the lane plan (U-E2's ``LaneWalk``) states the same extent and level for each inserted core.

Nothing else moves: the planner's one-source rule for a segment's order facts still reads the fused program wherever a
segment holds an inserted core (the walk's records predate it), the runtime soft core keeps the source-perceptron
proxy for its residency class (the finalizer keyed no class on the inserted core's values), and so the U1 program
of every lane chip is the one it was, pass for pass and placement for placement (controls, green before the change
except for the new module they import).
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.legacy_convert import neural_core_to_soft_core
from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS
from mimarsinan.mapping.ir.lane_law import INSERTED_RECORD, inserted_record
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_neural_core
from mimarsinan.mapping.packing.hybrid_build_pool import segment_specs_for

from unit.mapping.lanes.lane_planes import network_of
from unit.mapping.pool_lane_multiplexing import vehicles as V

#: (network, chip): a pool lane alone (the residual block), two reductions and a pool lane (transition), U-K2's network
#: whose ReLU module runs at two call sites, on a scheduled chip with on-core bias and on one with bias rows.
CASES = [("residual", dict(axons=256, neurons=64, count=8, passes=4096, coalescing=True, hardware_bias=True)),
         ("transition", dict(axons=1024, neurons=32, count=8, passes=4096)),
         ("shared", dict(axons=256, neurons=32, count=24, passes=4096))]
IDS = ["residual_on_core_bias", "transition_bias_rows", "shared_module"]


def _case(network: str, geometry: dict, multiplexing: bool = False):
    platform = V.chip(V.POOL, multiplexing=multiplexing, **geometry)
    return platform, V.program(network, V.POOL, platform)


def _stripped(ir):
    """The fused IR as the fusion left it before the change: no record on an inserted core."""
    ir = copy.deepcopy(ir)
    walked = [n for n in ir.get_neural_cores() if not inserted_record(n)]
    count = 1 + max(int(n.layout_softcore_index) for n in walked)
    for node in ir.get_neural_cores():
        if inserted_record(node):
            node.layout_softcore_index = None
            delattr(node, INSERTED_RECORD)
    ir.layout_softcores = list(ir.layout_softcores)[:count]
    return ir


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_every_inserted_core_carries_a_record_of_its_own(network, geometry):
    _platform, prog = _case(network, geometry)
    records = list(prog.ir.layout_softcores)
    inserted = V.lane_cores(prog)
    assert inserted, "the vehicle fuses a pool lane or a reduction"
    for core in prog.ir.get_neural_cores():
        assert core.layout_softcore_index is not None and 0 <= int(core.layout_softcore_index) < len(records)
        assert inserted_record(core) == (getattr(core, LANE_OUTPUTS, None) is not None)
    walked = max(int(c.layout_softcore_index) for c in prog.ir.get_neural_cores() if not inserted_record(c))
    assert sorted(int(c.layout_softcore_index) for c in inserted) == list(range(walked + 1, len(records)))


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_an_inserted_record_states_the_core_as_the_fused_program_holds_it(network, geometry):
    _platform, prog = _case(network, geometry)
    by_id = {int(n.id): n for n in prog.ir.get_neural_cores()}
    for core in V.lane_cores(prog):
        record = prog.ir.layout_softcores[int(core.layout_softcore_index)]
        rebuilt = spec_from_neural_core(core, hardware_bias=core.hardware_bias is not None,
                                        fallback_residency_class_id=0)
        assert (record.input_count, record.output_count, record.bank_id, record.perceptron_index, record.name) == (
            rebuilt.input_count, rebuilt.output_count, None, rebuilt.perceptron_index, core.name)
        assert record.latency_tag == int(core.latency)
        producers = {int(s.node_id) for s in core.input_sources.flatten() if int(s.node_id) in by_id}
        assert producers and all(int(by_id[p].latency) < record.latency_tag for p in producers)


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_the_shape_only_walk_states_each_inserted_core_alike(network, geometry):
    """The candidate plane's lane-aware walk emits a spec for each inserted core under the fusion's name (``_lane<g>``,
    ``_reduce<g>``): the same extent and the same dependency level as the builder's record."""
    from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import layout_walk
    from mimarsinan.mapping.lanes.shape_walk import LaneWalk

    platform, prog = _case(network, geometry)
    network_ = network_of(prog, V.NETWORKS[network][2])
    lanes = LaneWalk.of(network_.mapper_repr, platform)
    walk = layout_walk(network_, platform, wire_census=False, lanes=lanes)
    walked = {walk.layout_softcores[walk._node_id_to_softcore_idx[n]].name:
              walk.layout_softcores[walk._node_id_to_softcore_idx[n]] for n in lanes.inserted}
    records = {r.name: r for r in (prog.ir.layout_softcores[int(c.layout_softcore_index)] for c in V.lane_cores(prog))}
    assert set(walked) == set(records) and records
    assert {k: (r.input_count, r.output_count, r.latency_tag) for k, r in records.items()} == \
        {k: (s.input_count, s.output_count, s.latency_tag) for k, s in walked.items()}


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_control_the_planner_reads_the_specs_it_read_before(network, geometry):
    """Control: per segment, the pass planner's specs (``_segment_specs``) on the fused IR equal those on the same IR
    without the inserted records, field for field but the residency class's value (a unique id either way: the same
    partition of the segment's cores)."""
    platform, prog = _case(network, geometry)
    reader = segment_specs_for(platform["cores"])
    before = _stripped(prog.ir)
    for now_segment, old_segment in zip(*(
            [s for s in partition_ir_graph(ir) if isinstance(s, NeuralSegment)] for ir in (prog.ir, before))):
        now = reader(now_segment.nodes, ir_graph=prog.ir, hardware_bias=geometry.get("hardware_bias", False))[0]
        old = reader(old_segment.nodes, ir_graph=before, hardware_bias=geometry.get("hardware_bias", False))[0]
        assert [dataclasses.replace(s, residency_class_id=0) for s in now] == \
            [dataclasses.replace(s, residency_class_id=0) for s in old]
        assert _partition(now) == _partition(old)


def _partition(specs) -> list:
    groups: dict = {}
    for i, spec in enumerate(specs):
        groups.setdefault(spec.residency_class_id, []).append(i)
    return sorted(groups.values())


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_control_the_runtime_soft_core_keeps_the_proxy_class(network, geometry):
    """Control: an inserted core's soft core reads the source-perceptron proxy (the finalizer keyed no class on its
    values); a walked core reads its record's class."""
    _platform, prog = _case(network, geometry)
    for core in prog.ir.get_neural_cores():
        soft = neural_core_to_soft_core(core, prog.ir)
        if inserted_record(core):
            assert soft.residency_class_id == core.perceptron_index
        else:
            assert soft.residency_class_id == prog.ir.layout_softcores[int(core.layout_softcore_index)].residency_class_id


@pytest.mark.parametrize("network,geometry", CASES, ids=IDS)
def test_control_the_u1_program_is_the_one_it_was(network, geometry):
    """Control: the scheduled U1 program of the fused IR equals the one built from the IR without the inserted
    records: the same passes, each placing the same soft cores at the same offsets, resident alike."""
    platform, prog = _case(network, geometry)
    now, old = V.build(prog, platform), V.build(type(prog)(prog.model, _stripped(prog.ir), prog.input_shape), platform)
    assert _placements(now) == _placements(old)


def _placements(hybrid) -> list:
    return [(bool(s.schedule_weights_resident), tuple(
        tuple(sorted((int(p["ir_node_id"]), int(p["axon_offset"]), int(p["neuron_offset"]), int(p["axons"]),
                      int(p["neurons"])) for p in ps)) for ps in s.hard_core_mapping.soft_core_placements_per_hard_core))
            for s in V.neural(hybrid)]


def test_a_lane_free_ir_keeps_only_walk_records():
    """Control: without a pool lane or a reduction the fusion inserts nothing and writes no record."""
    platform = V.chip(V.HOST_POOL, multiplexing=False, **CASES[0][1])
    prog = V.program("residual", V.HOST_POOL, platform)
    assert not V.lane_cores(prog)
    assert not any(inserted_record(n) for n in prog.ir.get_neural_cores() if isinstance(n, NeuralCore))

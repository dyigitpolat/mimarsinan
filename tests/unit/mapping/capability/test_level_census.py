"""Per-level census on hand-built IRs: segments, boundary bytes, lane operations, dynamic payload, field map."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS
from mimarsinan.mapping.capability.census import ladder_census, level_census
from mimarsinan.mapping.capability.mac_split import MacCounters
from mimarsinan.mapping.capability.pricing_inputs import (
    FIELD_MAP_FIELDS,
    PENDING_CATALOG_FIELDS,
    ROTATION_LABEL,
    level_period_census,
    level_quantities,
)
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
from mimarsinan.mapping.capability.ladder import (
    HOST_PAYLOAD_WRITE,
    LADDER,
    ON_CHIP_WRITE_BACK,
    PLACEMENT_LABEL,
    capability_level,
)
from mimarsinan.mapping.layout.segmentation import HostSegment, NeuralSegment, partition_ir_graph

from .ir_fixtures import (
    contraction_graph,
    grouped_graph,
    independent_boundary_elements,
    interleaved_branches_graph,
    lane_graph,
    shared_readout_graph,
)

BITS = 8


def _row(ir, level, payload_write=ON_CHIP_WRITE_BACK):
    return level_census(ir, capability_level(level), value_bits=BITS, payload_write=payload_write)


@pytest.mark.parametrize("ir_factory", [
    lane_graph, contraction_graph, shared_readout_graph, interleaved_branches_graph, grouped_graph])
def test_c0_equals_the_partition_and_the_independent_boundary_census(ir_factory):
    ir = ir_factory()
    row = _row(ir, "C0")
    segments = partition_ir_graph(ir)
    assert row.neural_segments == sum(isinstance(s, NeuralSegment) for s in segments)
    assert row.host_segments == sum(isinstance(s, HostSegment) for s in segments)
    assert row.host_ops == len(ir.get_compute_ops())
    assert row.boundary_bytes == independent_boundary_elements(ir)


def test_lane_graph_hand_counts():
    ir = lane_graph()
    c0, c1, c5 = (_row(ir, name) for name in ("C0", "C1", "C5"))
    assert (c0.neural_segments, c0.host_segments, c0.host_ops) == (4, 5, 5)
    assert (c0.boundary.entry, c0.boundary.carry_out, c0.boundary.carry_in, c0.boundary.readout) == (4, 12, 8, 2)
    assert c0.boundary_bytes == 26
    assert c0.host_kinds == {"clamp": 2, "affine": 1, "add": 1, "smooth": 1}
    assert (c1.neural_segments, c1.host_segments, c1.boundary_bytes) == (2, 2, 14)
    assert c1.host_kinds == {"smooth": 1, "clamp": 1}
    assert (c5.neural_segments, c5.host_segments, c5.boundary_bytes) == (1, 0, 6)


def test_lane_ops_are_zero_at_c0_and_grow_with_the_level():
    rows = [row for row in ladder_census(lane_graph(), value_bits=BITS)]
    by_level = {}
    for row in rows:
        by_level.setdefault(row.level, set()).add(row.lane_ops)
    assert all(len(values) == 1 for values in by_level.values())
    series = [by_level[level.name].pop() for level in LADDER]
    assert series == [0, 12, 12, 12, 12, 16, 16]


def test_the_boundary_bytes_scale_with_the_declared_value_width():
    ir = lane_graph()
    assert level_census(ir, capability_level("C0"), value_bits=16).boundary_bytes == 52
    assert level_census(ir, capability_level("C0"), value_bits=4).boundary_bytes == 26, "one byte per element floor"


def test_dynamic_operand_placement_is_reported_twice():
    ir = contraction_graph()
    c2 = _row(ir, "C2")
    write_back = _row(ir, "C3", ON_CHIP_WRITE_BACK)
    host_write = _row(ir, "C3", HOST_PAYLOAD_WRITE)
    assert (c2.neural_segments, c2.host_segments, c2.boundary_bytes) == (2, 1, 36)
    assert c2.host_kinds == {"dmm": 1} and c2.dynamic_payload_elements == 0
    assert (write_back.neural_segments, write_back.host_segments, write_back.boundary_bytes) == (1, 0, 10)
    assert write_back.host_ops == 0 and write_back.dynamic_payload_elements == 12
    assert (host_write.neural_segments, host_write.host_segments, host_write.boundary_bytes) == (2, 1, 34)
    assert host_write.host_kinds == {"payload_write": 1} and host_write.dynamic_payload_elements == 12
    assert write_back.payload_write == ON_CHIP_WRITE_BACK and host_write.payload_write == HOST_PAYLOAD_WRITE
    assert c2.payload_write is None


def test_the_ladder_census_carries_both_d_rows_and_the_placement_label():
    rows = ladder_census(contraction_graph(), value_bits=BITS)
    names = [(row.level, row.payload_write) for row in rows]
    assert names[:3] == [("C0", None), ("C1", None), ("C2", None)]
    for level in ("C3", "C4", "C5", "C6"):
        assert (level, ON_CHIP_WRITE_BACK) in names and (level, HOST_PAYLOAD_WRITE) in names
    for row in rows:
        assert row.label == (None if row.level == "C0" else PLACEMENT_LABEL)


def _one_byte_per_cell(cells: int) -> int:
    return int(cells)


def _period(ir, level, payload_write=ON_CHIP_WRITE_BACK, bytes_of_cells=_one_byte_per_cell):
    return level_period_census(ir, capability_level(level), value_bits=BITS, payload_write=payload_write,
                               bytes_of_cells=bytes_of_cells)


def test_the_level_period_census_is_s0s_pass_period_census():
    census = _period(contraction_graph(), "C3")
    assert isinstance(census, PassPeriodCensus) and all(isinstance(p, PassCensus) for p in census.passes)


def test_the_period_census_rewrites_every_pass_by_the_executed_rotation():
    """Finding 4 (a): each execution stage loads its cores from its type's first core, so a level of two or more
    neural segments rewrites every pass's cores once per period (no residency plan pins them), while a level of one
    neural segment (a lone writing pass) is never displaced and writes nothing."""
    ir = contraction_graph()
    c0, c3 = _period(ir, "C0"), _period(ir, "C3")
    assert [(p.cores, p.written_cells_backing_store, p.payload_bytes, p.resident) for p in c0.passes] == [
        (2, 8 * 8 + 8 * 12, 8 * 8 + 8 * 12, False), (1, 6 * 2, 6 * 2, False)], "[Q K] P [O]: both passes rewrite"
    assert all(p.resident_cores == 0 for p in c0.passes)
    assert [(p.cores, p.written_cells_backing_store, p.resident) for p in c3.passes] == [(3, 0, True)], (
        "one neural segment at C3: a lone writing pass, loaded once")
    lane = [_period(lane_graph(), level) for level in ("C0", "C1", "C5")]
    assert [len(c.passes) for c in lane] == [4, 2, 1]
    assert [sum(p.written_cells_backing_store for p in c.passes) for c in lane] == [56, 56, 0]
    assert ROTATION_LABEL in c0.label and "C0" in c0.label


def test_the_payload_bytes_are_priced_core_by_core_by_the_callers_rule():
    census = _period(contraction_graph(), "C0", bytes_of_cells=lambda cells: -(-cells * 4 // 8))
    assert [p.payload_bytes for p in census.passes] == [32 + 48, 6], "4-bit weights, each core rounded up"


def test_the_period_census_carries_producer_sourced_payload_only_where_d_is_on_chip():
    ir = contraction_graph()
    c0 = _period(ir, "C0")
    c3 = _period(ir, "C3")
    assert sum(p.written_cells_producer for p in c0.passes) == 0
    assert sum(p.written_cells_producer for p in c3.passes) == 12, "the stationary K (4 x 3), from the operand shapes"
    assert sum(p.written_cells_backing_store for p in c3.passes) == 0
    assert sum(p.cores for p in c3.passes) == 3
    for census in (c0, c3):
        assert [p.position for p in census.passes] == list(range(len(census.passes)))
        assert [p.segment_index for p in census.passes] == list(range(len(census.passes)))
        assert census.boundaries == (), "a host-op boundary crosses segments: it is carry, in the field map"
    assert c3.label.startswith("C3 on_chip_write_back") and c0.label.startswith("C0")


def test_the_period_census_under_a_host_payload_write():
    ir = contraction_graph()
    census = _period(ir, "C3", HOST_PAYLOAD_WRITE)
    assert sum(p.written_cells_producer for p in census.passes) == 12, "the same stationary K, written by the host"
    assert [(p.cores, p.compute_steps) for p in census.passes] == [(2, 1), (1, 2)]
    assert all(not p.resident for p in census.passes), "the payload write splits the level into two writing passes"


def test_the_field_map_names_every_field_of_the_plan():
    assert set(FIELD_MAP_FIELDS) == {
        "host_macs", "onchip_macs", "compute_op_count", "macs", "cells_used", "synaptic_events",
        "latency_steps", "segment_cores", "pass_count", "sync_count", "carry_in_bytes", "carry_out_bytes",
        "lane_ops", "lanes_per_tile"}
    assert set(FIELD_MAP_FIELDS) - set(QUANTITY_SPECS) == set(PENDING_CATALOG_FIELDS) == set(), (
        "every field-map name is a catalog key since G3's lane rows joined the catalog at integration")
    assert {"lane_ops", "lanes_per_tile"} <= set(QUANTITY_SPECS)


def test_quantities_are_absent_unless_produced():
    row = _row(lane_graph(), "C1")
    bare = level_quantities(row)
    assert set(bare) == set(FIELD_MAP_FIELDS) - {"host_macs", "onchip_macs", "lanes_per_tile"}
    full = level_quantities(row, counters=MacCounters(0, 0, 100, False), lanes_per_tile=64)
    assert set(full) == set(FIELD_MAP_FIELDS)
    assert (full["onchip_macs"], full["host_macs"]) == (100, 0)
    assert full["lane_ops"] == 12 and full["lanes_per_tile"] == 64
    assert full["compute_op_count"] == row.host_ops and full["pass_count"] == row.neural_segments
    assert full["macs"] == full["cells_used"] == row.cells_committed == 4 * 4 + 4 * 4 + 4 * 4 + 4 * 2
    assert level_quantities(_row(lane_graph(), "C0"), lanes_per_tile=64)["lanes_per_tile"] == 0, (
        "no lane class declared at C0")


def test_latency_steps_hand_counts():
    lane = lane_graph()
    assert [_row(lane, level).latency_steps for level in ("C0", "C1", "C5")] == [4, 4, 4]
    assert [s.latency_steps for s in _row(lane, "C1").segments] == [3, 1], "A, D (after A), F (after E)"
    assert [s.latency_steps for s in _row(lane, "C0").segments] == [1, 1, 1, 1]
    ir = contraction_graph()
    assert _row(ir, "C0").latency_steps == 2
    assert _row(ir, "C3", ON_CHIP_WRITE_BACK).latency_steps == 3, "Q and K, then P, then O"
    assert [s.latency_steps for s in _row(ir, "C3", HOST_PAYLOAD_WRITE).segments] == [1, 2]
    assert level_quantities(_row(lane, "C1"))["latency_steps"] == 4


def test_synaptic_events_are_committed_cells_plus_on_chip_operator_macs():
    lane = lane_graph()
    assert {level_quantities(_row(lane, level))["synaptic_events"] for level in ("C0", "C1", "C5")} == {56}
    grouped = grouped_graph()
    assert level_quantities(_row(grouped, "C0"))["synaptic_events"] == 80, "the grouped conv runs on the host"
    assert level_quantities(_row(grouped, "C2"))["synaptic_events"] == 80 + 8, "its 8 MACs join on chip"


def test_the_readout_counts_an_on_chip_output_once_when_a_host_op_also_reads_it():
    c0, c1 = _row(shared_readout_graph(), "C0"), _row(shared_readout_graph(), "C1")
    assert (c0.boundary.entry, c0.boundary.carry_out, c0.boundary.carry_in, c0.boundary.readout) == (4, 4, 4, 2)
    assert c0.boundary_bytes == 14
    assert (c1.boundary.entry, c1.boundary.carry_out, c1.boundary.carry_in, c1.boundary.readout) == (4, 0, 0, 6)


def test_the_dependency_segmentation_merges_branches_the_in_order_partition_splits():
    row = _row(interleaved_branches_graph(), "C0")
    assert (row.neural_segments, row.host_segments) == (3, 2)
    assert (row.dependency_neural_segments, row.dependency_host_segments) == (2, 1)


def test_carry_quantities_exclude_image_entry_and_logit_readout():
    q = level_quantities(_row(lane_graph(), "C0"))
    assert (q["carry_out_bytes"], q["carry_in_bytes"]) == (12, 8)


def test_a_host_placed_grouped_convolution_carries_its_host_macs_as_a_stated_boundary():
    ir = grouped_graph()
    c0, c2 = _row(ir, "C0"), _row(ir, "C2")
    assert c0.host_kinds == {"mmg": 1}
    assert c0.stated_boundaries == {"grouped convolution on the host until N5": 8}, (
        "2 output channels x 4 positions x 1 input channel per group x a 1 x 1 kernel, from count_host_ops")
    assert c0.host_macs_by_kind == {"mmg": 8} and c0.host_macs == 8
    assert c2.stated_boundaries == {} and c2.host_macs == 0
    assert _row(lane_graph(), "C0").stated_boundaries == {}

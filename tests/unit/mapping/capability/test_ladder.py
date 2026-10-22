"""The capability ladder C0-C6 and the placement rule on the toolchain IR."""

from __future__ import annotations

import json
import operator
from pathlib import Path

import pytest
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.ladder import (
    CHIP,
    HOST,
    HOST_PAYLOAD_WRITE,
    LADDER,
    ON_CHIP_WRITE_BACK,
    PLACEMENT_LABEL,
    TRANSPARENT,
    UnknownCapabilityLevelError,
    capability_level,
    payload_writes,
    place_level,
)
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

from .ir_fixtures import INPUT, contraction_graph, core, graph, lane_graph, op, outputs

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "requirement_census"


def _sides(ir, level_name, payload_write=ON_CHIP_WRITE_BACK):
    nodes = place_level(ir, capability_level(level_name), ir_op_kinds(ir), payload_write=payload_write)
    return {node.id: node.side for node in nodes}, nodes


def test_ladder_equals_the_scout_ladder():
    oracle = json.loads((FIXTURES / "oracle_kind_map.json").read_text())["ladder"]
    assert [level.title for level in LADDER] == [name for name, _ in oracle]
    assert [sorted(level.classes) for level in LADDER] == [classes for _, classes in oracle]
    assert [level.name for level in LADDER] == [f"C{i}" for i in range(7)]


def test_levels_are_nested():
    for lower, upper in zip(LADDER, LADDER[1:]):
        assert lower.classes < upper.classes


def test_an_unknown_level_refuses_by_name():
    with pytest.raises(UnknownCapabilityLevelError) as caught:
        capability_level("C9")
    assert isinstance(caught.value, ToolchainRefusal) and "C9" in str(caught.value)


def test_d_is_reported_with_both_payload_writes_exactly_where_declared():
    for level in LADDER:
        expected = (ON_CHIP_WRITE_BACK, HOST_PAYLOAD_WRITE) if "D" in level.classes else (None,)
        assert payload_writes(level) == expected


def test_the_placement_label_is_the_plan_sentence():
    assert PLACEMENT_LABEL == "requirement placement on the toolchain IR (no lowering yet)"


def test_c0_places_cores_on_chip_and_every_compute_op_on_the_host():
    sides, _ = _sides(lane_graph(), "C0")
    assert sides == {0: CHIP, 1: HOST, 2: HOST, 3: CHIP, 4: HOST, 5: CHIP, 6: HOST, 7: HOST, 8: CHIP}


def test_a_lane_op_needs_its_classes_and_every_producer_on_chip():
    c1, _ = _sides(lane_graph(), "C1")
    assert [c1[i] for i in (1, 2, 4)] == [CHIP] * 3
    assert c1[6] == HOST, "GELU needs T"
    assert c1[7] == HOST, "the ReLU after the GELU has a host producer"
    c5, _ = _sides(lane_graph(), "C5")
    assert set(c5.values()) == {CHIP}


def test_a_lane_op_reading_the_raw_input_stays_on_the_host():
    ir = graph([op(0, nn.ReLU(), outputs(INPUT, 4), input_shape=(4,), output_shape=(4,)),
                core(1, outputs(0, 4), 2)], outputs(1, 2))
    sides, _ = _sides(ir, "C6")
    assert sides[0] == HOST


def test_a_crossbar_kind_needs_only_its_classes():
    ir = graph([op(0, nn.ConvTranspose2d(1, 1, 2), outputs(INPUT, 4), input_shape=(1, 2, 2),
                   output_shape=(1, 3, 3)),
                core(1, outputs(0, 9), 2)], outputs(1, 2))
    assert _sides(ir, "C1")[0][0] == HOST
    assert _sides(ir, "C2")[0][0] == CHIP, "S declared: on chip although its producer is the raw input"


def test_a_wiring_completion_is_transparent_off_chip_and_on_chip_with_chip_producers():
    slice_op = ComputeAdapter(operator.getitem, extra_args=((slice(None), slice(0, 2)),))
    ir = graph([op(0, slice_op, outputs(INPUT, 4), input_shape=(4,), output_shape=(2,)),
                core(1, outputs(0, 2), 2),
                op(2, ComputeAdapter(operator.getitem, extra_args=((slice(None), slice(0, 1)),)), outputs(1, 2),
                   input_shape=(2,), output_shape=(1,)),
                core(3, outputs(2, 1), 1)], outputs(3, 1))
    c0, _ = _sides(ir, "C0")
    assert (c0[0], c0[2]) == (HOST, HOST)
    c1, _ = _sides(ir, "C1")
    assert c1[0] == TRANSPARENT, "W declared, raw-input producer: host wiring, not a host op"
    assert c1[2] == CHIP


def test_a_host_payload_write_inserts_one_host_node_before_each_on_chip_product():
    ir = contraction_graph()
    sides, nodes = _sides(ir, "C3", HOST_PAYLOAD_WRITE)
    assert sides[2] == CHIP
    order = [node.id for node in nodes]
    virtual = [node for node in nodes if node.ir_node is None]
    assert len(virtual) == 1 and virtual[0].side == HOST and virtual[0].kind == "payload_write"
    assert order.index(virtual[0].id) == order.index(2) - 1
    assert sorted(set(virtual[0].source_ids.tolist())) == [1], "it reads the stationary operand K"
    product = next(node for node in nodes if node.id == 2)
    assert set(product.source_ids.tolist()) == {0, virtual[0].id}


def test_an_on_chip_write_back_adds_no_node():
    ir = contraction_graph()
    _, nodes = _sides(ir, "C3", ON_CHIP_WRITE_BACK)
    assert [node.id for node in nodes] == [node.id for node in ir.nodes]

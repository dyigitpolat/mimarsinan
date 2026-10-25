"""A capability level's requirement placement scheduled on a core budget: passes, tiles, the carry at every pass cut
and the pass spans, on hand-built IRs with hand counts (CL4's chip space). The period rule is G3's, priced on the
level plane (``tests/unit/deployment_record/test_level_pricing.py``)."""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.ladder import HOST_PAYLOAD_WRITE, capability_level
from mimarsinan.mapping.capability.level_schedule import (
    CoreGeometry,
    LevelScheduleError,
    level_schedule,
    prepare_level,
)

from .ir_fixtures import contraction_graph, lane_graph

TILE4 = CoreGeometry(max_axons=4, max_neurons=4, allow_coalescing=False, allow_splitting=False)
SPLIT4 = CoreGeometry(max_axons=4, max_neurons=4, allow_coalescing=True, allow_splitting=True)


def _schedule(ir, level, budget, geometry=TILE4, **kwargs):
    return level_schedule(ir, capability_level(level), budget=budget, geometry=geometry, **kwargs)


def test_a_program_that_fits_its_budget_is_pinned_one_pass_per_neural_segment():
    """C0 of the lane graph: [A] B C [D] E [F] G H [I], four neural segments of one 4 x 4 tile each (I is 4 x 2)."""
    schedule = _schedule(lane_graph(), "C0", 4)
    assert schedule.tiles == 4 and schedule.pinned
    assert [(p.segment_index, p.tile_cells, p.tile_extents) for p in schedule.passes] == [
        (0, (16,), ((4, 4),)), (1, (16,), ((4, 4),)), (2, (16,), ((4, 4),)), (3, (8,), ((4, 2),))]
    assert schedule.boundary_widths == (), "every pass is its own neural segment: host-op carry is the field map's"
    assert not _schedule(lane_graph(), "C0", 3).pinned


def test_a_segment_above_its_budget_runs_in_greedy_passes_with_its_live_set_carried_at_each_cut():
    """C5 places every op on chip (one segment). At C = 2: [A D] [F I]; E (A's ReLU-affine path plus D) feeds F, so 4
    elements cross. At C = 1: [A] [D] [F] [I]; after A, D reads A and E reads C (8); after D, F reads E (4); after F,
    I reads H (4). The same width both ways."""
    two = _schedule(lane_graph(), "C5", 2)
    assert [p.tile_cells for p in two.passes] == [(16, 16), (16, 8)] and not two.pinned
    assert two.boundary_widths == ((0, 4, 4),)
    one = _schedule(lane_graph(), "C5", 1)
    assert one.boundary_widths == ((0, 8, 8), (1, 4, 4), (2, 4, 4))
    fit = _schedule(lane_graph(), "C5", 4)
    assert len(fit.passes) == 1 and fit.pinned and fit.boundary_widths == ()


def test_each_pass_states_its_depth_span():
    """One segment of depth 3 in one pass spans 3; split into [A D] [F I] each pass spans one level; into four, 0."""
    assert [p.stage_latency for p in _schedule(lane_graph(), "C5", 4).passes] == [3]
    assert [p.stage_latency for p in _schedule(lane_graph(), "C5", 2).passes] == [1, 1]
    assert [p.stage_latency for p in _schedule(lane_graph(), "C5", 1).passes] == [0, 0, 0, 0]


def test_an_on_chip_dynamic_product_holds_its_stationary_operand_in_producer_written_tiles():
    """Q 8x8 (2x2 tiles), K 8x12 (2x3), the stationary K 4x3 (1x1, producer-written, 12 cells per inference),
    O 6x2 (2x1: 4x2 and 2x2 blocks)."""
    schedule = _schedule(contraction_graph(), "C3", 16, geometry=SPLIT4)
    (only,) = schedule.passes
    assert schedule.tiles == 4 + 6 + 1 + 2 and schedule.pinned
    assert only.tile_cells == (16,) * 10 + (8, 4) and only.producer_tiles == 1 and only.producer_cells == 12
    assert only.tile_extents is not None and only.tile_extents[-2:] == ((4, 2), (2, 2))
    host_write = _schedule(contraction_graph(), "C3", 16, geometry=SPLIT4, payload_write=HOST_PAYLOAD_WRITE)
    assert [(p.segment_index, len(p.tile_cells), p.producer_tiles, p.producer_cells) for p in host_write.passes] == [
        (0, 10, 0, 0), (1, 2, 1, 12)], "the host payload write splits the level into two segments"
    assert _schedule(contraction_graph(), "C0", 16, geometry=SPLIT4).tiles == 4 + 6 + 2, "P on the host at C0"


def test_the_schedules_at_several_budgets_equal_one_schedule_each():
    prepared = prepare_level(lane_graph(), capability_level("C5"), geometry=TILE4)
    for budget in (1, 2, 4):
        assert prepared.schedule(budget) == _schedule(lane_graph(), "C5", budget)


def test_a_prepared_level_packs_on_any_budget_and_names_its_widest_core():
    split = CoreGeometry(max_axons=2, max_neurons=2, allow_coalescing=True, allow_splitting=True)
    prepared = prepare_level(lane_graph(), capability_level("C0"), geometry=split)
    assert prepared.widest == 4, "a 4 x 4 core on 2 x 2 tiles"
    assert prepared.schedule(4).tiles == 14
    with pytest.raises(LevelScheduleError, match="exceeds the core budget"):
        prepared.schedule(3)


def test_a_level_without_d_states_no_payload_write():
    assert _schedule(lane_graph(), "C1", 4, payload_write=HOST_PAYLOAD_WRITE).payload_write is None
    assert _schedule(contraction_graph(), "C3", 16, geometry=SPLIT4).label.startswith("C3 on_chip_write_back on 16")


def test_a_core_wider_than_the_declared_core_needs_the_targets_split_permissions():
    narrow = CoreGeometry(max_axons=2, max_neurons=2, allow_coalescing=False, allow_splitting=True)
    with pytest.raises(LevelScheduleError, match="coalescing") as caught:
        _schedule(lane_graph(), "C0", 64, geometry=narrow)
    assert isinstance(caught.value, ToolchainRefusal)
    with pytest.raises(LevelScheduleError, match="splitting"):
        _schedule(lane_graph(), "C0", 64, geometry=CoreGeometry(4, 2, allow_coalescing=True, allow_splitting=False))
    split = CoreGeometry(max_axons=2, max_neurons=2, allow_coalescing=True, allow_splitting=True)
    schedule = _schedule(lane_graph(), "C0", 64, geometry=split)
    assert schedule.tiles == 4 + 4 + 4 + 2 and schedule.passes[0].tile_cells == (4, 4, 4, 4)


def test_a_core_of_more_tiles_than_the_budget_refuses_by_name():
    split = CoreGeometry(max_axons=2, max_neurons=2, allow_coalescing=True, allow_splitting=True)
    with pytest.raises(LevelScheduleError, match="exceeds the core budget"):
        _schedule(lane_graph(), "C0", 3, geometry=split)


def test_a_budget_must_be_a_positive_count():
    with pytest.raises(LevelScheduleError):
        _schedule(lane_graph(), "C0", 0)

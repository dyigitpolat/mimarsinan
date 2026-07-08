"""NoC message taxonomy and tile packet aggregation."""

from __future__ import annotations

from mimarsinan.chip_simulation.sanafe.records import SanafeConnectivityEdge
from mimarsinan.chip_simulation.sanafe.analysis import (
    _compute_noc_link_load_per_cycle,
    _compute_tile_packets_per_cycle,
    _count_cross_tile_connectivity_edges,
    _summarize_message_trace,
)


def test_cross_tile_connectivity_count():
    edges = [
        SanafeConnectivityEdge(0, 10, 1.0, 1),
        SanafeConnectivityEdge(0, 0, 1.0, 1),
        SanafeConnectivityEdge(8, 17, 1.0, 1),
    ]
    assert _count_cross_tile_connectivity_edges(edges, cores_per_tile=8) == 2


def test_tile_packets_per_cycle():
    mt = [
        [{"dest_tile_id": 2, "placeholder": False}, {"dest_tile_id": 2, "placeholder": False}],
        [{"dest_tile_id": 3, "placeholder": False}],
    ]
    cycles = _compute_tile_packets_per_cycle(mt)
    assert cycles[0][2] == 2
    assert cycles[1][3] == 1


def test_noc_link_load_per_cycle_walks_xy_routes():
    """Per-cycle mesh-edge load: XY routing (x first, then y), one entry
    per traversed edge per cycle — the time-scrubbed congestion record."""
    mt = [
        [{"src_x": 0, "src_y": 0, "dest_x": 2, "dest_y": 0, "placeholder": False}],
        [],
        [
            {"src_x": 1, "src_y": 0, "dest_x": 1, "dest_y": 1, "placeholder": False},
            {"src_x": 1, "src_y": 0, "dest_x": 1, "dest_y": 1, "placeholder": False},
        ],
    ]
    cycles = _compute_noc_link_load_per_cycle(mt)
    assert len(cycles) == 3
    assert sorted(cycles[0]) == [[0, 0, 1, 0, 1], [1, 0, 2, 0, 1]]
    assert cycles[1] == []
    assert cycles[2] == [[1, 0, 1, 1, 2]]


def test_noc_link_load_per_cycle_skips_local_and_unplaced_messages():
    mt = [[
        {"src_x": 1, "src_y": 1, "dest_x": 1, "dest_y": 1, "placeholder": False},
        {"src_x": -1, "src_y": 0, "dest_x": 2, "dest_y": 0, "placeholder": False},
        {"placeholder": True, "src_x": 0, "src_y": 0, "dest_x": 3, "dest_y": 0},
    ]]
    assert _compute_noc_link_load_per_cycle(mt) == [[]]


def test_noc_link_load_per_cycle_empty_trace():
    assert _compute_noc_link_load_per_cycle(None) == []
    assert _compute_noc_link_load_per_cycle([]) == []


def test_101111_style_all_intra_input_path():
    """Mirror run 101111: same-tile, input-group messages only."""
    mt = [[
        {
            "src_tile_id": 2, "dest_tile_id": 2,
            "src_neuron_group_id": "core23_in",
            "placeholder": False,
        },
    ]] * 3
    s = _summarize_message_trace(mt)
    assert s["inter_tile_packets"] == 0
    assert s["intra_tile_packets"] == 3
    assert s["input_path_packets"] == 3

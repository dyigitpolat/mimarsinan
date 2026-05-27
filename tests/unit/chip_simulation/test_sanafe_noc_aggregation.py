"""NoC message taxonomy and tile packet aggregation."""

from __future__ import annotations

from mimarsinan.chip_simulation.sanafe.records import SanafeConnectivityEdge
from mimarsinan.chip_simulation.sanafe.analysis import (
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

"""Cross-tile connectivity edge counting."""

from __future__ import annotations

from mimarsinan.chip_simulation.sanafe.records import SanafeConnectivityEdge
from mimarsinan.chip_simulation.sanafe.runner import (
    _count_cross_tile_connectivity_edges,
)


def test_cross_tile_connectivity_matches_tile_partition():
    edges = [
        SanafeConnectivityEdge(0, 10, 1.0, 1),
        SanafeConnectivityEdge(3, 4, 1.0, 1),
        SanafeConnectivityEdge(8, 17, 1.0, 1),
    ]
    assert _count_cross_tile_connectivity_edges(edges, cores_per_tile=8) == 2

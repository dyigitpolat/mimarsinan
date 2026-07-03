"""_compute_connectivity_edges fails loud on unindexable core matrices."""

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe.analysis.connectivity import (
    _compute_connectivity_edges,
)


class _Src:
    def __init__(self, core):
        self.is_off_ = False
        self.is_input_ = False
        self.is_always_on_ = False
        self.core_ = core


class _Core:
    def __init__(self, core_matrix, n_axons_used=1):
        self.axons_per_core = n_axons_used
        self.available_axons = 0
        self.core_matrix = core_matrix
        self.axon_sources = [_Src(0) for _ in range(n_axons_used)]


class _HCM:
    def __init__(self, cores):
        self.cores = cores


class _ExplodingMatrix:
    def __getitem__(self, key):
        raise RuntimeError("matrix backend exploded")


def test_edges_computed_from_valid_matrix():
    edges = _compute_connectivity_edges(_HCM([_Core(np.ones((1, 3)))]))
    assert len(edges) == 1
    assert edges[0].weight_sum_abs == 3.0
    assert edges[0].fan_count == 1


def test_unindexable_core_matrix_raises():
    with pytest.raises(RuntimeError, match="matrix backend exploded"):
        _compute_connectivity_edges(_HCM([_Core(_ExplodingMatrix())]))

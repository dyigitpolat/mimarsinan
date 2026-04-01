"""
Unit tests for true core coalescing in IRMapping.

When allow_coalescing is True, wide layers (in_features > max_axons) 
should simply generate a single wider NeuralCore rather than falling back 
to partial sum decomposition.

Configuration notes:
  - max_axons=20, in_features=40, out_features=2
    → coalescing: 1 core (outputs fit in 1 tile)
    → psum:      5 cores (2 tiles * 2 pos/neg + 1 accum)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping import IRMapping


def _make_sources(n: int) -> np.ndarray:
    return np.array([IRSource(node_id=-2, index=i) for i in range(n)])


def _make_mapping(
    *,
    max_axons: int,
    max_neurons: int | None = None,
    allow_coalescing: bool = True,
    hardware_bias: bool = False,
) -> IRMapping:
    return IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    )


def _neural_cores(m: IRMapping) -> list[NeuralCore]:
    return [n for n in m.nodes if isinstance(n, NeuralCore)]


class TestCoreCoalescing:
    def test_coalesced_core_is_wide(self):
        """When coalescing is allowed, a single wide core is produced."""
        m = _make_mapping(max_axons=20, max_neurons=64, allow_coalescing=True)
        m.map_fc(_make_sources(40), np.array([2]), torch.randn(2, 40))
        cores = _neural_cores(m)
        assert len(cores) == 1, f"Expected 1 wide core, got {len(cores)}"
        
        # Core has 40 inputs, exceeding max_axons=20
        assert cores[0].get_input_count() == 40

    def test_wide_core_without_coalescing_psum_decomposition(self):
        """When coalescing is disabled, wide layers use psum decomposition."""
        m = _make_mapping(max_axons=20, max_neurons=64, allow_coalescing=False)
        m.map_fc(_make_sources(40), np.array([2]), torch.randn(2, 40))
        cores = _neural_cores(m)
        # Psum produces: pos/neg partial cores per tile + accumulator cores
        assert len(cores) > 1, f"Expected psum decomposition, got {len(cores)} cores"
        roles = {getattr(c, 'psum_role', None) for c in cores}
        assert 'partial_pos' in roles
        assert 'partial_neg' in roles
        assert 'accum' in roles
        # No coalescing metadata on psum cores
        for c in cores:
            assert c.coalescing_group_id is None
            assert c.coalescing_role is None

    def test_coalescing_tiles_outputs_when_necessary(self):
        """Even with wide cores allowed, it must tile if neurons exceed max_neurons."""
        m = _make_mapping(max_axons=20, max_neurons=32, allow_coalescing=True)
        # out_features=40 > max_neurons=32 -> should split into 2 cores
        m.map_fc(_make_sources(40), np.array([40]), torch.randn(40, 40))
        cores = _neural_cores(m)
        assert len(cores) == 2, f"Expected 2 tiled wide cores, got {len(cores)}"
        
        # Both tiles have the full 40 inputs
        assert cores[0].get_input_count() == 40
        assert cores[1].get_input_count() == 40
        
        # The output neurons are split (e.g. 32 and 8)
        assert cores[0].get_output_count() == 32
        assert cores[1].get_output_count() == 8

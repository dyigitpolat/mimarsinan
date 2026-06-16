"""
Unit tests for true core coalescing in IRMapping.

When allow_coalescing is True, wide layers (in_features > max_axons)
generate a single wider NeuralCore (the packer fuses N hard cores into one wider
crossbar). When allow_coalescing is False the chip lacks inter-core membrane
transfer, so a wide layer is unmappable (the lossy partial-sum fallback was
removed) and raises ``WideFanInUnsupportedError``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.mapping_structure import WideFanInUnsupportedError


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

    def test_wide_core_without_flag_is_unmappable(self):
        """Wide layers need coalescing (inter-core membrane transfer); with the flag
        off they are unmappable — the lossy firing partial-sum fallback was removed."""
        m = _make_mapping(max_axons=20, max_neurons=64, allow_coalescing=False)
        with pytest.raises(WideFanInUnsupportedError):
            m.map_fc(_make_sources(40), np.array([2]), torch.randn(2, 40))

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

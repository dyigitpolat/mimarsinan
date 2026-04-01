"""Tests for mapping_structure.py shared structural helpers.

Verifies:
1. compute_core_input_count respects hardware_bias flag.
2. compute_fc_tiling_mode returns correct modes for all input combinations.
3. compute_psum_params returns identical results to the old inline logic.
4. Layout and IR mapping produce identical core counts for FC layers (regression).
"""

from __future__ import annotations

import math
import pytest

from mimarsinan.mapping.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
    compute_psum_params,
)


class TestComputeCoreInputCount:
    def test_no_bias(self):
        assert compute_core_input_count(64, False, False) == 64
        assert compute_core_input_count(64, False, True) == 64

    def test_legacy_bias(self):
        assert compute_core_input_count(64, True, False) == 65

    def test_hardware_bias(self):
        assert compute_core_input_count(64, True, True) == 64

    def test_zero_sources(self):
        assert compute_core_input_count(0, True, False) == 1
        assert compute_core_input_count(0, True, True) == 0


class TestComputeFcTilingMode:
    def test_single_fits(self):
        mode = compute_fc_tiling_mode(32, 16, 64, 64, False, False, False)
        assert mode == "single"

    def test_single_with_bias_still_fits(self):
        mode = compute_fc_tiling_mode(63, 16, 64, 64, True, False, False)
        assert mode == "single"

    def test_wide_triggers_psum_without_coalescing(self):
        mode = compute_fc_tiling_mode(128, 16, 64, 64, False, False, False)
        assert mode == "psum"

    def test_wide_triggers_coalescing(self):
        mode = compute_fc_tiling_mode(128, 16, 64, 64, False, False, True)
        assert mode == "coalescing"

    def test_bias_pushes_over_max_axons_legacy(self):
        mode = compute_fc_tiling_mode(64, 16, 64, 64, True, False, False)
        assert mode == "psum"

    def test_bias_does_not_push_over_with_hardware_bias(self):
        mode = compute_fc_tiling_mode(64, 16, 64, 64, True, True, False)
        assert mode == "single"

    def test_output_tiled(self):
        mode = compute_fc_tiling_mode(32, 128, 64, 64, False, False, False)
        assert mode == "output_tiled"

    def test_wide_takes_priority_over_output_tiled(self):
        mode = compute_fc_tiling_mode(128, 128, 64, 64, False, False, False)
        assert mode == "psum"

    def test_no_max_axons(self):
        mode = compute_fc_tiling_mode(999, 16, None, 64, False, False, False)
        assert mode == "single"

    def test_no_max_neurons(self):
        mode = compute_fc_tiling_mode(32, 999, 64, None, False, False, False)
        assert mode == "single"


class TestComputePsumParams:
    def test_basic(self):
        pp = compute_psum_params(128, 32, 64, 64, False, False)
        assert pp.tile_count == 2
        assert len(pp.tile_slices) == 2
        assert pp.tile_slices[0] == (0, 64)
        assert pp.tile_slices[1] == (64, 128)
        assert pp.accum_bias_axons == 0
        assert pp.out_block_size <= 64
        assert pp.out_block_size == (64 - 0) // (2 * 2)  # 16

    def test_with_bias_legacy(self):
        pp = compute_psum_params(128, 32, 64, 64, True, False)
        assert pp.accum_bias_axons == 1
        assert pp.out_block_size == (64 - 1) // (2 * 2)  # 15

    def test_with_bias_hardware(self):
        pp = compute_psum_params(128, 32, 64, 64, True, True)
        assert pp.accum_bias_axons == 0
        assert pp.out_block_size == 64 // (2 * 2)  # 16

    def test_tile_count_matches_slices(self):
        pp = compute_psum_params(200, 50, 64, 64, False, False)
        assert pp.tile_count == len(pp.tile_slices)
        assert pp.tile_slices[0][0] == 0
        assert pp.tile_slices[-1][1] == 200

    def test_impossible_raises(self):
        with pytest.raises(ValueError, match="Cannot build psum accumulator"):
            compute_psum_params(10000, 32, 8, 8, False, False)

    def test_out_block_capped_by_max_neurons(self):
        pp = compute_psum_params(128, 500, 64, 4, False, False)
        assert pp.out_block_size <= 4


class TestLayoutIRConsistency:
    """Verify layout and IR produce identical core counts for FC layers."""

    def _count_layout_cores(self, in_f, out_f, max_ax, max_ne, has_bias, hw_bias, coalesce):
        import numpy as np
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
        from mimarsinan.mapping.ir import IRSource

        mapping = LayoutIRMapping(
            max_axons=max_ax, max_neurons=max_ne,
            allow_coalescing=coalesce, hardware_bias=hw_bias,
        )
        src = np.array([IRSource(node_id=-2, index=i) for i in range(in_f)])
        w = np.zeros((out_f, in_f))
        b = np.zeros(out_f) if has_bias else None
        mapping.map_fc(src, np.array([out_f]), w, b)
        return len(mapping.layout_softcores)

    def _count_ir_cores(self, in_f, out_f, max_ax, max_ne, has_bias, hw_bias, coalesce):
        import numpy as np
        import torch
        from mimarsinan.mapping.ir_mapping import IRMapping
        from mimarsinan.mapping.ir import IRSource

        mapping = IRMapping(
            max_axons=max_ax, max_neurons=max_ne,
            allow_coalescing=coalesce, hardware_bias=hw_bias,
        )
        src = np.array([IRSource(node_id=-2, index=i) for i in range(in_f)])
        w = np.zeros((out_f, in_f))
        b = np.zeros(out_f) if has_bias else None
        mapping.map_fc(src, np.array([out_f]), w, b)
        return len(mapping.nodes)

    @pytest.mark.parametrize("has_bias,hw_bias,coalesce", [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (True, True, True),
    ])
    def test_single_core(self, has_bias, hw_bias, coalesce):
        assert self._count_layout_cores(32, 16, 64, 64, has_bias, hw_bias, coalesce) == \
               self._count_ir_cores(32, 16, 64, 64, has_bias, hw_bias, coalesce)

    @pytest.mark.parametrize("has_bias,hw_bias", [
        (False, False),
        (True, False),
        (True, True),
    ])
    def test_psum(self, has_bias, hw_bias):
        assert self._count_layout_cores(128, 32, 64, 64, has_bias, hw_bias, False) == \
               self._count_ir_cores(128, 32, 64, 64, has_bias, hw_bias, False)

    @pytest.mark.parametrize("has_bias,hw_bias", [
        (False, False),
        (True, False),
        (True, True),
    ])
    def test_output_tiled(self, has_bias, hw_bias):
        assert self._count_layout_cores(32, 128, 64, 64, has_bias, hw_bias, False) == \
               self._count_ir_cores(32, 128, 64, 64, has_bias, hw_bias, False)

    @pytest.mark.parametrize("has_bias,hw_bias", [
        (False, False),
        (True, False),
        (True, True),
    ])
    def test_coalescing(self, has_bias, hw_bias):
        assert self._count_layout_cores(128, 32, 64, 64, has_bias, hw_bias, True) == \
               self._count_ir_cores(128, 32, 64, 64, has_bias, hw_bias, True)

    def test_bias_boundary_legacy(self):
        """Bias pushes exactly to max_axons -> should still be 'single'."""
        assert self._count_layout_cores(63, 16, 64, 64, True, False, False) == 1
        assert self._count_ir_cores(63, 16, 64, 64, True, False, False) == 1

    def test_bias_boundary_psum_trigger(self):
        """64 features + legacy bias = 65 > 64 -> should trigger psum."""
        n_layout = self._count_layout_cores(64, 16, 64, 64, True, False, False)
        n_ir = self._count_ir_cores(64, 16, 64, 64, True, False, False)
        assert n_layout == n_ir
        assert n_layout > 1

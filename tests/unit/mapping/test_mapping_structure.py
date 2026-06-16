"""Tests for mapping_structure.py shared structural helpers.

Verifies:
1. compute_core_input_count respects hardware_bias flag.
2. compute_fc_tiling_mode returns correct modes for all input combinations —
   a wide fan-in maps via ``coalescing`` (the bit-exact fuse) when allowed, and
   raises when ``allow_coalescing=False`` (the chip lacks membrane transfer; the
   lossy firing partial-sum fallback was removed).
3. Layout and IR mapping produce identical core counts for FC layers (regression).
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.platform.mapping_structure import (
    WideFanInUnsupportedError,
    compute_core_input_count,
    compute_fc_tiling_mode,
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
        assert compute_fc_tiling_mode(32, 16, 64, 64, False, False, False) == "single"

    def test_single_with_bias_still_fits(self):
        assert compute_fc_tiling_mode(63, 16, 64, 64, True, False, False) == "single"

    def test_wide_with_coalescing_fuses(self):
        """A wide fan-in coalesces (fuse into one wider crossbar) when the chip
        supports inter-core membrane transfer."""
        assert compute_fc_tiling_mode(128, 16, 64, 64, False, False, True) == "coalescing"

    def test_wide_without_coalescing_raises(self):
        """Without coalescing capability a wide fan-in is unmappable — it must fail
        loudly, not silently emit a mapping the chip cannot run."""
        with pytest.raises(WideFanInUnsupportedError):
            compute_fc_tiling_mode(128, 16, 64, 64, False, False, False)

    def test_bias_pushes_over_max_axons_legacy_raises(self):
        """64 features + legacy bias = 65 > 64 -> wide -> needs coalescing."""
        with pytest.raises(WideFanInUnsupportedError):
            compute_fc_tiling_mode(64, 16, 64, 64, True, False, False)
        assert compute_fc_tiling_mode(64, 16, 64, 64, True, False, True) == "coalescing"

    def test_bias_does_not_push_over_with_hardware_bias(self):
        assert compute_fc_tiling_mode(64, 16, 64, 64, True, True, False) == "single"

    def test_output_tiled(self):
        assert compute_fc_tiling_mode(32, 128, 64, 64, False, False, False) == "output_tiled"

    def test_wide_takes_priority_over_output_tiled(self):
        assert compute_fc_tiling_mode(128, 128, 64, 64, False, False, True) == "coalescing"
        with pytest.raises(WideFanInUnsupportedError):
            compute_fc_tiling_mode(128, 128, 64, 64, False, False, False)

    def test_no_max_axons(self):
        assert compute_fc_tiling_mode(999, 16, None, 64, False, False, False) == "single"

    def test_no_max_neurons(self):
        assert compute_fc_tiling_mode(32, 999, 64, None, False, False, False) == "single"


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
        from mimarsinan.mapping.ir_mapping_class import IRMapping
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
    def test_wide_without_coalescing_is_unmappable(self, has_bias, hw_bias):
        """A wide layer with coalescing off is unmappable in both layout and IR —
        the lossy firing partial-sum fallback was removed."""
        with pytest.raises(WideFanInUnsupportedError):
            self._count_layout_cores(128, 32, 64, 64, has_bias, hw_bias, False)
        with pytest.raises(WideFanInUnsupportedError):
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

    def test_bias_boundary_coalescing_trigger(self):
        """64 features + legacy bias = 65 > 64 -> wide -> one coalescing core
        (fused at pack time) when coalescing is enabled. Layout and IR agree."""
        n_layout = self._count_layout_cores(64, 16, 64, 64, True, False, True)
        n_ir = self._count_ir_cores(64, 16, 64, 64, True, False, True)
        assert n_layout == n_ir == 1

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
    ChipCapabilities,
    MappingStrategy,
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


class TestChipCapabilities:
    def test_defaults_are_conservative(self):
        caps = ChipCapabilities()
        assert caps.max_axons is None
        assert caps.max_neurons is None
        assert caps.hardware_bias is False
        assert caps.allow_coalescing is False
        assert caps.allow_neuron_splitting is False
        assert caps.allow_scheduling is False
        assert caps.allow_per_layer_s is False

    def test_frozen(self):
        caps = ChipCapabilities(max_axons=64)
        with pytest.raises(Exception):
            caps.max_axons = 128  # type: ignore[misc]

    def test_carries_grid_and_permissions(self):
        caps = ChipCapabilities(
            max_axons=64,
            max_neurons=32,
            hardware_bias=True,
            allow_coalescing=True,
            allow_neuron_splitting=True,
            allow_scheduling=True,
        )
        assert caps.max_axons == 64
        assert caps.max_neurons == 32
        assert caps.hardware_bias is True
        assert caps.allow_coalescing is True
        assert caps.allow_neuron_splitting is True
        assert caps.allow_scheduling is True

    def test_from_platform_constraints_reads_three_bits(self):
        caps = ChipCapabilities.from_platform_constraints(
            {
                "allow_coalescing": True,
                "allow_neuron_splitting": True,
                "allow_scheduling": False,
                "cores": [{"max_axons": 64, "max_neurons": 64, "count": 4}],
            }
        )
        assert caps.allow_coalescing is True
        assert caps.allow_neuron_splitting is True
        assert caps.allow_scheduling is False
        # Grid is carried separately (cores/core_types), not by the capability factory.
        assert caps.max_axons is None
        assert caps.max_neurons is None

    def test_from_platform_constraints_defaults_false(self):
        caps = ChipCapabilities.from_platform_constraints({})
        assert caps.allow_coalescing is False
        assert caps.allow_neuron_splitting is False
        assert caps.allow_scheduling is False

    def test_from_platform_constraints_coerces_truthy(self):
        caps = ChipCapabilities.from_platform_constraints(
            {"allow_coalescing": 1, "allow_neuron_splitting": 0, "allow_scheduling": "x"}
        )
        assert caps.allow_coalescing is True
        assert caps.allow_neuron_splitting is False
        assert caps.allow_scheduling is True

    def test_permission_kwargs_match_helper_signature(self):
        caps = ChipCapabilities(
            allow_coalescing=True,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        assert caps.permission_kwargs() == {
            "allow_neuron_splitting": False,
            "allow_coalescing": True,
            "allow_scheduling": True,
        }

    def test_allow_per_layer_s_gate_default_false(self):
        # EW1 RESERVED capability: declared, defaults False, carried + readable.
        assert ChipCapabilities().allow_per_layer_s is False
        assert ChipCapabilities(allow_per_layer_s=True).allow_per_layer_s is True

    def test_from_platform_constraints_reads_allow_per_layer_s(self):
        caps = ChipCapabilities.from_platform_constraints({"allow_per_layer_s": True})
        assert caps.allow_per_layer_s is True
        # Absent / falsey ⇒ False (byte-identical default).
        assert ChipCapabilities.from_platform_constraints({}).allow_per_layer_s is False
        assert ChipCapabilities.from_platform_constraints(
            {"allow_per_layer_s": 0}
        ).allow_per_layer_s is False

    def test_permission_kwargs_excludes_per_layer_s(self):
        # allow_per_layer_s is a TEMPORAL gate, not a layout/verify kwarg — the leaf
        # helper signatures stay unchanged (byte-identical).
        caps = ChipCapabilities(allow_per_layer_s=True)
        assert "allow_per_layer_s" not in caps.permission_kwargs()

    def test_allow_weight_reuse_gate_default_false(self):
        # RESERVED weight-reuse capability: declared, defaults False, carried + readable
        # (the allow_per_layer_s pattern). Default False ⇒ schedule treats every pass
        # as a reprogram ⇒ byte-identical.
        assert ChipCapabilities().allow_weight_reuse is False
        assert ChipCapabilities(allow_weight_reuse=True).allow_weight_reuse is True

    def test_from_platform_constraints_reads_allow_weight_reuse(self):
        caps = ChipCapabilities.from_platform_constraints({"allow_weight_reuse": True})
        assert caps.allow_weight_reuse is True
        # Absent / falsey ⇒ False (byte-identical default).
        assert (
            ChipCapabilities.from_platform_constraints({}).allow_weight_reuse is False
        )
        assert ChipCapabilities.from_platform_constraints(
            {"allow_weight_reuse": 0}
        ).allow_weight_reuse is False

    def test_permission_kwargs_excludes_weight_reuse(self):
        # allow_weight_reuse is a SCHEDULING/cost gate, not a layout/verify kwarg — the
        # leaf helper signatures stay unchanged (byte-identical).
        caps = ChipCapabilities(allow_weight_reuse=True)
        assert "allow_weight_reuse" not in caps.permission_kwargs()


class TestMappingStrategy:
    def test_permission_accessors_mirror_capabilities(self):
        caps = ChipCapabilities(
            allow_coalescing=True,
            allow_neuron_splitting=False,
            allow_scheduling=True,
        )
        strat = MappingStrategy.resolve(caps)
        assert strat.allow_coalescing is True
        assert strat.allow_neuron_splitting is False
        assert strat.allow_scheduling is True
        assert strat.capabilities is caps

    def test_permission_kwargs_delegate_to_capabilities(self):
        caps = ChipCapabilities(
            allow_coalescing=True,
            allow_neuron_splitting=True,
            allow_scheduling=False,
        )
        strat = MappingStrategy.resolve(caps)
        assert strat.permission_kwargs() == caps.permission_kwargs()

    def test_allow_per_layer_s_accessor_mirrors_capabilities(self):
        caps = ChipCapabilities(allow_per_layer_s=True)
        assert MappingStrategy.resolve(caps).allow_per_layer_s is True
        assert MappingStrategy.resolve(
            ChipCapabilities()
        ).allow_per_layer_s is False

    def test_allow_weight_reuse_accessor_mirrors_capabilities(self):
        caps = ChipCapabilities(allow_weight_reuse=True)
        assert MappingStrategy.resolve(caps).allow_weight_reuse is True
        assert MappingStrategy.resolve(
            ChipCapabilities()
        ).allow_weight_reuse is False

    @pytest.mark.parametrize(
        "in_f,out_f,max_ax,max_ne,has_bias,hw_bias,coalesce",
        [
            (32, 16, 64, 64, False, False, False),    # single
            (63, 16, 64, 64, True, False, False),     # single (bias fits)
            (128, 16, 64, 64, False, False, True),    # coalescing
            (32, 128, 64, 64, False, False, False),   # output_tiled
            (128, 128, 64, 64, False, False, True),   # coalescing > output_tiled
            (64, 16, 64, 64, True, True, False),      # single (hw bias)
            (999, 16, None, 64, False, False, False), # no max_axons
            (32, 999, 64, None, False, False, False), # no max_neurons
        ],
    )
    def test_tiling_mode_matches_compute_fc_tiling_mode(
        self, in_f, out_f, max_ax, max_ne, has_bias, hw_bias, coalesce
    ):
        """The derived strategy decision is byte-identical to the standalone fn."""
        caps = ChipCapabilities(
            max_axons=max_ax,
            max_neurons=max_ne,
            hardware_bias=hw_bias,
            allow_coalescing=coalesce,
        )
        strat = MappingStrategy.resolve(caps)
        expected = compute_fc_tiling_mode(
            in_f, out_f, max_ax, max_ne, has_bias, hw_bias, coalesce
        )
        assert strat.tiling_mode(in_f, out_f, has_bias) == expected

    def test_tiling_mode_raises_when_wide_and_no_coalescing(self):
        caps = ChipCapabilities(max_axons=64, max_neurons=64, allow_coalescing=False)
        strat = MappingStrategy.resolve(caps)
        with pytest.raises(WideFanInUnsupportedError):
            strat.tiling_mode(128, 16, False)


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

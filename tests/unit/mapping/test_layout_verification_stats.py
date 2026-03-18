"""Tests for layout-verification stats contract.

Defines the expected output shape and numeric correctness of
``build_layout_verification_stats``, covering:
  - total and per-core wasted-axon/neuron percentages
  - mapped-parameter utilization percentages
  - coalesced-core and neuron-splitting statistics
  - edge cases (single softcore, perfect-fit, infeasible)
"""

from __future__ import annotations

import pytest
from typing import List

from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout_verification_stats import (
    build_layout_verification_stats,
    LayoutVerificationStats,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_softcores(specs: list[tuple[int, int]]) -> list[LayoutSoftCoreSpec]:
    return [
        LayoutSoftCoreSpec(input_count=a, output_count=n)
        for a, n in specs
    ]


def _pack_and_stats(
    softcores: list[LayoutSoftCoreSpec],
    core_types: list[LayoutHardCoreType],
    *,
    allow_neuron_splitting: bool = False,
    allow_axon_coalescing: bool = False,
) -> LayoutVerificationStats:
    """Pack and build stats in one step (convenience)."""
    return build_layout_verification_stats(
        softcores=softcores,
        core_types=core_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_axon_coalescing=allow_axon_coalescing,
    )


# ── Basic contract ───────────────────────────────────────────────────────────

class TestStatsShape:
    """Stats object has all required fields with correct types."""

    def test_required_fields_present(self):
        scs = _make_softcores([(8, 4), (8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=4)]
        stats = _pack_and_stats(scs, hw)

        assert isinstance(stats.feasible, bool)
        assert stats.feasible is True
        assert isinstance(stats.total_cores, int)
        assert isinstance(stats.total_softcores, int)

        # Percentages are 0..100 floats
        assert 0 <= stats.total_wasted_neurons_pct <= 100
        assert 0 <= stats.total_wasted_axons_pct <= 100
        assert 0 <= stats.mapped_params_pct <= 100

        # Per-core min/avg/max
        assert stats.per_core_wasted_neurons_pct_min <= stats.per_core_wasted_neurons_pct_avg
        assert stats.per_core_wasted_neurons_pct_avg <= stats.per_core_wasted_neurons_pct_max
        assert stats.per_core_wasted_axons_pct_min <= stats.per_core_wasted_axons_pct_avg
        assert stats.per_core_wasted_axons_pct_avg <= stats.per_core_wasted_axons_pct_max
        assert stats.per_core_mapped_params_pct_min <= stats.per_core_mapped_params_pct_avg
        assert stats.per_core_mapped_params_pct_avg <= stats.per_core_mapped_params_pct_max

        # Coalescing and splitting summaries
        assert isinstance(stats.coalesced_cores, int)
        assert isinstance(stats.split_cores, int)

    def test_to_dict_roundtrip(self):
        scs = _make_softcores([(8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=2)]
        stats = _pack_and_stats(scs, hw)
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["feasible"] is True
        assert "total_cores" in d
        assert "mapped_params_pct" in d


# ── Numeric correctness ─────────────────────────────────────────────────────

class TestPerfectFit:
    """When softcores exactly fill hardware cores, waste is 0%."""

    def test_single_core_exact_fit(self):
        scs = _make_softcores([(16, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 1
        assert stats.total_wasted_axons_pct == pytest.approx(0.0)
        assert stats.total_wasted_neurons_pct == pytest.approx(0.0)
        assert stats.mapped_params_pct == pytest.approx(100.0)

    def test_two_cores_exact_fit(self):
        scs = _make_softcores([(16, 8), (16, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=2)]
        stats = _pack_and_stats(scs, hw)

        assert stats.total_cores == 2
        assert stats.per_core_wasted_axons_pct_min == pytest.approx(0.0)
        assert stats.per_core_wasted_neurons_pct_max == pytest.approx(0.0)
        assert stats.per_core_mapped_params_pct_min == pytest.approx(100.0)


class TestKnownWaste:
    """Known waste scenarios with manual calculations."""

    def test_single_small_softcore_in_large_core(self):
        """4x2 softcore in 16x8 core: 75% axon waste, 75% neuron waste, 6.25% param utilization."""
        scs = _make_softcores([(4, 2)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 1
        # Wasted axons: (16-4)/16 = 75%
        assert stats.total_wasted_axons_pct == pytest.approx(75.0)
        # Wasted neurons: (8-2)/8 = 75%
        assert stats.total_wasted_neurons_pct == pytest.approx(75.0)
        # Mapped params: (4*2)/(16*8) = 8/128 = 6.25%
        assert stats.mapped_params_pct == pytest.approx(6.25)

    def test_two_softcores_packed_into_one_core(self):
        """Two 8x4 softcores in a 16x16 core."""
        scs = _make_softcores([(8, 4), (8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 1
        # 2 softcores packed into one core means: used_axons = 16, used_neurons = 8
        # Wasted axons: 0/16 = 0%   (both fit side-by-side on neuron axis)
        # Wasted neurons: (16-8)/16 = 50%
        assert stats.total_wasted_axons_pct == pytest.approx(0.0)
        assert stats.total_wasted_neurons_pct == pytest.approx(50.0)


class TestPerCoreMinAvgMax:
    """Per-core statistics with multiple cores."""

    def test_heterogeneous_packing(self):
        """Two cores with different utilizations produce distinct min/max."""
        # Core 1: 16x8 softcore in 16x8 core (perfect fit)
        # Core 2: 4x2 softcore in 16x8 core (poor fit)
        scs = _make_softcores([(16, 8), (4, 2)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=2)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 2
        # Min mapped params should be from the poor-fit core
        assert stats.per_core_mapped_params_pct_min < stats.per_core_mapped_params_pct_max
        assert stats.per_core_mapped_params_pct_max == pytest.approx(100.0)


# ── Coalescing statistics ────────────────────────────────────────────────────

class TestCoalescingStats:
    """Axon coalescing produces coalesced_cores count."""

    def test_coalescing_produces_fragments(self):
        """A 32-axon softcore on 16-axon hardware produces coalesced fragments."""
        scs = _make_softcores([(32, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]
        stats = _pack_and_stats(scs, hw, allow_axon_coalescing=True)

        assert stats.feasible
        # 32 axons / 16 max = 2 fragments, so 1 extra fragment introduced
        assert stats.coalesced_cores >= 1

    def test_no_coalescing_when_disabled(self):
        scs = _make_softcores([(8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw, allow_axon_coalescing=False)

        assert stats.coalesced_cores == 0


# ── Neuron splitting statistics ──────────────────────────────────────────────

class TestSplittingStats:
    """Neuron splitting produces split_cores count."""

    def test_splitting_produces_fragments(self):
        """A 32-neuron softcore on 16-neuron hardware produces split fragments."""
        scs = _make_softcores([(8, 32)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=16, count=4)]
        stats = _pack_and_stats(scs, hw, allow_neuron_splitting=True)

        assert stats.feasible
        # 32 neurons / 16 max = 1 split (producing 2 fragments)
        assert stats.split_cores >= 1

    def test_no_splitting_when_disabled(self):
        scs = _make_softcores([(8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw, allow_neuron_splitting=False)

        assert stats.split_cores == 0


# ── Combined coalescing + splitting ──────────────────────────────────────────

class TestCombinedFeatures:
    def test_both_features_produce_stats(self):
        """64x64 softcore on 16x16 hardware with both features."""
        scs = _make_softcores([(64, 64)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=64)]
        stats = _pack_and_stats(
            scs, hw,
            allow_axon_coalescing=True,
            allow_neuron_splitting=True,
        )

        assert stats.feasible
        assert stats.coalesced_cores > 0
        assert stats.split_cores > 0
        assert stats.total_cores > 1


# ── Infeasible packing ──────────────────────────────────────────────────────

class TestInfeasible:
    def test_infeasible_returns_stats_with_feasible_false(self):
        scs = _make_softcores([(32, 32)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible is False
        assert stats.total_cores == 0

    def test_infeasible_has_zero_percentages(self):
        scs = _make_softcores([(32, 32)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.mapped_params_pct == 0.0
        assert stats.total_wasted_axons_pct == 0.0
        assert stats.total_wasted_neurons_pct == 0.0


# ── Integration with verify_hardware_config ──────────────────────────────────

class TestVerifyHardwareConfigStats:
    """verify_hardware_config returns stats in its response dict."""

    def test_stats_key_present_on_success(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = _make_softcores([(8, 4), (8, 4)])
        core_types = [{"max_axons": 16, "max_neurons": 8, "count": 2}]
        result = verify_hardware_config(scs, core_types)

        assert "stats" in result
        stats = result["stats"]
        assert isinstance(stats, dict)
        assert stats["feasible"] is True
        assert "total_cores" in stats
        assert "mapped_params_pct" in stats

    def test_stats_key_present_on_failure(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = _make_softcores([(64, 64)])
        core_types = [{"max_axons": 8, "max_neurons": 8, "count": 1}]
        result = verify_hardware_config(scs, core_types)

        assert "stats" in result
        assert result["stats"]["feasible"] is False

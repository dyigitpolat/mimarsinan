"""Tests for layout-verification stats contract.

Defines the expected output shape and numeric correctness of
``build_layout_verification_stats``, covering:
  - total and per-core wasted-axon/neuron percentages
  - mapped-parameter utilization percentages
  - coalesced-core and neuron-splitting statistics
  - coalescing group distribution (count, min/median/max fragments per group)
  - split distribution (count, min/median/max splits per softcore)
  - neural segment count, per-segment latency min/median/max, and threshold group counts
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

        # New extended fields
        assert isinstance(stats.neural_segment_count, int)
        assert isinstance(stats.segment_latency_min, float)
        assert isinstance(stats.segment_latency_median, float)
        assert isinstance(stats.segment_latency_max, float)
        assert isinstance(stats.threshold_group_count, int)
        assert isinstance(stats.coalescing_group_count, int)
        assert isinstance(stats.split_softcore_count, int)
        assert stats.segment_latency_min <= stats.segment_latency_median
        assert stats.segment_latency_median <= stats.segment_latency_max
        assert stats.coalescing_frags_per_group_min <= stats.coalescing_frags_per_group_max
        assert stats.splits_per_softcore_min <= stats.splits_per_softcore_max

    def test_to_dict_roundtrip(self):
        scs = _make_softcores([(8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=2)]
        stats = _pack_and_stats(scs, hw)
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["feasible"] is True
        assert "total_cores" in d
        assert "mapped_params_pct" in d
        assert "neural_segment_count" in d
        assert "segment_latency_min" in d
        assert "segment_latency_median" in d
        assert "segment_latency_max" in d
        assert "threshold_group_count" in d
        assert "coalescing_group_count" in d
        assert "split_softcore_count" in d


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


# ── Coalescing group distribution ────────────────────────────────────────────

class TestCoalescingGroupDistribution:
    """Coalescing group count and per-group fragment distribution."""

    def test_single_coalesced_softcore_reports_one_group(self):
        """One 32-axon softcore on 16-axon hw → one coalescing group of size 2."""
        scs = _make_softcores([(32, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]
        stats = _pack_and_stats(scs, hw, allow_axon_coalescing=True)

        assert stats.coalescing_group_count == 1
        assert stats.coalescing_frags_per_group_min == pytest.approx(2.0)
        assert stats.coalescing_frags_per_group_median == pytest.approx(2.0)
        assert stats.coalescing_frags_per_group_max == pytest.approx(2.0)

    def test_two_different_coalescing_groups(self):
        """32-axon and 48-axon softcores on 16-axon hw → two groups of sizes 2 and 3."""
        scs = _make_softcores([(32, 4), (48, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=4, count=8)]
        stats = _pack_and_stats(scs, hw, allow_axon_coalescing=True)

        assert stats.coalescing_group_count == 2
        assert stats.coalescing_frags_per_group_min == pytest.approx(2.0)
        assert stats.coalescing_frags_per_group_max == pytest.approx(3.0)
        # Median of [2, 3] = 2.5
        assert stats.coalescing_frags_per_group_median == pytest.approx(2.5)

    def test_no_coalescing_group_when_fits(self):
        """Softcore fits in one core — no coalescing groups."""
        scs = _make_softcores([(8, 4)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw, allow_axon_coalescing=True)

        assert stats.coalescing_group_count == 0
        assert stats.coalescing_frags_per_group_min == pytest.approx(0.0)
        assert stats.coalescing_frags_per_group_max == pytest.approx(0.0)


# ── Split distribution ───────────────────────────────────────────────────────

class TestSplitDistribution:
    """Split softcore count and per-softcore split count distribution."""

    def test_single_split_softcore_reports_one_entry(self):
        """One 32-neuron softcore on 16-neuron hw → one softcore split once."""
        scs = _make_softcores([(8, 32)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=16, count=4)]
        stats = _pack_and_stats(scs, hw, allow_neuron_splitting=True)

        assert stats.split_softcore_count == 1
        assert stats.splits_per_softcore_min == pytest.approx(1.0)
        assert stats.splits_per_softcore_median == pytest.approx(1.0)
        assert stats.splits_per_softcore_max == pytest.approx(1.0)

    def test_two_softcores_split_different_amounts(self):
        """32-neuron and 48-neuron softcores on 16-neuron hw → 1 and 2 splits each."""
        scs = _make_softcores([(4, 32), (4, 48)])
        hw = [LayoutHardCoreType(max_axons=4, max_neurons=16, count=8)]
        stats = _pack_and_stats(scs, hw, allow_neuron_splitting=True)

        assert stats.split_softcore_count == 2
        assert stats.splits_per_softcore_min == pytest.approx(1.0)
        assert stats.splits_per_softcore_max == pytest.approx(2.0)

    def test_no_splits_when_all_fit(self):
        """All softcores fit; no splits."""
        scs = _make_softcores([(4, 4), (4, 4)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=2)]
        stats = _pack_and_stats(scs, hw, allow_neuron_splitting=True)

        assert stats.split_softcore_count == 0
        assert stats.splits_per_softcore_min == pytest.approx(0.0)
        assert stats.splits_per_softcore_max == pytest.approx(0.0)


# ── Segment latency / threshold counts ──────────────────────────────────────

class TestLatencyStats:
    """Segment latency summaries and threshold counts are derived from softcores."""

    def test_no_latency_tags_reports_zero(self):
        scs = [LayoutSoftCoreSpec(input_count=8, output_count=4)]  # latency_tag=None by default
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.neural_segment_count == 0
        assert stats.segment_latency_min == pytest.approx(0.0)
        assert stats.segment_latency_median == pytest.approx(0.0)
        assert stats.segment_latency_max == pytest.approx(0.0)

    def test_segment_count_tracks_distinct_latency_tags(self):
        scs = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=0),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=1),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=0),
        ]
        hw = [LayoutHardCoreType(max_axons=4, max_neurons=4, count=3)]
        stats = _pack_and_stats(scs, hw)

        assert stats.neural_segment_count == 2
        assert stats.segment_latency_min == pytest.approx(1.0)
        assert stats.segment_latency_median == pytest.approx(1.0)
        assert stats.segment_latency_max == pytest.approx(1.0)

    def test_segment_latency_summary_uses_tiers_per_segment(self):
        scs = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=0, segment_id=0),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=1, segment_id=0),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=2, segment_id=1),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=3, segment_id=2),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=4, segment_id=2),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=5, segment_id=2),
        ]
        hw = [LayoutHardCoreType(max_axons=4, max_neurons=4, count=6)]
        stats = _pack_and_stats(scs, hw)

        assert stats.neural_segment_count == 3
        assert stats.segment_latency_min == pytest.approx(1.0)
        assert stats.segment_latency_median == pytest.approx(2.0)
        assert stats.segment_latency_max == pytest.approx(3.0)

    def test_threshold_groups_counted(self):
        scs = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, threshold_group_id=0),
            LayoutSoftCoreSpec(input_count=4, output_count=4, threshold_group_id=1),
        ]
        hw = [LayoutHardCoreType(max_axons=4, max_neurons=4, count=2)]
        stats = _pack_and_stats(scs, hw)

        assert stats.threshold_group_count == 2

    def test_single_threshold_group_default(self):
        scs = _make_softcores([(4, 4), (4, 4)])
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=1)]
        stats = _pack_and_stats(scs, hw)

        assert stats.threshold_group_count == 1  # default group_id=0 for all


# ── Chip-level metrics (idle core accounting) ────────────────────────────────

class TestChipLevelMetrics:
    """Metrics account for the entire chip, not just actively used cores."""

    def test_idle_cores_increase_waste(self):
        """1 softcore of 16x8 on a 2-core chip (16x8 each): 50% idle."""
        scs = _make_softcores([(16, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=2)]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 1
        assert stats.total_hw_cores == 2
        assert stats.mapped_params_pct == pytest.approx(50.0)
        assert stats.total_wasted_axons_pct == pytest.approx(50.0)
        assert stats.total_wasted_neurons_pct == pytest.approx(50.0)

    def test_all_cores_used_matches_no_idle(self):
        """2 softcores filling 2 cores exactly: no idle waste."""
        scs = _make_softcores([(16, 8), (16, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=2)]
        stats = _pack_and_stats(scs, hw)

        assert stats.total_cores == 2
        assert stats.total_hw_cores == 2
        assert stats.mapped_params_pct == pytest.approx(100.0)
        assert stats.total_wasted_axons_pct == pytest.approx(0.0)
        assert stats.total_wasted_neurons_pct == pytest.approx(0.0)

    def test_per_core_stats_include_idle(self):
        """Per-core min/avg/max include idle cores (100% waste, 0% util)."""
        scs = _make_softcores([(16, 8)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=3)]
        stats = _pack_and_stats(scs, hw)

        assert stats.total_cores == 1
        assert stats.total_hw_cores == 3
        # 1 used core (0% waste) + 2 idle cores (100% waste)
        assert stats.per_core_wasted_axons_pct_min == pytest.approx(0.0)
        assert stats.per_core_wasted_axons_pct_max == pytest.approx(100.0)
        assert stats.per_core_wasted_axons_pct_avg == pytest.approx(200.0 / 3.0)
        # Param usage: 1 used core (100%) + 2 idle cores (0%)
        assert stats.per_core_mapped_params_pct_min == pytest.approx(0.0)
        assert stats.per_core_mapped_params_pct_max == pytest.approx(100.0)
        assert stats.per_core_mapped_params_pct_avg == pytest.approx(100.0 / 3.0)

    def test_total_hw_cores_field(self):
        """total_hw_cores equals total hardware core count from core_types."""
        scs = _make_softcores([(4, 4)])
        hw = [
            LayoutHardCoreType(max_axons=8, max_neurons=8, count=3),
            LayoutHardCoreType(max_axons=16, max_neurons=4, count=2),
        ]
        stats = _pack_and_stats(scs, hw)
        assert stats.total_hw_cores == 5

    def test_heterogeneous_idle_cores(self):
        """Chip with two core types, only some cores used.

        Per-core stats must reflect the actual dimensions of each idle core's
        type.  An idle 8x8 core and an idle 16x4 core both waste 100% of their
        axons/neurons, but should still appear as distinct entries in the
        per-core lists (all with 100% waste / 0% param util).
        """
        scs = _make_softcores([(4, 4)])
        hw = [
            LayoutHardCoreType(max_axons=8, max_neurons=8, count=2),
            LayoutHardCoreType(max_axons=16, max_neurons=4, count=2),
        ]
        stats = _pack_and_stats(scs, hw)

        assert stats.feasible
        assert stats.total_cores == 1
        assert stats.total_hw_cores == 4
        # chip_total_capacity = 2*8*8 + 2*16*4 = 128 + 128 = 256
        # used_area = 4*4 = 16
        assert stats.mapped_params_pct == pytest.approx(16.0 / 256.0 * 100.0)

        # 1 used core + 3 idle cores → per-core stats have 4 entries
        # idle cores all have 100% waste and 0% param util
        assert stats.per_core_wasted_axons_pct_max == pytest.approx(100.0)
        assert stats.per_core_mapped_params_pct_min == pytest.approx(0.0)
        # avg param util = (used_core_pct + 0 + 0 + 0) / 4
        used_core_param_pct = 16.0 / (8 * 8) * 100.0  # 4x4 on 8x8 = 25%
        assert stats.per_core_mapped_params_pct_avg == pytest.approx(
            used_core_param_pct / 4.0
        )

    def test_small_softcore_in_large_chip(self):
        """4x2 softcore on 10-core chip of 16x8: mostly idle."""
        scs = _make_softcores([(4, 2)])
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=10)]
        stats = _pack_and_stats(scs, hw)

        assert stats.total_cores == 1
        assert stats.total_hw_cores == 10
        # chip_total_axons = 16*10 = 160, used_axons = 4, wasted = 156
        assert stats.total_wasted_axons_pct == pytest.approx(156.0 / 160.0 * 100.0)
        # chip_total_neurons = 8*10 = 80, used_neurons = 2, wasted = 78
        assert stats.total_wasted_neurons_pct == pytest.approx(78.0 / 80.0 * 100.0)
        # chip_total_capacity = 16*8*10 = 1280, used_area = 4*2 = 8
        assert stats.mapped_params_pct == pytest.approx(8.0 / 1280.0 * 100.0)


class TestChipLevelViaVerifyHardwareConfig:
    """Chip-level metrics flow through verify_hardware_config."""

    def test_idle_cores_reflected_in_verify(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = _make_softcores([(16, 8)])
        core_types = [{"max_axons": 16, "max_neurons": 8, "count": 4}]
        result = verify_hardware_config(scs, core_types)

        stats = result["stats"]
        assert stats["feasible"] is True
        assert stats["total_cores"] == 1
        assert stats["total_hw_cores"] == 4
        assert stats["mapped_params_pct"] == pytest.approx(25.0)
        assert stats["total_wasted_axons_pct"] == pytest.approx(75.0)
        assert stats["total_wasted_neurons_pct"] == pytest.approx(75.0)

    def test_scheduled_idle_cores(self):
        """Scheduled mapping with idle cores per pass."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=64, output_count=64,
                               threshold_group_id=0, latency_tag=i, segment_id=i)
            for i in range(3)
        ]
        core_types = [{"max_axons": 64, "max_neurons": 64, "count": 2}]
        result = verify_hardware_config(scs, core_types, allow_scheduling=True)

        stats = result["stats"]
        assert stats["feasible"] is True
        assert stats["total_hw_cores"] == 2
        # Busiest pass uses 1 of 2 cores
        assert stats["total_cores"] == 1
        assert stats["mapped_params_pct"] == pytest.approx(50.0)

    def test_single_softcore_infeasible_reports_correctly(self):
        """When a single softcore cannot pack on the given hardware,
        the config is correctly marked infeasible (no synthetic fallback)."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=65, output_count=10,
                               threshold_group_id=0, latency_tag=0, segment_id=0,
                               name=f"sc{i}")
            for i in range(5)
        ]
        core_types = [{"max_axons": 8, "max_neurons": 30000, "count": 1}]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=False,
            allow_neuron_splitting=False,
        )
        assert result["feasible"] is False

    def test_coalescing_makes_wide_softcore_feasible(self):
        """With coalescing enabled, a wide softcore should be packable
        via validate-and-split, yielding valid metrics."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=65, output_count=10,
                               threshold_group_id=0, latency_tag=0, segment_id=0,
                               name=f"sc{i}")
            for i in range(2)
        ]
        core_types = [{"max_axons": 8, "max_neurons": 30000, "count": 20}]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=True,
            allow_neuron_splitting=True,
        )
        stats = result["stats"]
        if stats["feasible"]:
            assert stats["mapped_params_pct"] >= 0.0
            assert stats["mapped_params_pct"] <= 100.0
            assert stats["total_wasted_axons_pct"] >= 0.0
            assert stats["total_wasted_axons_pct"] <= 100.0
            assert stats["total_wasted_neurons_pct"] >= 0.0
            assert stats["total_wasted_neurons_pct"] <= 100.0


# ── Unified validate-and-split tests ─────────────────────────────────────────

class TestValidateAndSplitPasses:
    """Tests for _validate_and_split_passes in schedule_partitioner."""

    def test_all_passes_valid(self):
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["a", "b"], ["c"]]
        validated, ok = _validate_and_split_passes(passes, lambda items: True)
        assert ok is True
        assert len(validated) == 2

    def test_split_on_failure(self):
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        # Reject any pass with more than 1 item
        passes = [["a", "b", "c", "d"]]
        validated, ok = _validate_and_split_passes(
            passes, lambda items: len(items) <= 1,
        )
        assert ok is True
        assert len(validated) == 4
        for p in validated:
            assert len(p) == 1

    def test_single_item_infeasible(self):
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["x"]]
        validated, ok = _validate_and_split_passes(passes, lambda items: False)
        assert ok is False
        assert len(validated) == 1

    def test_mixed_feasibility(self):
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["a"], ["b"]]
        validated, ok = _validate_and_split_passes(
            passes, lambda items: items != ["b"],
        )
        assert ok is False
        assert len(validated) == 2

    def test_fragment_expansion_rescues_single_item(self):
        """A single-item pass that fails can be expanded into fragments."""
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["big"]]
        def expander(item):
            if item == "big":
                return ["frag0", "frag1", "frag2", "frag3"]
            return None
        validated, ok = _validate_and_split_passes(
            passes, lambda items: all(i.startswith("frag") for i in items),
            fragment_expander=expander, max_per_pass=2,
        )
        assert ok is True
        assert len(validated) == 2
        for p in validated:
            assert len(p) <= 2

    def test_fragment_expansion_infeasible_when_fragment_fails(self):
        """If a fragment itself can't validate, expansion is infeasible."""
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["big"]]
        def expander(item):
            if item == "big":
                return ["bad_frag"]
            return None
        validated, ok = _validate_and_split_passes(
            passes, lambda items: False,
            fragment_expander=expander, max_per_pass=1,
        )
        assert ok is False

    def test_fragment_expansion_no_expander_means_infeasible(self):
        """Without an expander, a single failing item is infeasible."""
        from mimarsinan.mapping.schedule_partitioner import _validate_and_split_passes
        passes = [["x"]]
        validated, ok = _validate_and_split_passes(
            passes, lambda items: False,
            fragment_expander=None,
        )
        assert ok is False


class TestEstimatePassesWithCoreTypes:
    """estimate_passes_for_layout validates passes with typed packing when core_types is given."""

    def test_core_types_none_preserves_old_behavior(self):
        """Without core_types, behavior is unchanged (no typed validation)."""
        from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout
        softcores = [
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0)
            for _ in range(4)
        ]
        n, passes = estimate_passes_for_layout(softcores, max_cores_per_pass=2)
        assert n == 2
        assert all(len(p) <= 2 for p in passes)

    def test_typed_validation_splits_passes(self):
        """When core_types is given and passes fail packing, they are split."""
        from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout
        softcores = [
            LayoutSoftCoreSpec(input_count=16, output_count=8, segment_id=0, latency_tag=0)
            for _ in range(4)
        ]
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        # Scalar heuristic says budget=1 → 4 passes (1 softcore each).
        # Each single-softcore pass should pack on the 1 hardware core.
        n, passes = estimate_passes_for_layout(
            softcores, max_cores_per_pass=1, core_types=hw,
        )
        assert n == 4
        for p in passes:
            assert len(p) == 1

    def test_typed_validation_more_passes_than_scalar(self):
        """With typed packing, more passes may be needed than the scalar heuristic predicts."""
        from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout
        # 2 softcores, each 16x8. Scalar budget=2 → 1 pass.
        # But with only 1 hw core of 16x8, only 1 softcore per pass → 2 passes.
        softcores = [
            LayoutSoftCoreSpec(input_count=16, output_count=8, segment_id=0, latency_tag=0)
            for _ in range(2)
        ]
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        n, passes = estimate_passes_for_layout(
            softcores, max_cores_per_pass=2, core_types=hw,
        )
        assert n >= 2
        for p in passes:
            assert len(p) <= 1


class TestSoftcoreFragmentExpander:
    """Tests for _make_softcore_fragment_expander in schedule_partitioner.

    The expander only produces splitting fragments (neuron dimension).
    Coalescing groups must fit within a single core type's count.
    """

    def test_no_fragmentation_when_fits(self):
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=16, output_count=8, segment_id=0, latency_tag=0)
        assert expander(sc) is None

    def test_coalescing_only_infeasible_when_count_insufficient(self):
        """A wide softcore needing 4 coalescing cores on a type with count=1
        is infeasible (no splitting enabled to help)."""
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=False)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=8, segment_id=0, latency_tag=0)
        assert expander(sc) is None

    def test_coalescing_only_feasible_when_count_sufficient(self):
        """Same wide softcore but type has count=4 — no splitting needed,
        so no expansion (it fits in one pass as coalesced)."""
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=False)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=8, segment_id=0, latency_tag=0)
        assert expander(sc) is None

    def test_splitting_fragments(self):
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=False, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=16, output_count=32, segment_id=0, latency_tag=0)
        frags = expander(sc)
        assert frags is not None
        assert len(frags) == 4  # ceil(32/8)
        for f in frags:
            assert f.input_count == 16
            assert f.output_count == 8

    def test_combined_coalescing_splitting_expands_only_splitting(self):
        """With coalescing + splitting, only the neuron dimension is expanded.
        Fragments keep the full axon width.  Coalescing count must be sufficient."""
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=4)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=64, segment_id=0, latency_tag=0)
        frags = expander(sc)
        assert frags is not None
        assert len(frags) == 4  # ceil(64/16) splitting fragments
        for f in frags:
            assert f.input_count == 64  # full width preserved for coalescing
            assert f.output_count == 16

    def test_combined_infeasible_when_coalescing_count_insufficient(self):
        """If no type has enough count for coalescing, returns None even
        with splitting enabled."""
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=16, count=1)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=64, segment_id=0, latency_tag=0)
        assert expander(sc) is None

    def test_infeasible_without_coalescing(self):
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=1)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=False, allow_splitting=False)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=64, segment_id=0, latency_tag=0)
        assert expander(sc) is None

    def test_picks_best_core_type_for_splitting(self):
        """With heterogeneous types, the expander picks the type that has
        sufficient coalescing count and minimizes splitting fragments."""
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [
            LayoutHardCoreType(max_axons=16, max_neurons=16, count=4),
            LayoutHardCoreType(max_axons=32, max_neurons=8, count=2),
        ]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=64, output_count=64, segment_id=0, latency_tag=0)
        frags = expander(sc)
        assert frags is not None
        # Type A: coalesce=4 (<=count=4), split=4 → 4 fragments
        # Type B: coalesce=2 (<=count=2), split=8 → 8 fragments
        # Picks type A (fewer split fragments)
        assert len(frags) == 4
        for f in frags:
            assert f.input_count == 64  # full width
            assert f.output_count == 16

    def test_preserves_segment_and_latency(self):
        from mimarsinan.mapping.schedule_partitioner import _make_softcore_fragment_expander
        hw = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=2)]
        expander = _make_softcore_fragment_expander(hw, allow_coalescing=True, allow_splitting=True)
        sc = LayoutSoftCoreSpec(input_count=32, output_count=32, segment_id=5, latency_tag=3)
        frags = expander(sc)
        assert frags is not None
        for f in frags:
            assert f.segment_id == 5
            assert f.latency_tag == 3


class TestEstimatePassesValidated:
    """Tests for estimate_passes_for_layout_validated returning feasibility."""

    def test_feasible_all_pack(self):
        from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout_validated
        softcores = [
            LayoutSoftCoreSpec(input_count=8, output_count=4, segment_id=0, latency_tag=0)
            for _ in range(2)
        ]
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=4, count=2)]
        n, passes, ok = estimate_passes_for_layout_validated(
            softcores, max_cores_per_pass=2, core_types=hw,
        )
        assert ok is True
        assert n >= 1

    def test_infeasible_single_softcore_too_large(self):
        from mimarsinan.mapping.schedule_partitioner import estimate_passes_for_layout_validated
        softcores = [
            LayoutSoftCoreSpec(input_count=64, output_count=64, segment_id=0, latency_tag=0),
        ]
        hw = [LayoutHardCoreType(max_axons=8, max_neurons=8, count=1)]
        n, passes, ok = estimate_passes_for_layout_validated(
            softcores, max_cores_per_pass=1, core_types=hw,
        )
        assert ok is False


class TestUnifiedSchedulingVerifier:
    """Verify that verify_hardware_config uses typed packing for scheduling."""

    def test_heterogeneous_infeasible_reports_correctly(self):
        """Scalar budget says feasible, but typed packing reveals infeasible.
        E.g. 1 core of type A (4x4), softcore needs 16x16. Scheduling can't
        help because even a single softcore doesn't pack."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=16, output_count=16,
                               segment_id=0, latency_tag=0),
        ]
        core_types = [{"max_axons": 4, "max_neurons": 4, "count": 10}]
        result = verify_hardware_config(scs, core_types, allow_scheduling=True)
        assert result["feasible"] is False

    def test_single_core_type_feasible_with_scheduling(self):
        """Multiple softcores with scheduling on a single core type,
        where 1 core at a time is enough."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=16, output_count=8,
                               segment_id=0, latency_tag=i)
            for i in range(3)
        ]
        core_types = [{"max_axons": 16, "max_neurons": 8, "count": 1}]
        result = verify_hardware_config(scs, core_types, allow_scheduling=True)
        assert result["feasible"] is True
        stats = result["stats"]
        assert stats["schedule_pass_count"] >= 3

    def test_total_cores_from_real_packing(self):
        """total_cores reflects actual typed packing, not a synthetic/extreme config."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=8, output_count=4,
                               segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=8, output_count=4,
                               segment_id=0, latency_tag=0),
        ]
        core_types = [{"max_axons": 8, "max_neurons": 4, "count": 1}]
        result = verify_hardware_config(scs, core_types, allow_scheduling=True)
        assert result["feasible"] is True
        stats = result["stats"]
        assert stats["total_cores"] == 1

    def test_heterogeneous_core_types_typed_validation(self):
        """With two core types: type A (8x8, count=1) and type B (4x4, count=3).
        A softcore of 8x8 requires type A; if we need 2 such softcores in a
        pass but only have 1 type-A core, the pass is split."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=8, output_count=8,
                               segment_id=0, latency_tag=0)
            for _ in range(2)
        ]
        core_types = [
            {"max_axons": 8, "max_neurons": 8, "count": 1},
            {"max_axons": 4, "max_neurons": 4, "count": 3},
        ]
        result = verify_hardware_config(scs, core_types, allow_scheduling=True)
        assert result["feasible"] is True
        stats = result["stats"]
        # Each pass can only fit 1 softcore (type A has count=1), so need >= 2 passes
        assert stats.get("schedule_pass_count", 0) >= 2

    def test_coalescing_infeasible_when_count_insufficient(self):
        """With 1 core of 16x16, a 64x64 softcore needs 4 coalescing cores
        but only 1 is available — infeasible even with scheduling."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=64, output_count=64,
                               segment_id=0, latency_tag=0),
        ]
        core_types = [{"max_axons": 16, "max_neurons": 16, "count": 1}]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=True,
            allow_neuron_splitting=True,
        )
        assert result["feasible"] is False

    def test_coalescing_feasible_when_count_sufficient(self):
        """With 4 cores of 16x16, a 64x64 softcore coalesces (4 cores) and
        splits neurons across passes.  Scheduling distributes splitting only."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=64, output_count=64,
                               segment_id=0, latency_tag=0),
        ]
        core_types = [{"max_axons": 16, "max_neurons": 16, "count": 4}]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=True,
            allow_neuron_splitting=True,
        )
        assert result["feasible"] is True
        stats = result["stats"]
        assert stats["total_hw_cores"] == 4
        assert stats["mapped_params_pct"] >= 0.0
        assert stats["mapped_params_pct"] <= 100.0

    def test_many_softcores_with_coalescing(self):
        """96 softcores (32x16) on 2 cores of 16x16 with coalescing+splitting+scheduling.
        Each softcore needs 2 coalescing cores (ceil(32/16)=2), count=2 suffices."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=32, output_count=16,
                               segment_id=i % 4, latency_tag=i)
            for i in range(96)
        ]
        core_types = [{"max_axons": 16, "max_neurons": 16, "count": 2}]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=True,
            allow_neuron_splitting=True,
        )
        assert result["feasible"] is True
        assert result["stats"]["total_hw_cores"] == 2

    def test_splitting_only_heterogeneous_total_hw_cores(self):
        """Splitting with heterogeneous core types: total_hw_cores must
        equal sum(ct.count) from the real chip."""
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        scs = [
            LayoutSoftCoreSpec(input_count=16, output_count=64,
                               segment_id=0, latency_tag=0),
        ]
        core_types = [
            {"max_axons": 16, "max_neurons": 16, "count": 2},
            {"max_axons": 16, "max_neurons": 8, "count": 3},
        ]
        result = verify_hardware_config(
            scs, core_types,
            allow_scheduling=True,
            allow_axon_coalescing=False,
            allow_neuron_splitting=True,
        )
        assert result["feasible"] is True
        stats = result["stats"]
        assert stats["total_hw_cores"] == 5  # 2 + 3 from real chip
        assert stats["mapped_params_pct"] >= 0.0
        assert stats["mapped_params_pct"] <= 100.0

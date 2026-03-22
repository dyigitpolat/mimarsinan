"""Tests for schedule_partitioner: partitioning neural cores into schedule passes."""

import pytest
import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.schedule_partitioner import (
    partition_segment_into_passes,
    estimate_passes_for_layout,
    _build_atomic_units,
    _bin_pack_units_costed,
    _hw_cost_layout,
)


def _make_core(id, input_from=None, n_in=4, n_out=2, coalescing_group_id=None):
    """Create a minimal NeuralCore for testing."""
    if input_from is None:
        sources = np.array(
            [IRSource(-2, i) for i in range(n_in)] + [IRSource(-3, 0)],
            dtype=object,
        )
    else:
        sources = np.array(
            [IRSource(input_from, i) for i in range(n_in)] + [IRSource(-3, 0)],
            dtype=object,
        )
    w = np.ones((n_in + 1, n_out), dtype=np.float32) * 0.1
    core = NeuralCore(id=id, name=f"c{id}", input_sources=sources, core_matrix=w)
    if coalescing_group_id is not None:
        core.coalescing_group_id = coalescing_group_id
        core.coalescing_role = "partial_pos"
    return core


class TestPartitionSegmentIntoPasses:
    def test_all_fit_single_pass(self):
        """When all cores fit, should produce exactly one pass."""
        cores = [_make_core(0), _make_core(1), _make_core(2)]
        passes = partition_segment_into_passes(cores, max_cores_per_pass=10)
        assert len(passes) == 1
        assert len(passes[0]) == 3

    def test_split_into_two_passes_by_latency(self):
        """Cores at different latencies should split when they exceed budget."""
        c0 = _make_core(0)
        c1 = _make_core(1)
        c2 = _make_core(2)
        c3 = _make_core(3, input_from=0, n_in=2)
        c4 = _make_core(4, input_from=1, n_in=2)
        c5 = _make_core(5, input_from=2, n_in=2)

        passes = partition_segment_into_passes(
            [c0, c1, c2, c3, c4, c5], max_cores_per_pass=3,
        )
        assert len(passes) == 2
        pass0_ids = {c.id for c in passes[0]}
        assert pass0_ids == {0, 1, 2}
        pass1_ids = {c.id for c in passes[1]}
        assert pass1_ids == {3, 4, 5}

    def test_single_latency_group_split(self):
        """An oversized single latency group should be split into sub-passes."""
        cores = [_make_core(i) for i in range(6)]
        passes = partition_segment_into_passes(cores, max_cores_per_pass=2)
        assert len(passes) == 3
        for p in passes:
            assert len(p) <= 2

    def test_empty_cores(self):
        passes = partition_segment_into_passes([], max_cores_per_pass=10)
        assert passes == []

    def test_invalid_max_cores(self):
        with pytest.raises(ValueError):
            partition_segment_into_passes([_make_core(0)], max_cores_per_pass=0)

    def test_coalescing_group_can_split_across_passes(self):
        """Coalescing groups CAN be split across passes (state buffer handles it)."""
        c0 = _make_core(0, coalescing_group_id=42)
        c1 = _make_core(1, coalescing_group_id=42)
        c2 = _make_core(2)
        c3 = _make_core(3)

        passes = partition_segment_into_passes([c0, c1, c2, c3], max_cores_per_pass=2)
        assert len(passes) == 2
        total = sum(len(p) for p in passes)
        assert total == 4

    def test_large_coalescing_group_distributes_across_passes(self):
        """A coalescing group larger than budget is distributed across passes."""
        cores = [_make_core(i, coalescing_group_id=1) for i in range(5)]
        passes = partition_segment_into_passes(cores, max_cores_per_pass=2)
        assert len(passes) == 3  # ceil(5/2) = 3
        total = sum(len(p) for p in passes)
        assert total == 5

    def test_mixed_latencies_greedy_fill(self):
        """Latency groups should greedily fill passes up to the budget."""
        c0 = _make_core(0)
        c1 = _make_core(1)
        c2 = _make_core(2, input_from=0, n_in=2)
        c3 = _make_core(3, input_from=2, n_in=2)
        c4 = _make_core(4, input_from=2, n_in=2)

        passes = partition_segment_into_passes(
            [c0, c1, c2, c3, c4], max_cores_per_pass=3,
        )
        assert len(passes) == 2
        assert len(passes[0]) == 3
        assert len(passes[1]) == 2

    def test_hw_cost_aware_partitioning(self):
        """With hw dimensions, wide softcores should cost more than 1."""
        # 4 cores, each with 512+1 axons (including always-on) and 10 neurons
        # With max_hw_axons=256 and coalescing, each costs ceil(513/256)=3 hw cores
        # Budget of 4 → only 1 softcore (cost 3) per pass → 4 passes
        cores = [_make_core(i, n_in=512, n_out=10) for i in range(4)]
        passes = partition_segment_into_passes(
            cores, max_cores_per_pass=4,
            max_hw_axons=256, max_hw_neurons=256,
            allow_coalescing=True,
        )
        # Each core costs 3 hw cores, budget is 4, so 1 core per pass
        assert len(passes) == 4
        # Without hw cost info, would be 1 pass (4 softcores ≤ 4 budget)
        passes_naive = partition_segment_into_passes(cores, max_cores_per_pass=4)
        assert len(passes_naive) == 1


class TestBuildAtomicUnits:
    def test_all_standalone(self):
        cores = [_make_core(0), _make_core(1), _make_core(2)]
        units = _build_atomic_units(cores)
        assert len(units) == 3
        for u in units:
            assert len(u) == 1

    def test_coalescing_group(self):
        c0 = _make_core(0, coalescing_group_id=1)
        c1 = _make_core(1, coalescing_group_id=1)
        c2 = _make_core(2)
        units = _build_atomic_units([c0, c1, c2])
        sizes = sorted(len(u) for u in units)
        assert sizes == [1, 2]


class TestBinPackUnitsCosted:
    def test_simple_packing(self):
        c0 = _make_core(0)
        c1 = _make_core(1)
        c2 = _make_core(2)
        units = [[c0], [c1], [c2]]
        passes = _bin_pack_units_costed(units, max_per_pass=2, cost_fn=lambda c: 1)
        assert len(passes) == 2
        total = sum(len(p) for p in passes)
        assert total == 3

    def test_oversized_unit_gets_own_pass(self):
        """A single unit exceeding budget gets its own pass (no raise)."""
        cores = [_make_core(i, coalescing_group_id=1) for i in range(3)]
        units = [cores]
        passes = _bin_pack_units_costed(units, max_per_pass=2, cost_fn=lambda c: 1)
        assert len(passes) == 1
        assert len(passes[0]) == 3


class TestHwCostLayout:
    def test_no_expansion(self):
        sc = LayoutSoftCoreSpec(input_count=100, output_count=50)
        assert _hw_cost_layout(sc, 256, 256, allow_coalescing=False, allow_splitting=False) == 1

    def test_coalescing_expansion(self):
        sc = LayoutSoftCoreSpec(input_count=600, output_count=50)
        # ceil(600/256) = 3 axon fragments
        assert _hw_cost_layout(sc, 256, 256, allow_coalescing=True, allow_splitting=False) == 3

    def test_splitting_expansion(self):
        sc = LayoutSoftCoreSpec(input_count=100, output_count=600)
        # ceil(600/256) = 3 neuron fragments
        assert _hw_cost_layout(sc, 256, 256, allow_coalescing=False, allow_splitting=True) == 3

    def test_both_expansion(self):
        sc = LayoutSoftCoreSpec(input_count=600, output_count=600)
        # ceil(600/256) * ceil(600/256) = 3 * 3 = 9
        assert _hw_cost_layout(sc, 256, 256, allow_coalescing=True, allow_splitting=True) == 9


class TestEstimatePassesForLayout:
    def test_single_pass(self):
        softcores = [
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
        ]
        num_passes, pass_lists = estimate_passes_for_layout(softcores, max_cores_per_pass=10)
        assert num_passes == 1
        assert len(pass_lists) == 1
        assert len(pass_lists[0]) == 2

    def test_multiple_passes_by_latency(self):
        softcores = [
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=1),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=1),
        ]
        num_passes, pass_lists = estimate_passes_for_layout(softcores, max_cores_per_pass=2)
        assert num_passes == 2
        assert all(len(p) <= 2 for p in pass_lists)

    def test_multiple_segments(self):
        softcores = [
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=1, latency_tag=2),
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=1, latency_tag=2),
        ]
        num_passes, pass_lists = estimate_passes_for_layout(softcores, max_cores_per_pass=1)
        assert num_passes == 4

    def test_empty(self):
        num_passes, pass_lists = estimate_passes_for_layout([], max_cores_per_pass=10)
        assert num_passes == 0
        assert pass_lists == []

    def test_oversized_latency_group(self):
        softcores = [
            LayoutSoftCoreSpec(input_count=10, output_count=5, segment_id=0, latency_tag=0)
            for _ in range(6)
        ]
        num_passes, pass_lists = estimate_passes_for_layout(softcores, max_cores_per_pass=2)
        assert num_passes == 3
        assert all(len(p) <= 2 for p in pass_lists)

    def test_hw_cost_aware_more_passes_needed(self):
        """With hw dimensions and coalescing, wide softcores need more passes."""
        softcores = [
            LayoutSoftCoreSpec(input_count=600, output_count=50, segment_id=0, latency_tag=0)
            for _ in range(4)
        ]
        # Without hw info: 4 softcores / budget 4 = 1 pass
        num_passes_naive, _ = estimate_passes_for_layout(softcores, max_cores_per_pass=4)
        assert num_passes_naive == 1

        # With hw info: each costs ceil(600/256)=3 hw cores, budget 4
        # Only 1 softcore (cost 3) fits per pass → 4 passes
        num_passes_aware, _ = estimate_passes_for_layout(
            softcores, max_cores_per_pass=4,
            max_hw_axons=256, max_hw_neurons=256,
            allow_coalescing=True,
        )
        assert num_passes_aware == 4

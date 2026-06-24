"""Static placement-capacity estimate: hard-core count from an IR graph WITHOUT placement.

``estimate_cores_needed`` is the E4 capacity diagnostic: a pure, fast, sound
LOWER bound on the number of hard cores the greedy placement engine would need,
computed straight from the IR graph and the platform core budget, so an
infeasible config (e.g. VGG16@224 needing hundreds of thousands of cores on a
1000-core budget) is rejected EARLY with a clear ``CapacityExceededError`` instead
of a late, non-diagnosable ``RuntimeError("No more hard cores available")``.

The estimate mirrors the diagonal packer: a hard core stacks softcores along the
diagonal, consuming both axons and neurons, so a neural segment needs at least
``max(ceil(Σ axons / max_axons), ceil(Σ neurons / max_neurons), max per-core
frags·groups)`` cores. Oversized cores (input > max_axons, output > max_neurons)
contribute their forced coalescing-fragment × neuron-group count.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.capacity import (
    CapacityEstimate,
    CapacityExceededError,
    estimate_cores_needed,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _core(node_id, name, in_count, out_count):
    """A NeuralCore with ``in_count`` input axons and ``out_count`` output neurons."""
    return NeuralCore(
        id=node_id,
        name=name,
        input_sources=_src([(-2, i) for i in range(in_count)]),
        core_matrix=np.ones((in_count, out_count), dtype=np.float64),
        threshold=1.0,
        latency=0,
    )


def _cores(max_axons, max_neurons, count):
    return [{"max_axons": max_axons, "max_neurons": max_neurons, "count": count,
             "has_bias": True}]


class TestHandComputed:
    def test_single_fitting_core_needs_one(self):
        a = _core(0, "A", in_count=4, out_count=4)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 16)})
        assert est.cores_needed == 1
        assert est.cores_available == 16
        assert est.feasible is True
        assert est.overflowing_segment is None

    def test_many_small_cores_copack_by_diagonal_bound(self):
        """10 cores of (2 axons, 3 neurons) on an 8x8 core: the diagonal bound is
        max(ceil(20/8)=3, ceil(30/8)=4, 1) = 4 — co-packing, NOT 10 separate cores."""
        nodes = [_core(i, f"C{i}", in_count=2, out_count=3) for i in range(10)]
        graph = IRGraph(nodes=nodes, output_sources=_src([(9, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 64)})
        assert est.cores_needed == 4
        assert est.feasible is True

    def test_wide_fanin_forces_coalescing_fragments(self):
        """One core of (20 axons, 3 neurons), max_axons=8 → ceil(20/8)=3 fragments;
        diagonal bound = max(ceil(20/8)=3, ceil(3/8)=1, frags·groups=3·1=3) = 3."""
        a = _core(0, "wide", in_count=20, out_count=3)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 64)})
        assert est.cores_needed == 3

    def test_tall_fanout_forces_neuron_groups(self):
        """One core of (3 axons, 20 neurons), max_neurons=8 → ceil(20/8)=3 groups;
        diagonal bound = max(ceil(3/8)=1, ceil(20/8)=3, 1·3=3) = 3."""
        a = _core(0, "tall", in_count=3, out_count=20)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 64)})
        assert est.cores_needed == 3

    def test_oversized_both_dims_frags_times_groups(self):
        """(20 axons, 20 neurons), 8x8 core: frags=3, groups=3 → per-core 9;
        diagonal bound = max(ceil(20/8)=3, ceil(20/8)=3, 9) = 9."""
        a = _core(0, "big", in_count=20, out_count=20)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 64)})
        assert est.cores_needed == 9

    def test_segments_sum_and_report_per_segment(self):
        """Two neural segments split by a ComputeOp barrier; the per-segment bounds
        sum, and ``per_segment`` carries each segment's count keyed by label."""
        # Segment 1: two (2,3) cores → max(ceil(4/8)=1, ceil(6/8)=1, 1) = 1
        a = _core(0, "A", 2, 3)
        b = _core(1, "B", 2, 3)
        barrier = ComputeOp(
            id=2, name="pool", input_sources=_src([(1, 0)]),
            op_type="identity", input_shape=(3,), output_shape=(3,),
        )
        # Segment 2: one (20,3) core → 3 fragments
        c = _core(3, "C", 20, 3)
        c.input_sources = _src([(2, i) for i in range(20)])
        graph = IRGraph(nodes=[a, b, barrier, c], output_sources=_src([(3, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 64)})
        assert est.cores_needed == 1 + 3
        assert len(est.per_segment) == 2
        assert sum(est.per_segment.values()) == est.cores_needed
        # labels follow partition_ir_graph: first ends at the barrier, second is final
        assert any("pool" in label for label in est.per_segment)
        assert "neural_segment_final" in est.per_segment

    def test_effective_max_axons_loses_one_without_hardware_bias(self):
        """When ``has_bias`` is False the bias row steals an axon slot: a core of
        8 axons no longer fits an 8-axon crossbar (effective max_axons = 7)."""
        a = _core(0, "A", in_count=8, out_count=3)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        cores = [{"max_axons": 8, "max_neurons": 8, "count": 64, "has_bias": False}]
        est = estimate_cores_needed(graph, {"cores": cores})
        # effective max_axons = 7 → ceil(8/7) = 2 fragments
        assert est.cores_needed == 2


class TestFeasibilityVerdict:
    def test_infeasible_names_overflowing_segment(self):
        # Segment "small" fits in 1 core; segment "huge" needs many → budget=2.
        a = _core(0, "small", 2, 2)
        barrier = ComputeOp(
            id=1, name="barrier", input_sources=_src([(0, 0)]),
            op_type="identity", input_shape=(2,), output_shape=(2,),
        )
        huge = _core(2, "huge", in_count=80, out_count=80)
        huge.input_sources = _src([(1, i) for i in range(80)])
        graph = IRGraph(nodes=[a, barrier, huge], output_sources=_src([(2, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 2)})
        assert est.feasible is False
        # huge: frags=ceil(80/8)=10, groups=10 → 100 cores; small=1 → 101 total
        assert est.cores_needed == 1 + 100
        assert est.cores_available == 2
        # the "huge" core lives in the trailing segment (no barrier after it);
        # cumulative first exceeds the budget there. Segments are labelled by the
        # barrier that closes them (partition_ir_graph SSOT), so the overflowing
        # segment is the final one, and it carries the 100-core requirement.
        assert est.overflowing_segment == "neural_segment_final"
        assert est.per_segment["neural_segment_final"] == 100

    def test_raise_if_infeasible_includes_counts_and_segment(self):
        huge = _core(0, "huge", in_count=80, out_count=80)
        graph = IRGraph(nodes=[huge], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 2)})
        with pytest.raises(CapacityExceededError) as exc:
            est.raise_if_infeasible()
        msg = str(exc.value)
        assert "100" in msg  # cores_needed
        assert "2" in msg    # cores_available
        assert "neural_segment_final" in msg  # overflowing segment named
        assert exc.value.cores_needed == 100
        assert exc.value.cores_available == 2
        assert exc.value.overflowing_segment == "neural_segment_final"

    def test_raise_if_infeasible_is_noop_when_feasible(self):
        a = _core(0, "A", 2, 2)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 16)})
        est.raise_if_infeasible()  # must not raise

    def test_budget_sums_across_mixed_core_types(self):
        a = _core(0, "A", 2, 2)
        graph = IRGraph(nodes=[a], output_sources=_src([(0, 0)]))
        cores = [
            {"max_axons": 8, "max_neurons": 8, "count": 10, "has_bias": True},
            {"max_axons": 4, "max_neurons": 4, "count": 5, "has_bias": True},
        ]
        est = estimate_cores_needed(graph, {"cores": cores})
        assert est.cores_available == 15


class TestEmptyAndEdgeCases:
    def test_empty_graph_is_feasible_zero(self):
        graph = IRGraph(nodes=[], output_sources=_src([(-1, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 4)})
        assert est.cores_needed == 0
        assert est.feasible is True
        assert est.overflowing_segment is None

    def test_capacity_estimate_is_frozen_dataclass(self):
        graph = IRGraph(nodes=[_core(0, "A", 2, 2)], output_sources=_src([(0, 0)]))
        est = estimate_cores_needed(graph, {"cores": _cores(8, 8, 4)})
        assert isinstance(est, CapacityEstimate)
        with pytest.raises(Exception):
            est.cores_needed = 999

"""Tests for model-level pruning boundary policy."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import (
    PruningBoundaryPolicy,
    assert_unified_ir_for_pruning,
    build_computeop_producer_map,
    build_computeop_referenced_neurons,
    compute_model_io_boundary_policy,
    compute_perceptron_io_exemption_indices,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _two_segment_graph():
    w0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    core0 = NeuralCore(
        id=0,
        name="c0",
        input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=w0.copy(),
        threshold=1.0,
        latency=0,
        perceptron_index=0,
    )
    op = ComputeOp(
        id=1,
        name="op",
        input_sources=_src([(0, 0), (0, 1)]),
        op_type="identity",
    )
    w1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    core1 = NeuralCore(
        id=2,
        name="c1",
        input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=w1.copy(),
        threshold=1.0,
        latency=1,
        perceptron_index=1,
    )
    return IRGraph(
        nodes=[core0, op, core1],
        output_sources=_src([(2, 0), (2, 1)]),
    )


class TestModelIOBoundaryPolicy:
    def test_segment_exit_core_not_output_exempt(self):
        graph = _two_segment_graph()
        policy = compute_model_io_boundary_policy(graph)
        assert policy.exempt_cols_per_node[0] == frozenset()
        assert policy.exempt_cols_per_node[2] == frozenset({0, 1})

    def test_segment_entry_core_not_input_exempt(self):
        graph = _two_segment_graph()
        policy = compute_model_io_boundary_policy(graph)
        assert policy.exempt_rows_per_node[0] == frozenset({0, 1})
        assert policy.exempt_rows_per_node[2] == frozenset()

    def test_computeop_producer_map(self):
        graph = _two_segment_graph()
        producer_map = build_computeop_producer_map(graph)
        assert producer_map[(1, 0)] == (0, 0)
        assert producer_map[(1, 1)] == (0, 1)

    def test_computeop_producer_map_excludes_non_identity_ops(self):
        """A general op's output index has no positional correspondence with its
        inputs; only declared-identity ops may relay deadness 1:1."""
        graph = _two_segment_graph()
        op = next(n for n in graph.nodes if isinstance(n, ComputeOp))
        op.op_type = "MaxPool2d"
        assert build_computeop_producer_map(graph) == {}

    def test_computeop_producer_map_excludes_identity_with_mismatched_width(self):
        """Identity relay requires the flat input count to equal the output count."""
        graph = _two_segment_graph()
        op = next(n for n in graph.nodes if isinstance(n, ComputeOp))
        op.output_shape = (1,)
        assert build_computeop_producer_map(graph) == {}

    def test_computeop_referenced_neurons(self):
        graph = _two_segment_graph()
        refs = build_computeop_referenced_neurons(graph)
        assert refs == frozenset({(0, 0), (0, 1)})

    def test_perceptron_exemption_indices(self):
        graph = _two_segment_graph()
        exempt_in, exempt_out = compute_perceptron_io_exemption_indices(graph)
        assert exempt_in == {0}
        assert exempt_out == {1}

    def test_assert_unified_ir_rejects_segment_subgraph(self):
        graph = _two_segment_graph()
        graph._is_segment_subgraph = True
        with pytest.raises(ValueError, match="unified pre-segmentation"):
            assert_unified_ir_for_pruning(graph)


class TestPruningBoundaryPolicyType:
    def test_frozen_dataclass(self):
        policy = compute_model_io_boundary_policy(_two_segment_graph())
        assert isinstance(policy, PruningBoundaryPolicy)

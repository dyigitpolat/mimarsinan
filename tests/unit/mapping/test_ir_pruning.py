"""Tests for IR graph pruning: zeroed row/column elimination and source rewiring."""

import pytest
import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_pruning import prune_ir_graph


def _make_source_array(specs):
    """Helper: build an np.ndarray of IRSource from a list of (node_id, index) tuples."""
    return np.array([IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object)


class TestPruneIRGraph:
    def test_preserves_nonzero_structure(self):
        """A core with no zeroed rows or cols should be unchanged."""
        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])  # 2 inputs + bias
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape == (3, 2)
        assert len(pruned_core.input_sources.flatten()) == 3

    def test_removes_zero_rows(self):
        """A fully zeroed row (axon) should be removed along with its input source."""
        w = np.array([
            [1.0, 2.0],
            [0.0, 0.0],  # zeroed row
            [3.0, 4.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape[0] == 2, "Should have 2 rows after removing 1 zeroed row"
        assert len(pruned_core.input_sources.flatten()) == 2

    def test_removes_zero_columns(self):
        """A fully zeroed column (neuron) should be removed, and downstream sources
        referencing it should be remapped to off."""
        w = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        # Output references all 3 neurons; neuron 1 is zeroed
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape[1] == 2, "Should have 2 cols after removing 1 zeroed col"

        # The output source that referenced neuron 1 should now be off
        pruned_out = pruned.output_sources.flatten()
        off_count = sum(1 for s in pruned_out if s.is_off())
        assert off_count == 1, "One output source should be remapped to off"

    def test_rewires_downstream_sources(self):
        """When a column is removed, downstream core input sources referencing
        higher indices should be reindexed."""
        # Core 0: 3 neurons, middle one (idx 1) is zeroed
        w0 = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        src0 = _make_source_array([(-2, 0), (-3, 0)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src0, core_matrix=w0, threshold=1.0, latency=0)

        # Core 1: takes input from core 0's neurons
        w1 = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2), (-3, 0)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)

        out_src = _make_source_array([(1, 0), (1, 1)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        
        # Core 1 should have its input source for (0, 1) set to off,
        # and (0, 2) reindexed to (0, 1)
        pruned_core1 = pruned.nodes[1]
        flat_src = pruned_core1.input_sources.flatten()

        # The source that was (0, 1) should be off
        old_mid = flat_src[1]
        assert old_mid.is_off(), f"Source for pruned neuron should be off, got node_id={old_mid.node_id}"
        
        # The source that was (0, 2) should now reference index 1
        old_last = flat_src[2]
        assert old_last.node_id == 0 and old_last.index == 1, \
            f"Source for neuron 2 should be reindexed to 1, got ({old_last.node_id}, {old_last.index})"

        # Core 1 row count: the off-source row still has non-zero weights,
        # so it is NOT removed. All 4 rows remain.
        assert pruned_core1.core_matrix.shape[0] == 4, \
            f"Core 1 should have 4 rows (off-source row still has non-zero weights)"

    def test_graph_validates_after_pruning(self):
        """The pruned graph should pass validation."""
        w = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        errors = pruned.validate()
        assert errors == [], f"Pruned graph should validate cleanly, got errors: {errors}"

    def test_empty_graph_is_noop(self):
        """An empty graph should not crash."""
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        pruned = prune_ir_graph(graph)
        assert len(pruned.nodes) == 0

    def test_sets_pre_pruning_snapshot_for_gui(self):
        """When rows/cols are pruned, nodes get pre_pruning_heatmap and masks for soft-core viz."""
        w = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.pre_pruning_heatmap is not None
        assert isinstance(pruned_core.pre_pruning_heatmap, list)
        assert pruned_core.pruned_row_mask is not None
        assert pruned_core.pruned_col_mask is not None
        assert len(pruned_core.pruned_row_mask) == 3
        assert len(pruned_core.pruned_col_mask) == 3
        assert pruned_core.pruned_row_mask[1] is True
        assert pruned_core.pruned_col_mask[1] is True

    def test_propagative_pruning_expands_pruned_set(self):
        """A row that only feeds pruned columns is pruned by propagation.
        Cols 2,3 are below threshold (tiny values from row 3); row 3 only feeds those cols -> pruned."""
        thresh = 1e-8
        tiny = 1e-10  # below thresh so cols 2,3 count as zero
        w = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, tiny, tiny],
        ], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1e-8, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2), (0, 3)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        pruned = prune_ir_graph(graph, zero_threshold=thresh)
        pruned_core = pruned.nodes[0]
        # zero_cols = {2,3}, zero_rows = {2}. Row 3 only feeds {2,3} -> pruned by propagation.
        assert pruned_core.core_matrix.shape == (2, 2), "Propagation should prune row 2 and 3, cols 2 and 3"
        assert pruned_core.core_matrix[0, 0] == 1.0 and pruned_core.core_matrix[1, 1] == 1.0
        assert len(pruned_core.input_sources.flatten()) == 2

    def test_initial_pruned_per_node_drives_compaction(self):
        """When initial_pruned_per_node is provided, those masks are used and propagated."""
        w = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)
        # Prune row 1 and col 2 by model mask (True = pruned)
        row_mask = [False, True, False]
        col_mask = [False, False, True]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask, col_mask)},
        )
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape == (2, 2)
        assert pruned_core.core_matrix[0, 0] == 1.0 and pruned_core.core_matrix[0, 1] == 2.0
        assert pruned_core.core_matrix[1, 0] == 7.0 and pruned_core.core_matrix[1, 1] == 8.0

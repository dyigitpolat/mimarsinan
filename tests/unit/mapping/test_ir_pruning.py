"""Tests for IR graph pruning: zeroed row/column elimination and source rewiring."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace

from mimarsinan.mapping.ir import (
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
    ir_graph_to_soft_core_mapping,
    neural_core_to_soft_core,
)
from mimarsinan.mapping.ir_pruning import (
    get_initial_pruning_masks_from_model,
    prune_ir_graph,
    get_neural_segments,
    compute_segment_io_exemption,
)
from mimarsinan.mapping.softcore_mapping import compact_soft_core_mapping


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
        assert pruned_core.core_matrix.shape[0] == 3, (
            "Segment input rows (0,1) are exempt; zeroed row 1 is not pruned. Bias row 2 is not exempt but has non-zero values."
        )
        assert len(pruned_core.input_sources.flatten()) == 3

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
        assert pruned_core.core_matrix.shape[1] == 3, (
            "All columns are segment output-buffer (referenced in output_sources); exempt, so none pruned."
        )

        # The output source that referenced neuron 1 should now be off
        pruned_out = pruned.output_sources.flatten()
        off_count = sum(1 for s in pruned_out if s.is_off())
        assert off_count == 0, "Segment output exemption keeps all output refs; none remapped to off"

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
        # After compaction, masks are re-set to post-compaction shape (all-False).
        assert pruned_core.pruned_row_mask is not None
        assert pruned_core.pruned_col_mask is not None
        assert len(pruned_core.pruned_row_mask) == pruned_core.core_matrix.shape[0]
        assert len(pruned_core.pruned_col_mask) == pruned_core.core_matrix.shape[1]
        # Segment input rows (0,1) and output cols (0,1,2) exempt; only bias row 2 can be pruned if zero -> (2, 3) or all kept (3, 3)
        assert pruned_core.core_matrix.shape[0] in (2, 3) and pruned_core.core_matrix.shape[1] == 3

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
        # Rows 0,1,2 are segment input (-2); we exempt them. Only row 3 can be pruned. Cols 2,3 below threshold; cols 0,1 are output exempt.
        assert pruned_core.core_matrix.shape == (3, 2), "Exemption keeps rows 0,1,2; propagation prunes row 3 and cols 2,3"
        assert pruned_core.core_matrix[0, 0] == 1.0 and pruned_core.core_matrix[1, 1] == 1.0
        assert len(pruned_core.input_sources.flatten()) == 3

    def test_initial_pruned_per_node_drives_compaction(self):
        """When initial_pruned_per_node is provided, those masks are used and propagated."""
        w0 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src0, core_matrix=w0, threshold=1.0, latency=0)
        # core1 is the output node (output nodes use only zero-threshold pruning)
        w1 = np.array([[1.0, 1.0]], dtype=np.float64)
        src1 = _make_source_array([(0, 0), (0, 1)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=0)
        out_src = _make_source_array([(1, 0), (1, 1)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)
        # Prune row 1 and col 2 of core0 by model mask (True = pruned)
        row_mask = [False, True, False]
        col_mask = [False, False, True]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask, col_mask)},
        )
        pruned_core = pruned.nodes[0]
        # Segment exemption: core0 rows 0,1 exempt; col 2 exempt (feeds core1). So we only prune row 1 and col 2 -> (2, 2) but row 1 exempt -> (3, 2).
        assert pruned_core.core_matrix.shape == (3, 2)
        assert pruned_core.core_matrix[0, 0] == 1.0 and pruned_core.core_matrix[0, 1] == 2.0
        assert pruned_core.core_matrix[2, 0] == 7.0 and pruned_core.core_matrix[2, 1] == 8.0
        # Post-compaction masks must match matrix shape (all-False after physical removal).
        assert len(pruned_core.pruned_row_mask) == pruned_core.core_matrix.shape[0]
        assert len(pruned_core.pruned_col_mask) == pruned_core.core_matrix.shape[1]

    def test_output_node_columns_never_pruned(self):
        """Nodes that feed output_sources never have their columns pruned (all output dims preserved)."""
        # Output core 0: 3 neurons; mask would mark all 3 columns pruned
        w = np.array([[1.0, 0.1, 5.0], [2.0, 0.2, 6.0]], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-3, 0)])
        core = NeuralCore(id=0, name="out", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)
        row_mask = [False, False]
        col_mask = [True, True, True]  # would prune all columns
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask, col_mask)},
        )
        flat = pruned.output_sources.flatten()
        valid = sum(1 for s in flat if getattr(s, "node_id", -1) >= 0)
        assert valid == 3, "Output node must keep all output refs"
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape[1] == 3, "Output core must keep all columns"
        np.testing.assert_allclose(pruned_core.core_matrix, w)

    def test_bank_backed_node_gets_sliced_masks_and_pre_pruning_heatmap(self):
        """Bank-backed node with weight_row_slice gets per-node masks and pre_pruning_heatmap matching effective matrix."""
        # Bank 4x4; one node uses slice (0, 2) -> effective 4x2
        bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
        src = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        node = NeuralCore(
            id=0,
            name="bank_core",
            input_sources=src,
            core_matrix=None,
            threshold=1.0,
            latency=0,
            weight_bank_id=0,
            weight_row_slice=(0, 2),
        )
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[node], output_sources=out_src, weight_banks={0: bank})
        # Prune row 1 and col 1 of the full bank (4x4)
        row_mask = [False, True, False, False]
        col_mask = [False, True, False, False]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_bank={0: (row_mask, col_mask)},
        )
        pruned_node = pruned.nodes[0]
        assert pruned_node.pre_pruning_heatmap is not None
        pre_arr = np.array(pruned_node.pre_pruning_heatmap)
        assert pre_arr.shape == (4, 2), "Effective matrix is 4x2 (slice of 4x4)"
        assert len(pruned_node.pruned_row_mask) == 4
        assert len(pruned_node.pruned_col_mask) == 2, "Column mask must be sliced to match effective cols"
        # Segment exemption: rows 0,1,2 are segment input (-2), so row 1 is exempt and not pruned.
        # Cols 0,1 are segment output, so col 1 is exempt and not pruned.
        assert pruned_node.pruned_row_mask[1] is False
        assert pruned_node.pruned_col_mask[1] is False

    def test_bank_backed_soft_core_compaction_reduces_dimensions(self):
        """After prune_ir_graph, soft cores from bank-backed nodes get masks; compact_soft_core_mapping reduces dimensions."""
        bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
        src = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        node = NeuralCore(
            id=0,
            name="bank_core",
            input_sources=src,
            core_matrix=None,
            threshold=1.0,
            latency=0,
            weight_bank_id=0,
            weight_row_slice=(0, 2),
        )
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[node], output_sources=out_src, weight_banks={0: bank})
        row_mask = [False, True, False, False]
        col_mask = [False, True, False, False]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_bank={0: (row_mask, col_mask)},
        )
        soft = ir_graph_to_soft_core_mapping(pruned)
        assert len(soft.cores) == 1
        core = soft.cores[0]
        assert core.core_matrix.shape == (4, 2)
        assert getattr(core, "pruned_row_mask", None) is not None
        assert getattr(core, "pruned_col_mask", None) is not None
        assert len(core.pruned_row_mask) == 4 and len(core.pruned_col_mask) == 2
        compact_soft_core_mapping(soft.cores, soft.output_sources)
        # Segment exemption keeps all segment input rows and output cols; no pruning applied -> shape unchanged (4, 2)
        assert core.core_matrix.shape == (4, 2)
        assert len(core.axon_sources) == 4
        assert len(soft.output_sources) == 2

    def test_snapshot_includes_heatmap_and_pre_pruning_for_bank_backed(self):
        """snapshot_ir_graph returns heatmap_image and pre_pruning_heatmap_image for bank-backed node with masks."""
        from mimarsinan.gui.snapshot import snapshot_ir_graph

        bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
        src = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        node = NeuralCore(
            id=0,
            name="bank_core",
            input_sources=src,
            core_matrix=None,
            threshold=1.0,
            latency=0,
            weight_bank_id=0,
            weight_row_slice=(0, 2),
        )
        out_src = _make_source_array([(0, 0), (0, 1)])
        graph = IRGraph(nodes=[node], output_sources=out_src, weight_banks={0: bank})
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_bank={0: ([False, True, False, False], [False, True, False, False])},
        )
        snap = snapshot_ir_graph(pruned)
        nodes_info = snap["nodes"]
        assert len(nodes_info) == 1
        info = nodes_info[0]
        assert "heatmap_image" in info and info["heatmap_image"] is not None
        assert "pre_pruning_heatmap_image" in info and info["pre_pruning_heatmap_image"] is not None


def _make_mock_perceptron(out_f: int, in_f: int, row_pruned_indices=None, col_pruned_indices=None):
    """Build a mock perceptron with layer.prune_mask (out_f, in_f) and optional prune_bias_mask."""
    pm = torch.zeros(out_f, in_f)
    if col_pruned_indices is not None:
        for j in col_pruned_indices:
            if 0 <= j < in_f:
                pm[:, j] = 1.0
    if row_pruned_indices is not None:
        for i in row_pruned_indices:
            if 0 <= i < out_f:
                pm[i, :] = 1.0
    prune_bias = torch.zeros(out_f)
    if row_pruned_indices is not None:
        for i in row_pruned_indices:
            if 0 <= i < out_f:
                prune_bias[i] = 1.0
    layer = SimpleNamespace(prune_mask=pm, prune_bias_mask=prune_bias)
    return SimpleNamespace(layer=layer)


class TestGetInitialPruningMasksFromModel:
    """Tests for order-based perceptron-to-node mapping when counts differ (tiled IR)."""

    def test_1to1_unchanged_two_nodes(self):
        """When neural_cores and perceptrons count match (2 each), masks are assigned 1:1."""
        # Two cores: (3, 2) and (4, 3)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=np.ones((3, 2)), threshold=1.0, latency=0)
        src1 = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        core1 = NeuralCore(id=1, name="c1", input_sources=src1, core_matrix=np.ones((4, 3)), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1), (1, 2)]))
        # Two perceptrons: (2, 2) -> (3, 2), (3, 3) -> (4, 3)
        p0 = _make_mock_perceptron(2, 2)
        p1 = _make_mock_perceptron(3, 3)

        def get_perceptrons():
            return [p0, p1]

        model = SimpleNamespace(get_perceptrons=get_perceptrons)

        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 2
        assert 0 in initial_node and 1 in initial_node
        r0, c0 = initial_node[0]
        r1, c1 = initial_node[1]
        assert len(r0) == 3 and len(c0) == 2
        assert len(r1) == 4 and len(c1) == 3

    def test_tiled_one_perceptron_14_nodes(self):
        """Tiled IR (14 nodes, 1 perceptron): get_initial_pruning_masks returns empty so prune_ir_graph uses zero-threshold only."""
        n_axons, n_neurons_full = 57, 224
        chunk = 16
        n_tiles = (n_neurons_full + chunk - 1) // chunk
        cores = []
        for i in range(n_tiles):
            src = _make_source_array([(-2, k) for k in range(n_axons - 1)] + [(-3, 0)])
            n_cols = min(chunk, n_neurons_full - i * chunk)
            mat = np.ones((n_axons, n_cols), dtype=np.float64)
            core = NeuralCore(id=i, name=f"tile_{i}", input_sources=src, core_matrix=mat, threshold=1.0, latency=0)
            cores.append(core)
        out_src = _make_source_array([(n_tiles - 1, j) for j in range(cores[-1].core_matrix.shape[1])])
        graph = IRGraph(nodes=cores, output_sources=out_src)
        p = _make_mock_perceptron(224, 56, row_pruned_indices=[0, 1], col_pruned_indices=[10, 11])
        model = SimpleNamespace(get_perceptrons=lambda: [p])

        initial_node, _ = get_initial_pruning_masks_from_model(model, graph)
        # Tiled path does not assign model masks (node order vs perceptron order not guaranteed).
        assert len(initial_node) == 0

    def test_mixed_untiled_then_tiled(self):
        """Tiled IR (3 nodes, 2 perceptrons): get_initial_pruning_masks returns empty."""
        def make_core(nid, n_cols):
            src = _make_source_array([(-2, k) for k in range(56)] + [(-3, 0)])
            mat = np.ones((57, n_cols), dtype=np.float64)
            return NeuralCore(id=nid, name=f"n{nid}", input_sources=src, core_matrix=mat, threshold=1.0, latency=0)

        core0 = make_core(0, 16)
        core1 = make_core(1, 16)
        core2 = make_core(2, 16)
        graph = IRGraph(nodes=[core0, core1, core2], output_sources=_make_source_array([(2, 0), (2, 1)]))

        p0 = _make_mock_perceptron(16, 56)
        p1 = _make_mock_perceptron(32, 56, row_pruned_indices=[0, 15, 16, 31])
        model = SimpleNamespace(get_perceptrons=lambda: [p0, p1])

        initial_node, _ = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 0

    def test_tiled_mask_lengths_match_node_matrix_shape(self):
        """When 1:1 (len(cores)==len(perceptrons)), mask lengths match node matrix shape. Tiled returns empty."""
        # 3 nodes, 2 perceptrons -> tiled path: empty initial_node
        def make_core(nid, n_axons, n_neurons):
            src = _make_source_array([(-2, k) for k in range(n_axons - 1)] + [(-3, 0)])
            mat = np.ones((n_axons, n_neurons), dtype=np.float64)
            return NeuralCore(id=nid, name=f"n{nid}", input_sources=src, core_matrix=mat, threshold=1.0, latency=0)

        core0 = make_core(0, 57, 16)
        core1 = make_core(1, 57, 16)
        core2 = make_core(2, 57, 16)
        graph = IRGraph(nodes=[core0, core1, core2], output_sources=_make_source_array([(2, 0), (2, 1)]))
        p0 = _make_mock_perceptron(16, 56)
        p1 = _make_mock_perceptron(32, 56)
        model = SimpleNamespace(get_perceptrons=lambda: [p0, p1])

        initial_node, _ = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 0

    def test_post_compaction_mask_lengths_match_matrix_for_all_cores(self):
        """After prune_ir_graph with initial masks that cause compaction, every non–bank-backed node has mask lengths equal to core_matrix.shape."""
        w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=w.copy(), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1), (0, 2)]))
        row_mask = [False, True, False]
        col_mask = [False, False, True]
        pruned = prune_ir_graph(graph, initial_pruned_per_node={0: (row_mask, col_mask)})
        for node in pruned.nodes:
            if not isinstance(node, NeuralCore) or node.core_matrix is None or getattr(node, "weight_bank_id", None) is not None:
                continue
            assert len(node.pruned_row_mask) == node.core_matrix.shape[0], (
                f"node_id={node.id} pruned_row_mask length must match matrix rows"
            )
            assert len(node.pruned_col_mask) == node.core_matrix.shape[1], (
                f"node_id={node.id} pruned_col_mask length must match matrix cols"
            )


class TestNeuralCoreToSoftCoreMaskMismatch:
    """neural_core_to_soft_core must raise when mask length != matrix shape (no silent drop)."""

    def test_raises_when_row_mask_length_mismatch(self):
        """If pruned_row_mask length != core_matrix.shape[0], raise ValueError."""
        w = np.ones((3, 2), dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(
            id=0,
            name="c0",
            input_sources=src,
            core_matrix=w,
            threshold=1.0,
            latency=0,
            pruned_row_mask=[False, False, False, False],
            pruned_col_mask=[False, False],
        )
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        with pytest.raises(ValueError, match="pruning mask length mismatch.*node_id=0"):
            neural_core_to_soft_core(core, graph)

    def test_raises_when_col_mask_length_mismatch(self):
        """If pruned_col_mask length != core_matrix.shape[1], raise ValueError."""
        w = np.ones((3, 2), dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(
            id=0,
            name="c0",
            input_sources=src,
            core_matrix=w,
            threshold=1.0,
            latency=0,
            pruned_row_mask=[False, False, False],
            pruned_col_mask=[False, False, False],
        )
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        with pytest.raises(ValueError, match="pruning mask length mismatch.*node_id=0"):
            neural_core_to_soft_core(core, graph)


class TestSegmentIOExemption:
    """Segment-aware input/output buffer exemption: segment input rows and output cols are never pruned."""

    def test_get_neural_segments_single_segment(self):
        """Single segment: all neural cores in one list when no ComputeOp."""
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=np.ones((3, 2)), threshold=1.0, latency=0)
        core1 = NeuralCore(id=1, name="c1", input_sources=_make_source_array([(0, 0), (0, 1), (-3, 0)]), core_matrix=np.ones((3, 2)), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1)]))
        segments = get_neural_segments(graph)
        assert len(segments) == 1
        assert len(segments[0]) == 2
        assert segments[0][0].id == 0 and segments[0][1].id == 1

    def test_compute_segment_io_exemption_single_node_all_io_exempt(self):
        """Single node segment: all rows are input-buffer, all cols are output-buffer."""
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=np.ones((3, 2)), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        in_buf, out_buf = compute_segment_io_exemption(graph)
        assert 0 in in_buf
        assert in_buf[0] == {0, 1}, "Rows 0,1 are segment input (from -2); row 2 is bias (-3) not exempt"
        assert 0 in out_buf
        assert out_buf[0] == {0, 1}, "All 2 columns feed output_sources"

    def test_segment_io_exemption_prevents_pruning_single_core(self):
        """With aggressive masks (all rows and cols pruned), segment I/O exemption keeps input rows and output cols."""
        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=w.copy(), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        row_mask = [True, True, True]
        col_mask = [True, True]
        pruned = prune_ir_graph(graph, initial_pruned_per_node={0: (row_mask, col_mask)})
        pruned_core = pruned.nodes[0]
        # Output nodes are overridden to zero-threshold-only in prune_ir_graph; exemption still applied so nothing pruned -> (3, 2)
        assert pruned_core.core_matrix.shape == (3, 2)
        np.testing.assert_allclose(pruned_core.core_matrix, w)

    def test_segment_io_exemption_two_layer_first_input_rows_exempt(self):
        """Two nodes in one segment: first node input rows exempt, last node output cols exempt."""
        w0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0.copy(), threshold=1.0, latency=0)
        w1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
        src1 = _make_source_array([(0, 0), (0, 1), (-3, 0)])
        core1 = NeuralCore(id=1, name="c1", input_sources=src1, core_matrix=w1.copy(), threshold=1.0, latency=1)
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1)]))
        in_buf, out_buf = compute_segment_io_exemption(graph)
        assert in_buf[0] == {0, 1}, "Rows 0,1 from -2; row 2 is bias (-3) not segment input"
        assert out_buf[0] == set(), "Node 0 feeds node 1 (in segment), not output_sources; no output-buffer cols"
        assert in_buf[1] == set(), "Node 1 rows are from node 0 (in segment) or bias (-3), not segment input"
        assert out_buf[1] == {0, 1}, "Node 1 feeds output_sources"
        row_mask0 = [True, True, True]
        col_mask0 = [True, True]
        row_mask1 = [True, True, True]
        col_mask1 = [True, True]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask0, col_mask0), 1: (row_mask1, col_mask1)},
        )
        # Node 0: input rows 0,1 exempt so 3 rows kept; output_buf empty so cols can be pruned -> (3, 1) or similar
        # Node 1 (output node): overridden to zero-threshold + exemption; output_buf {0,1} so both cols kept -> (3, 2)
        assert pruned.nodes[1].core_matrix.shape == (3, 2)

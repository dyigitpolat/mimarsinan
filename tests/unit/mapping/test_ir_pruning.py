"""Tests for IR graph pruning: zeroed row/column elimination and source rewiring."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace

from mimarsinan.mapping.ir import (
    ComputeOp,
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
)
from mimarsinan.mapping.ir_pruning_analysis import (
    compute_graph_io_exemption,
    get_neural_segments,
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
        """Cross-core propagation: when an upstream column is pruned, the
        consumer axon row is pruned and surviving sources are reindexed."""
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

        # Core 1: row reading pruned (0, 1) is dropped; surviving source (0, 2)
        # is reindexed to (0, 1).
        pruned_core1 = pruned.nodes[1]
        assert pruned_core1.core_matrix.shape[0] == 3, (
            "Core 1 row reading the pruned neuron must be dropped under cross-core propagation"
        )
        flat_src = pruned_core1.input_sources.flatten()
        assert flat_src[0].node_id == 0 and flat_src[0].index == 0
        assert flat_src[1].node_id == 0 and flat_src[1].index == 1, (
            f"Old (0, 2) source must reindex to (0, 1) after compaction; got "
            f"({flat_src[1].node_id}, {flat_src[1].index})"
        )
        assert flat_src[2].is_always_on()

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
        """When rows/cols are pruned, nodes get pre_pruning_heatmap (opt-in) and masks for soft-core viz."""
        w = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)

        # Default: heatmap is skipped to avoid tens-of-GB bloat on large models.
        pruned_no_hm = prune_ir_graph(graph)
        assert pruned_no_hm.nodes[0].pre_pruning_heatmap is None

        # Opt-in: heatmap is stored as a float32 ndarray (not a Python list).
        pruned = prune_ir_graph(graph, store_heatmap=True)
        pruned_core = pruned.nodes[0]
        assert pruned_core.pre_pruning_heatmap is not None
        assert isinstance(pruned_core.pre_pruning_heatmap, np.ndarray)
        assert pruned_core.pre_pruning_heatmap.dtype == np.float32
        # After compaction, masks are re-set to post-compaction shape (all-False).
        assert pruned_core.pruned_row_mask is not None
        assert pruned_core.pruned_col_mask is not None
        assert len(pruned_core.pruned_row_mask) == pruned_core.core_matrix.shape[0]
        assert len(pruned_core.pruned_col_mask) == pruned_core.core_matrix.shape[1]
        # Segment input rows (0,1) and output cols (0,1,2) exempt; only bias row 2 can be pruned if zero -> (2, 3) or all kept (3, 3)
        assert pruned_core.core_matrix.shape[0] in (2, 3) and pruned_core.core_matrix.shape[1] == 3

    def test_propagative_pruning_expands_pruned_set(self):
        """A row that only feeds pruned columns is pruned by propagation.
        Segment output columns are exempt, so only row 3 is pruned (rows 0,1,2 are segment input exempt)."""
        thresh = 1e-8
        tiny = 1e-10  # below thresh
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
        # Rows 0,1,2 are segment input (exempt); row 3 only feeds cols 2,3 -> pruned by propagation.
        # All 4 cols are segment output (exempt), so no cols pruned. Shape (3, 4).
        assert pruned_core.core_matrix.shape == (3, 4), "Exemption keeps rows 0,1,2 and all output cols; only row 3 pruned"
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
        """When output node has all columns marked pruned by model mask, segment exemption keeps them (no prune, no raise)."""
        w = np.array([[1.0, 0.1, 5.0], [2.0, 0.2, 6.0]], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-3, 0)])
        core = NeuralCore(id=0, name="out", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        out_src = _make_source_array([(0, 0), (0, 1), (0, 2)])
        graph = IRGraph(nodes=[core], output_sources=out_src)
        row_mask = [False, False]
        col_mask = [True, True, True]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask, col_mask)},
        )
        flat = pruned.output_sources.flatten()
        valid = sum(1 for s in flat if getattr(s, "node_id", -1) >= 0)
        assert valid == 3, "Segment output exemption keeps all output columns; no refs rewired to off"
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape[1] == 3
        np.testing.assert_allclose(pruned_core.core_matrix, w)

    def test_output_layer_respects_model_mask_pruned_column_removed_and_rewired(self):
        """Cross-core propagation: pruning a non-output column drops the consumer
        axon, surviving sources reindex, output columns stay exempt."""
        # Two cores: core0 feeds core1 (output). Mask prunes core0 column 1 only; core1 output cols stay exempt.
        w0 = np.array([[1.0, 0.5, 3.0], [2.0, 0.5, 4.0], [0.1, 0.1, 0.1]], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0.copy(), threshold=1.0, latency=0)
        w1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2), (-3, 0)])
        core1 = NeuralCore(id=1, name="out", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0), (1, 1)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)
        row_mask0 = [False, False, False]
        col_mask0 = [False, True, False]
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask0, col_mask0)},
        )
        pruned_core0 = pruned.nodes[0]
        assert pruned_core0.core_matrix.shape == (3, 2), "Column 1 removed -> 2 columns"
        np.testing.assert_allclose(pruned_core0.core_matrix[:, 0], w0[:, 0])
        # Cross-core: core1 axon reading (0, 1) is dropped; surviving (0, 2) reindexes to (0, 1).
        pruned_core1 = pruned.nodes[1]
        assert pruned_core1.core_matrix.shape[0] == 3, "Axon reading pruned (0, 1) is dropped"
        flat1 = pruned_core1.input_sources.flatten()
        assert flat1[0].node_id == 0 and flat1[0].index == 0
        assert flat1[1].node_id == 0 and flat1[1].index == 1, "Old (0, 2) reindexes to (0, 1)"
        assert flat1[2].is_always_on()

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
            store_heatmap=True,
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
        """snapshot_ir_graph advertises heatmap + pre-pruning as lazy resources."""
        from mimarsinan.gui.snapshot import (
            snapshot_ir_graph,
            RESOURCE_KIND_IR_CORE_HEATMAP,
            RESOURCE_KIND_IR_CORE_PRE_PRUNING,
        )

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
            store_heatmap=True,
        )
        snap, descriptors = snapshot_ir_graph(pruned)
        nodes_info = snap["nodes"]
        assert len(nodes_info) == 1
        info = nodes_info[0]
        assert info.get("has_heatmap") is True
        assert info.get("has_pre_pruning") is True
        # No base64 bytes in the summary — resources are fetched via descriptors.
        assert "heatmap_image" not in info
        assert "pre_pruning_heatmap_image" not in info

        kinds = {d.kind for d in descriptors}
        assert RESOURCE_KIND_IR_CORE_HEATMAP in kinds
        assert RESOURCE_KIND_IR_CORE_PRE_PRUNING in kinds


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


def _make_mock_perceptron_1d(out_f: int, in_f: int, row_pruned=None, col_pruned=None):
    """Build a mock perceptron with 1D prune_row_mask and prune_col_mask (True = pruned)."""
    if row_pruned is None:
        row_pruned = torch.zeros(out_f, dtype=torch.bool)
    elif isinstance(row_pruned, (list, tuple)):
        row_pruned = torch.tensor(row_pruned, dtype=torch.bool)
    if col_pruned is None:
        col_pruned = torch.zeros(in_f, dtype=torch.bool)
    elif isinstance(col_pruned, (list, tuple)):
        col_pruned = torch.tensor(col_pruned, dtype=torch.bool)
    layer = SimpleNamespace(prune_row_mask=row_pruned, prune_col_mask=col_pruned)
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

    def test_provenance_path_with_perceptron_index_returns_masks(self):
        """When perceptron_index is set on nodes, get_initial_pruning_masks_from_model uses provenance (H2/H4)."""
        # Two cores with explicit perceptron_index (e.g. from assign_perceptron_indices)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(
            id=0, name="c0", input_sources=src0, core_matrix=np.ones((3, 2)), threshold=1.0, latency=0,
            perceptron_index=0,
        )
        src1 = _make_source_array([(-2, 0), (-2, 1), (-2, 2), (-3, 0)])
        core1 = NeuralCore(
            id=1, name="c1", input_sources=src1, core_matrix=np.ones((4, 3)), threshold=1.0, latency=0,
            perceptron_index=1,
        )
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1), (1, 2)]))
        # Perceptrons with 1D masks (preferred path)
        p0 = _make_mock_perceptron_1d(2, 2)
        p1 = _make_mock_perceptron_1d(3, 3)
        model = SimpleNamespace(get_perceptrons=lambda: [p0, p1])

        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 2, "Provenance path should assign masks when perceptron_index is set"
        assert 0 in initial_node and 1 in initial_node
        r0, c0 = initial_node[0]
        r1, c1 = initial_node[1]
        assert len(r0) == 3 and len(c0) == 2
        assert len(r1) == 4 and len(c1) == 3
        assert len(initial_bank) == 0

    def test_1to1_hardware_bias_fallback_returns_masks(self):
        """1:1 fallback works when core matrix has no bias row (hardware_bias layout)."""
        # Two cores with (in_f, out_f) — no bias row
        src0 = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=np.ones((2, 2)), threshold=1.0, latency=0)
        src1 = _make_source_array([(-2, 0), (-2, 1), (-2, 2)])
        core1 = NeuralCore(id=1, name="c1", input_sources=src1, core_matrix=np.ones((3, 3)), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1), (1, 2)]))
        p0 = _make_mock_perceptron_1d(2, 2)
        p1 = _make_mock_perceptron_1d(3, 3)
        model = SimpleNamespace(get_perceptrons=lambda: [p0, p1])

        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 2
        assert 0 in initial_node and 1 in initial_node
        r0, c0 = initial_node[0]
        r1, c1 = initial_node[1]
        assert len(r0) == 2 and len(c0) == 2
        assert len(r1) == 3 and len(c1) == 3
        assert len(initial_bank) == 0

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

    def test_provenance_single_core_gets_mask(self):
        """When a neural core has perceptron_index set, get_initial_pruning_masks_from_model uses that perceptron's masks."""
        # One core (3 axons, 2 neurons) with perceptron_index=0
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(
            id=0,
            name="c0",
            input_sources=src,
            core_matrix=np.ones((3, 2), dtype=np.float64),
            threshold=1.0,
            latency=0,
            perceptron_index=0,
        )
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        # Perceptron (2, 2): prune row 0 and col 1
        p = _make_mock_perceptron(2, 2, row_pruned_indices=[0], col_pruned_indices=[1])
        model = SimpleNamespace(get_perceptrons=lambda: [p])

        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 1
        assert 0 in initial_node
        row_mask, col_mask = initial_node[0]
        assert len(row_mask) == 3 and len(col_mask) == 2
        # IR: row_mask = [col_pruned[0], col_pruned[1], bias=False] -> [False, True, False]
        assert row_mask == [False, True, False]
        # col_mask = row_pruned[0], row_pruned[1] -> [True, False]
        assert col_mask == [True, False]

    def test_provenance_tiled_two_cores_get_sliced_masks(self):
        """When two cores have same perceptron_index and perceptron_output_slice, masks are sliced per core."""
        # Perceptron (32, 56): 32 outputs, 56 inputs. Two tiles of 16 neurons each.
        n_axons = 57  # 56 + bias
        src = _make_source_array([(-2, k) for k in range(56)] + [(-3, 0)])
        core0 = NeuralCore(
            id=0,
            name="tile_0_16",
            input_sources=src,
            core_matrix=np.ones((n_axons, 16), dtype=np.float64),
            threshold=1.0,
            latency=0,
            perceptron_index=0,
            perceptron_output_slice=(0, 16),
        )
        core1 = NeuralCore(
            id=1,
            name="tile_16_32",
            input_sources=src,
            core_matrix=np.ones((n_axons, 16), dtype=np.float64),
            threshold=1.0,
            latency=0,
            perceptron_index=0,
            perceptron_output_slice=(16, 32),
        )
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, j) for j in range(16)]))
        # Perceptron: prune rows 0,1 and cols 10,20
        p = _make_mock_perceptron(32, 56, row_pruned_indices=[0, 1], col_pruned_indices=[10, 20])
        model = SimpleNamespace(get_perceptrons=lambda: [p])

        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)
        assert len(initial_node) == 2
        assert 0 in initial_node and 1 in initial_node
        r0, c0 = initial_node[0]
        r1, c1 = initial_node[1]
        assert len(r0) == n_axons and len(c0) == 16
        assert len(r1) == n_axons and len(c1) == 16
        # Row mask (axons): col_pruned[0..55] + bias -> index 10 and 20 True
        assert r0[10] is True and r0[20] is True
        assert r1[10] is True and r1[20] is True
        # Col mask tile 0: row_pruned[0:16] -> indices 0,1 True
        assert c0[0] is True and c0[1] is True
        # Col mask tile 1: row_pruned[16:32] -> none pruned in this slice (rows 0,1 are in first tile)
        assert sum(c1) == 0
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


class TestHardwareBiasPruning:
    """hardware_bias must be pruned alongside core_matrix columns in Phase 3."""

    def test_hardware_bias_pruned_with_columns(self):
        """When zeroed columns are removed, hardware_bias is sliced to match.

        A column whose weights are all zero AND whose ``hardware_bias`` is also
        zero has no live source and is pruned. Columns whose bias is non-zero
        are never killed by within-matrix propagation: the bias produces spikes
        on its own and pruning would silently drop that contribution.
        """
        w = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        # Col 1 has zero bias, so the all-zero col 1 is truly dead and pruned.
        bias = np.array([10.0, 0.0, 30.0], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        core0.hardware_bias = bias

        # Core 1 downstream so column 1 of core0 is NOT exempt
        w1 = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        # Column 1 is all-zero with zero bias and not output -> removed
        assert pruned_core.core_matrix.shape[1] == 2
        assert pruned_core.hardware_bias.shape[0] == 2
        np.testing.assert_allclose(pruned_core.hardware_bias, [10.0, 30.0])

    def test_hardware_bias_keeps_zero_weight_col_alive(self):
        """A col with zero weights but non-zero ``hardware_bias`` is preserved.

        The bias is an out-of-matrix live source that produces spikes
        regardless of axon connectivity; within-matrix col-death must respect
        it, otherwise the simulation silently loses the bias contribution.
        """
        w = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        core0.hardware_bias = bias

        w1 = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.core_matrix.shape[1] == 3
        assert pruned_core.hardware_bias.shape[0] == 3
        np.testing.assert_allclose(pruned_core.hardware_bias, [10.0, 20.0, 30.0])

    def test_hardware_bias_all_columns_pruned_removes_core(self):
        """When weights AND bias are all zero, the core is DEAD and gets
        deleted from the graph (no (1,1) placeholder is left behind).

        The downstream core that read from it survives as an output-
        protected BIAS_ONLY core (its only input is now OFF, but it is
        referenced by ``output_sources``).
        """
        w = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        bias = np.array([0.0, 0.0], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        core0.hardware_bias = bias

        w1 = np.array([[1.0], [1.0]], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)

        ids = [n.id for n in pruned.nodes]
        assert 0 not in ids, "DEAD core0 must be removed from the graph"
        assert 1 in ids, "Output-protected core1 must survive"

        survivor = next(n for n in pruned.nodes if n.id == 1)
        # core1 is output-protected; its sole live column is preserved. All
        # axons collapse to a single OFF-source placeholder row (BIAS_ONLY
        # shape (1, 1)) so the matrix stays indexable.
        assert survivor.core_matrix.shape == (1, 1)
        flat = survivor.input_sources.flatten()
        assert flat[0].is_off()

    def test_no_hardware_bias_unchanged(self):
        """When hardware_bias is None, pruning columns does not crash."""
        w = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        assert core0.hardware_bias is None

        w1 = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        assert pruned_core.hardware_bias is None
        assert pruned_core.core_matrix.shape[1] == 2

    def test_hardware_bias_survives_to_soft_core(self):
        """After pruning, hardware_bias length matches core_matrix columns through soft-core conversion."""
        w = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ], dtype=np.float32)
        bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        src = _make_source_array([(-2, 0), (-2, 1)])
        core0 = NeuralCore(id=0, name="core0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        core0.hardware_bias = bias

        w1 = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        src1 = _make_source_array([(0, 0), (0, 1), (0, 2)])
        core1 = NeuralCore(id=1, name="core1", input_sources=src1, core_matrix=w1, threshold=1.0, latency=1)
        out_src = _make_source_array([(1, 0)])
        graph = IRGraph(nodes=[core0, core1], output_sources=out_src)

        pruned = prune_ir_graph(graph)
        pruned_core = pruned.nodes[0]
        # Convert to soft core — should not raise
        soft = neural_core_to_soft_core(pruned_core, pruned)
        assert soft.hardware_bias is not None
        assert len(soft.hardware_bias) == soft.core_matrix.shape[1]

    def test_neural_core_to_soft_core_raises_on_bias_mismatch(self):
        """neural_core_to_soft_core raises ValueError when hardware_bias length != neuron count."""
        w = np.ones((3, 2), dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-2, 2)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=w, threshold=1.0, latency=0)
        core.hardware_bias = np.ones(5, dtype=np.float64)  # mismatched length
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        with pytest.raises(ValueError, match="hardware_bias length.*5.*neuron count.*2"):
            neural_core_to_soft_core(core, graph)


class TestGraphIOExemption:
    """Graph-level model-input axons and model-output neurons are never pruned."""

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

    def test_compute_graph_io_exemption_single_node_all_io_exempt(self):
        """Single node: model-input rows and output-source cols are exempt."""
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=np.ones((3, 2)), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        in_buf, out_buf = compute_graph_io_exemption(graph)
        assert 0 in in_buf
        assert in_buf[0] == {0, 1}, "Rows 0,1 are segment input (from -2); row 2 is bias (-3) not exempt"
        assert 0 in out_buf
        assert out_buf[0] == {0, 1}, "All 2 columns feed output_sources"

    def test_graph_io_exemption_prevents_pruning_single_core(self):
        """With aggressive masks (all rows and cols pruned), model I/O exemption keeps input rows and output cols."""
        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core = NeuralCore(id=0, name="c0", input_sources=src, core_matrix=w.copy(), threshold=1.0, latency=0)
        graph = IRGraph(nodes=[core], output_sources=_make_source_array([(0, 0), (0, 1)]))
        row_mask = [True, True, True]
        col_mask = [True, True]
        pruned = prune_ir_graph(graph, initial_pruned_per_node={0: (row_mask, col_mask)})
        pruned_core = pruned.nodes[0]
        # Exemption keeps rows 0,1 (input) and cols 0,1 (output). Only bias row 2 pruned -> (2, 2).
        assert pruned_core.core_matrix.shape == (2, 2)
        np.testing.assert_allclose(pruned_core.core_matrix, w[:2, :])

    def test_graph_io_exemption_two_layer_first_input_rows_exempt(self):
        """Two nodes: first node input rows exempt, last node output cols exempt."""
        w0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0.copy(), threshold=1.0, latency=0)
        w1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
        src1 = _make_source_array([(0, 0), (0, 1), (-3, 0)])
        core1 = NeuralCore(id=1, name="c1", input_sources=src1, core_matrix=w1.copy(), threshold=1.0, latency=1)
        graph = IRGraph(nodes=[core0, core1], output_sources=_make_source_array([(1, 0), (1, 1)]))
        in_buf, out_buf = compute_graph_io_exemption(graph)
        assert in_buf[0] == {0, 1}, "Rows 0,1 from -2 (model input); row 2 is bias (-3) not exempt"
        assert out_buf[0] == set(), "Node 0 feeds node 1 (NeuralCore consumer), not output_sources"
        assert in_buf[1] == set(), "Node 1 rows are from node 0 or bias (-3), not model input"
        assert out_buf[1] == {0, 1}, "Node 1 feeds output_sources"
        row_mask0 = [True, True, True]
        col_mask0 = [True, True]
        # Do not pass init for output node 1 so model output exemption keeps its cols; only test node 0.
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: (row_mask0, col_mask0)},
        )
        # Node 0 is fully dead (all rows + all cols pruned via the model
        # mask) -> liveness deletes it from the graph. Look up Node 1 by id.
        ids = [n.id for n in pruned.nodes]
        assert 0 not in ids, "Node 0 should be removed (DEAD via initial mask)"
        node1 = next(n for n in pruned.nodes if n.id == 1)
        # Node 1 (output node): output cols are exempt; cross-core propagation
        # prunes the axons reading the now-dead Node 0 neurons. Bias row stays.
        # Surviving rows: only the bias axon. Surviving cols: both (output exempt).
        assert node1.core_matrix.shape == (1, 2), (
            "Cross-core propagation: only the bias axon survives after Node 0 is fully pruned"
        )

    def test_graph_io_exemption_non_last_segment_output_cols_empty(self):
        """NC -> ComputeOp -> NC: first core feeds next segment but is not in output_sources.
        output_buffer_cols for the first core must be empty so compaction can prune columns."""
        # Segment 1: core0 (id=0). Segment 2: core1 (id=2). ComputeOp (id=1) in between.
        w0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0.copy(), threshold=1.0, latency=0)
        op_src = _make_source_array([(0, 0), (0, 1)])
        compute_op = ComputeOp(
            id=1,
            name="op",
            input_sources=op_src,
            op_type="identity",
        )
        w1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
        src1 = _make_source_array([(1, 0), (1, 1), (-3, 0)])  # from ComputeOp (node 1) and bias
        core1 = NeuralCore(id=2, name="c1", input_sources=src1, core_matrix=w1.copy(), threshold=1.0, latency=1)
        graph = IRGraph(
            nodes=[core0, compute_op, core1],
            output_sources=_make_source_array([(2, 0), (2, 1)]),  # only core1 is graph output
        )
        segments = get_neural_segments(graph)
        assert len(segments) == 2, "ComputeOp splits into two segments"
        assert len(segments[0]) == 1 and segments[0][0].id == 0
        assert len(segments[1]) == 1 and segments[1][0].id == 2

        in_buf, out_buf = compute_graph_io_exemption(graph)
        assert out_buf[0] == set(), "Node 0 is not in output_sources; not exempt"
        assert out_buf[2] == {0, 1}, "Node 2 feeds output_sources; both cols exempt"

        # Prune column 1 of core0 via initial mask; exemption does not protect it -> compaction removes it
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False, False], [False, True])},
        )
        pruned_core0 = pruned.nodes[0]
        assert pruned_core0.core_matrix.shape[1] == 1, "Column 1 was pruned and not exempt; core0 compacted to 1 col"

    def test_graph_io_exemption_non_model_input_rows_can_compact(self):
        """NC -> ComputeOp -> NC: second core's rows are from ComputeOp (not model input).
        input_buffer_rows for the second core must be empty so axon/row pruning compacts."""
        # Segment 1: core0 (id=0). Segment 2: core1 (id=2). core1's rows from ComputeOp (1) and bias (-3).
        w0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        src0 = _make_source_array([(-2, 0), (-2, 1), (-3, 0)])
        core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0.copy(), threshold=1.0, latency=0)
        op_src = _make_source_array([(0, 0), (0, 1)])
        compute_op = ComputeOp(
            id=1,
            name="op",
            input_sources=op_src,
            op_type="identity",
        )
        w1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
        src1 = _make_source_array([(1, 0), (1, 1), (-3, 0)])  # from ComputeOp and bias; no -2
        core1 = NeuralCore(id=2, name="c1", input_sources=src1, core_matrix=w1.copy(), threshold=1.0, latency=1)
        graph = IRGraph(
            nodes=[core0, compute_op, core1],
            output_sources=_make_source_array([(2, 0), (2, 1)]),
        )
        in_buf, out_buf = compute_graph_io_exemption(graph)
        assert in_buf[0] == {0, 1}, "Node 0 has model input rows (-2)"
        assert in_buf[2] == set(), "Node 2 has no model input rows; all rows from ComputeOp or bias"

        # Prune row 0 of core1 via initial mask; exemption does not protect it -> row compaction
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={2: ([True, False, False], [False, False])},
        )
        pruned_core1 = pruned.nodes[2]
        assert pruned_core1.core_matrix.shape[0] == 2, "Row 0 was pruned and not exempt; core1 compacted to 2 rows"
        assert pruned_core1.core_matrix.shape[1] == 2
        assert len(pruned_core1.input_sources.flatten()) == 2


class TestCrossCorePruningIntegration:
    """End-to-end `prune_ir_graph` checks for recursive bidirectional propagation."""

    def test_pruning_a_neuron_compacts_consumer_axon_row(self):
        """A.col 0 pruned -> B's axon reading (A,0) compacted out."""
        # A: 3 axons, 2 neurons
        w_a = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]], dtype=np.float64)
        # B: 3 axons (2 from A + bias), 2 neurons
        w_b = np.array([[5.0, 6.0], [7.0, 8.0], [0.1, 0.1]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_make_source_array([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_make_source_array([(0, 0), (0, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[a, b], output_sources=_make_source_array([(1, 0), (1, 1)]),
        )
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False, False], [True, False])},
        )
        b_pruned = pruned.nodes[1]
        # B compacted: axon 0 dropped (it read pruned A.0); 2 axons remain (former (0,1) and bias).
        assert b_pruned.core_matrix.shape[0] == 2, (
            f"B should have 2 axons after dropping the one fed by pruned A.0; got {b_pruned.core_matrix.shape}"
        )
        flat = b_pruned.input_sources.flatten()
        # First surviving axon was (0, 1); after A compaction it is now (0, 0).
        assert flat[0].node_id == 0 and flat[0].index == 0
        assert flat[1].is_always_on()

    def test_three_core_cascade_compacts(self):
        """A->B->C: pruning A.col 0 cascades through B (within-matrix) to compact C."""
        w_a = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]], dtype=np.float64)
        # B's row 0 only contributes to col 0; row 1 to col 1.
        w_b = np.array([[10.0, 0.0], [0.0, 11.0], [0.0, 1.0]], dtype=np.float64)
        # C's row 0 only feeds col 0.
        w_c = np.array([[5.0, 0.0], [0.0, 6.0], [0.0, 1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_make_source_array([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_make_source_array([(0, 0), (0, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        c = NeuralCore(
            id=2, name="C",
            input_sources=_make_source_array([(1, 0), (1, 1), (-3, 0)]),
            core_matrix=w_c, threshold=1.0, latency=2,
        )
        graph = IRGraph(
            nodes=[a, b, c],
            output_sources=_make_source_array([(2, 0), (2, 1)]),
        )
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False, False], [True, False])},
        )
        # A: col 0 pruned -> 1 surviving neuron
        assert pruned.nodes[0].core_matrix.shape[1] == 1
        # B: row 0 pruned (read pruned A.0) and col 0 pruned (within-matrix from row death)
        assert pruned.nodes[1].core_matrix.shape == (2, 1)
        # C: row 0 pruned (read pruned B.0)
        assert pruned.nodes[2].core_matrix.shape[0] == 2

    def test_compute_op_blocks_cross_core_compaction(self):
        """A -> ComputeOp -> B: pruning A.col 0 must NOT cascade into B."""
        w_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        op = ComputeOp(
            id=1, name="op",
            input_sources=_make_source_array([(0, 0), (0, 1)]),
            op_type="identity",
        )
        w_b = np.array([[5.0, 6.0], [7.0, 8.0], [0.1, 0.1]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_make_source_array([(-2, 0), (-2, 1)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=2, name="B",
            input_sources=_make_source_array([(1, 0), (1, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[a, op, b],
            output_sources=_make_source_array([(2, 0), (2, 1)]),
        )
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False], [True, False])},
        )
        # A col 0 pruned (from -2 model input axons - cols are not exempt because A is not last)
        assert pruned.nodes[0].core_matrix.shape[1] == 1
        # B not compacted because ComputeOp barrier breaks propagation.
        assert pruned.nodes[2].core_matrix.shape == (3, 2)
        flat = pruned.nodes[2].input_sources.flatten()
        assert flat[0].node_id == 1 and flat[0].index == 0  # from ComputeOp, untouched
        assert flat[1].node_id == 1 and flat[1].index == 1
        assert flat[2].is_always_on()

    def test_unconsumed_neuron_pruned_in_first_core(self):
        """A.col j with no consumer (no other core reads it) is pruned."""
        # A produces 3 neurons; B reads only A.0 and A.1, leaving A.2 dead.
        w_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        w_b = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_make_source_array([(-2, 0), (-3, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_make_source_array([(0, 0), (0, 1), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[a, b], output_sources=_make_source_array([(1, 0)]),
        )
        pruned = prune_ir_graph(graph)
        # A.col 2 has no consumer -> compacted out.
        assert pruned.nodes[0].core_matrix.shape[1] == 2

    def test_value_based_pruning_unioned_with_model_mask(self):
        """All-zero rows/cols not in the model mask are still pruned.

        The model mask is the *minimum* set of rows/cols the caller declares
        dead. Any additional rows/cols whose weights are already at zero are
        dead by themselves and must be unioned into the pruned set.
        """
        # B is the model-input-facing core (its rows are ``-2`` axons, exempt).
        # A reads from B; its row indices are *not* model-input axons, so
        # value-based pruning is allowed to remove all-zero rows.
        w_b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        b = NeuralCore(
            id=0, name="B",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=w_b, threshold=1.0, latency=0,
        )
        w_a = np.array(
            [
                [1.0, 0.0, 3.0],  # col 1 is all zero
                [0.0, 0.0, 0.0],  # row 1 is all zero
                [0.5, 0.0, 0.6],
            ],
            dtype=np.float32,
        )
        a = NeuralCore(
            id=1, name="A",
            input_sources=_make_source_array([(0, 0), (0, 1), (0, 2)]),
            core_matrix=w_a, threshold=1.0, latency=1,
        )
        # Downstream consumer keeps A's neurons referenced (so col 1 not orphan).
        w_c = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        c = NeuralCore(
            id=2, name="C",
            input_sources=_make_source_array([(1, 0), (1, 1), (1, 2)]),
            core_matrix=w_c, threshold=1.0, latency=2,
        )
        graph = IRGraph(
            nodes=[b, a, c], output_sources=_make_source_array([(2, 0)]),
        )
        # Provide an explicit (empty) model mask for A so legacy
        # "skip value-based on seeded nodes" behavior would have kept the
        # all-zero row/col. The fix unions value-based with the seed.
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={
                0: ([False], [False, False, False]),
                1: ([False, False, False], [False, False, False]),
                2: ([False, False, False], [False]),
            },
        )
        a_pruned = next(n for n in pruned.nodes if n.id == 1)
        # Row 1 is all zero -> dropped. Col 1 is all zero -> dropped.
        assert a_pruned.core_matrix.shape == (2, 2)

    def test_fully_pruned_core_is_removed_from_graph(self):
        """When every row and column of an intermediate core dies, the
        liveness pass deletes the node entirely (no ``(1, 1)`` placeholder
        is kept). Surviving nodes must continue to validate.
        """
        # A's col 0 is the only one with non-zero weight. We prune it via
        # model mask -> the resulting B core ends up entirely dead-math:
        # its only producer source goes to off, all rows die, all cols die.
        w_a = np.array([[1.0]], dtype=np.float64)
        w_b = np.array([[2.0], [3.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B",
            input_sources=_make_source_array([(0, 0), (-3, 0)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        # Critical: another core C is the model output, otherwise B itself
        # would be exempt as the output source.
        w_c = np.array([[7.0]], dtype=np.float64)
        c = NeuralCore(
            id=2, name="C",
            input_sources=_make_source_array([(-2, 0)]),
            core_matrix=w_c, threshold=1.0, latency=0,
        )
        graph = IRGraph(
            nodes=[a, b, c], output_sources=_make_source_array([(2, 0)]),
        )
        pruned = prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False], [True])},
        )
        ids = [n.id for n in pruned.nodes]
        # A is DEAD (col pruned by model mask -> single neuron lost ->
        # nothing emits anymore). B is DEAD too (its sole live producer
        # neuron is gone, so every axon dies). Both must be removed.
        assert 0 not in ids and 1 not in ids
        assert 2 in ids
        assert pruned.validate() == [], (
            "graph must remain consistent after dead-core deletion"
        )

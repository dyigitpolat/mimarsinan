"""Unit tests for snapshot_pruning_layers_from_ir (the deployed pruning view)."""

import numpy as np

from mimarsinan.gui.snapshot.ir_graph.ir_pruning_layers import (
    snapshot_pruning_layers_from_ir,
)
from mimarsinan.gui.snapshot.util.constants import (
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
    RESOURCE_KIND_PRUNING_MASK_MAP,
)
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.source import IRSource


def _core(node_id, perceptron_index, out_slice, col_mask, row_mask, n_axons, n_neurons,
          psum_role=None):
    return NeuralCore(
        id=node_id,
        name=f"c{node_id}",
        input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(n_axons)]),
        core_matrix=np.zeros((n_axons, n_neurons), dtype=np.float64),
        perceptron_index=perceptron_index,
        perceptron_output_slice=out_slice,
        pre_pruning_col_mask=col_mask,
        pre_pruning_row_mask=row_mask,
        psum_role=psum_role,
    )


def _graph(nodes):
    return IRGraph(
        nodes=nodes,
        output_sources=np.array([IRSource(node_id=nodes[0].id, index=0)]),
    )


def test_deployed_dims_and_sparsity_from_pruned_ir():
    # 6 original neurons, 2 pruned; 5 original axons, 1 pruned.
    col = [False, True, False, False, True, False]
    row = [False, False, True, False, False]
    graph = _graph([_core(0, 0, (0, 6), col, row, n_axons=5, n_neurons=4)])
    summary, descriptors = snapshot_pruning_layers_from_ir(
        graph, configured_fraction=0.2,
    )
    (layer,) = summary["layers"]
    assert summary["source"] == "deployed_ir"
    assert (layer["pre_neurons"], layer["post_neurons"]) == (6, 4)
    assert (layer["pre_axons"], layer["post_axons"]) == (5, 4)
    assert abs(layer["achieved_sparsity_neurons"] - 2 / 6) < 1e-12
    assert summary["configured_fraction"] == 0.2
    kinds = {d.kind for d in descriptors}
    assert RESOURCE_KIND_PRUNING_MASK_MAP in kinds
    assert RESOURCE_KIND_PRUNING_LAYER_HEATMAP in kinds


def test_multi_tile_perceptron_concatenates_neuron_masks():
    graph = _graph([
        _core(0, 0, (0, 3), [False, True, False], None, n_axons=4, n_neurons=2),
        _core(1, 0, (3, 6), [True, False, False], None, n_axons=4, n_neurons=2),
    ])
    summary, _ = snapshot_pruning_layers_from_ir(graph)
    (layer,) = summary["layers"]
    assert (layer["pre_neurons"], layer["post_neurons"]) == (6, 4)
    # multi-tile: no single weight heatmap, mask map still present
    assert layer["has_heatmap"] is False
    assert "mask_map_resource" in layer


def test_unpruned_graph_reports_zero_sparsity():
    graph = _graph([_core(0, 0, (0, 4), None, None, n_axons=3, n_neurons=4)])
    summary, _ = snapshot_pruning_layers_from_ir(graph)
    (layer,) = summary["layers"]
    assert layer["pruned_rows"] == 0 and layer["pruned_cols"] == 0
    assert layer["achieved_sparsity_neurons"] == 0.0


def test_relays_and_psum_pos_are_not_layers():
    graph = _graph([
        _core(0, 0, (0, 4), None, None, n_axons=3, n_neurons=4),
        _core(1, None, None, None, None, n_axons=2, n_neurons=2),          # relay
        _core(2, 0, (0, 4), None, None, n_axons=3, n_neurons=4, psum_role="pos"),
    ])
    summary, _ = snapshot_pruning_layers_from_ir(graph)
    assert len(summary["layers"]) == 1


def test_bank_backed_perceptron_is_skipped_loudly():
    node = _core(0, 0, (0, 4), None, None, n_axons=3, n_neurons=4)
    banked = NeuralCore(
        id=1, name="conv_pos0",
        input_sources=np.array([IRSource(-2, 0)]),
        core_matrix=None,
        perceptron_index=1,
    )
    summary, _ = snapshot_pruning_layers_from_ir(_graph([node, banked]))
    assert len(summary["layers"]) == 1
    (skip,) = summary["skipped"]
    assert skip["layer"] == "conv_pos0" and "bank-backed" in skip["reason"]

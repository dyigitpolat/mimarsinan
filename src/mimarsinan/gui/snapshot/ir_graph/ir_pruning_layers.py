"""Deployed per-perceptron pruning summary from the pruned IR graph (the deployment authority)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mimarsinan.gui.resources import HeatmapSource, ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import HeatmapScaleFamily
from mimarsinan.gui.snapshot.model_snapshot import _mask_map_matrix
from mimarsinan.gui.snapshot.util.constants import (
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
    RESOURCE_KIND_PRUNING_MASK_MAP,
)
from mimarsinan.mapping.ir import NeuralCore


def _perceptron_tiles(ir_graph: Any) -> dict[int, list[Any]]:
    """Accum NeuralCores per perceptron, ordered by output-tile offset."""
    tiles: dict[int, list[Any]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.psum_role not in (None, "accum"):
            continue
        pi = node.perceptron_index
        if pi is None or pi < 0:
            continue
        tiles.setdefault(int(pi), []).append(node)
    for nodes in tiles.values():
        nodes.sort(key=lambda n: (
            (n.perceptron_output_slice or (0, 0))[0], n.id,
        ))
    return tiles


def _tile_masks(node: Any) -> tuple[list[bool], list[bool]]:
    """(axon_mask, neuron_mask) at pre-compaction width; all-kept when unpruned."""
    mat = node.core_matrix
    n_axons = int(mat.shape[0]) if mat is not None else 0
    n_neurons = int(mat.shape[1]) if mat is not None else 0
    row = node.pre_pruning_row_mask
    col = node.pre_pruning_col_mask
    axon_mask = [bool(v) for v in row] if row is not None else [False] * n_axons
    neuron_mask = [bool(v) for v in col] if col is not None else [False] * n_neurons
    return axon_mask, neuron_mask


def snapshot_pruning_layers_from_ir(
    ir_graph: Any, *, configured_fraction: Any = None
) -> tuple[dict, list[ResourceDescriptor]]:
    """Per-perceptron DEPLOYED pruning (pre -> post, sparsity, mask map) from the
    pruned IR. Complements the Pruning Adaptation view: the IR is where structured
    pruning actually commits, so this is what deploys."""
    layers_out: list[dict] = []
    descriptors: list[ResourceDescriptor] = []
    family = HeatmapScaleFamily("pruning_layers")

    for pi, nodes in sorted(_perceptron_tiles(ir_graph).items()):
        axon_masks = [_tile_masks(n)[0] for n in nodes]
        neuron_mask: list[bool] = []
        for n in nodes:
            neuron_mask.extend(_tile_masks(n)[1])
        # Tiles slice the neuron axis; the axon axis is shared (widest tile wins
        # so a multi-tile perceptron never under-reports its fan-in).
        axon_mask = max(axon_masks, key=len) if axon_masks else []

        pre_neurons = len(neuron_mask)
        post_neurons = pre_neurons - sum(neuron_mask)
        pre_axons = len(axon_mask)
        post_axons = pre_axons - sum(axon_mask)
        name = str(nodes[0].name) if nodes else f"perceptron_{pi}"
        rid = f"ir/{pi}"

        entry: dict[str, Any] = {
            "layer_index": int(pi),
            "layer_name": name,
            "shape": [pre_neurons, pre_axons],
            "pre_neurons": pre_neurons,
            "post_neurons": post_neurons,
            "pre_axons": pre_axons,
            "post_axons": post_axons,
            "pruned_rows": int(sum(neuron_mask)),
            "pruned_cols": int(sum(axon_mask)),
            "achieved_sparsity_neurons": (
                float(sum(neuron_mask) / pre_neurons) if pre_neurons else 0.0
            ),
            "achieved_sparsity_axons": (
                float(sum(axon_mask) / pre_axons) if pre_axons else 0.0
            ),
            "has_heatmap": False,
            "mask_map_resource": {
                "kind": RESOURCE_KIND_PRUNING_MASK_MAP,
                "rid": rid,
            },
        }
        descriptors.append(ResourceDescriptor(
            kind=RESOURCE_KIND_PRUNING_MASK_MAP,
            rid=rid,
            source=HeatmapSource(_mask_map_matrix(neuron_mask, axon_mask)),
            media_type="image/png",
        ))
        if len(nodes) == 1 and nodes[0].core_matrix is not None:
            weight = np.asarray(nodes[0].core_matrix, dtype=np.float64).T
            source = HeatmapSource(weight)
            family.adopt(source)
            entry["has_heatmap"] = True
            entry["heatmap_resource"] = {
                "kind": RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
                "rid": rid,
            }
            descriptors.append(ResourceDescriptor(
                kind=RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
                rid=rid,
                source=source,
                media_type="image/png",
            ))
        layers_out.append(entry)

    summary: dict = {
        "layers": layers_out,
        "skipped": [],
        "configured_fraction": (
            None if configured_fraction is None else float(configured_fraction)
        ),
        "source": "deployed_ir",
    }
    heatmap_scale = family.finalize(descriptors)
    if heatmap_scale is not None:
        summary["heatmap_scale"] = heatmap_scale
    return summary, descriptors

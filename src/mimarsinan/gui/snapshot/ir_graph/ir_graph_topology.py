"""IR graph topology snapshot orchestration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import _make_heatmap_producer
from mimarsinan.gui.snapshot.util.helpers import _histogram
from mimarsinan.gui.snapshot.ir_graph.ir_graph_nodes import process_ir_graph_node
from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    RESOURCE_KIND_IR_BANK_HEATMAP,
    make_resource_ref,
)
from mimarsinan.gui.snapshot.ir_graph.ir_graph_topology_groups import (
    _build_layer_groups,
    _deduplicate_edges,
)

logger = logging.getLogger("mimarsinan.gui")


def snapshot_ir_graph(
    ir_graph: Any,
    *,
    source_step_name: str | None = None,
) -> tuple[dict, list[ResourceDescriptor]]:
    """Extract IR topology, core stats, and lazy heatmap resource descriptors."""
    from mimarsinan.mapping.ir import NeuralCore

    nodes_info: list[dict] = []
    edges: list[dict] = []
    descriptors: list[ResourceDescriptor] = []
    register_descriptors = source_step_name is None

    neural_cores: list[dict] = []
    compute_ops: list[dict] = []

    weight_banks: dict = {}
    try:
        for bank_id, bank in getattr(ir_graph, "weight_banks", {}).items():
            try:
                rid = f"bank/{int(bank_id)}"
                weight_banks[int(bank_id)] = {
                    "has_heatmap": True,
                    "heatmap_resource": make_resource_ref(
                        source_step_name, RESOURCE_KIND_IR_BANK_HEATMAP, rid,
                    ),
                }
                if register_descriptors:
                    descriptors.append(ResourceDescriptor(
                        kind=RESOURCE_KIND_IR_BANK_HEATMAP,
                        rid=rid,
                        producer=_make_heatmap_producer(bank.core_matrix, copy=False),
                        media_type="image/png",
                    ))
            except Exception:
                logger.debug(
                    "Failed to register heatmap for weight bank %s", bank_id, exc_info=True,
                )
    except Exception:
        pass

    for topo_idx, node in enumerate(ir_graph.nodes):
        process_ir_graph_node(
            topo_idx=topo_idx,
            node=node,
            ir_graph=ir_graph,
            NeuralCore=NeuralCore,
            register_descriptors=register_descriptors,
            descriptors=descriptors,
            source_step_name=source_step_name,
            neural_cores=neural_cores,
            compute_ops=compute_ops,
            nodes_info=nodes_info,
            edges=edges,
        )

    try:
        for src in ir_graph.output_sources.flatten():
            if src.node_id >= 0:
                edges.append({"from": int(src.node_id), "to": "output"})
    except Exception:
        logger.debug("Failed to extract output_sources from ir_graph", exc_info=True)

    deduped_edges = _deduplicate_edges(edges)

    node_group_map: dict = {n["id"]: n.get("layer_group", n["name"]) for n in nodes_info}
    node_group_map["input"] = "input"
    node_group_map["const1"] = "const1"
    node_group_map["output"] = "output"

    groups, group_edges = _build_layer_groups(nodes_info, deduped_edges, node_group_map)

    thresholds = [c["threshold"] for c in neural_cores]
    latencies = [c["latency"] for c in neural_cores if c["latency"] is not None]
    axon_counts = [c["axons"] for c in neural_cores]
    neuron_counts = [c["neurons"] for c in neural_cores]

    summary = {
        "num_nodes": len(ir_graph.nodes),
        "num_neural_cores": len(neural_cores),
        "num_compute_ops": len(compute_ops),
        "max_latency": max(latencies) if latencies else 0,
        "nodes": nodes_info,
        "edges": deduped_edges,
        "groups": groups,
        "group_edges": group_edges,
        "weight_banks": weight_banks,
        "threshold_distribution": _histogram(np.array(thresholds)) if thresholds else None,
        "latency_distribution": _histogram(np.array(latencies)) if latencies else None,
        "axon_counts": axon_counts,
        "neuron_counts": neuron_counts,
        "compute_op_types": list({c["op_type"] for c in compute_ops}),
    }
    return summary, descriptors

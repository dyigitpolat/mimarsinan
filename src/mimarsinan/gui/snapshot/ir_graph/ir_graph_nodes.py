"""GUI snapshot module."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _t, _histogram
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import (
    _detect_neural_core_liveness,
    _make_bias_strip_producer,
    _make_heatmap_producer,
)
from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    LIVENESS_BIAS_ONLY,
    RESOURCE_KIND_IR_CORE_BIAS,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    make_resource_ref,
)


def process_ir_graph_node(
    *,
    topo_idx: int,
    node,
    ir_graph,
    NeuralCore,
    register_descriptors: bool,
    descriptors,
    source_step_name: str | None,
    neural_cores: list,
    compute_ops: list,
    nodes_info: list,
    edges: list,
) -> None:
    info: dict = {
            "id": node.id,
            "name": node.name,
            "type": "neural_core" if isinstance(node, NeuralCore) else "compute_op",
            "topo_order": topo_idx,
    }

    if isinstance(node, NeuralCore):
            mat = node.get_core_matrix(ir_graph)
            base_group = layer_key_from_node_name(node.name)
            if node.latency is not None:
                group_key = f"{base_group} (L{node.latency})"
            else:
                group_key = base_group

            info["layer_group"] = group_key
            info["axons"] = int(mat.shape[0])
            info["neurons"] = int(mat.shape[1])
            info["threshold"] = float(node.threshold)
            info["activation_scale"] = _t(node.activation_scale)
            info["parameter_scale"] = _t(node.parameter_scale)
            info["input_activation_scale"] = _t(node.input_activation_scale)
            info["latency"] = node.latency
            info["normalization_type"] = getattr(node, "normalization_type", None)
            info["activation_type"] = getattr(node, "activation_type", None)
            info["weight_stats"] = {
                "mean": float(np.mean(mat)),
                "std": float(np.std(mat)),
                "min": float(np.min(mat)),
                "max": float(np.max(mat)),
                "sparsity": float(np.mean(np.abs(mat) < 1e-8)),
                "histogram": _histogram(mat),
            }
            info["psum_group_id"] = node.psum_group_id
            info["psum_role"] = node.psum_role
            info["coalescing_group_id"] = node.coalescing_group_id
            info["coalescing_role"] = node.coalescing_role
            if node.weight_bank_id is not None:
                info["weight_bank_id"] = int(node.weight_bank_id)

            liveness = _detect_neural_core_liveness(node, mat)
            info["liveness"] = liveness

            bias_arr = getattr(node, "hardware_bias", None)
            if bias_arr is not None:
                try:
                    bias_np = np.asarray(bias_arr, dtype=np.float64)
                    if bias_np.size:
                        info["hardware_bias_stats"] = {
                            "size": int(bias_np.size),
                            "min": float(np.min(bias_np)),
                            "max": float(np.max(bias_np)),
                            "abs_max": float(np.max(np.abs(bias_np))),
                            "nonzero": int(np.count_nonzero(bias_np)),
                        }
                except Exception:
                    pass

            core_rid = f"core/{int(node.id)}"
            info["has_heatmap"] = True
            info["heatmap_resource"] = make_resource_ref(source_step_name, RESOURCE_KIND_IR_CORE_HEATMAP, core_rid)
            if register_descriptors:
                descriptors.append(ResourceDescriptor(
                    kind=RESOURCE_KIND_IR_CORE_HEATMAP,
                    rid=core_rid,
                    producer=_make_heatmap_producer(mat, copy=False),
                    media_type="image/png",
                ))

            if (
                liveness == LIVENESS_BIAS_ONLY
                and bias_arr is not None
                and np.asarray(bias_arr).size
            ):
                info["has_bias_resource"] = True
                info["bias_resource"] = make_resource_ref(source_step_name, RESOURCE_KIND_IR_CORE_BIAS, core_rid)
                if register_descriptors:
                    descriptors.append(ResourceDescriptor(
                        kind=RESOURCE_KIND_IR_CORE_BIAS,
                        rid=core_rid,
                        producer=_make_bias_strip_producer(bias_arr),
                        media_type="image/png",
                    ))

            pre = getattr(node, "pre_pruning_heatmap", None)
            row_mask = getattr(node, "pre_pruning_row_mask", None) or getattr(node, "pruned_row_mask", None)
            col_mask = getattr(node, "pre_pruning_col_mask", None) or getattr(node, "pruned_col_mask", None)
            if pre is not None and row_mask is not None and col_mask is not None:
                try:
                    pre_arr = np.array(pre, dtype=np.float64)
                    if pre_arr.shape[0] == len(row_mask) and pre_arr.shape[1] == len(col_mask):
                        info["has_pre_pruning"] = True
                        info["pre_pruning_resource"] = make_resource_ref(source_step_name, 
                            RESOURCE_KIND_IR_CORE_PRE_PRUNING, core_rid,
                        )
                        info["pre_pruning_axons"] = int(pre_arr.shape[0])
                        info["pre_pruning_neurons"] = int(pre_arr.shape[1])
                        if register_descriptors:
                            descriptors.append(ResourceDescriptor(
                                kind=RESOURCE_KIND_IR_CORE_PRE_PRUNING,
                                rid=core_rid,
                                producer=_make_heatmap_producer(
                                    pre_arr,
                                    pruned_row_mask=list(row_mask),
                                    pruned_col_mask=list(col_mask),
                                    copy=False,
                                ),
                                media_type="image/png",
                            ))
                except Exception:
                    logger.debug("Failed to register pre-pruning heatmap for core %s", node.id, exc_info=True)
            neural_cores.append(info)
    else:
            info["layer_group"] = node.name
            info["op_type"] = node.op_type
            info["input_shape"] = list(node.input_shape) if node.input_shape else None
            info["output_shape"] = list(node.output_shape) if node.output_shape else None
            try:
                info["latency"] = node.latency
            except Exception:
                pass
            try:
                params = node.params
                if params is not None:
                    info["params"] = str(params)[:200]
            except Exception:
                pass
            compute_ops.append(info)

    nodes_info.append(info)

    for src in node.input_sources.flatten():
            if src.node_id >= 0:
                edges.append({"from": int(src.node_id), "to": int(node.id)})
            elif src.node_id == -2:
                edges.append({"from": "input", "to": int(node.id)})
            elif src.node_id == -3:
                edges.append({"from": "const1", "to": int(node.id)})

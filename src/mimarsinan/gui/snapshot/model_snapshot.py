"""GUI snapshot module."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch.nn as nn

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _histogram, _safe_scalar
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import _make_heatmap_producer

RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"

LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"

def snapshot_model(model: Any) -> dict:
    """Extract per-layer weight/bias statistics and architecture info."""
    layers: list[dict] = []
    total_params = 0

    perceptrons = _get_model_perceptrons(model)

    for idx, p in enumerate(perceptrons):
        layer_info: dict = {"index": idx, "name": getattr(p, "name", f"perceptron_{idx}")}
        try:
            w = p.layer.weight.data.detach().cpu().numpy()
            layer_info["weight"] = {
                "shape": list(w.shape),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "histogram": _histogram(w),
                "sparsity": float(np.mean(np.abs(w) < 1e-8)),
            }
            total_params += w.size
        except Exception:
            layer_info["weight"] = None

        try:
            b = p.layer.bias.data.detach().cpu().numpy()
            layer_info["bias"] = {
                "shape": list(b.shape),
                "mean": float(np.mean(b)),
                "std": float(np.std(b)),
                "min": float(np.min(b)),
                "max": float(np.max(b)),
                "histogram": _histogram(b),
            }
            total_params += b.size
        except Exception:
            layer_info["bias"] = None

        layer_info["activation_scale"] = _safe_scalar(p, "activation_scale")
        layer_info["parameter_scale"] = _safe_scalar(p, "parameter_scale")

        layers.append(layer_info)

    try:
        total_params_torch = sum(p.numel() for p in model.parameters())
    except Exception:
        total_params_torch = total_params

    return {
        "total_params": int(total_params_torch),
        "num_layers": len(layers),
        "layers": layers,
    }


def _get_model_perceptrons(model: Any) -> list:
    """Try multiple strategies to extract layer-like objects from the model."""
    try:
        perceptrons = model.get_perceptrons()
        if perceptrons:
            return perceptrons
    except Exception:
        pass

    try:
        if hasattr(model, "perceptrons"):
            return list(model.perceptrons)
    except Exception:
        pass

    try:
        children = list(model.children())
        if children:
            linear_layers = []
            for i, child in enumerate(children):
                if isinstance(child, nn.Linear):
                    wrapper = type("_Wrapper", (), {"layer": child, "name": f"linear_{i}"})()
                    wrapper.activation_scale = None
                    wrapper.parameter_scale = None
                    linear_layers.append(wrapper)
            if linear_layers:
                return linear_layers
    except Exception:
        pass

    logger.debug("Could not extract perceptrons/layers from model %s", type(model).__name__)
    return []


def snapshot_pruning_layers(model: Any) -> tuple[dict, list[ResourceDescriptor]]:
    """Extract per-layer weight-heatmap summaries and lazy heatmap descriptors."""
    perceptrons = _get_model_perceptrons(model)
    layers_out: list[dict] = []
    descriptors: list[ResourceDescriptor] = []

    for idx, p in enumerate(perceptrons):
        layer = getattr(p, "layer", None)
        if layer is None or not hasattr(layer, "weight"):
            continue
        weight = layer.weight.data.detach().cpu().numpy()
        out_f, in_f = weight.shape
        prune_row = getattr(layer, "prune_row_mask", None)
        prune_col = getattr(layer, "prune_col_mask", None)
        if prune_row is None or prune_col is None:
            continue
        row_list = prune_row.detach().cpu().tolist()
        col_list = prune_col.detach().cpu().tolist()
        if len(row_list) != out_f or len(col_list) != in_f:
            continue
        layer_name = getattr(p, "name", f"perceptron_{idx}")
        pruned_rows = sum(1 for x in row_list if x)
        pruned_cols = sum(1 for x in col_list if x)
        rid = f"layer/{idx}"
        layers_out.append({
            "layer_index": idx,
            "layer_name": str(layer_name),
            "shape": [int(out_f), int(in_f)],
            "pruned_rows": int(pruned_rows),
            "pruned_cols": int(pruned_cols),
            "has_heatmap": True,
            "heatmap_resource": {
                "kind": RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
                "rid": rid,
            },
        })
        descriptors.append(ResourceDescriptor(
            kind=RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
            rid=rid,
            producer=_make_heatmap_producer(
                weight,
                pruned_row_mask=row_list,
                pruned_col_mask=col_list,
            ),
            media_type="image/png",
        ))

    return {"layers": layers_out}, descriptors



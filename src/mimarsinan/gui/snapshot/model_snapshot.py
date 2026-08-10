"""Model and pruning-layer snapshot builders for the GUI."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch.nn as nn

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.snapshot.util.helpers import _histogram, _safe_scalar
from mimarsinan.gui.snapshot.util.constants import (
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
    RESOURCE_KIND_PRUNING_MASK_MAP,
)
from mimarsinan.gui.resources import HeatmapSource, ResourceDescriptor

# Layer-name stand-in for model-level skip diagnostics (no single layer to blame).
_MODEL_LEVEL = "<model>"


def snapshot_model(model: Any) -> dict:
    """Extract per-layer weight/bias statistics and architecture info."""
    layers: list[dict] = []
    total_params = 0

    perceptrons, _source = _get_model_perceptrons(model)

    for idx, p in enumerate(perceptrons):
        layer_info: dict = {"index": idx, "name": getattr(p, "name", f"perceptron_{idx}")}
        layer_info["weight"] = None
        with best_effort(f"extract weight stats for layer {idx}", logger=logger):
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

        layer_info["bias"] = None
        with best_effort(f"extract bias stats for layer {idx}", logger=logger):
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

        layer_info["activation_scale"] = _safe_scalar(p, "activation_scale")
        layer_info["parameter_scale"] = _safe_scalar(p, "parameter_scale")

        layers.append(layer_info)

    total_params_torch = total_params
    with best_effort("sum model.parameters() for total_params", logger=logger):
        total_params_torch = sum(p.numel() for p in model.parameters())

    return {
        "total_params": int(total_params_torch),
        "num_layers": len(layers),
        "layers": layers,
    }


def _get_model_perceptrons(model: Any) -> tuple[list, str]:
    """Extract layer-like objects from the model; returns ``(layers, source)``.

    ``source`` names the strategy that produced the list: ``"get_perceptrons"``,
    ``"perceptrons_attr"``, ``"linear_fallback"`` (synthesized wrappers that
    never carry pruning masks), or ``"none"``.
    """
    with best_effort("model.get_perceptrons() lookup", logger=logger):
        perceptrons = model.get_perceptrons()
        if perceptrons:
            return list(perceptrons), "get_perceptrons"

    with best_effort("model.perceptrons attribute lookup", logger=logger):
        if hasattr(model, "perceptrons"):
            return list(model.perceptrons), "perceptrons_attr"

    with best_effort("model.children() nn.Linear fallback lookup", logger=logger):
        children = list(model.children())
        if children:
            linear_layers = []
            for i, child in enumerate(children):
                if isinstance(child, nn.Linear):
                    wrapper = type("_Wrapper", (), {
                        "layer": child,
                        "name": f"linear_{i}",
                        "activation_scale": None,
                        "parameter_scale": None,
                    })()
                    linear_layers.append(wrapper)
            if linear_layers:
                return linear_layers, "linear_fallback"

    logger.debug("Could not extract perceptrons/layers from model %s", type(model).__name__)
    return [], "none"


def pruning_layers_unavailable(reason: str, configured_fraction: Any = None) -> dict:
    """A pruning summary whose only content is a model-level skip diagnostic."""
    return {
        "layers": [],
        "skipped": [{"layer": _MODEL_LEVEL, "reason": reason}],
        "configured_fraction": _fraction_or_none(configured_fraction),
    }


def _fraction_or_none(configured_fraction: Any) -> float | None:
    return None if configured_fraction is None else float(configured_fraction)


def _mask_map_matrix(row_mask: list, col_mask: list) -> np.ndarray:
    """Outer product of the keep-masks: +1 = weight kept, -1 = its row or col pruned."""
    keep_rows = ~np.asarray(row_mask, dtype=bool)
    keep_cols = ~np.asarray(col_mask, dtype=bool)
    return np.where(np.outer(keep_rows, keep_cols), 1.0, -1.0)


def snapshot_pruning_layers(
    model: Any, *, configured_fraction: Any = None
) -> tuple[dict, list[ResourceDescriptor]]:
    """Per-layer pruning summaries (dims, sparsity, mask map) + lazy descriptors.

    Masks follow the model convention True = PRUNED; committed weights stay
    full-width, so pre dims come from ``weight.shape`` and post dims are the
    kept counts. Every skip path appends a ``{layer, reason}`` entry to
    ``skipped`` — the tab renders these as visible diagnostics.
    """
    perceptrons, source = _get_model_perceptrons(model)
    layers_out: list[dict] = []
    skipped: list[dict] = []
    descriptors: list[ResourceDescriptor] = []

    if source == "linear_fallback":
        skipped.append({
            "layer": _MODEL_LEVEL,
            "reason": "model exposes no perceptrons; the bare nn.Linear fallback "
                      "synthesizes wrappers that never carry pruning masks",
        })
    if not perceptrons:
        skipped.append({
            "layer": _MODEL_LEVEL,
            "reason": "no perceptron-like layers extractable from the model",
        })

    for idx, p in enumerate(perceptrons):
        layer_name = str(getattr(p, "name", f"perceptron_{idx}"))
        layer = getattr(p, "layer", None)
        if layer is None or not hasattr(layer, "weight"):
            skipped.append({"layer": layer_name, "reason": "no weight tensor on the layer"})
            continue
        weight = layer.weight.data.detach().cpu().numpy()
        out_f, in_f = weight.shape
        prune_row = getattr(layer, "prune_row_mask", None)
        prune_col = getattr(layer, "prune_col_mask", None)
        if prune_row is None or prune_col is None:
            missing = [
                name for name, value in
                (("prune_row_mask", prune_row), ("prune_col_mask", prune_col))
                if value is None
            ]
            skipped.append({
                "layer": layer_name,
                "reason": f"missing {' and '.join(missing)} buffer(s)",
            })
            continue
        row_list = prune_row.detach().cpu().tolist()
        col_list = prune_col.detach().cpu().tolist()
        if len(row_list) != out_f or len(col_list) != in_f:
            skipped.append({
                "layer": layer_name,
                "reason": f"mask length mismatch: row mask {len(row_list)} vs "
                          f"{out_f} neurons, col mask {len(col_list)} vs {in_f} axons",
            })
            continue
        pruned_rows = sum(1 for x in row_list if x)
        pruned_cols = sum(1 for x in col_list if x)
        rid = f"layer/{idx}"
        layers_out.append({
            "layer_index": idx,
            "layer_name": layer_name,
            "shape": [int(out_f), int(in_f)],
            "pre_neurons": int(out_f),
            "pre_axons": int(in_f),
            "post_neurons": int(out_f - pruned_rows),
            "post_axons": int(in_f - pruned_cols),
            "pruned_rows": int(pruned_rows),
            "pruned_cols": int(pruned_cols),
            "achieved_sparsity_neurons": float(pruned_rows / out_f) if out_f else 0.0,
            "achieved_sparsity_axons": float(pruned_cols / in_f) if in_f else 0.0,
            "has_heatmap": True,
            "heatmap_resource": {
                "kind": RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
                "rid": rid,
            },
            "mask_map_resource": {
                "kind": RESOURCE_KIND_PRUNING_MASK_MAP,
                "rid": rid,
            },
        })
        descriptors.append(ResourceDescriptor(
            kind=RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
            rid=rid,
            source=HeatmapSource(
                weight,
                pruned_row_mask=row_list,
                pruned_col_mask=col_list,
            ),
            media_type="image/png",
        ))
        descriptors.append(ResourceDescriptor(
            kind=RESOURCE_KIND_PRUNING_MASK_MAP,
            rid=rid,
            source=HeatmapSource(
                _mask_map_matrix(row_list, col_list),
                pruned_row_mask=row_list,
                pruned_col_mask=col_list,
            ),
            media_type="image/png",
        ))

    summary = {
        "layers": layers_out,
        "skipped": skipped,
        "configured_fraction": _fraction_or_none(configured_fraction),
    }
    return summary, descriptors

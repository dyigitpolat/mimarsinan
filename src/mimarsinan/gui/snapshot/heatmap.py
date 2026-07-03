"""GUI snapshot module."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.heatmap_renderer import render_heatmap_png_bytes

logger = logging.getLogger("mimarsinan.gui")

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

def _detect_neural_core_liveness(node: Any, mat: Any) -> str:
    """Classify a NeuralCore's liveness for the GUI snapshot.

    Best-effort: falls back to LIVE when masks are missing or inconsistent.
    """
    from mimarsinan.mapping.ir import IRSource

    pre_row = getattr(node, "pre_pruning_row_mask", None)
    pre_col = getattr(node, "pre_pruning_col_mask", None)
    pre_row_all_true = bool(pre_row) and all(pre_row)
    pre_col_all_true = bool(pre_col) and all(pre_col)

    if pre_row_all_true and pre_col_all_true:
        return LIVENESS_DEAD_LEGACY

    if pre_row_all_true and pre_col is not None and not pre_col_all_true:
        return LIVENESS_BIAS_ONLY

    shape = None
    with best_effort("read core matrix shape for liveness detection", logger=logger):
        shape = tuple(int(d) for d in mat.shape)

    flat_src = None
    with best_effort("flatten input_sources for liveness detection", logger=logger):
        flat_src = node.input_sources.flatten()

    rmask = getattr(node, "pruned_row_mask", None)
    cmask = getattr(node, "pruned_col_mask", None)
    rmask_all_true = bool(rmask) and all(rmask)
    cmask_all_true = bool(cmask) and all(cmask)

    if shape == (1, 1) and rmask_all_true and cmask_all_true:
        return LIVENESS_DEAD_LEGACY

    if (
        shape is not None
        and shape[0] == 1
        and flat_src is not None
        and len(flat_src) == 1
        and isinstance(flat_src[0], IRSource)
        and flat_src[0].is_off()
    ):
        return LIVENESS_BIAS_ONLY

    return LIVENESS_LIVE


def _make_heatmap_producer(
    matrix: Any,
    *,
    pruned_row_mask: list | None = None,
    pruned_col_mask: list | None = None,
    copy: bool = True,
):
    """Build a zero-arg closure that renders *matrix* to PNG bytes lazily."""
    matrix_copy: Any = matrix
    with best_effort("prepare heatmap matrix", logger=logger):
        matrix_copy = np.asarray(matrix).copy() if copy else np.asarray(matrix)
    rr = list(pruned_row_mask) if pruned_row_mask is not None else None
    cc = list(pruned_col_mask) if pruned_col_mask is not None else None

    def produce() -> bytes:
        return render_heatmap_png_bytes(
            matrix_copy,
            pruned_row_mask=rr,
            pruned_col_mask=cc,
        )

    return produce


def _make_bias_strip_producer(bias: Any):
    """Render a ``hardware_bias`` vector as a 1-row colormap PNG (for BIAS_ONLY cores)."""
    bias_copy: Any = bias
    with best_effort("prepare bias strip array", logger=logger):
        bias_copy = np.asarray(bias, dtype=np.float64).copy()

    def produce() -> bytes:
        arr = np.asarray(bias_copy)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return render_heatmap_png_bytes(arr)

    return produce


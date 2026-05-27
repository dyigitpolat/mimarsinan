"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor

# Bump cautiously: frontend URL builders hard-code these.
RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"


# Per-NeuralCore liveness tags surfaced in the GUI (must match
# ``mimarsinan.mapping.pruning.ir_liveness.NodeLiveness`` for current runs).
LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"  # only for old pickles still containing (1,1) placeholders

def _detect_neural_core_liveness(node: Any, mat: Any) -> str:
    """Classify a NeuralCore's liveness for the GUI snapshot.

    Returns one of:

    - :data:`LIVENESS_DEAD_LEGACY` for old pickles that still contain
      ``(1, 1)`` zero placeholders -- detected by an all-True
      ``pre_pruning_*_mask`` (or a fully-pruned post-compaction mask
      when no pre-pruning data was stored).
    - :data:`LIVENESS_BIAS_ONLY` for cores whose every original axon was
      pruned but live columns remain (the post-compaction shape is
      ``(1, N)`` with a single OFF-source axon).
    - :data:`LIVENESS_LIVE` otherwise.

    The function is best-effort: when masks are missing or inconsistent
    we fall back to ``LIVE`` rather than fail the snapshot build.
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

    try:
        shape = tuple(int(d) for d in mat.shape)
    except Exception:
        shape = None

    flat_src = None
    try:
        flat_src = node.input_sources.flatten()
    except Exception:
        pass

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
    try:
        if copy:
            matrix_copy: Any = np.asarray(matrix).copy()
        else:
            matrix_copy = np.asarray(matrix)
    except Exception:
        matrix_copy = matrix
    rr = list(pruned_row_mask) if pruned_row_mask is not None else None
    cc = list(pruned_col_mask) if pruned_col_mask is not None else None

    def produce() -> bytes:
        from mimarsinan.gui.heatmap_renderer import render_heatmap_png_bytes
        return render_heatmap_png_bytes(
            matrix_copy,
            pruned_row_mask=rr,
            pruned_col_mask=cc,
        )

    return produce


def _make_bias_strip_producer(bias: Any):
    """Render a ``hardware_bias`` vector as a 1-row colormap PNG.

    Used for BIAS_ONLY cores so the UI can show that the core fires from
    bias drive even though its weight matrix is empty.
    """
    try:
        bias_copy: Any = np.asarray(bias, dtype=np.float64).copy()
    except Exception:
        bias_copy = bias

    def produce() -> bytes:
        from mimarsinan.gui.heatmap_renderer import render_heatmap_png_bytes
        arr = np.asarray(bias_copy)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return render_heatmap_png_bytes(arr)

    return produce


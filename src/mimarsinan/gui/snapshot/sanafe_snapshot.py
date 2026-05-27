"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
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


def snapshot_sanafe_simulation(
    report: Any,
) -> tuple[dict, list[ResourceDescriptor]]:
    """Build the snapshot and resource descriptors for the SANA-FE GUI tab."""
    snap = report.to_snapshot_dict() if report is not None else {
        "arch_preset": "", "sample_indices": [], "aggregate": {}, "per_sample": [],
    }
    return snap, []


def _find_ir_graph_promiser(pipeline: Any) -> str | None:
    """Return the name of the pipeline step whose ``promises`` includes
    ``ir_graph``, i.e. the step that owns the IR-level resource folder.

    Returns ``None`` when no such step exists (e.g. ad-hoc pipelines used
    in tests). Callers must treat ``None`` as "register descriptors locally"
    rather than silently dropping them.
    """
    try:
        steps = getattr(pipeline, "steps", ())
    except Exception:
        return None
    for name, s in steps:
        try:
            if "ir_graph" in set(getattr(s, "promises", ())):
                return str(name)
        except Exception:
            continue
    return None


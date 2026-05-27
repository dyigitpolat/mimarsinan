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
# ``mimarsinan.mapping.ir_liveness.NodeLiveness`` for current runs).
LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"  # only for old pickles still containing (1,1) placeholders




def snapshot_adaptation_manager(manager: Any) -> dict:
    """Extract current adaptation rates."""
    return {
        "clamp_rate": getattr(manager, "clamp_rate", None),
        "shift_rate": getattr(manager, "shift_rate", None),
        "quantization_rate": getattr(manager, "quantization_rate", None),
        "scale_rate": getattr(manager, "scale_rate", None),
        "noise_rate": getattr(manager, "noise_rate", None),
    }


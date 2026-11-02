"""Convenience wrapper to map a model representation to an IRGraph; [U-DOOR] and the NF twin's comb keys read from that
mapping (``mapper_phase_keys``)."""

from __future__ import annotations

import weakref
from typing import Any, Dict

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.phase_channels import node_phase_channels, source_channels

def map_model_to_ir(
    model_representation,
    q_max: float = 1.0,
    firing_mode: str = "Default",
    max_axons: int | None = None,
    max_neurons: int | None = None,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> IRGraph:
    """Convenience wrapper around ``IRMapping.map``."""
    return IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    ).map(model_representation)


#: Per mapper graph: (its structure, its nodes' keys); kept while the graph lives (a key walk maps the whole graph).
_KEYS: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()


def _structure(model_representation: Any) -> tuple:
    """What decides the IR's node partition: the nodes in execution order and which perceptrons are encoding layers
    (a host node rather than a core)."""
    return tuple((id(n), type(n).__name__, bool(getattr(getattr(n, "perceptron", None), "is_encoding_layer", False)))
                 for n in model_representation.execution_order())


def mapper_phase_keys(model_representation: Any) -> Dict[int, np.ndarray]:
    """[U-DOOR] ``id(node) -> channels``: every mapper node's channel at unbounded geometry per element of one sample
    of its value, the NF twin's comb keys: the node's IR sources (``map_to_ir``) through
    ``phase_channels.node_phase_channels``, the table the deployed program keys its stage inputs by
    (``phase_channels.source_channels``), so both twins key from one rule. The graph is mapped as the identity program
    maps it (unbounded cores, no coalescing: E1's channels do not move with a cost-only decision), with the perceptron
    indices Soft Core Mapping assigns (an idempotent walk); its walk memos are cleared after. Kept per graph while its
    structure holds. A graph the mapping refuses refuses here by the mapping's own name."""
    structure = _structure(model_representation)
    kept = _KEYS.get(model_representation)
    if kept is not None and kept[0] == structure:
        return kept[1]
    try:
        with torch.no_grad():
            model_representation.assign_perceptron_indices()
            table = node_phase_channels(map_model_to_ir(model_representation))
            keys = {id(n): source_channels(table, sources) for n in model_representation.execution_order()
                    if (sources := getattr(n, "_ir_sources", None)) is not None}
    finally:
        model_representation.clear_ir_caches()
    _KEYS[model_representation] = (structure, keys)
    return keys

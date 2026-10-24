"""Rewire every reader of a removed IR node to the sources its output elements became (lane fusion)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource
from mimarsinan.mapping.ir.gather_plan import invalidate_gather_plan
from mimarsinan.mapping.ir.source import encode_ir_sources


def _rewired(sources: np.ndarray, replace: Dict[int, np.ndarray], removed: np.ndarray) -> np.ndarray:
    encoded = encode_ir_sources(sources)
    ids = encoded[2] if isinstance(encoded, tuple) else np.array([int(s.node_id) for s in sources.flat])
    hits = np.flatnonzero(np.isin(ids, removed))
    if hits.size == 0:
        return sources
    flat: list = list(np.asarray(sources, dtype=object).reshape(-1))
    for position in hits.tolist():
        source = flat[position]
        flat[position] = replace[int(source.node_id)][int(source.index)]
    return np.array(flat, dtype=object).reshape(np.shape(sources))


def rewire_removed(ir_graph: IRGraph, replace: Dict[int, np.ndarray]) -> None:
    """Point every source naming a node of ``replace`` (by element) at its replacement, the outputs included."""
    if not replace:
        return
    removed = np.array(sorted(replace), dtype=np.int64)
    for node in ir_graph.nodes:
        rewired = _rewired(node.input_sources, replace, removed)
        if rewired is not node.input_sources:
            node.input_sources = rewired
            if hasattr(node, "_source_counts_cache"):
                delattr(node, "_source_counts_cache")
            invalidate_gather_plan(node)
    ir_graph.output_sources = _rewired(np.asarray(ir_graph.output_sources, dtype=object), replace, removed)
    for source in np.asarray(ir_graph.output_sources).flat:
        if isinstance(source, IRSource) and int(source.node_id) in replace:
            raise AssertionError("a rewired output still names a removed node")

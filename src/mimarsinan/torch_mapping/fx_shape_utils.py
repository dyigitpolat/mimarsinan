"""Shared FX shape-metadata extraction helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch.fx as fx


def node_input_shapes(node: fx.Node) -> List[Optional[Tuple[int, ...]]]:
    """One shape per ``fx.Node`` argument (batch dim included); ``None`` for missing metadata."""
    shapes: List[Optional[Tuple[int, ...]]] = []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            shapes.append(_meta_shape(arg))
    return shapes


def node_output_shape(node: fx.Node) -> Optional[Tuple[int, ...]]:
    return _meta_shape(node)


def strip_batch(shape: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    if shape is None or len(shape) < 2:
        return None
    return tuple(shape[1:])


def _meta_shape(node: fx.Node) -> Optional[Tuple[int, ...]]:
    meta = node.meta.get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        return None
    return tuple(meta.shape)

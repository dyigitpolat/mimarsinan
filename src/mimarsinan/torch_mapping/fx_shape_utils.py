"""Shared FX shape-metadata extraction and literal-argument coercion helpers."""

from __future__ import annotations

from typing import List, Optional, SupportsInt, Tuple, cast

import torch.fx as fx


def fx_literal_int(value: object) -> int:
    """Coerce a numeric FX literal to int; a Node means a dynamic value and is rejected."""
    if isinstance(value, fx.Node):
        raise TypeError(
            f"FX argument {value!r} is a Node (dynamic value), not a numeric literal"
        )
    return int(cast(SupportsInt, value))


def node_target_str(node: fx.Node) -> str:
    """Target of a call_module/call_method/get_attr node; FX guarantees these are strings."""
    target = node.target
    if not isinstance(target, str):
        raise TypeError(
            f"expected a string target on {node.op} node {node.name}; got {target!r}"
        )
    return target


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

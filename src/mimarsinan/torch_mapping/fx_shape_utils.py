"""Shared FX shape-metadata extraction, literal-argument coercion and call-site state helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, SupportsInt, Tuple, cast

import torch
import torch.fx as fx
import torch.nn as nn


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


def node_state(node: fx.Node, gm: fx.GraphModule, modules: Dict[str, nn.Module]) -> List[torch.Tensor]:
    """Parameters a call_module or get_attr node reads: the state whose per-site copy would untie a weight.

    Buffers are constants at inference (a copy per site loses no tie), so they are not call-site state.
    """
    if node.op == "call_module":
        mod = modules.get(node_target_str(node))
        return [] if mod is None else list(mod.parameters())
    if node.op == "get_attr":
        obj: object = gm
        for part in node_target_str(node).split("."):
            obj = getattr(obj, part, None)
        return [obj] if isinstance(obj, nn.Parameter) else []
    return []


def shared_state_sites(
    gm: fx.GraphModule, modules: Dict[str, nn.Module],
) -> List[Tuple[fx.Node, List[str]]]:
    """``(first node, call sites)`` per target whose state another FX call site also reads, in graph order."""
    nodes = list(gm.graph.nodes)
    order = {node.name: i for i, node in enumerate(nodes)}
    sites: Dict[int, List[str]] = {}
    for node in nodes:
        for tensor in node_state(node, gm, modules):
            names = sites.setdefault(id(tensor), [])
            if node.name not in names:
                names.append(node.name)
    shared: List[Tuple[fx.Node, List[str]]] = []
    seen: set[str] = set()
    for node in nodes:
        reached = {n for t in node_state(node, gm, modules) if len(sites[id(t)]) > 1 for n in sites[id(t)]}
        if reached and str(node.target) not in seen:
            seen.add(str(node.target))
            shared.append((node, sorted(reached, key=order.__getitem__)))
    return shared

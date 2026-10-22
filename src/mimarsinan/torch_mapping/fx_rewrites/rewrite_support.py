"""What every FX rewrite shares: the nodes it may touch, the extents it may write as literals, and the traced
metadata of a node it adds."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Set

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import map_arg
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    COPY_GUARD,
    KEPT_OPERANDS,
    RUN_AFTER,
    declared_writes,
)

_WRITE_KEYS = (RUN_AFTER, COPY_GUARD, KEPT_OPERANDS)


def write_ordered(gm: fx.GraphModule) -> Set[str]:
    """The names of every node a write in place orders or that orders one (``operand_writes``), and of every node
    writing in place: a rewrite leaves them as traced."""
    modules: Mapping[str, nn.Module] = dict(gm.named_modules())
    names: Set[str] = set()
    for node in gm.graph.nodes:
        ordered = [name for key in _WRITE_KEYS for name in (node.meta.get(key) or ()) if isinstance(name, str)]
        if ordered or any(node.meta.get(key) for key in _WRITE_KEYS) or declared_writes(node, modules):
            names.update([node.name, *ordered])
    return names


def rewritable(nodes, ordered: Set[str]) -> bool:
    """None of ``nodes`` nor their users is ordered by a write in place."""
    return not any(node.name in ordered or any(user.name in ordered for user in node.users) for node in nodes)


class StaticExtents:
    """The extents of ``gm``'s values a rewrite may write into the graph as literals (a ``narrow``'s pieces, an
    attention's ``1 / sqrt(E)``): static extents the batch domain proves, ``BatchLayouts.layout`` (a pure program's
    proof that every seeded trace agrees with), never a traced extent alone. Any axis may carry the batch (axis 0 of an
    activation; another after ``torch.stack(..., 0)``, ``movedim``, ``swapaxes`` or ``t``), and a call outside the
    declared operator set may change an extent with the batch, so an impure program proves none (``symbolic/purity.py``).
    The layouts are read on first use. A rewrite keeps every value, so they hold across rewrites; one that replaces a
    call without a declared batch rule by declared operators (the SDPA rewrite) extends what the domain proves
    downstream, and ``invalidate`` makes the next use read them again."""

    def __init__(self, gm: fx.GraphModule):
        self._gm = gm
        self._layouts: Optional[BatchLayouts] = None

    def holds(self, node: fx.Node, axis: int, extent: int) -> bool:
        """Axis ``axis`` of ``node`` has the proved static extent ``extent`` (never the unproven-extent marker)."""
        if self._layouts is None:
            self._layouts = BatchLayouts(self._gm)
        layout = self._layouts.layout(node)
        if layout is None or not -len(layout) <= axis < len(layout):
            return False
        found = layout[axis]
        return type(found) is int and found == extent

    def invalidate(self) -> None:
        """The graph now proves more (a call without a batch rule was rewritten): read the layouts again on next use."""
        self._layouts = None


def _meta_value(arg: Any) -> Any:
    meta = arg.meta.get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        raise ValueError(f"FX node {arg.name!r} has no traced tensor metadata")
    return torch.empty_strided(tuple(meta.shape), tuple(meta.stride), dtype=meta.dtype, device="meta")


def trace_meta(node: fx.Node) -> None:
    """``node``'s traced metadata, from its call on meta tensors of its operands' traced metadata (no data, no random
    state): the shape and stride the shape propagation would record."""
    args = map_arg(node.args, _meta_value)
    kwargs = map_arg(node.kwargs, _meta_value)
    if node.op == "call_method":
        result = getattr(args[0], str(node.target))(*args[1:], **kwargs)
    else:
        result = node.target(*args, **kwargs)  # type: ignore[operator]  # a call_function node's callable
    node.meta["tensor_meta"] = _extract_tensor_metadata(result)
    node.meta["type"] = type(result)

"""The batch-axis push-forward: a permute that moves the batch off axis 0, read only back into a batch-leading form.

``y = x.permute(p)`` with ``p[0] != 0`` (``transpose`` alike) on a batch-leading activation ``x`` does not carry the
batch on axis 0, so Wave 0 refuses it by name (``_require_batch_leading``). When every consumer of ``y`` restores a
batch-leading value, the consumers are rewritten onto ``x`` and ``y`` is erased:

- ``y[i]`` (an integer ``getitem``, also through ``y.unbind(0)``) is ``x.select(p[0], i).permute(q)``, ``q`` the order
  of ``p[1:]`` among the remaining axes, provided ``q[0] == 0`` (the batch leads again): the timm-style
  ``qkv.permute(2, 0, 3, 1, 4)[0]``. It is the same view of ``x`` (offset and strides).
- ``y.permute(r)`` (or a transpose) is ``x.permute(p[r])`` when ``p[r[0]] == 0``, ``x`` itself when that is the
  identity.

Any other consumer (a reshape, an ``unsqueeze``, a host op) leaves ``y`` as traced, so it still refuses by name; the
rewrite never routes a batch axis around that check.
"""

from __future__ import annotations

import operator
from typing import List, Optional, Sequence, Set, Tuple, cast

import torch
import torch.fx as fx

from mimarsinan.torch_mapping.fx_rewrites.rewrite_support import rewritable, trace_meta

_LAYOUTS = {"permute": torch.permute, "transpose": torch.transpose}


def _rank(node: fx.Node) -> Optional[int]:
    shape = getattr(node.meta.get("tensor_meta"), "shape", None)
    return None if shape is None else len(shape)


def permutation(node: fx.Node) -> Optional[Tuple[int, ...]]:
    """The axis order of a ``permute``/``transpose`` (method or function) with literal dims; ``None`` otherwise."""
    name = node.target if node.op == "call_method" else next(
        (key for key, fn in _LAYOUTS.items() if node.op == "call_function" and node.target is fn), None)
    rank = _rank(node)
    if name not in _LAYOUTS or rank is None or not node.args or node.kwargs:
        return None
    found = list(node.args[1:])
    if name == "permute":
        found = list(found[0]) if len(found) == 1 and isinstance(found[0], (tuple, list)) else found
    if any(type(d) is not int for d in found) or len(found) != (rank if name == "permute" else 2):
        return None
    dims = [cast(int, d) for d in found]
    if name == "permute":
        return tuple(d % rank for d in dims)
    order = list(range(rank))
    a, b = dims[0] % rank, dims[1] % rank
    order[a], order[b] = order[b], order[a]
    return tuple(order)


def _index(user: fx.Node, node: fx.Node) -> Optional[int]:
    if user.op == "call_function" and user.target is operator.getitem and len(user.args) == 2 \
            and user.args[0] is node and type(user.args[1]) is int:
        return int(user.args[1])
    return None


def _unbound(user: fx.Node) -> Optional[List[fx.Node]]:
    """The integer ``getitem`` users of ``user`` when it is an ``unbind`` along axis 0 read only through them."""
    method = user.op == "call_method" and user.target == "unbind"
    function = user.op == "call_function" and user.target is torch.unbind
    dim = (user.args[1] if len(user.args) > 1 else user.kwargs.get("dim", 0)) if method or function else None
    readers = list(user.users)
    if dim != 0 or not readers or any(_index(reader, user) is None for reader in readers):
        return None
    return readers


def _selected(p: Sequence[int]) -> Optional[Tuple[int, ...]]:
    rest = [axis for axis in range(len(p)) if axis != p[0]]
    order = tuple(rest.index(axis) for axis in p[1:])
    return order if order and order[0] == 0 else None


def _plan(user: fx.Node, node: fx.Node, p: Tuple[int, ...]) -> Optional[list]:
    """``[(reader, ("select", i) | ("permute", r))]`` restoring a batch-leading value; ``None`` when one cannot."""
    if _index(user, node) is not None:
        return [(user, ("select", _index(user, node)))] if _selected(p) is not None else None
    readers = _unbound(user)
    if readers is not None and _selected(p) is not None:
        return [(reader, ("select", _index(reader, user))) for reader in readers]
    r = permutation(user) if user.args and user.args[0] is node else None
    if r is not None and p[r[0]] == 0:
        return [(user, ("permute", tuple(p[axis] for axis in r)))]
    return None


def _emit(gm: fx.GraphModule, at: fx.Node, source: fx.Node, step: tuple, p: Tuple[int, ...]) -> fx.Node:
    with gm.graph.inserting_before(at):
        if step[0] == "permute":
            if step[1] == tuple(range(len(step[1]))):
                return source
            value = gm.graph.call_method("permute", (source, *step[1]))
            trace_meta(value)
            return value
        value = gm.graph.call_method("select", (source, p[0], step[1]))
        trace_meta(value)
        order = _selected(p)
        if order is not None and order != tuple(range(len(order))):
            value = gm.graph.call_method("permute", (value, *order))
            trace_meta(value)
        return value


def push_batch_axis_forward(gm: fx.GraphModule, ordered: Set[str]) -> bool:
    """Rewrite every batch-moving permute whose consumers all restore a batch-leading value; True when one was."""
    changed = False
    for node in list(gm.graph.nodes):
        p = permutation(node)
        source = node.args[0] if p is not None else None
        if p is None or p[0] == 0 or not isinstance(source, fx.Node) or _rank(source) != len(p):
            continue
        users = list(node.users)
        plans = [_plan(user, node, p) for user in users]
        if not users or any(plan is None for plan in plans) or not rewritable([node, *users], ordered):
            continue
        for user, plan in zip(users, plans):
            for reader, step in plan or ():
                reader.replace_all_uses_with(_emit(gm, reader, source, step, p))
                gm.graph.erase_node(reader)
            if user in gm.graph.nodes and not user.users:
                gm.graph.erase_node(user)
        gm.graph.erase_node(node)
        changed = True
    return changed

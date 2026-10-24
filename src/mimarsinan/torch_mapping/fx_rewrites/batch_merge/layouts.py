"""The layout rules of the batch-merge push-forward (coverage W7, ``axes.py`` states the rules): how a merged value's
reshape, permutation, index and named-axis operations act on its split value."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from torch.fx.node import map_arg

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine

__all__ = ["DROPOUTS", "Extents", "FIBERS", "IDENTITIES", "Infos", "LAYOUTS", "Merged", "PERMUTES", "Plan",
           "along_plan", "argument", "axis_of", "index_plan", "name_of", "permute_plan", "reshape_target",
           "scalar_operands", "shape_operands", "split_axis"]


@dataclass(frozen=True)
class Merged:
    """A value merged on axis ``at``, of extent ``a*B``, rank ``rank`` (the original value's); ``elements``: a tuple
    of such values (``unbind``), read by an integer index."""

    at: int
    a: int
    rank: int
    elements: bool = False


@dataclass(frozen=True)
class Plan:
    """The operation on the split values: ``op``, ``target``, ``args``, ``kwargs`` (original nodes stand for their
    split values); ``out``: the result's merge (``None``: a value without the merged axis, or a batch-leading end)."""

    op: str
    target: Any
    args: tuple
    kwargs: Dict[str, Any]
    out: Optional[Merged]
    end: bool = False


Extents = Callable[[fx.Node], Optional[Tuple[Any, ...]]]
Infos = Mapping[fx.Node, Merged]

LAYOUTS = {"view", "reshape", "flatten"}
PERMUTES = {"permute": torch.permute, "transpose": torch.transpose}
IDENTITIES = {"contiguous", "clone"}
FIBERS = {"softmax": 1, "log_softmax": 1, "softmin": 1, "cumsum": 1, "cumprod": 1, "logcumsumexp": 1,
           "normalize": 2}
"""Operations computed on every fiber along ``dim`` independently: the position of ``dim`` (the input at 0)."""
DROPOUTS = {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"}


def name_of(node: fx.Node) -> str:
    return str(node.target) if node.op == "call_method" else str(getattr(node.target, "__name__", node.target))


def argument(node: fx.Node, name: str, position: int, default: Any = None) -> Any:
    return node.kwargs[name] if name in node.kwargs else node.args[position] if len(node.args) > position else default


def split_axis(axis: int, at: int) -> int:
    """An original axis other than the merged one, in the split value."""
    return axis if axis < at else axis + 1


def axis_of(dim: Any, rank: int) -> Optional[int]:
    return dim % rank if type(dim) is int and -rank <= dim < rank else None


def shape_operands(node: fx.Node) -> bool:
    """Every operand of a ``view``, ``reshape`` or ``flatten`` after its input is an extent: an ``int``, a traced
    extent, or a tuple or list of them (``view(torch.int32)`` reinterprets the elements, no reshape)."""
    def extent(value: Any) -> bool:
        if isinstance(value, (tuple, list)):
            return all(map(extent, value))
        return type(value) is int or isinstance(value, fx.Node) and "tensor_meta" not in value.meta
    return all(map(extent, (*node.args[1:], *node.kwargs.values())))


def static(dims: Optional[Sequence[Any]], skip: int = -1) -> bool:
    return dims is not None and all(type(d) is int for i, d in enumerate(dims) if i != skip)


def reshape_target(source: Merged, dims: Optional[Tuple[Any, ...]], prefix: Optional[Tuple[Any, ...]]
                   ) -> Optional[Tuple[Optional[Merged], bool, Tuple[int, ...]]]:
    """A reshape of a value merged as ``source`` to the proved extents ``dims``: ``(merge, end, target)``. A source
    merged on axis 0 is sample-major in its whole element order, so any output with ``a2*B`` rows is merged (``a2 >=
    2``, target ``(-1, a2, ...)``) or batch-leading (``a2 == 1``, an end: its own target); one merged on a later axis
    keeps its prefix (``prefix``, the source's leading extents) and its axis there. ``None`` otherwise."""
    if dims is None or not dims or len(dims) <= source.at:
        return None
    lead = dims[source.at]
    if not isinstance(lead, Affine) or lead.c != 0 or lead.a < 1 or not static(dims, source.at):
        return None
    if source.at > 0 and (prefix is None or tuple(dims[:source.at]) != tuple(prefix)):
        return None
    if lead.a == 1:
        return (None, True, ()) if source.at == 0 else None
    rest = tuple(int(d) for d in dims[source.at + 1:])
    return Merged(source.at, lead.a, len(dims)), False, (*tuple(int(d) for d in dims[:source.at]), -1, lead.a, *rest)


def _permutation(node: fx.Node, rank: int) -> Optional[Tuple[int, ...]]:
    found = list(node.args[1:])
    if name_of(node) == "permute":
        found = list(found[0]) if len(found) == 1 and isinstance(found[0], (tuple, list)) else found
        dims = [axis_of(d, rank) for d in found]
        return None if None in dims or sorted(dims) != list(range(rank)) else tuple(dims)  # type: ignore[type-var]
    dims = [axis_of(d, rank) for d in found]
    if len(dims) != 2 or None in dims:
        return None
    order = list(range(rank))
    order[dims[0]], order[dims[1]] = order[dims[1]], order[dims[0]]  # type: ignore[index]
    return tuple(order)


def permute_plan(node: fx.Node, merged: Merged) -> Optional[Plan]:
    p = _permutation(node, merged.rank)
    if p is None or node.kwargs:
        return None
    expanded = [axis for k in p for axis in (
        (merged.at, merged.at + 1) if k == merged.at else (split_axis(k, merged.at),))]
    return Plan("call_method", "permute", (node.args[0], *expanded), {}, Merged(p.index(merged.at), merged.a,
                                                                                   merged.rank))


def index_plan(node: fx.Node, merged: Merged) -> Optional[Plan]:
    """``x[i]`` or ``x[e_0, ..., e_k]`` of integers and slices, the merged axis's entry ``:`` or absent."""
    index = node.args[1]
    entries = index if type(index) is tuple else (index,)
    literal = all(type(e) is int or type(e) is slice and all(
        b is None or type(b) is int for b in (e.start, e.stop, e.step)) for e in entries)
    if not literal or len(entries) > merged.rank:
        return None
    at, split = merged.at, list(entries)
    if len(entries) > merged.at:
        whole = entries[merged.at]
        if type(whole) is not slice or (whole.start, whole.stop, whole.step) != (None, None, None):
            return None
        split.insert(merged.at, slice(None))
    removed = sum(1 for e in entries[:merged.at] if type(e) is int)
    out = Merged(at - removed, merged.a, merged.rank - sum(1 for e in entries if type(e) is int))
    return Plan("call_function", operator.getitem, (node.args[0], tuple(split) if type(index) is tuple or len(
        split) > 1 else split[0]), {}, out)


def scalar_operands(node: fx.Node) -> bool:
    """No operand after the first is a tensor (``torch.masked.softmax``'s ``mask`` would broadcast against the split
    value at other axes)."""
    found: list = []
    map_arg((node.args[1:], node.kwargs), found.append)
    return not any(hasattr(arg.meta.get("tensor_meta"), "shape") or isinstance(arg.meta.get("tensor_meta"), (
        tuple, list)) for arg in found)


def along_plan(node: fx.Node, merged: Merged, position: int, dropped: bool, default: Any = None) -> Optional[Plan]:
    """An operation along one named axis ``dim`` other than the merged one (``dropped``: the axis is removed;
    ``default``: ``dim`` when the call does not pass it), with no tensor operand but the merged one."""
    dim = axis_of(argument(node, "dim", position, default), merged.rank)
    if dim is None or dim == merged.at or not scalar_operands(node):
        return None
    args, kwargs = list(node.args), dict(node.kwargs)
    if "dim" in kwargs or len(args) <= position:
        kwargs["dim"] = split_axis(dim, merged.at)
    else:
        args[position] = split_axis(dim, merged.at)
    at = merged.at - (1 if dropped and dim < merged.at else 0)
    return Plan(node.op, node.target, tuple(args), kwargs, Merged(at, merged.a, merged.rank - (1 if dropped else 0)))


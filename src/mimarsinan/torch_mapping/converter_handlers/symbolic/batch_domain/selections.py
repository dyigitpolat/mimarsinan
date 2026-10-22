"""Declared batch rules of torch's joining, splitting and indexing kernels: cat, stack, chunk, split, unbind,
narrow, select, index_select, diff, gather and basic indexing (review round 14).

As in ``layouts.py``: each rule computes its output's extents in the affine batch domain, as torch's kernel defines
them (a pass-through, a sum for ``cat``, a slice's length clamped to its axis by ``sym_max`` and ``sym_min``, which fold
only where the choice is the same at every batch); an extent the domain does not compute is ``Opaque``.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Optional

import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Opaque, form
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.operations import evaluate
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Call, Shape, axis_of as _axis, free

__all__ = ["SELECTION_RULES"]


def _joined(call: Call) -> Optional[list]:
    tensors = call.arg(0, "tensors")
    if type(tensors) not in (tuple, list) or not tensors or not all(isinstance(t, Shape) for t in tensors):
        return None
    shapes = [t.dims for t in tensors if t.dims != (0,)] or [tensors[0].dims]
    rank = len(shapes[0])
    if call.name == "stack":
        axis = _axis(call.arg(1, "dim", 0), rank + 1)
        return None if axis is None else [shapes[0][:axis] + (len(tensors),) + shapes[0][axis:]]
    axis = _axis(call.arg(1, "dim", 0), rank)
    if axis is None or any(len(s) != rank for s in shapes):
        return None
    total: Any = 0
    for s in shapes:
        total = evaluate(operator.add, [total, s[axis]])
    return [shapes[0][:axis] + (total,) + shapes[0][axis + 1:]]


def _pieces(call: Call) -> Optional[list]:
    """``chunk``, ``split`` and ``unbind`` along a batch-free axis: torch's pieces, each computed here."""
    dims = call.args[0].dims
    axis = _axis(call.arg(2 if call.name != "unbind" else 1, "dim", 0), len(dims))
    if axis is None or type(dims[axis]) is not int:
        return None
    extent, how = dims[axis], call.arg(1, "chunks" if call.name == "chunk" else "split_size_or_sections")
    if call.name == "unbind":
        return [dims[:axis] + dims[axis + 1:]] * extent
    if call.name == "chunk" and type(how) is int and how > 0:
        size = -(-extent // how)
        sizes = [size] * (extent // size) + ([extent % size] if extent % size else []) if size else [0]
    elif type(how) is int and how > 0:
        sizes = [how] * (extent // how) + ([extent % how] if extent % how or not extent else [])
    elif isinstance(how, (tuple, list)) and all(type(s) is int for s in how) and sum(how) == extent:
        sizes = list(how)
    else:
        return None
    return [dims[:axis] + (size,) + dims[axis + 1:] for size in sizes]


def _narrow(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    axis = _axis(call.arg(1, "dim"), len(dims))
    length = call.arg(3, "length")
    if axis is None or isinstance(length, Shape) or (form(length) is None and not isinstance(length, Opaque)):
        return None
    return [dims[:axis] + (length,) + dims[axis + 1:]]


def _select(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    axis = _axis(call.arg(1, "dim"), len(dims))
    if axis is None:
        return None
    if call.name == "index_select":
        index = call.arg(2, "index")
        if not isinstance(index, Shape) or len(index.dims) > 1:
            return None
        return [dims[:axis] + (index.dims[0] if index.dims else 1,) + dims[axis + 1:]]
    return [dims[:axis] + dims[axis + 1:]]


def _diff(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    steps, axis = call.arg(1, "n", 1), _axis(call.arg(2, "dim", -1), len(dims))
    if axis is None or type(steps) is not int or call.arg(3, "prepend") is not None or \
            call.arg(4, "append") is not None:
        return None
    left = evaluate(operator.sub, [dims[axis], steps])
    kept = evaluate(operator.ge, [left, 0])
    extent = left if kept is True else Opaque("a diff whose length changes sign with the batch")
    return [dims[:axis] + (extent,) + dims[axis + 1:]]


def _gather(call: Call) -> Optional[list]:
    index = call.arg(2, "index")
    return [index.dims] if isinstance(index, Shape) else None


def _clamped(value: Any, extent: Any) -> Any:
    """A slice bound as Python normalizes it on an axis of ``extent``: a negative one counted from the end, then
    clamped to ``[0, extent]``."""
    negative = evaluate(operator.lt, [value, 0])
    if isinstance(negative, Opaque):
        return negative
    shifted = evaluate(operator.add, [value, extent]) if negative else value
    return evaluate(torch.sym_max, [0, evaluate(torch.sym_min, [shifted, extent])])


def _slice_length(extent: Any, piece: slice) -> Any:
    step = 1 if piece.step is None else piece.step
    if type(step) is not int or step < 1 or isinstance(extent, Opaque):
        return Opaque("a slice with a step the domain does not take")
    if free((extent, piece.start, piece.stop)):
        return len(range(*slice(piece.start, piece.stop, step).indices(extent)))
    start = 0 if piece.start is None else _clamped(piece.start, extent)
    stop = extent if piece.stop is None else _clamped(piece.stop, extent)
    span = evaluate(torch.sym_max, [0, evaluate(operator.sub, [stop, start])])
    length = span if step == 1 else evaluate(operator.floordiv, [evaluate(operator.add, [span, step - 1]), step])
    if isinstance(length, Opaque) and length.origin is None:
        return Opaque(f"the length of a slice of a batch-dependent extent ({length.reason})")
    return length


def _index(call: Call) -> Optional[list]:
    dims, index = call.args[0].dims, call.args[1]
    items = list(index) if type(index) is tuple else [index]
    if sum(isinstance(item, Shape) for item in items) > 1 or any(isinstance(item, list) for item in items) or \
            sum(item is Ellipsis for item in items) > 1:
        return None  # one index tensor at most: its extents take its axis's place (an integer index's; a Boolean
    #                  mask's count is data-dependent, so torch's run disagrees and the size is opaque)
    consumed = sum(1 for item in items if item is not None and item is not Ellipsis)
    if consumed > len(dims):
        return None
    out, axis = [], 0
    for item in items:
        if item is None:
            out.append(1)
        elif item is Ellipsis:
            out.extend(dims[axis:axis + len(dims) - consumed])
            axis += len(dims) - consumed
        elif type(item) is slice:
            out.append(_slice_length(dims[axis], item))
            axis += 1
        elif isinstance(item, Shape):
            out.extend(item.dims)
            axis += 1
        elif form(item) is not None or isinstance(item, Opaque):
            axis += 1  # an integer index drops its axis
        else:
            return None
    return [tuple(out) + dims[axis:]]


SELECTION_RULES: Dict[str, Callable[[Call], Optional[list]]] = {
    "cat": _joined, "concat": _joined, "concatenate": _joined, "stack": _joined, "chunk": _pieces, "split": _pieces,
    "unbind": _pieces, "narrow": _narrow, "select": _select, "index_select": _select, "diff": _diff,
    "gather": _gather, "getitem": _index,
}
"""The joining, splitting and indexing kernels by name (``getitem``: indexing)."""

"""Declared batch rules of torch's reshaping kernels: view and reshape with domain targets, flatten, unflatten,
permute, transpose, movedim, squeeze, unsqueeze, expand, repeat, tile and repeat_interleave (review round 14).

Each rule computes its output's extents in the affine batch domain (``operations.evaluate``) from its operands' extents
and size arguments, as torch's kernel defines them: a pass-through, a permutation, a product (``flatten``, ``repeat``)
or a quotient that is exact in the domain (the ``-1`` of a reshape). An extent the domain does not compute is
``Opaque``. A rule returns one extent tuple per tensor the kernel returns, or ``None`` when it states no output (the
kernel's every size is then opaque). ``selections.py`` holds the joining, splitting and indexing kernels, ``kernels.py``
the other kernels and the lemma.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List, Optional

import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Opaque, form
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.operations import evaluate
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.selections import SELECTION_RULES
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import (
    Call,
    Dims,
    Shape,
    axis_of as _axis,
    product,
    quotient,
    sizes_of as _sizes,
)

__all__ = ["LAYOUT_RULES"]


def _infer(dims: Dims, sizes: list) -> Optional[Dims]:
    """The target extents of a reshape of ``dims`` to ``sizes``: one ``-1`` is the element count over the product of
    the other sizes, computed after cancelling equal domain values (``view(B, -1)`` of ``(B, 6)`` is 6)."""
    if any(isinstance(size, (torch.dtype, Shape)) for size in sizes):
        return None
    unknown = [i for i, size in enumerate(sizes) if type(size) is int and size == -1]
    if len(unknown) > 1:
        return None
    if not unknown:
        return tuple(sizes)
    numerators = [d for d in dims if not isinstance(d, Opaque)]
    denominators = [s for i, s in enumerate(sizes) if i != unknown[0]]
    if len(numerators) < len(dims) or any(isinstance(s, Opaque) for s in denominators):
        inferred: Any = Opaque("the inferred size of a reshape of a value whose size is opaque")
    else:
        for value in list(denominators):
            if value in numerators:
                numerators.remove(value)
                denominators.remove(value)
        found = quotient(product(numerators), product(denominators))
        inferred = found if found is not None else Opaque(
            "the inferred size of a reshape whose element count the domain does not divide exactly")
    return tuple(inferred if i == unknown[0] else size for i, size in enumerate(sizes))


def _reshape(call: Call) -> Optional[list]:
    sizes = _sizes(call, 1, "shape", "size")
    found = _infer(call.args[0].dims, sizes)
    return None if found is None else [found]


def _flattened(dims: Dims, start: Any, end: Any) -> Optional[Dims]:
    if not dims:
        return (1,)
    s, e = _axis(start, len(dims)), _axis(end, len(dims))
    if s is None or e is None or s > e:
        return None
    return dims[:s] + (product(dims[s:e + 1]),) + dims[e + 1:]


def _flatten(call: Call) -> Optional[list]:
    if call.module is not None:
        start, end = call.module.start_dim, call.module.end_dim
    else:
        start, end = call.arg(1, "start_dim", 0), call.arg(2, "end_dim", -1)
    found = _flattened(call.args[0].dims, start, end)
    return None if found is None else [found]


def _unflatten(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    if call.module is not None:
        axis, sizes = call.module.dim, list(call.module.unflattened_size)
    else:
        axis, sizes = call.arg(1, "dim"), list(call.arg(2, "sizes", ()))
    d = _axis(axis, len(dims))
    if d is None:
        return None
    inner = _infer((dims[d],), sizes)
    return None if inner is None else [dims[:d] + inner + dims[d + 1:]]


def _permute(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    found = [_axis(d, len(dims)) for d in _sizes(call, 1, "dims")]
    order = [d for d in found if d is not None]
    if len(order) != len(found) or sorted(order) != list(range(len(dims))):
        return None
    return [tuple(dims[d] for d in order)]


def _transpose(call: Call) -> Optional[list]:
    dims = list(call.args[0].dims)
    if call.name == "t" and len(dims) < 2:
        return [tuple(dims)]
    first = call.arg(1, "dim0", 0 if call.name == "t" else None)
    second = call.arg(2, "dim1", 1 if call.name == "t" else None)
    a, b = _axis(first, len(dims)), _axis(second, len(dims))
    if a is None or b is None:
        return None
    dims[a], dims[b] = dims[b], dims[a]
    return [tuple(dims)]


def _movedim(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    source, target = call.arg(1, "source"), call.arg(2, "destination")
    source = list(source) if type(source) in (tuple, list) else [source]
    target = list(target) if type(target) in (tuple, list) else [target]
    src = [a for a in (_axis(d, len(dims)) for d in source) if a is not None]
    dst = [a for a in (_axis(d, len(dims)) for d in target) if a is not None]
    if len(src) != len(source) or len(dst) != len(target) or len(src) != len(dst) or len(set(src)) != len(src):
        return None
    order: List[Optional[int]] = [None] * len(dims)
    for s, d in zip(src, dst):
        order[d] = s
    rest = iter(d for d in range(len(dims)) if d not in src)
    return [tuple(dims[o if o is not None else next(rest)] for o in order)]


def _squeeze(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    named = call.arg(1, "dim")
    axes = range(len(dims)) if named is None else [_axis(d, len(dims)) for d in (
        named if type(named) in (tuple, list) else [named])]
    dropped = set()
    for axis in axes:
        if axis is None:
            return None
        one = evaluate(operator.eq, [dims[axis], 1])
        if isinstance(one, Opaque):
            return None  # whether the axis is dropped changes with the batch (B is 1 at batch 1)
        if one:
            dropped.add(axis)
    return [tuple(d for i, d in enumerate(dims) if i not in dropped)]


def _unsqueeze(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    axis = _axis(call.arg(1, "dim"), len(dims) + 1)
    return None if axis is None else [dims[:axis] + (1,) + dims[axis:]]


def _expand(call: Call) -> Optional[list]:
    dims = call.args[0].dims
    if call.name == "expand_as":
        return [call.args[1].dims]
    sizes = _sizes(call, 1, "size")
    lead = len(sizes) - len(dims)
    if lead < 0 or any(type(s) is int and s == -1 for s in sizes[:lead]) or any(isinstance(s, Shape) for s in sizes):
        return None  # a tensor given as a size is data (review round 15)
    return [tuple(dims[i - lead] if type(s) is int and s == -1 else s for i, s in enumerate(sizes))]


def _repeat(call: Call) -> Optional[list]:
    dims, counts = call.args[0].dims, _sizes(call, 1, "repeats", "dims")
    if call.name == "tile" and len(counts) < len(dims):
        counts = [1] * (len(dims) - len(counts)) + counts
    if len(counts) < len(dims) or any(isinstance(c, Shape) for c in counts):
        return None
    padded = (1,) * (len(counts) - len(dims)) + tuple(dims)
    return [tuple(evaluate(operator.mul, [d, c]) for d, c in zip(padded, counts))]


def _repeat_interleave(call: Call) -> Optional[list]:
    dims, repeats, axis = call.args[0].dims, call.arg(1, "repeats"), call.arg(2, "dim")
    if isinstance(repeats, Shape) or form(repeats) is None:
        return None
    if axis is None:
        return [(evaluate(operator.mul, [product(dims), repeats]),)]
    d = _axis(axis, len(dims))
    return None if d is None else [dims[:d] + (evaluate(operator.mul, [dims[d], repeats]),) + dims[d + 1:]]


def _as(call: Call) -> Optional[list]:
    other = call.arg(1, "other")
    return [other.dims] if isinstance(other, Shape) else None


def _transposed(last_two: bool) -> Callable[[Call], Optional[list]]:
    def rule(call: Call) -> Optional[list]:
        dims = call.args[0].dims
        if not last_two:
            return [tuple(reversed(dims))]
        return None if len(dims) < 2 else [dims[:-2] + (dims[-1], dims[-2])]
    return rule


LAYOUT_RULES: Dict[str, Callable[[Call], Optional[list]]] = {
    "view": _reshape, "reshape": _reshape, "flatten": _flatten, "Flatten": _flatten, "unflatten": _unflatten,
    "Unflatten": _unflatten, "permute": _permute, "transpose": _transpose, "swapaxes": _transpose,
    "swapdims": _transpose, "t": _transpose, "movedim": _movedim, "moveaxis": _movedim, "squeeze": _squeeze,
    "unsqueeze": _unsqueeze, "expand": _expand, "expand_as": _expand, "broadcast_to": _expand, "repeat": _repeat,
    "tile": _repeat, "repeat_interleave": _repeat_interleave, "view_as": _as, "reshape_as": _as,
    "T": _transposed(False), "H": _transposed(False), "mT": _transposed(True), "mH": _transposed(True),
    **SELECTION_RULES,
}
"""The layout kernels by name: functions and methods of torch's namespaces, ``nn.Flatten`` and ``nn.Unflatten``,
indexing (``getitem``) and the transposing attributes."""

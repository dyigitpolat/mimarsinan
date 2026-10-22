"""The batch rules of torch's tensor kernels other than the layouts (``kernels.py`` declares which kernel has which,
and states the lemma; review round 14): same shape, broadcast, reduction, leading axes, matrix products, ``einsum``,
the factories, ``arange``, ``tensor`` and the data-dependent kernels (``module_rules.py``: attention, recurrent and
transformer modules, ``GLU``). Every rule reads its call bound to the kernel's signature (``binding.py``, review round
16): ``tensors_of`` lists the tensor operands in the signature's order.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Opaque, form
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.operations import evaluate
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Call, Shape, free

__all__ = [
    "DATA_DEPENDENT", "Rule", "arange", "bilinear", "broadcast", "data", "einsum", "factory", "hinted", "leading",
    "matmul", "padded", "pointwise", "reduce", "same", "spatial", "tensor", "tensors_of",
]

Rule = Callable[[Call], Optional[list]]

DATA_DEPENDENT = Opaque("a data-dependent size")
"""The extent a rule states as torch's data-dependent size (``nonzero``'s row count)."""


def tensors_of(call: Call) -> List[Shape]:
    """The call's tensor operands in its operands' order (the signature's, ``binding.py``), through containers."""
    found: List[Shape] = []

    def walk(value: Any) -> None:
        if isinstance(value, Shape):
            found.append(value)
        elif type(value) in (tuple, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk((call.args, dict(call.kwargs)))
    return found


def _holds_tensor(value: Any) -> bool:
    if isinstance(value, Shape):
        return True
    if type(value) in (tuple, list):
        return any(_holds_tensor(item) for item in value)
    return isinstance(value, dict) and any(_holds_tensor(item) for item in value.values())


def _scalars_free(call: Call, operands: Tuple[str, ...] = ("input",)) -> bool:
    """Every operand but the kernel's tensor operands (``operands``: their names, in positional order) is batch-free
    and holds no tensor: a tensor where the kernel reads a number (a pad amount, an output size, a class count, a
    0-dim tensor) is data the domain does not compute (review round 15)."""
    rest = [arg for index, arg in enumerate(call.args) if not (index < len(operands) and isinstance(arg, Shape))]
    rest += [value for key, value in call.kwargs.items() if not (key in operands and isinstance(value, Shape))]
    return free(rest) and not _holds_tensor(rest)


def hinted(extent: Any) -> Any:
    """Torch's extent at the hint batch; an extent the pass made opaque (the seeded traces', a sample) stays one."""
    if isinstance(extent, Opaque):
        return extent
    return int(extent.node.require_hint()) if isinstance(extent, torch.SymInt) else int(extent)


def same(call: Call) -> Optional[list]:
    tensors = tensors_of(call)
    count = len(call.out) if call.out is not None else 1
    return [tensors[0].dims] * count if tensors else None


def broadcast(shapes: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    rank = max(len(s) for s in shapes)
    out = []
    for axis in range(rank):
        column = [s[len(s) - rank + axis] for s in shapes if len(s) - rank + axis >= 0]
        wide = [d for d in column if not (type(d) is int and d == 1)]
        if not wide:
            out.append(1)
        elif all(d == wide[0] and not isinstance(d, Opaque) for d in wide):
            out.append(wide[0])
        elif len(wide) == 1:
            out.append(wide[0])  # an opaque extent broadcast against ones
        else:
            out.append(Opaque("a broadcast of sizes the domain does not prove equal"))
    return tuple(out)


def pointwise(call: Call) -> Optional[list]:
    tensors = tensors_of(call)
    return [broadcast([t.dims for t in tensors])] if tensors else None


def _reduced(dims: Tuple[Any, ...], axes: Any, keep: Any) -> Optional[Tuple[Any, ...]]:
    rank = len(dims)
    if axes is None or (type(axes) in (tuple, list) and not axes):
        chosen = set(range(rank))
    else:
        listed = list(axes) if type(axes) in (tuple, list) else [axes]
        if not all(type(a) is int and -max(rank, 1) <= a < max(rank, 1) for a in listed):
            return None
        chosen = {a % rank if rank else 0 for a in listed}
    if type(keep) is not bool:
        return None
    return tuple((1 if keep else None) if i in chosen else d for i, d in enumerate(dims) if keep or i not in chosen)


_REDUCE_POSITIONS = {"std": (1, 3), "var": (1, 3), "norm": (2, 3), "count_nonzero": (1, None)}


def reduce(call: Call) -> Optional[list]:
    tensors = tensors_of(call)
    if len(tensors) != 1:
        return pointwise(call) if call.name in ("max", "min") else None  # max(a, b) is pointwise
    at_dim, at_keep = _REDUCE_POSITIONS.get(call.name, (1, 2))
    axes = call.arg(at_dim, "dim")
    if type(axes) is bool:  # std(x, unbiased)
        axes = None
    keep = call.arg(at_keep, "keepdim", False) if at_keep is not None else False
    found = _reduced(tensors[0].dims, axes, keep)
    count = len(call.out) if call.out is not None else 1
    return None if found is None else [found] * count


def leading(k: Optional[int], tail_from: Callable[[Call], Optional[int]] = lambda call: None,
            operands: Tuple[str, ...] = ("input",)) -> Rule:
    """Rule ``LEADING(k)``: ``k`` (or ``tail_from(call)``) trailing axes of the first operand are consumed; the
    kernel's tensor operands are ``operands`` (``linear``: input, weight, bias), every other operand a number."""
    def rule(call: Call) -> Optional[list]:
        tensors = tensors_of(call)
        consumed = k if k is not None else tail_from(call)
        if not tensors or consumed is None or call.out is None:
            return None
        dims = tensors[0].dims
        keep = len(dims) - consumed
        if keep < 0 or not free(dims[keep:]) or not all(free(t.dims) for t in tensors[1:]) or \
                not _scalars_free(call, operands):
            return None
        return [dims[:keep] + tuple(hinted(d) for d in out[keep:]) if len(out) >= keep else None for out in call.out]
    return rule


def bilinear(call: Call) -> Optional[list]:
    """``F.bilinear`` and ``nn.Bilinear``: ``(*, in1)`` and ``(*, in2)`` give ``(*, out)``; the leading axes of the
    first input pass through (the second's are the same where the kernel returns), the weight ``(out, in1, in2)`` and
    bias are batch-free: ``out`` is the module's ``out_features`` or the weight's first extent (review round 15: the
    operands state it, as torch's FakeTensor run of ``_trilinear`` raises)."""
    tensors = tensors_of(call)
    if len(tensors) < 2 or call.out is None or not free(tensors[0].dims[-1:]) or not free(tensors[1].dims[-1:]):
        return None
    if not all(free(t.dims) for t in tensors[2:]) or not _scalars_free(call, ("input1", "input2", "weight", "bias")):
        return None
    out = getattr(call.module, "out_features", None) if call.module is not None else (
        tensors[2].dims[0] if len(tensors) > 2 and tensors[2].dims else None)
    return None if type(out) is not int else [tensors[0].dims[:-1] + (out,) for _ in call.out]


def spatial(call: Call) -> Optional[int]:
    tensors = tensors_of(call)
    return len(tensors[0].dims) - 2 if tensors else None


def padded(call: Call) -> Optional[int]:
    """The padded axes: a pad of plain integers (a tensor amount is data, review round 15)."""
    pad = call.module.padding if call.module is not None else call.arg(1, "pad")
    return len(pad) // 2 if type(pad) in (tuple, list) and all(type(p) is int for p in pad) else None


def _matrix(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    if not a or not b:
        return None
    if len(a) == 1 and len(b) == 1:
        return ()
    if len(a) == 1:
        return b[:-2] + b[-1:]
    if len(b) == 1:
        return a[:-1]
    batch = broadcast([a[:-2], b[:-2]]) if (a[:-2] or b[:-2]) else ()
    return batch + (a[-2], b[-1])


def matmul(call: Call) -> Optional[list]:
    tensors = tensors_of(call)
    if call.name in ("addmm", "baddbmm", "addbmm", "addmv"):
        tensors = tensors[1:]
    if len(tensors) != 2:
        return None
    found = _matrix(tensors[0].dims, tensors[1].dims)
    return None if found is None else [found]


def einsum(call: Call) -> Optional[list]:
    equation = call.args[0] if call.args else None
    operands = tensors_of(call)
    if type(equation) is not str or "->" not in equation or "." in equation:
        return None
    inputs, output = equation.replace(" ", "").split("->")
    terms = inputs.split(",")
    if len(terms) != len(operands) or any(len(t) != len(o.dims) for t, o in zip(terms, operands)):
        return None
    extents: Dict[str, list] = {}
    for term, operand in zip(terms, operands):
        for letter, d in zip(term, operand.dims):
            extents.setdefault(letter, []).append((d,))
    if not all(letter in extents for letter in output):
        return None
    return [tuple(broadcast(extents[letter])[0] for letter in output)]  # a letter's extents broadcast (review round 16)


def factory(call: Call) -> Optional[list]:
    start = 1 if call.name.startswith("new_") else 0
    if call.name == "full" or call.name == "new_full":
        sizes = call.arg(start, "size")
        sizes = list(sizes) if type(sizes) in (tuple, list, torch.Size) else None
    else:
        rest = list(call.args[start:])
        sizes = (list(call.kwargs["size"]) if "size" in call.kwargs else
                 list(rest[0]) if len(rest) == 1 and type(rest[0]) in (tuple, list, torch.Size) else rest)
    if sizes is None or not all(form(s) is not None or isinstance(s, Opaque) for s in sizes):
        return None
    return [tuple(sizes)]


def arange(call: Call) -> Optional[list]:
    named = [call.kwargs.get(key) for key in ("start", "end", "step")]
    positional = list(call.args)
    if len(positional) == 1 and named[0] is None:
        start, end, step = 0, positional[0], named[2] if named[2] is not None else 1
    elif len(positional) in (2, 3):
        start, end, step = positional[0], positional[1], positional[2] if len(positional) == 3 else 1
    else:
        return None
    dtype = call.kwargs.get("dtype")
    floating = isinstance(dtype, torch.dtype) and (dtype.is_floating_point or dtype.is_complex)
    if type(step) is not int or step != 1 or form(start) is None or form(end) is None or floating:
        return None  # a floating arange computes its length in doubles: only an integer one is stated
    span = evaluate(operator.sub, [end, start])
    return [(evaluate(torch.sym_max, [span, 0]),)]


def tensor(call: Call) -> Optional[list]:
    def shape(value: Any) -> Optional[Tuple[int, ...]]:
        if type(value) not in (tuple, list):
            return () if not isinstance(value, Shape) else None
        inner = [shape(item) for item in value]
        if any(i is None for i in inner) or len(set(inner)) > 1:
            return None
        return (len(value), *((inner[0] or ()) if inner else ()))
    found = shape(call.args[0]) if call.args else None
    return None if found is None else [found]


def data(call: Call) -> Optional[list]:
    tensors = tensors_of(call)
    if not tensors or call.name == "nonzero" and call.kwargs.get("as_tuple"):
        return None
    return [(DATA_DEPENDENT,) + ((len(tensors[0].dims),) if call.name in ("nonzero", "argwhere") else ())]

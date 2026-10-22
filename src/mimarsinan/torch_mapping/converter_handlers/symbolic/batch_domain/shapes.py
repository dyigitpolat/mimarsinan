"""Tensors in the affine batch domain: a value's extents (``Shape``), a kernel call (``Call``) and the extent
arithmetic the declared batch rules share (review round 14).

A tensor's domain value is its tuple of extents, each a batch-free ``int``, an ``Affine`` form or an ``Opaque`` value.
A shape query reads them (``query``): ``size``, ``shape``, ``dim``, ``numel``; the element count is their product in the
domain, so it is opaque where a product of two forms or an opaque extent is.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Affine, Opaque, affine, form
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.operations import evaluate

__all__ = ["Call", "Dims", "Shape", "axis_of", "free", "product", "query", "quotient", "sizes_of"]

Dims = Tuple[Any, ...]


@dataclass(frozen=True)
class Shape:
    """A tensor in the domain: its extents, each an ``int``, an ``Affine`` form or an ``Opaque`` value."""

    dims: Dims


@dataclass(frozen=True)
class Call:
    """A kernel call in the domain: its name (a function's or method's, or a module's class), its operands as domain
    values (tensors as ``Shape``, a method's receiver first), torch's output tensors' extents (``out``; ``None`` when
    torch's run raised) and the called module."""

    name: str
    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    out: Optional[Tuple[Tuple[Any, ...], ...]] = None
    module: Any = None

    def arg(self, index: int, name: str, default: Any = None) -> Any:
        if name in self.kwargs:
            return self.kwargs[name]
        return self.args[index] if len(self.args) > index else default


def _leaves(value: Any) -> list:
    if isinstance(value, Shape):
        return list(value.dims)
    if type(value) in (tuple, list, torch.Size):
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if type(value) is slice:
        return _leaves((value.start, value.stop, value.step))
    return [value]


def free(value: Any) -> bool:
    """No extent or scalar of ``value`` depends on the batch (no ``Affine`` or ``Opaque`` leaf)."""
    return not any(isinstance(leaf, (Affine, Opaque)) for leaf in _leaves(value))


def product(values: Sequence[Any]) -> Any:
    total: Any = 1
    for value in values:
        total = evaluate(operator.mul, [total, value])
    return total


def quotient(numerator: Any, denominator: Any) -> Optional[Any]:
    """``numerator / denominator`` when it is a batch-free integer or an affine form at every batch (an exact
    division in the domain); ``None`` otherwise."""
    n, d = form(numerator), form(denominator)
    if n is None or d is None:
        return None
    if d[0] == 0:
        k = d[1]
        return None if k <= 0 or n[0] % k or n[1] % k else affine(n[0] // k, n[1] // k)
    if n[0] % d[0]:
        return None
    k = n[0] // d[0]
    return k if n[1] == k * d[1] else None


def axis_of(index: Any, rank: int) -> Optional[int]:
    if type(index) is not int or not -rank <= index < max(rank, 1):
        return None
    return index % rank if rank else 0


def sizes_of(call: Call, start: int, *names: str) -> list:
    """The size arguments ``*sizes`` or one sequence (or a keyword in ``names``) after the receiver."""
    for name in names:
        if name in call.kwargs:
            value = call.kwargs[name]
            return list(value) if type(value) in (tuple, list, torch.Size) else [value]
    rest = list(call.args[start:])
    return list(rest[0]) if len(rest) == 1 and type(rest[0]) in (tuple, list, torch.Size) else rest


_RANKS = ("dim", "ndimension", "ndim")


def query(name: str, dims: Dims, rest: Sequence[Any]) -> Any:
    """The domain value of the shape query ``name`` (``rest``: its operands after the tensor) of a tensor of extents
    ``dims``; ``NotImplemented`` for a query this rule does not answer."""
    if name in ("size", "shape") and not rest:
        return torch.Size(dims) if free(Shape(dims)) else tuple(dims)
    if name == "size" and len(rest) == 1 and type(rest[0]) is int and -len(dims) <= rest[0] < len(dims):
        return dims[rest[0]]
    if name in _RANKS and not rest:
        return len(dims)
    if name in ("numel", "nelement") and not rest:
        return product(dims)
    return NotImplemented

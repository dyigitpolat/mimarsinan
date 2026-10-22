"""Torch-side values of the symbolic-batch pass (review round 14): the vehicle torch's kernels run on, built from the
affine batch domain's values, never read back for a fold.

The pass computes every batch-dependent size in the domain (``batch_domain``). Torch's kernels still run, on
FakeTensors, to execute the program and to cross-check the declared batch rules; their operands are built here from
the domain: a batch-free value as itself, an affine form ``a*B + c`` as the SymInt ``a*s0 + c`` of the batch symbol
``s0``, and an opaque value as a fresh backed symbol of its decision (``DecisionValues.symbols``), its hint the native
value at the hint batch. A value the pass does not compute is the seeded traces' structure (``from_traces``): its rank
and dtype, every extent a fresh symbol of its decision.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch.fx as fx

import sympy
import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils import _pytree as pytree

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Affine, Opaque
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import DATA_DEPENDENT, Decision
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import BATCH, SampledTensor, Trace, sample

__all__ = ["DecisionValues", "SYMBOLIC_TYPES", "SymbolEnv", "fp64", "fp64_like", "hint_of", "is_symbolic_size"]

SYMBOLIC_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)
_NUMBERS = (int, float, bool)


def is_symbolic_size(leaf: Any) -> bool:
    return isinstance(leaf, SYMBOLIC_TYPES) and not leaf.node.expr.is_number


def fp64(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype.is_floating_point else dtype


def fp64_like(tensor: torch.Tensor) -> torch.Tensor:
    """A FakeTensor (under the active mode) of ``tensor``'s shape in fp64, ``requires_grad`` kept: a module's path
    can depend on it (``nn.MultiheadAttention``'s fast path)."""
    return torch.empty(tensor.shape, dtype=fp64(tensor.dtype), device=tensor.device,
                       requires_grad=tensor.requires_grad and fp64(tensor.dtype).is_floating_point)


def hint_of(value: Any) -> Any:
    """``value`` with every symbolic scalar its hint (its native value at the hint batch; ``None`` for a
    data-dependent one, which has none), through containers."""
    def leaf(item: Any) -> Any:
        if isinstance(item, SYMBOLIC_TYPES):
            kind = {torch.SymInt: int, torch.SymFloat: float, torch.SymBool: bool}[type(item)]
            return kind(item.node.require_hint()) if item.node.has_hint() else None
        return item
    return pytree.tree_map(leaf, value)


class SymbolEnv(ShapeEnv):
    """The pass's ShapeEnv: torch's kernels run under it with guards suppressed (a guard is evaluated at the hint and
    never installed); its fresh symbols stand for opaque values."""

    def call(self, fn: Callable[[], Any]) -> Tuple[Any, Optional[BaseException]]:
        """``fn()`` with guards suppressed: its value or exception."""
        ShapeEnv._inner_evaluate_expr.cache_clear()
        with self.suppress_guards():
            return guarded(fn)

    def fresh(self, hint: Any, name: str) -> Any:
        """A new backed symbol whose hint is ``hint`` (an ``int``, ``float`` or ``bool``)."""
        source = ConstantSource(name)
        if type(hint) is float:
            return self.create_symfloatnode(self.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC),
                                            hint=hint, source=source)
        symbol = self.create_symbol(int(hint), source, DimDynamic.DYNAMIC, positive=None,
                                    do_not_specialize_zero_one=True)
        if type(hint) is bool:
            return self.create_symboolnode(cast(sympy.Expr, sympy.Eq(symbol, 1)))
        return self.create_symintnode(symbol, hint=int(hint), source=source)


class DecisionValues:
    """The decision symbols of one run of the pass, and the torch-side values it builds (under the active fake mode)."""

    def __init__(self, env: SymbolEnv, batch: torch.SymInt, device: torch.device):
        self.env, self.batch, self.device = env, batch, device
        self.symbols: Dict[sympy.Symbol, Decision] = {}

    def fresh(self, hint: Any, decision: Decision) -> Any:
        """A fresh symbol of ``decision`` (``hint``'s type; a non-number ``hint`` is itself)."""
        if type(hint) not in _NUMBERS:
            return hint
        value = self.env.fresh(hint, f"decision_{decision.node.name}_{len(self.symbols)}")
        for symbol in value.node.expr.free_symbols:
            self.symbols.setdefault(symbol, decision)
        return value

    def extent(self, value: Any, hint: int) -> Any:
        """The torch-side extent of a domain extent (``hint``: its native value at the hint batch)."""
        if isinstance(value, Affine):
            return self.batch * value.a + value.c
        if isinstance(value, Opaque):
            return self.fresh(hint, value.origin) if isinstance(value.origin, Decision) else hint
        return value

    def scalar(self, value: Any, hint: Any) -> Any:
        """The torch-side value of a domain scalar or container (``hint``: the native value at the hint batch)."""
        if isinstance(value, (Affine, Opaque)):
            return self.extent(value, hint)
        if type(value) in (tuple, list, torch.Size) and type(hint) in (tuple, list, torch.Size) \
                and len(value) == len(hint):
            items = [self.scalar(v, h) for v, h in zip(value, hint)]
            return type(hint)(items)
        return value

    def tensor(self, like: torch.Tensor, dims: Tuple[Any, ...], hints: Tuple[int, ...],
               kept: Tuple[bool, ...]) -> torch.Tensor:
        """A FakeTensor like ``like`` whose extents are the domain's ``dims`` (``kept``: per axis, torch's own extent
        already is it); ``like`` itself when every extent is its own."""
        built = [own if keep else self.extent(d, h) for d, h, own, keep in zip(dims, hints, like.shape, kept)]
        same = len(built) == like.dim() and all(b is o or (type(b) is int and type(o) is int and b == o) or (
            isinstance(b, torch.SymInt) and isinstance(o, torch.SymInt) and b.node.expr == o.node.expr)
            for b, o in zip(built, like.shape))
        return like if same else torch.empty(built, dtype=like.dtype, device=like.device)

    def from_traces(self, sampled, decision: Optional[Decision]) -> Any:
        """The traces' structure (``trace_values.sample``) as a torch-side value: a tensor of the traced rank and dtype
        (fp64 for a floating one) with every extent a fresh symbol of ``decision`` (B on a batch axis, the traced extent
        on a static one, when ``decision`` is ``None``: a restored reshape), a scalar a fresh symbol; ``None`` without
        traces."""
        if sampled is None:
            return None
        built = []
        for leaf in sampled[1]:
            if isinstance(leaf, SampledTensor):
                if decision is None:
                    dims = [self.batch if axis is BATCH else axis for axis in leaf.axes]
                else:
                    dims = [self.fresh(first, decision) for first in leaf.first]
                built.append(torch.empty(dims, dtype=fp64(leaf.dtype), device=self.device))
            else:
                hint = leaf[1] if type(leaf) is tuple else leaf
                built.append(hint if decision is None else self.fresh(hint, decision))
        return pytree.tree_unflatten(built, sampled[0])

    def stand_in(self, node: fx.Node, traces: Tuple[Trace, ...]) -> Tuple[Any, Dict[int, Tuple[Any, ...]]]:
        """Where torch's FakeTensor run of ``node`` raised: its tensors at batch 1 as the seeded trace shows them (the
        program's vehicle; no rule reads their extents, each ``Opaque`` to it, review round 15), and their shapes in
        every trace (the cross-check); ``(None, {})`` without one. [corrected in review round 15: "a stand-in a rule
        reads batch-free extents of"]"""
        sampled = sample(node, traces)
        if sampled is None or not all(isinstance(leaf, SampledTensor) for leaf in sampled[1]):
            return None, {}
        stand_in = [torch.empty(leaf.first, dtype=fp64(leaf.dtype), device=self.device) for leaf in sampled[1]]
        runs = {trace.batch: tuple(getattr(leaf, "shape", ()) for leaf in trace.values[node][1]) for trace in traces}
        return pytree.tree_unflatten(stand_in, sampled[0]), runs

    def kept(self, dims: Tuple[Any, ...], leaf: torch.Tensor) -> Tuple[bool, ...]:
        """Per axis, torch's own extent is the domain's: an opaque one already a symbol of the same decision, or a
        data-dependent one."""
        kept = []
        for d, got in zip(dims, leaf.shape):
            symbols = got.node.expr.free_symbols if isinstance(got, torch.SymInt) else set()
            same_decision = isinstance(d, Opaque) and bool(symbols) and all(
                self.symbols.get(s) is d.origin for s in symbols)
            kept.append(same_decision or (isinstance(d, Opaque) and d.origin is DATA_DEPENDENT))
        return tuple(kept)

    def decisions_in(self, value: Any) -> set:
        """The decisions whose symbols ``value``'s scalars or extents hold."""
        found = set()
        for leaf in pytree.tree_leaves(value):
            extents = leaf.shape if isinstance(leaf, torch.Tensor) else [leaf]
            for extent in extents:
                if isinstance(extent, SYMBOLIC_TYPES):
                    found |= {self.symbols[s] for s in extent.node.expr.free_symbols if s in self.symbols}
        return found

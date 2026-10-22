"""Which per-sample shapes a batch decision reaches (review rounds 8 and 9).

A batch decision (``symbolic_batch.Decision``) is a value the affine batch domain does not compute (an opaque value).
Every scalar or extent computed from it is opaque with the same decision (review round 14: ``batch_domain.Opaque``'s
origin), so the pass itself computes which values a decision reaches: a value whose domain shape (a tensor) or domain
value (a scalar) holds an opaque leaf of a decision is shape-decided by it, and a value a decision only scales
(``y * g``) is not. Where the link breaks (a value taken from the seeded traces), a value is shape-decided
conservatively when an input is.

Pure reshapes (``RESHAPE_METHODS``, ``torch.flatten``, ``torch.reshape``, ``torch.squeeze``, ``torch.unsqueeze``,
``nn.Flatten``) keep the row-major order and the element count of their input. One whose size arguments are not decided
restores a per-sample shape independent of the decision when it reads only its input's count (``view``, ``reshape``,
a flatten of every per-sample axis) and that count is not decided, or when it reads its input's axes (a partial
flatten, ``squeeze``, ``unsqueeze``) and their shape is not decided: ``torch.flatten(z.view(B, 6 // g, g), 1)``
restores ``(6,)`` whatever ``g`` is. Any other operator's count is decided wherever its shape is.

The count a sized reshape keeps is its input's TOTAL count; it is a per-sample count only where axis 0 is the batch.
A sized reshape a decision reaches whose axis 0 the domain does not prove to be B (``y.view(B // (1 + 8 * (B > 8)),
-1)``: 1 row from batch 9) has its axis 0 decided (``batch_axis``), and so has every pure reshape that keeps that axis:
a flatten of every per-sample axis of it restores nothing (its per-sample count is the total over the decided rows), and
a query of its axis 0 is no batch-extent query (``LayoutMixin._unrestored_consumer``).
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, List, Mapping, Optional, Set, TypeVar

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.torch_mapping.converter_handlers.refusals import BatchDecision

__all__ = ["RESHAPE_METHODS", "ShapeDecisions", "reshape_kind", "reshape_sizes"]

RESHAPE_METHODS = frozenset({"view", "reshape", "flatten", "squeeze", "unsqueeze"})
_RESHAPE_FUNCTIONS = {id(fn): fn.__name__ for fn in (torch.flatten, torch.reshape, torch.squeeze, torch.unsqueeze)}

D = TypeVar("D", bound=BatchDecision)

COUNT = "count"
"""A pure reshape whose per-sample result shape is fixed by its size arguments and its input's per-sample count."""

AXES = "axes"
"""A pure reshape whose per-sample result shape is fixed by its arguments and its input's per-sample axes."""


def _flatten_kind(node: fx.Node, start, end) -> str:
    source = node.args[0] if node.args else None
    shape = getattr(source.meta.get("tensor_meta"), "shape", None) if isinstance(source, fx.Node) else None
    rank = len(shape) if shape is not None else None
    if rank is None or type(start) is not int or type(end) is not int or rank < 2:
        return AXES
    return COUNT if start % rank == 1 and end % rank == rank - 1 else AXES


def reshape_kind(node: fx.Node, modules: Mapping[str, nn.Module]) -> Optional[str]:
    """``COUNT`` or ``AXES`` for a pure reshape, ``None`` for any other node."""
    if node.op == "call_method" and node.target in RESHAPE_METHODS:
        name = str(node.target)
    elif node.op == "call_function" and id(node.target) in _RESHAPE_FUNCTIONS:
        name = _RESHAPE_FUNCTIONS[id(node.target)]
    elif node.op == "call_module" and type(modules.get(str(node.target))) is nn.Flatten:
        module = modules[str(node.target)]
        return _flatten_kind(node, module.start_dim, module.end_dim)
    else:
        return None
    if name in ("view", "reshape"):
        return COUNT
    if name != "flatten":
        return AXES
    rest = list(node.args[1:3])
    start = rest[0] if rest else node.kwargs.get("start_dim", 0)
    end = rest[1] if len(rest) > 1 else node.kwargs.get("end_dim", -1)
    return _flatten_kind(node, start, end)


def reshape_sizes(node: fx.Node) -> Optional[list]:
    """The size arguments of a sized reshape (``view``, ``reshape``, ``torch.reshape``); ``None`` for any other node."""
    if node.op == "call_method" and node.target in ("view", "reshape"):
        sizes = list(node.args[1:])
        sizes = list(sizes[0]) if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) else sizes
    elif node.op == "call_function" and node.target is torch.reshape:
        shape = node.args[1] if len(node.args) > 1 else node.kwargs.get("shape")
        sizes = list(shape) if isinstance(shape, (tuple, list)) else []
    else:
        return None
    return sizes if sizes and not any(isinstance(size, torch.dtype) for size in sizes) else None


class ShapeDecisions(Generic[D]):
    """Per node, the decision its per-sample shape (or scalar value) depends on, filled in traced order."""

    def __init__(self, modules: Mapping[str, nn.Module], order: Mapping[fx.Node, int]):
        self._modules = modules
        self._order = order
        self.decided: Dict[fx.Node, D] = {}
        self.batch_axis: Dict[fx.Node, D] = {}
        self._count: Dict[fx.Node, D] = {}
        self._visible: Set[fx.Node] = set()

    def _first(self, found: Iterable[Optional[D]]) -> Optional[D]:
        present = [decision for decision in found if decision is not None]
        return min(present, key=lambda decision: self._order[decision.node]) if present else None

    def record(self, node: fx.Node, own: Optional[D], symbol_decisions: Iterable[D], from_traces: bool,
               keeps_batch_axis: Callable[[], bool]) -> bool:
        """Record ``node`` (``own``: its own decision; ``symbol_decisions``: the decisions whose opaque values its
        domain value holds; ``from_traces``: its value was taken from the traces; ``keeps_batch_axis``: whether a sized
        reshape's axis 0 is provably B, asked only of a decided one). True when
        ``node`` is a pure reshape that restores a per-sample shape although its symbolic value holds decision symbols
        (the pass then takes its value from the traces)."""
        symbols: List[D] = list(symbol_decisions)
        found: List[Optional[D]]
        kind = reshape_kind(node, self._modules)
        source = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        pure = kind is not None and source is not None and not from_traces
        sized = pure and reshape_sizes(node) is not None
        if pure:
            assert source is not None
            found = [own] + [self.decided.get(arg) for arg in node.all_input_nodes if arg is not source]
            found.append(self._count.get(source) if kind == COUNT else self.decided.get(source))
            if not sized:  # it keeps its input's axis 0
                found.append(self.batch_axis.get(source))
            count = self._count.get(source)
        else:
            inherited = [self.decided.get(arg) for arg in node.all_input_nodes
                         if from_traces or arg not in self._visible]
            found = [own, *symbols, *inherited]
            count = None
        decision = self._first(found)
        if decision is not None:
            self.decided[node] = decision
            if decision in symbols:  # its value carries the decision's symbols: downstream values show it too
                self._visible.add(node)
            if sized:
                axis = None if keeps_batch_axis() else decision
            else:
                axis = self.batch_axis.get(source) if pure and source is not None else None
            if axis is not None:
                self.batch_axis[node] = axis
        count = count if pure else decision
        if count is not None:
            self._count[node] = count
        return kind is not None and decision is None and bool(symbols)

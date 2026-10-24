"""Rule (c) of a body capture (``body_capture``): the capture at a batch B, every extent static, is the same program as
the capture with batch symbols evaluated at that B (batch 1, and every batch at which a symbol's form is 0 or 1).

Dynamo assumes a batch symbol is not 0 or 1 (its 0/1 specialization), and size-oblivious reasoning in torch's own
meta functions does the same, without a guard: ``t.squeeze(0).ndim``, ``t.squeeze().dim()``, ``t.shape[0] == 1`` and
``t.t().is_contiguous()`` capture a graph that is wrong where the symbol is 1 alone (batch 1 for ``B``, batch 2 for
``B - 1``). The two captures are compared node for node in traced order: an integer-valued node of the first (an
extent, a value computed from extents) is its value at B, and must be the literal the second holds in its place;
every other node has the same kind, target and operands; the graph inputs correspond by what they read; the return
values have the same structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import sympy
import torch
import torch.fx as fx

from mimarsinan.torch_mapping.fx_rewrites.body_output import CapturedBody, GraphOutput, OperandOutput

__all__ = ["differs_at_batch"]

_SYMBOLIC = (torch.SymInt, torch.SymFloat, torch.SymBool)
_INPUT_OUTPUT = ("placeholder", "output")


def _key(binding: tuple) -> tuple:
    return ("constant", id(binding[1])) if binding[0] == "constant" else binding


def _name(target: Any) -> str:
    return target if isinstance(target, str) else str(getattr(target, "__name__", target))


class _Comparison:
    def __init__(self, at_batch: Dict[sympy.Symbol, sympy.Expr]):
        self.at_batch = at_batch
        self.env: Dict[fx.Node, fx.Node] = {}

    def valued(self, node: fx.Node, symbols: bool) -> bool:
        """An integer-, float- or bool-valued node whose value the batch symbols determine (``symbols``), or a Python
        number (the second capture's)."""
        value = node.meta.get("example_value")
        if isinstance(value, _SYMBOLIC):
            return symbols and value.node.expr.free_symbols <= set(self.at_batch)
        return type(value) in (int, float, bool)

    def value(self, node: fx.Node) -> Any:
        value = node.meta.get("example_value")
        if not isinstance(value, _SYMBOLIC):
            return value
        found = value.node.expr.xreplace(self.at_batch)
        kind = {torch.SymInt: int, torch.SymFloat: float, torch.SymBool: bool}[type(value)]
        return kind(found) if found.is_number or found in (sympy.true, sympy.false) else ("unresolved", str(found))

    def same(self, a: Any, b: Any) -> bool:
        if isinstance(a, fx.Node):
            if a in self.env:
                return self.env[a] is b
            return self.valued(a, True) and not isinstance(b, fx.Node) and _equal(self.value(a), b)
        if isinstance(b, fx.Node):
            return False
        if isinstance(a, (tuple, list)):  # FX holds a node's list operand as an ``immutable_list``, a list subclass
            return isinstance(b, (tuple, list)) and len(a) == len(b) and all(map(self.same, a, b))
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(self.same(a[k], b[k]) for k in a)
        if isinstance(a, slice):
            return isinstance(b, slice) and all(self.same(x, y) for x, y in (
                (a.start, b.start), (a.stop, b.stop), (a.step, b.step)))
        return _equal(a, b)

    def same_output(self, a: Any, b: Any, first: List[Any], second: List[Any]) -> bool:
        if isinstance(a, GraphOutput) or isinstance(b, GraphOutput):
            left = first[a.index] if isinstance(a, GraphOutput) else a
            right = second[b.index] if isinstance(b, GraphOutput) else b
            return self.same(left, right)
        if isinstance(a, OperandOutput) or isinstance(b, OperandOutput):
            return a == b
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return a is b
        if type(a) in (tuple, list):
            return type(b) is type(a) and len(a) == len(b) and all(
                self.same_output(x, y, first, second) for x, y in zip(a, b))
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(
                self.same_output(a[k], b[k], first, second) for k in a)
        return _equal(a, b)


def _equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    return a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))


def differs_at_batch(body: CapturedBody, other: CapturedBody, at_batch: Dict[sympy.Symbol, sympy.Expr]
                     ) -> Optional[str]:
    """How ``other`` (captured at a batch B, every extent static) differs from ``body`` with each batch symbol at its
    value at that B (``at_batch``); ``None`` when they are the same program."""
    compare = _Comparison(at_batch)
    second = {_key(b): n for n, b in zip(other.placeholders(), other.bindings)}
    seen = set()
    for node, binding in zip(body.placeholders(), body.bindings):
        if binding[0] in ("size", "stride", "offset"):
            continue
        key = _key(binding)
        if key not in second:
            return f"the input {binding[:2]} is not read at that batch"
        compare.env[node] = second[key]
        seen.add(key)
    if set(second) - seen:
        return "that batch reads an input the other batches do not"
    first_nodes = [n for n in body.gm.graph.nodes if n.op not in _INPUT_OUTPUT and not compare.valued(n, True)]
    second_nodes = [n for n in other.gm.graph.nodes if n.op not in _INPUT_OUTPUT and not compare.valued(n, False)]
    if len(first_nodes) != len(second_nodes):
        return f"{len(first_nodes)} nodes at the other batches, {len(second_nodes)} at that batch"
    for a, b in zip(first_nodes, second_nodes):
        if a.op != b.op or a.target != b.target:
            return (f"node {a.name} is {a.op} {_name(a.target)} at the other batches, {b.op} {_name(b.target)} at "
                    "that batch")
        if not (compare.same(a.args, b.args) and compare.same(a.kwargs, b.kwargs)):
            return f"node {a.name} ({_name(a.target)}) has other operands at that batch"
        compare.env[a] = b
    if not compare.same_output(body.output, other.output, body.outputs(), other.outputs()):
        return "its return value differs at that batch"
    return None

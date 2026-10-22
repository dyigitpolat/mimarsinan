"""The symbolic-batch pass's step for a tensor kernel (review round 14): its declared batch rule
(``batch_domain.kernels``) run on its operands' domain values, accepted where torch's run agrees
(``batch_domain.acceptance``); a kernel with no batch-dependent operand is batch-free, torch computing it from integers
as natively.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import Opaque
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.acceptance import (
    accept,
    data_dependent,
    hint_int,
    unbacked,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.binding import bound
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernels import rule_of, same
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.layouts import LAYOUT_RULES
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Call, Shape, free
from mimarsinan.torch_mapping.converter_handlers.symbolic.decision_values import (
    SYMBOLIC_TYPES,
    DecisionValues,
    SymbolEnv,
    hint_of,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import SYMBOLIC_HINT, Decision
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import Trace

__all__ = ["KernelStep", "MISSING", "leaves"]

MISSING = object()
"""The domain value of a node with no value (its run raised on batch-free operands, natively too)."""


def leaves(value: Any) -> list:
    if type(value) in (tuple, list, torch.Size):
        return [leaf for item in value for leaf in leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in leaves(item)]
    if type(value) is slice:
        return leaves((value.start, value.stop, value.step))
    if isinstance(value, Shape):
        return [value, *value.dims]
    return [value]


def _holds_shape(value: Any) -> bool:
    return any(isinstance(leaf, Shape) for leaf in leaves(value))


def origins(domain: Any) -> set:
    return {leaf.origin for leaf in leaves(domain) if isinstance(leaf, Opaque) and isinstance(leaf.origin, Decision)}


class KernelStep:
    """The kernel step of ``_Propagation`` (``propagation.py``), which holds these attributes and the decision
    methods."""

    env: SymbolEnv
    interpreter: fx.Interpreter
    modules: Mapping[str, Any]
    values: DecisionValues
    traces: Tuple[Trace, ...]
    order: Mapping[fx.Node, int]

    def _opaque(self, node: fx.Node, operator_name: str, found: Opaque) -> Opaque:
        raise NotImplementedError

    def _failed(self, node: fx.Node, operator_name: str, detail: str) -> Tuple[Any, Any]:
        raise NotImplementedError

    def _traced(self, node: fx.Node, value_leaves: list) -> list:
        """Per tensor of the FakeTensor run's value (``value_leaves``: its leaves), the seeded traces' extents of it by
        batch: each trace whose value has a tensor of the same rank at the same leaf. A tensor every trace that reached
        ``node`` with as many leaves shows as no tensor (an attention's weights with ``need_weights=False``: an empty
        tensor in torch's FakeTensor run, ``None`` natively) is no native value, which no node of the traced program
        reads (the run at batch 1 shows none): it is cross-checked at the hint alone (``None``)."""
        found: list = []
        reached = [trace for trace in self.traces if len(trace.values.get(node, (None, ()))[1]) == len(value_leaves)]
        for position, leaf in enumerate(value_leaves):
            if not isinstance(leaf, torch.Tensor):
                continue
            column = [(trace.batch, trace.values[node][1][position]) for trace in reached]
            shapes = {batch: tuple(got.shape) for batch, got in column
                      if hasattr(got, "shape") and len(got.shape) == leaf.dim()}
            native = not reached or any(hasattr(got, "shape") for _, got in column)  # no trace: two batches needed
            found.append(shapes if native else None)
        return found

    def _batch_free(self, node: fx.Node, operator_name: str, value: Any) -> Any:
        """The domain value of a batch-free kernel's result: its tensors' integer extents (an unbacked one is a
        data-dependent size); a scalar it returns is opaque (torch computes it in the pass's fp64), a non-number as
        torch gives it."""
        def leaf(item: Any) -> Any:
            if isinstance(item, torch.Tensor):
                return Shape(tuple(data_dependent(d) if unbacked(d) else hint_int(d) for d in item.shape))
            if isinstance(item, (*SYMBOLIC_TYPES, int, float, bool)) and node.op != "get_attr":
                return self._opaque(node, operator_name, Opaque(f"a value {operator_name} returns"))
            return item
        return pytree.tree_map(leaf, value)

    def _rule(self, node: fx.Node):
        if node.op == "call_function" and node.target is getattr and len(node.args) == 2:
            name = str(node.args[1])
            return (name, same) if name == "data" else (name, LAYOUT_RULES[name]) if name in LAYOUT_RULES else None
        module = self.modules.get(str(node.target)) if node.op == "call_module" else None
        return rule_of(node.op, node.target, module)

    def _kernel(self, node: fx.Node, operator_name: str, args: tuple, kwargs: dict) -> Tuple[Any, Any]:
        """A tensor kernel: its declared batch rule's extents on its call bound to the kernel's signature
        (``binding.py``), cross-checked against torch's run at the hint and the seeded traces (review round 16; where
        the run at the hint raised, against the traces alone, whose extents the rule never reads: each is ``Opaque``,
        review round 15)."""
        value, exc = self.env.call(lambda: self.interpreter.run_node(node))
        if free((args, kwargs)):
            return (None, MISSING) if exc is not None else (value, self._batch_free(node, operator_name, value))
        runs: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        if exc is not None:
            value, runs = self.values.stand_in(node, self.traces)
            if value is None:
                return self._failed(node, operator_name, type(exc).__name__)
        declared = self._rule(node)
        leaves, spec = pytree.tree_flatten(value)
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        module = self.modules.get(str(node.target)) if node.op == "call_module" else None
        out = tuple(tuple(t.shape) for t in tensors) if not runs else tuple(tuple(
            Opaque(f"an extent of {operator_name} the seeded traces show (a sample)") for _ in t.shape) for t in tensors)
        operands = bound(node.op, node.target, module, args, kwargs, _holds_shape)  # the signature's order
        call = Call(declared[0] if declared else operator_name, *(operands or (args, kwargs)), out, module)
        rule = declared[1] if declared is not None and operands is not None else None
        stated, rule_exc = guarded(lambda: rule(call)) if rule is not None else (None, None)
        traced = self._traced(node, leaves) if not runs else []  # the seeded traces (review round 16)
        if rule_exc is not None or not isinstance(stated, list) or len(stated) != len(tensors):
            stated = [None] * len(tensors)
        inherited = min(origins((args, kwargs)), key=lambda decision: self.order[decision.node], default=None)
        built, domains, index = [], [], 0
        for leaf in leaves:
            if not isinstance(leaf, torch.Tensor):
                opaque = leaf is not None and not isinstance(leaf, (torch.dtype, torch.device, str))
                domains.append(self._opaque(node, operator_name, Opaque(f"a value {operator_name} returns"))
                               if opaque else leaf)
                built.append(self.values.scalar(domains[-1], hint_of(leaf)) if opaque else leaf)
                continue
            if runs:  # torch's FakeTensor run raised: the seeded traces alone
                observed = {batch: shapes[index] for batch, shapes in runs.items()}
            else:  # torch's FakeTensor run at the hint, and the seeded traces (review round 16)
                observed = {**(traced[index] or {}), SYMBOLIC_HINT: tuple(leaf.shape)}
            dims = tuple(self._opaque(node, operator_name, d) if isinstance(d, Opaque) else d for d in accept(
                operator_name, stated[index], tuple(leaf.shape), observed, inherited, traced[index] is None if traced
                else False))
            index += 1
            domains.append(Shape(dims))
            kept = self.values.kept(dims, leaf) if exc is None else (False,) * len(dims)
            built.append(self.values.tensor(leaf, dims, tuple(hint_int(d) for d in leaf.shape), kept))
        return pytree.tree_unflatten(built, spec), pytree.tree_unflatten(domains, spec)

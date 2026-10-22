"""Locate the first converted mapper whose output diverges from the native FX node it represents."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity.comparison import relative_error


@dataclass(frozen=True)
class DivergingNode:
    """The FX node a refusal names, and its operator (module type, function or tensor method)."""

    name: str
    operator: str


class _NodeRecorder(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.values: Dict[str, torch.Tensor] = {}

    def run_node(self, n: fx.Node):
        out = super().run_node(n)
        # A tuple-returning module (attention, recurrent) is compared on the element its conversion carries.
        first = out[0] if isinstance(out, (tuple, list)) and out else out
        if isinstance(first, torch.Tensor):
            self.values[n.name] = first
        return out


def trace_native(reference: nn.Module, input_shape: Tuple[int, ...]) -> fx.GraphModule:
    """The reference's trace, whose node names are the converter's (a private copy: tracing moves the model); a
    traced program (the probe's witness program) is its own."""
    if isinstance(reference, fx.GraphModule):
        return reference
    return trace_model(copy.deepcopy(reference), input_shape, device="cpu")


def native_node_values(gm: fx.GraphModule, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Every tensor-valued FX node of a native trace on ``x``."""
    recorder = _NodeRecorder(gm)
    recorder.run(x)
    return recorder.values


def node_operator(gm: fx.GraphModule, name: str) -> str:
    """The operator of the named FX node: its module type, function name or tensor method."""
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.name != name:
            continue
        if node.op == "call_module":
            return type(modules[str(node.target)]).__name__
        if node.op == "call_method":
            return f"Tensor.{node.target}"
        target = node.target
        return getattr(target, "__name__", None) or str(target)
    return "unknown"


def mapper_label(mapper) -> Optional[str]:
    """The FX node name a mapper was emitted for (its own name, else its perceptron's)."""
    for owner in (mapper, getattr(mapper, "perceptron", None)):
        name = getattr(owner, "name", None)
        if isinstance(name, str):
            return name
    return None


def mapper_node_values(mapper_repr, x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """``(label, output)`` of every labelled mapper, in execution order, all values retained."""
    order, deps, _ = mapper_repr._ensure_exec_graph()
    values: dict = {}
    labelled: List[Tuple[str, torch.Tensor]] = []
    for node in order:
        sources = deps.get(node, [])
        if not sources:
            out = node.forward(x)
        elif len(sources) == 1:
            out = node.forward(values[sources[0]])
        else:
            out = node.forward(tuple(values[dep] for dep in sources))
        values[node] = out
        label = mapper_label(node)
        if label is not None and isinstance(out, torch.Tensor):
            labelled.append((label, out))
    return labelled


def _absorbed_by_target(absorption_plan: Mapping[str, str]) -> Dict[str, List[str]]:
    absorbed: Dict[str, List[str]] = {}
    for node_name, target in absorption_plan.items():
        absorbed.setdefault(target.split(":", 1)[-1], []).append(node_name)
    return absorbed


def _agrees(got: torch.Tensor, want: torch.Tensor, bound: float) -> bool:
    if got.numel() != want.numel():
        return False
    return relative_error(got.reshape(-1), want.reshape(-1)).relative_error <= bound


def first_disagreement(
    labelled: Iterable[Tuple[str, torch.Tensor]],
    native: Mapping[str, torch.Tensor],
    absorption_plan: Mapping[str, str],
    bound: float,
) -> Optional[str]:
    """First label (in the given order) agreeing with neither its FX node nor any node absorbed into it."""
    absorbed = _absorbed_by_target(absorption_plan)
    for label, value in labelled:
        names = (label, *absorbed.get(label, ()))
        candidates = [native[name] for name in names if name in native]
        if candidates and not any(_agrees(value, c, bound) for c in candidates):
            return label
    return None


def first_diverging_node(
    reference: nn.Module,
    flow: nn.Module,
    absorption_plan: Mapping[str, str],
    input_shape: Tuple[int, ...],
    x: torch.Tensor,
    bound: float,
) -> Optional[DivergingNode]:
    """First mapper (execution order) agreeing with neither its FX node nor any node absorbed into it."""
    get_mapper_repr = getattr(flow, "get_mapper_repr", None)
    if get_mapper_repr is None:
        return None
    was_training = flow.training
    flow.eval()
    try:
        with torch.no_grad():
            gm = trace_native(reference, input_shape)
            native = native_node_values(gm, x)
            converted = mapper_node_values(get_mapper_repr(), x)
    finally:
        if was_training:
            flow.train()
    label = first_disagreement(converted, native, absorption_plan, bound)
    return None if label is None else DivergingNode(label, node_operator(gm, label))

"""The axis rules of the batch-merge push-forward (coverage W7): for an operation reading a merged value, whether it is
independent along the merged axis, the axis it carries the merge on, and the operation on the split values.

A MERGED value of rank R has an axis ``at`` of extent ``a*B`` (``a >= 2``, the batch domain's proved form) whose index
``i`` belongs to sample ``i // a`` (sample-major); its SPLIT value has that axis as two, ``(B, a)`` at ``at, at + 1``,
the same elements in the same order (row-major). An operation is taken when it computes each index of the merged axis
from the same index of its merged operands alone, so that on the split operands it computes the split result:

- a layout: ``view``, ``reshape``, ``flatten`` (``reshape_target``: by the domain's proved output extents), ``permute``
  and ``transpose`` (the permutation with the merged axis as two), an integer or a slice index of another axis
  (``getitem``, ``select``, ``narrow``, ``unbind``), ``contiguous`` and ``clone``;
- a POINTWISE operation (torch's ``pointwise`` tag, Python's arithmetic and comparisons, an eval-mode dropout, a
  pointwise activation module): every other tensor operand is merged on the same axis, or computed with no input and
  reaching the merged axis, if at all, with an extent of 1 as its own first axis (``(1, h, N, N)`` against
  ``(B*n, h, N, N)``);
- an operation on trailing axes: ``linear`` and ``nn.Linear`` (the last axis, a weight with no input), ``layer_norm``
  and ``nn.LayerNorm`` (the normalized axes); a matrix product of operands merged alike, or of one merged operand and
  a matrix with no input; an operation along named axes other than the merged one (``softmax``, ``log_softmax``,
  ``softmin``, ``nn.Softmax``, ``normalize``, ``cumsum``, ``cumprod``, ``flip``, ``roll`` with axes); a concatenation
  of operands merged alike along another axis;
- a value without the merged axis: an extent (``size(d)``: at the merged axis ``a`` times the split value's), and a
  receiver factory or the ``dtype``/``device`` of a merged receiver (they read its dtype and device).

Anything else (a reduction, scan, flip, roll, softmax or normalization along the merged axis, an index or slice of it,
a contraction over it, a random draw, a call outside these rules) is not taken, and the merged reshape stays refused.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, Callable, List, Mapping, Optional

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import map_arg

from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import RECEIVER_FACTORIES
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import kernel_rules
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernels import rule_of
from mimarsinan.torch_mapping.fx_rewrites.batch_merge.layouts import (
    DROPOUTS, FIBERS, IDENTITIES, LAYOUTS, PERMUTES, Extents, Infos, Merged, Plan, along_plan, argument, axis_of,
    index_plan, name_of, permute_plan, reshape_target, scalar_operands, shape_operands, split_axis,
)

__all__ = ["module_kind", "plan_of", "pointwise"]


def _tensors(value: Any) -> List[Any]:
    found: List[Any] = []
    map_arg(value, found.append)
    return found


def _rank(node: fx.Node) -> Optional[int]:
    shape = getattr(node.meta.get("tensor_meta"), "shape", None)
    return None if shape is None else len(shape)


def _broadcast(node: fx.Node, infos: Infos, extents: Extents, free: Callable[[fx.Node], bool]) -> Optional[Merged]:
    """A pointwise operation's merge: every merged operand merged alike at the output's rank, every other operand a
    tensor computed with no input that reaches the merged axis, if at all, with an extent of 1 as its first axis."""
    rank, found = _rank(node), None
    for arg in _tensors((node.args, node.kwargs)):
        info = infos.get(arg)
        if isinstance(info, Merged):
            if info.rank != rank or found not in (None, info) or info.elements:
                return None
            found = info
    if found is None or rank is None:
        return None
    for arg in _tensors((node.args, node.kwargs)):
        if arg in infos:
            continue
        r, dims = _rank(arg), extents(arg)
        if r is None or not free(arg) or (rank - r < found.at and r > 0) or (
                rank - r == found.at and (not dims or dims[0] != 1)):
            return None
    return found


@functools.lru_cache(maxsize=None)
def pointwise(name: str) -> bool:
    """The aten operator of the name is elementwise: an overload carries torch's ``pointwise`` tag and none draws a
    random number (``nondeterministic_seeded``: ``rrelu``)."""
    packet = getattr(torch.ops.aten, name, None)
    try:
        tags = [getattr(packet, overload).tags for overload in packet.overloads()] if packet is not None else []
    except (AttributeError, RuntimeError):
        return False
    return any(torch.Tag.pointwise in found for found in tags) and not any(
        torch.Tag.nondeterministic_seeded in found for found in tags)


def _pointwise_call(node: fx.Node) -> bool:
    found = rule_of(node.op, node.target, None)
    return found is not None and found[1] is kernel_rules.pointwise or pointwise(name_of(node))


_DROPOUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
"""torch's dropout modules: in eval mode each returns its input."""

_ACTIVATION_OPERATORS = {
    nn.CELU: "celu", nn.ELU: "elu", nn.GELU: "gelu", nn.Hardshrink: "hardshrink", nn.Hardsigmoid: "hardsigmoid",
    nn.Hardtanh: "hardtanh", nn.Mish: "mish", nn.ReLU: "relu", nn.ReLU6: "hardtanh", nn.SELU: "selu", nn.SiLU: "silu",
    nn.Sigmoid: "sigmoid", nn.Softplus: "softplus", nn.Softshrink: "softshrink", nn.Tanh: "tanh",
    nn.Threshold: "threshold"}
"""torch's activation modules without state, each with the aten operator its call computes; one is taken as pointwise
only while that operator carries torch's ``pointwise`` tag (``pointwise``)."""


def module_kind(module: Optional[nn.Module]) -> Optional[str]:
    """The axis rule of a called module, by its exact class (a subclass may override ``forward``)."""
    if module is None:
        return None
    kind = type(module)
    if kind is nn.Linear:
        return "linear"
    if kind is nn.LayerNorm:
        return "layer_norm"
    if kind in (nn.Softmax, nn.LogSoftmax, nn.Softmin):
        return "fiber"
    if kind is nn.Identity or (kind in _DROPOUT_MODULES and not module.training):
        return "identity"
    operator_name = _ACTIVATION_OPERATORS.get(kind)
    if operator_name is not None and pointwise(operator_name):
        return "pointwise"  # ``nn.GELU``: its operator, elementwise, with no state
    return None


def _trailing(node: fx.Node, merged: Merged, count: Optional[int], free: Callable[[fx.Node], bool]) -> Optional[Plan]:
    """An operation on the last ``count`` axes, every other tensor operand computed with no input."""
    others = [arg for arg in _tensors((node.args[1:], node.kwargs))]
    if count is None or merged.at >= merged.rank - count or not all(free(arg) for arg in others):
        return None
    return Plan(node.op, node.target, node.args, dict(node.kwargs), merged)


def _matmul(node: fx.Node, infos: Infos, free: Callable[[fx.Node], bool]) -> Optional[Plan]:
    if len(node.args) != 2 or node.kwargs:
        return None
    left, right = (infos.get(arg) if isinstance(arg, fx.Node) else None for arg in node.args)
    merged = [info for info in (left, right) if isinstance(info, Merged)]
    other = [arg for arg, info in zip(node.args, (left, right)) if not isinstance(info, Merged)]
    if any(info.elements or info.at >= info.rank - 2 for info in merged) or len(set(merged)) != 1:
        return None
    if other and not (isinstance(other[0], fx.Node) and free(other[0]) and _rank(other[0]) == 2):
        return None
    return Plan(node.op, node.target, node.args, {}, merged[0])


def _cat(node: fx.Node, infos: Infos) -> Optional[Plan]:
    parts, dim = argument(node, "tensors", 0), argument(node, "dim", 1, 0)
    found = [infos.get(part) for part in parts] if isinstance(parts, (list, tuple)) and parts else [None]
    merged = found[0]
    if not isinstance(merged, Merged) or any(info != merged for info in found):
        return None
    axis = axis_of(dim, merged.rank)
    if merged.elements or axis is None or axis == merged.at:
        return None
    return Plan(node.op, node.target, (list(parts), split_axis(axis, merged.at)), {}, merged)


def dims_plan(node: fx.Node, merged: Merged, position: int) -> Optional[Plan]:
    """``flip`` and ``roll`` along named axes other than the merged one (``roll`` with no axes flattens: not taken)."""
    dims = argument(node, "dims", position)
    if node.op == "call_method" and node.target == "flip" and len(node.args) > 2:
        dims = tuple(node.args[1:])
    dims = (dims,) if type(dims) is int else dims
    if not isinstance(dims, (tuple, list)) or not dims or not scalar_operands(node):
        return None
    axes = [axis_of(d, merged.rank) for d in dims]
    if None in axes or merged.at in axes:
        return None
    split = [split_axis(int(axis), merged.at) for axis in axes]  # type: ignore[arg-type]
    args, kwargs = list(node.args[:position]), {key: value for key, value in node.kwargs.items() if key != "dims"}
    kwargs["dims"] = split
    return Plan(node.op, node.target, tuple(args), kwargs, merged)


def plan_of(node: fx.Node, infos: Infos, extents: Extents, free: Callable[[fx.Node], bool],
            modules: Mapping[str, nn.Module]) -> Optional[Plan]:
    """The plan of ``node``, a reader of a merged value (the module docstring); ``None`` when it is not taken."""
    source = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
    merged = infos.get(source) if source is not None else None
    name = name_of(node)
    if node.op == "call_module":
        kind = module_kind(modules.get(str(node.target)))
        module = modules.get(str(node.target))
        if not isinstance(merged, Merged) or merged.elements or kind is None or len(node.args) != 1 or node.kwargs:
            return None
        if kind == "fiber":  # the module's own ``dim`` must name the same axis of the split value
            dim = getattr(module, "dim", None)
            same = type(dim) is int and (0 <= dim < merged.at or -merged.rank <= dim < 0 and (
                merged.rank + dim > merged.at))
            return Plan(node.op, node.target, node.args, {}, merged) if same else None
        if kind in ("pointwise", "identity"):
            return Plan(node.op, node.target, node.args, {}, merged)
        count = 1 if kind == "linear" else len(getattr(module, "normalized_shape", ())) if kind == "layer_norm" else 0
        return _trailing(node, merged, count, free)
    if node.op == "call_function" and node.target is operator.getitem and isinstance(merged, Merged):
        if merged.elements:
            index = node.args[1]
            return Plan(node.op, node.target, node.args, {}, Merged(merged.at, merged.a, merged.rank)) \
                if type(index) is int else None
        return index_plan(node, merged)
    if node.op == "call_function" and node.target in (torch.matmul, operator.matmul) or (
            node.op == "call_method" and node.target == "matmul"):
        return _matmul(node, infos, free)
    if node.op == "call_function" and node.target is torch.cat:
        return _cat(node, infos)
    if not isinstance(merged, Merged) or merged.elements:
        return None
    if name in LAYOUTS and node.op == "call_method" or node.target in (torch.reshape, torch.flatten):
        if source is None or not shape_operands(node):
            return None
        before = extents(source)
        found = reshape_target(merged, extents(node), None if before is None else before[:merged.at])
        if found is None:
            return None
        out, end, target = found
        if end:
            return Plan(node.op, node.target, node.args, dict(node.kwargs), None, end=True)
        return Plan("call_method", "view" if node.target == "view" else "reshape", (source, *target), {}, out)
    if name == "size" and node.op == "call_method" and len(node.args) == 2 and not node.kwargs:
        axis = axis_of(node.args[1], merged.rank)
        return None if axis is None else Plan("size", axis, node.args, {}, None)
    if node.op == "call_function" and node.target is getattr:
        return Plan(node.op, node.target, node.args, {}, None) if node.args[1] in ("dtype", "device") else None
    if name in RECEIVER_FACTORIES and node.op == "call_method" and node.all_input_nodes == [source]:
        return Plan(node.op, node.target, node.args, dict(node.kwargs), None)
    if name in PERMUTES and (node.op == "call_method" or node.target is PERMUTES[name]):
        return permute_plan(node, merged)
    if name in IDENTITIES and node.op == "call_method" and len(node.args) == 1:
        return Plan(node.op, node.target, node.args, dict(node.kwargs), merged)
    if name in FIBERS:
        return along_plan(node, merged, FIBERS[name], dropped=False, default=1 if name == "normalize" else None)
    if name in ("select", "narrow", "unbind"):
        plan = along_plan(node, merged, 1, dropped=name != "narrow", default=0 if name == "unbind" else None)
        if plan is not None and name == "unbind":
            out = plan.out
            return Plan(plan.op, plan.target, plan.args, plan.kwargs,
                        Merged(out.at, out.a, out.rank, elements=True) if out is not None else None)
        return plan
    if name in ("flip", "roll"):
        return dims_plan(node, merged, 1 if name == "flip" else 2)
    if name in ("linear", "layer_norm") and node.op == "call_function":
        shape = argument(node, "normalized_shape", 1, ())
        count = 1 if name == "linear" else len(shape) if isinstance(shape, (tuple, list)) else None
        return _trailing(node, merged, count or None, free)
    if name in DROPOUTS and node.op == "call_function" and argument(node, "training", 2, True) is False:
        return Plan(node.op, node.target, node.args, dict(node.kwargs), merged)
    if _pointwise_call(node):
        out = _broadcast(node, infos, extents, free)
        return None if out is None else Plan(node.op, node.target, node.args, dict(node.kwargs), out)
    return None

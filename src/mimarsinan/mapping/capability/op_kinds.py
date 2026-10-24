"""Op kind of an IR node, read from its payload and operand structure; an operator with no rule refuses by name."""

from __future__ import annotations

import operator
from typing import Callable, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.capability.requirements import REQUIREMENTS, UnknownOpKindError
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, NeuralCore
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, _cat_along, cat_per_sample
from mimarsinan.mapping.support.operand_write import OperandWrite
from mimarsinan.mapping.support.scale_wrapper import unwrap_scale_wrapper
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import BatchNormAffine
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import torch_function_of

#: The §5.1 interface (G1's ``Contraction`` payload): ``op_type`` names it, ``operand_roles`` decides D against S.
CONTRACTION_OP_TYPE = "Contraction"
ACTIVATION_ROLE = "activation"
#: The requirement-table kind of N2's payload write host op (``ladder.PAYLOAD_WRITE_KIND`` names the same row).
PAYLOAD_WRITE_KIND = "payload_write"

_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_MODULE_TYPES: Tuple[Tuple[tuple, str], ...] = (
    ((nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.LeakyReLU, nn.PReLU, nn.Hardsigmoid), "clamp"),
    ((nn.Hardswish,), "hswish"),
    ((nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.Softplus, nn.Mish), "smooth"),
    ((nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d), "norm"),
    ((nn.Softmax,), "softmax"),
    ((nn.AvgPool1d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d), "pool"),
    ((nn.MaxPool1d, nn.MaxPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d), "maxp"),
    ((nn.modules.batchnorm._BatchNorm, BatchNormAffine), "affine"),
    ((nn.ConvTranspose1d, nn.ConvTranspose2d), "mms"),
    ((nn.PixelShuffle, nn.PixelUnshuffle), "wire+"),
    ((nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.RNNBase), "leaf"),
)
#: Module rules whose kind reads a module attribute (groups, interpolation mode); checked before the type table.
_MODULE_RULES: Tuple[Tuple[Callable[[nn.Module], bool], str], ...] = (
    (lambda m: isinstance(m, _CONVS) and int(m.groups) > 1, "mmg"),
    (lambda m: isinstance(m, _CONVS + (nn.Linear,)), "mms"),
    (lambda m: isinstance(m, nn.Upsample) and m.mode == "nearest", "wire+"),
    (lambda m: isinstance(m, nn.Upsample), "mms"),
)
_FUNCTION_KINDS: Mapping[Callable, str] = {
    **dict.fromkeys((F.relu, torch.relu, F.relu6, F.hardtanh, F.leaky_relu, torch.clamp, torch.clip,
                     F.hardsigmoid), "clamp"),
    F.hardswish: "hswish",
    **dict.fromkeys((F.gelu, F.silu, torch.sigmoid, F.sigmoid, torch.tanh, F.tanh, F.elu, F.softplus, F.mish,
                     torch.exp), "smooth"),
    **dict.fromkeys((F.softmax, torch.softmax), "softmax"),
    **dict.fromkeys((F.layer_norm, F.group_norm), "norm"),
    **dict.fromkeys((torch.mean, F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.avg_pool1d, F.avg_pool2d), "pool"),
    **dict.fromkeys((F.max_pool1d, F.max_pool2d, F.adaptive_max_pool2d), "maxp"),
    **dict.fromkeys((operator.getitem, torch.cat, torch.stack, torch.chunk, torch.split, torch.narrow, torch.select,
                     F.pad, torch.flatten, torch.reshape, torch.permute, torch.transpose, torch.squeeze, torch.unsqueeze,
                     cat_per_sample, _cat_along), "wire+"),
    F.scaled_dot_product_attention: "leaf",
}
#: Binary arithmetic: (kind with two activation operands, kind with one activation and a constant).
_BINARY_KINDS: Mapping[Callable, Tuple[str, str]] = {
    **dict.fromkeys((operator.add, torch.add, operator.sub, torch.sub), ("add", "affine")),
    **dict.fromkeys((operator.mul, torch.mul), ("prod", "affine")),
    **dict.fromkeys((operator.truediv, torch.div, torch.true_divide), ("data_division", "affine")),
    **dict.fromkeys((torch.matmul, operator.matmul, torch.bmm, torch.mm, torch.einsum, F.linear), ("dmm", "mms")),
}


def operand_spans(op: ComputeOp) -> Tuple[Tuple[int, int, Tuple[int, ...]], ...]:
    """``(offset, size, shape)`` of each activation operand in ``op.input_sources``, in the executed layout."""
    shapes = op.params.get("input_shapes")
    if shapes is None:
        count = int(op.input_sources.size)
        shape = tuple(op.input_shape) if op.input_shape is not None else (count,)
        return ((0, count, shape),)
    spans, offset = [], 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= int(dim)
        spans.append((offset, size, tuple(int(d) for d in shape)))
        offset += size
    return tuple(spans)


def stationary_operand(op: ComputeOp) -> Tuple[int, int, Tuple[int, ...]]:
    """The operand a dynamic product writes into its crossbar: the right-hand ``k x n`` one (the design note's K, V)."""
    return operand_spans(op)[-1]


def _payload(op: ComputeOp):
    return unwrap_scale_wrapper(op.params.get("module"))


def _contraction_kind(payload: object) -> str:
    roles = getattr(payload, "operand_roles", None)
    if not isinstance(roles, (tuple, list)) or not roles:
        return "mms"
    return "dmm" if all(role == ACTIVATION_ROLE for role in roles) else "mms"


def _function_of(fn: Callable) -> Callable:
    """The torch function a host op computes: a tensor method (``torch.Tensor.softmax``, which G1's method rule and the
    batch-axis push-forward emit) is the torch function of its name (``torch_function_of``, the method rule's own
    test), so it takes that function's kind; any other callable is itself."""
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and getattr(torch.Tensor, name, None) is fn:
        function = torch_function_of(name)
        if function is not None:
            return function
    return fn


def _adapter_kind(adapter: ComputeAdapter, op: ComputeOp) -> str | None:
    fn = _function_of(adapter.fn)
    binary = _BINARY_KINDS.get(fn)
    if binary is not None:
        return binary[0] if len(operand_spans(op)) >= 2 else binary[1]
    return _FUNCTION_KINDS.get(fn)


def _module_kind(module: nn.Module) -> str | None:
    for predicate, kind in _MODULE_RULES:
        if predicate(module):
            return kind
    for types, kind in _MODULE_TYPES:
        if isinstance(module, types):
            return kind
    return None


def op_kind(node: IRNode) -> str:
    """The requirement-table kind of one IR node."""
    if isinstance(node, NeuralCore):
        return "mm"
    if not isinstance(node, ComputeOp):
        raise UnknownOpKindError(f"IR node {node.name!r} is neither a NeuralCore nor a ComputeOp ({type(node).__name__})")
    payload = _payload(node)
    if isinstance(payload, OperandWrite):  # N2: the host write of a lowered product's operand banks
        kind = PAYLOAD_WRITE_KIND
    elif node.op_type == CONTRACTION_OP_TYPE or hasattr(payload, "operand_roles"):
        kind = _contraction_kind(payload)
    elif isinstance(payload, ComputeAdapter):
        kind = _adapter_kind(payload, node)
    elif isinstance(payload, nn.Module):
        kind = _module_kind(payload)
    else:
        kind = None
    if kind is None or kind not in REQUIREMENTS:
        raise UnknownOpKindError(
            f"host op {node.op_type!r} (IR node {node.name!r}) has no op-kind rule in mapping/capability/op_kinds.py; "
            "its capability requirement is unknown, so the census refuses it by name"
        )
    return kind


def ir_op_kinds(ir_graph: IRGraph) -> Dict[int, str]:
    """``{node id: kind}`` over every node; the first operator with no rule refuses by name."""
    return {node.id: op_kind(node) for node in ir_graph.nodes}

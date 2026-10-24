"""Forward-MAC SSOT: the one per-type rule set both MAC counters read (IR host ops and the forward estimator).

Every rule counts the multiply-accumulates of one invocation at the shapes it is given (batch axis leading, one
sample), every output position included: a convolution at every output position, a Linear at every leading position,
an attention, transformer-encoder or recurrent leaf over its whole sequence, a product in functional form (``F.linear``,
``matmul``, ``mm``, ``bmm``, a two-operand ``einsum``, as a host op evaluates it: coverage W10) at every output element.
A module that declares its own count (``forward_macs()``, the Contraction host op and the decomposed recurrence) is
counted by that declaration; element-wise and shape ops carry no MACs. No other module dispatches on ``nn.Module`` type
to compute MACs.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.contraction import contraction_macs
from mimarsinan.mapping.support.scale_wrapper import is_scale_wrapper

Shape = Optional[Tuple[int, ...]]

_RECURRENT_GATES = {"LSTM": 4, "GRU": 3}
"""Gate count per recurrent kind; ``nn.RNN`` (either nonlinearity) has one."""


def _numel(shape: Sequence[int]) -> int:
    return math.prod(int(d) for d in shape)


def _positions(features: int, shape: Shape) -> int:
    """How many vectors of ``features`` entries ``shape`` holds (its element count over ``features``); 1 without a
    shape or when it does not divide."""
    if shape is not None and features > 0 and _numel(shape) % features == 0:
        return max(1, _numel(shape) // features)
    return 1


def _linear_macs(layer: nn.Linear, first: Tuple[int, Shape], second: Tuple[int, Shape]) -> int:
    """``in · out`` per position, the positions read from the first ``(features, shape)`` pair that has a shape."""
    features, shape = first if first[1] is not None else second
    return int(layer.in_features) * int(layer.out_features) * _positions(features, shape)


def _conv_macs(module: nn.modules.conv._ConvNd, in_shape: Shape, out_shape: Shape) -> int:
    """A convolution at every output position, ``groups`` respected; a transposed one is its scatter form, every input
    element times every kernel weight of its group (``in_channels · out_channels / groups · prod(kernel)`` per input
    position)."""
    kernel = _numel(module.kernel_size)
    per_group = int(module.out_channels) // int(module.groups) if module.transposed else (
        int(module.in_channels) // int(module.groups))
    spatial, channels = (in_shape, module.in_channels) if module.transposed else (out_shape, module.out_channels)
    if spatial is None:
        return 0
    return int(channels) * _numel(tuple(spatial)[2:]) * per_group * kernel


def _attention_macs(module: nn.MultiheadAttention, in_shape: Shape, input_shapes: Sequence[Shape]) -> int:
    """Projections of query, key and value, the output projection, and the two products of the scores (query length
    ``Lq``, key length ``Lk``): ``Lq·E·E + Lk·E·kdim + Lk·E·vdim + Lq·E·E + 2·Lq·Lk·E``; with the query alone (self
    attention) ``4·E²·L + 2·L²·E``."""
    e = int(module.embed_dim)
    shapes = list(input_shapes) or [in_shape]
    query = shapes[0]
    if query is None:
        return 0
    lq = _positions(e, query)
    key = shapes[1] if len(shapes) > 1 else None
    lk = _positions(int(module.kdim), key) if key is not None else lq
    return int(lq * e * e + lk * e * int(module.kdim) + lk * e * int(module.vdim) + lq * e * e + 2 * lq * lk * e)


def _recurrent_macs(module: nn.RNNBase, in_shape: Shape) -> int:
    """``T · g · H · (F_l + H_r)`` per layer and direction (``g`` = 4, 3, 1 for LSTM, GRU, RNN; ``F_l`` the layer's
    input width; ``H_r`` the recurrent width, ``proj_size`` when set) plus ``T · H · H_r`` for an LSTM projection."""
    if in_shape is None:
        return 0
    gates = _RECURRENT_GATES.get(str(module.mode), 1)
    hidden, proj = int(module.hidden_size), int(getattr(module, "proj_size", 0) or 0)
    recurrent = proj or hidden
    directions = 2 if module.bidirectional else 1
    steps = _positions(int(module.input_size), in_shape)
    total = 0
    for layer in range(int(module.num_layers)):
        width = int(module.input_size) if layer == 0 else recurrent * directions
        per_step = gates * hidden * (width + recurrent) + (hidden * proj if proj else 0)
        total += steps * per_step * directions
    return int(total)


def _encoder_macs(module: nn.Module, in_shape: Shape) -> int:
    """A transformer encoder leaf: per layer its self attention and its two Linears, every one on the layer's
    sequence (the input's shape, which every layer keeps)."""
    layers = list(module.layers) if isinstance(module, nn.TransformerEncoder) else [module]
    total = 0
    for layer in cast(List[nn.TransformerEncoderLayer], layers):
        linear1, linear2 = cast(nn.Linear, layer.linear1), cast(nn.Linear, layer.linear2)
        hidden = None if in_shape is None else (*in_shape[:-1], int(linear1.out_features))
        total += _attention_macs(cast(nn.MultiheadAttention, layer.self_attn), in_shape, ()) + module_macs(
            linear1, in_shape, hidden) + module_macs(linear2, hidden, in_shape)
    return int(total)


def module_macs(module: nn.Module, in_shape: Shape, out_shape: Shape, input_shapes: Sequence[Shape] = ()) -> int:
    """Forward MACs of one invocation of ``module`` at full I/O shapes (batch axis leading). ``input_shapes``: every
    operand's shape of a multi-input op (the IR's ``params['input_shapes']``, or every input of the observed call),
    read where ``in_shape`` alone does not say (the attention leaf's key length)."""
    if is_scale_wrapper(module):
        return module_macs(cast(nn.Module, getattr(module, "module")), in_shape, out_shape, input_shapes)
    declared = getattr(module, "forward_macs", None)
    if callable(declared):
        return int(cast(int, declared()))
    if in_shape is None and input_shapes:
        in_shape = input_shapes[0]
    layer = getattr(module, "layer", None)
    if isinstance(layer, nn.Linear):  # a perceptron: its owner's I/O, so a perceptron conv's positions are its outputs
        return _linear_macs(layer, (int(layer.out_features), out_shape), (int(layer.in_features), in_shape))
    if isinstance(module, nn.Linear):
        return _linear_macs(module, (int(module.in_features), in_shape), (int(module.out_features), out_shape))
    if isinstance(module, nn.modules.conv._ConvNd):
        return _conv_macs(module, in_shape, out_shape)
    if isinstance(module, nn.MultiheadAttention):
        return _attention_macs(module, in_shape, input_shapes)
    if isinstance(module, nn.RNNBase):
        return _recurrent_macs(module, in_shape)
    if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerEncoder)):
        return _encoder_macs(module, in_shape)
    if _weight_op_package(module):
        return module_macs(cast(nn.Sequential, module)[0], in_shape, out_shape)
    if isinstance(module, ComputeAdapter):
        return _functional_product_macs(module, in_shape, out_shape, input_shapes)
    return 0


#: The products a host op evaluates in functional form: the requirement table's product row (``op_kinds``), a tensor
#: method by the torch function of its name.
_PRODUCTS = (F.linear, torch.matmul, operator.matmul, torch.mm, torch.bmm, torch.einsum)


def _product(fn: Any) -> Any:
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and not name.startswith("_") and getattr(torch.Tensor, name, None) is fn:
        fn = getattr(torch, name, None)
    return next((product for product in _PRODUCTS if fn is product), None)


def _product_macs(product: Any, args: Sequence[Any], kwargs: Dict[str, Any], out_shape: Tuple[int, ...]) -> int:
    """Every output element times the extent the call sums over: a Linear's ``in_features`` (its weight's last axis),
    a matrix product's inner axis (its first operand's last), a two-operand einsum's summed axes
    (``contraction_macs``); 0 for an einsum of other than two operands (no single product) or an operand of rank 0."""
    if product is torch.einsum:
        operands = list(args[1:])
        if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
            operands = list(operands[0])
        if len(operands) != 2 or not isinstance(args[0], str):
            return 0
        return contraction_macs(args[0], [tuple(t.shape) for t in operands], out_shape)
    key, position = ("weight", 1) if product is F.linear else ("input", 0)
    operand = kwargs.get(key, args[position] if len(args) > position else None)
    if not isinstance(operand, torch.Tensor) or operand.dim() == 0:
        return 0
    return _numel(out_shape) * int(operand.shape[-1])


def _functional_product_macs(adapter: ComputeAdapter, in_shape: Shape, out_shape: Shape,
                             input_shapes: Sequence[Shape]) -> int:
    """A host op evaluating a product (``_PRODUCTS``), with a bound weight or not (``_product_macs``), the operands'
    shapes read through the adapter's own call: its argument template on meta stand-ins of its inputs and bound
    tensors, a batch-free slot by its stored shape, never evaluated. 0 for any other call, or where a shape is
    unknown."""
    product = _product(adapter.fn)
    shapes = list(input_shapes) or [in_shape]
    if product is None or out_shape is None or any(shape is None for shape in shapes):
        return 0
    inputs = [torch.empty(tuple(cast(Tuple[int, ...], shape)), device="meta") for shape in shapes]
    bound = [torch.empty(tuple(getattr(adapter, f"bound_{i}").shape), device="meta")
             for i in range(int(adapter._bound_count))]
    if adapter.arg_template is None:
        args, kwargs = (*inputs, *bound, *adapter.extra_args), dict(adapter.kwargs)
    else:
        args, slots = adapter.arg_template.resolve(inputs, bound, adapter.extra_args)
        kwargs = {**adapter.kwargs, **slots}
    return _product_macs(product, args, kwargs, tuple(out_shape))


def _weight_op_package(module: nn.Module) -> bool:
    """The converter's host package of a weight op and its unfolded BatchNorm (``Sequential(op, bn)``), whose output
    has the weight op's shape."""
    return isinstance(module, nn.Sequential) and len(module) == 2 and isinstance(
        module[1], nn.modules.batchnorm._BatchNorm)


def _shape(value: Any) -> Shape:
    value = value[0] if isinstance(value, (tuple, list)) and value else value
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _call_shapes(inputs: Any, output: Any) -> Tuple[Shape, Shape, Tuple[Shape, ...]]:
    many = tuple(_shape(item) for item in inputs) if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else ()
    return _shape(inputs), _shape(output), many


def _observed_forward(node: nn.Module, seen: Dict[int, tuple]) -> Callable[..., Any]:
    call = type(node).forward.__get__(node)

    def forward(x: Any) -> Any:
        out = call(x)
        seen[id(node)] = _call_shapes(x, out)
        return out
    return forward


def observed_macs(units: Sequence[Tuple[nn.Module, nn.Module]], run: Callable[[], Any]) -> List[int]:
    """``module_macs`` of each ``(node, unit)`` pair at the shapes of one ``run()``: the node's own call (a mapper
    whose payload never runs as a module, a perceptron convolution through ``F.conv2d``, is counted at its real output,
    every position included), else the unit's own call (a flow whose forward does not walk its mapper nodes)."""
    seen: Dict[int, tuple] = {}
    hooked: Dict[int, tuple] = {}
    nodes = {id(node): node for node, _ in units}
    units_by_id = {id(unit): unit for _, unit in units}
    handles = [unit.register_forward_hook(lambda _m, i, o, key=key: hooked.__setitem__(key, _call_shapes(i, o)))
               for key, unit in units_by_id.items()]
    try:
        for node in nodes.values():
            setattr(node, "forward", _observed_forward(node, seen))
        run()
    finally:
        for node in nodes.values():
            delattr(node, "forward")
        for handle in handles:
            handle.remove()
    return [module_macs(unit, *seen.get(id(node), hooked.get(id(unit), (None, None, ())))) for node, unit in units]

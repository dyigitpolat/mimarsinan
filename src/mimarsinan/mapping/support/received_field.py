"""[U-RSHIFT] The received field: each shift's source carried to every reader of it by each node's own forward.

A host value that can go negative is shifted before an on-chip reader encodes it (Soft Core Mapping's shift, stamped
``_negative_shift``; the gauge seam's lift in the op, ``output_value_offset`` over its emitted gauge), and each reader's
bias removes it (``received_shift``, the one writer). An axon receives the shift only as its element does: a padding
slot, the padded share of a pool's border window, or an axon of a join whose producer is not shifted receives none or a
share of it. ``received_shifts`` is the one propagation: per source, over its downstream cone in execution order, one
zero sample's shapes, float64, each node by its own forward:

* a structural node moves each element's field as its forward moves the element (a join's unshifted part reads zero);
* a host op passes it by its declared response only: an element move or a linear op by its own deployed forward, a
  maximum where the field is constant on each window, a LayerNorm absorbs a field constant over its normalized group;
* a perceptron (a core, or an encoder on the host) or a host bias carrier is a reader: the field ends there;
* any other kind, an operand product, a forward-only module that changes the field, or a shifted op whose input already
  carries a field, refuses by name (``ReceivedShiftRefusal``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.module_mapper import ModuleMapper
from mimarsinan.mapping.mappers.operand.gather_chain import mapper_output_shapes
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.padded_output import ZeroPaddedOutput
from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_along
_MOVES = frozenset({"flatten", "reshape", "permute", "transpose", "getitem", "view", "contiguous", "squeeze",
                    "unsqueeze", "cat", "stack", "narrow"})
_LINEAR = frozenset({"mean", "add", "sum", "sub", "neg"})
_EXTREMA = frozenset({"amax", "amin"})
_AVERAGES = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
_MAXIMA = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)


class ReceivedShiftRefusal(ToolchainRefusal, ValueError):
    """A node kind that declares no response to a nonzero received field, or a reader no bias can correct for it."""


@dataclass(frozen=True)
class ReceivedShift:
    """``rows`` (positions, in_features), float64: the field each core position's axons read (a padding slot at zero);
    ``elements`` the field over one sample of the reader's input (None for a direct call); ``sources`` the names it
    carries; ``uniform`` today's every-axon shift where the rows are exactly it, else None; ``slots`` a convolution's
    patch table ``(table, elements)``; ``operands`` a host attention module's rows per operand (query, key, value)."""

    rows: torch.Tensor
    elements: Optional[torch.Tensor] = None
    sources: tuple = ()
    uniform: Optional[torch.Tensor] = None
    slots: Optional[tuple] = None
    operands: tuple = ()


def _name(node: Any) -> str:
    return str(getattr(node, "name", None) or getattr(getattr(node, "perceptron", None), "name", None)
               or f"{type(node).__name__}@{id(node):x}")


def _host_carrier(module: Any) -> bool:
    """A host module whose own bias can hold a correction: a biased Linear, or attention with a packed projection."""
    if isinstance(module, nn.Linear):
        return module.bias is not None
    return (isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None
            and module.in_proj_bias is not None)


def _host_kind(op: Any) -> str:
    """A host op's declared response to a field on its input: its kind, or its type's name (no response)."""
    module = getattr(op, "module", None)
    if isinstance(module, ZeroPaddedOutput):
        module = module.module
    if isinstance(module, ComputeAdapter):
        fn = getattr(module.fn, "__name__", "")
        return "move" if fn in _MOVES else "linear" if fn in _LINEAR else "extremum" if fn in _EXTREMA else fn
    if isinstance(module, (nn.Identity, nn.Flatten)):
        return "move"
    if isinstance(module, _AVERAGES):
        return "linear"
    if isinstance(module, _MAXIMA):
        return "extremum"
    if isinstance(module, nn.LayerNorm):
        return "layernorm"
    return "reader" if _host_carrier(module) else type(module).__name__


def stamped_value(node: Any) -> Optional[torch.Tensor]:
    """A host op's own shift as its readers receive it: Soft Core Mapping's stamp, or a lift over its emitted gauge."""
    stamp = getattr(node, "_negative_shift", None)
    if stamp is not None:
        return torch.as_tensor(np.asarray(stamp, dtype=np.float32))
    offset = getattr(node, "output_value_offset", None) if isinstance(node, ComputeOpMapper) else None
    if offset is None:
        return None
    kappa = float(torch.as_tensor(node.output_scale).float().mean()) if node.output_scale is not None else 1.0
    return torch.tensor(float(offset) / max(kappa, 1e-12))


def _own_field(node: Any, value: Any, shape: tuple, device: Any) -> torch.Tensor:
    """A source's own field over its value: one value, one per element of the last axis, or one per element."""
    v = torch.as_tensor(value).detach().to(device=device, dtype=torch.float64).reshape(-1)
    if v.numel() == int(np.prod(shape)) and v.numel() not in (1, int(shape[-1])):
        return v.reshape(shape).clone()
    if v.numel() not in (1, int(shape[-1])):
        raise ReceivedShiftRefusal(f"{_name(node)!r}: a shift of {v.numel()} values over a value of shape {shape}")
    return v.expand(*shape).clone()


def _response(node: Any, kind: str, ins: list) -> torch.Tensor:
    """A host op's response to its inputs' fields: its own deployed forward on the fields less on zeros (float64)."""
    x = ins[0] if len(ins) == 1 else tuple(ins)
    zero = torch.zeros_like(ins[0]) if len(ins) == 1 else tuple(torch.zeros_like(f) for f in ins)
    def run(value: Any) -> torch.Tensor:
        return cast(torch.Tensor, node.forward_scale_normalized(value))

    out = run(x) - run(zero)
    if kind == "extremum":
        low = -(run(-ins[0] if len(ins) == 1 else tuple(-f for f in ins)) - run(zero))
        if not torch.equal(out, low):
            raise ReceivedShiftRefusal(f"{_name(node)!r}: a maximum over a field that varies inside its windows (the "
                                       f"winner depends on the data, so no bias holds the correction)")
    return out.to(torch.float64)


def _absolute(node: Any, ins: list) -> tuple:
    """A host carrier's operands in the units its module reads: each field at its own input scale where the op is
    armed (the deployed wrapper's ``x * s_i``), else as it arrives."""
    scales = node.per_source_scales if node.output_scale is not None else None
    if scales is None:
        return tuple(ins)
    return tuple(f * broadcast_scale_along(torch.as_tensor(s, dtype=f.dtype, device=f.device), f,
                                           node.wrap_channel_axis) for f, s in zip(ins, scales))


def _carry(order, deps, shapes, device, source, value, shifted, wanted) -> list:
    """``[(reader node, field)]``: ``source``'s field carried through its cone in execution order, each node by its own
    forward (a structural node's elements moved as its forward moves them, a host op by its declared response)."""
    fields = {id(source): _own_field(source, value, (1, *shapes[id(source)]), device)}
    found = []
    for node in order:
        if node is source or (wanted is not None and id(node) not in wanted):
            continue
        sources = [d for d in deps.get(node, []) if d is not None]
        if not any(id(d) in fields for d in sources):
            continue
        ins = [fields[id(d)] if id(d) in fields else torch.zeros((1, *shapes[id(d)]), dtype=torch.float64,
                                                                 device=device) for d in sources]
        if getattr(node, "perceptron", None) is not None:
            found.append((node, ins[0]))
            continue
        if hasattr(node, "operand_perceptrons"):
            raise ReceivedShiftRefusal(f"{_name(node)!r}: an operand product reads a nonzero field (its response is "
                                       f"the other operand times the field)")
        kind = _host_kind(node) if isinstance(node, ComputeOpMapper) else "structural"
        if kind == "reader":
            found.append((node, _absolute(node, ins)))
            continue
        if kind == "layernorm":
            width = len(getattr(node.module, "normalized_shape"))
            group = ins[0].reshape(*ins[0].shape[:ins[0].dim() - width], -1)
            if not bool((group == group[..., :1]).all()):
                raise ReceivedShiftRefusal(f"{_name(node)!r}: a LayerNorm reads a field that varies over its features "
                                           f"(it absorbs only a constant)")
            continue
        if id(node) in shifted:
            raise ReceivedShiftRefusal(f"{_name(node)!r}: a shifted op whose input already carries a field "
                                       f"({_name(source)!r}'s); a second shift's response through it is not declared")
        if kind in ("move", "linear", "extremum"):
            fields[id(node)] = _response(node, kind, ins)
        elif kind != "structural":
            raise ReceivedShiftRefusal(f"{_name(node)!r}: a host op of kind {kind!r} declares no response to a "
                                       f"nonzero received field")
        else:
            try:
                out = node.forward(ins[0] if len(ins) == 1 else tuple(ins)).to(torch.float64)
            except (RuntimeError, TypeError, ValueError, IndexError) as exc:
                raise ReceivedShiftRefusal(f"{_name(node)!r}: its forward cannot carry the field ({exc})") from exc
            if isinstance(node, ModuleMapper) and not (out.shape == ins[0].shape and torch.equal(out, ins[0])):
                raise ReceivedShiftRefusal(f"{_name(node)!r}: a forward-only module that changes the field (the "
                                           f"program delivers it unchanged)")
            fields[id(node)] = out
    return found


def _sample_shapes(model_repr: Any, order: list) -> dict:
    """Every node's per-sample value shape from one zero sample on the model's device, in eval mode with the run's
    random state preserved (no running statistic moves)."""
    modules = {id(m): m for node in order for m in node.modules()}.values()
    modes = [(m, m.training) for m in modules]
    try:
        for m, _ in modes:
            m.training = False
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            return mapper_output_shapes(model_repr)
    finally:
        for m, training in modes:
            m.training = training


def received_shifts(model_repr: Any, sources: Optional[dict] = None, *, readers: Any = None,
                    refused: Optional[dict] = None) -> dict:
    """``{id(reader): (reader node, {source name: field}, {source name: own value})}``, the one propagation of the class.
    ``sources`` (``{node: value}``) defaults to every stamped shift and lift; ``readers`` (perceptrons) limits the cone
    to their paths; with ``refused`` a source whose field meets a refusal is listed there instead of raising. Nothing is
    computed where no source exists."""
    order, deps, _ = model_repr._ensure_exec_graph()
    stamped = {id(n): n for n in order if stamped_value(n) is not None}
    given = dict(sources) if sources is not None else {n: stamped_value(n) for n in stamped.values()}
    if not given:
        return {}
    wanted = None
    if readers is not None:
        targets = {id(p) for p in readers}
        wanted = {id(n) for n in order if id(getattr(n, "perceptron", None)) in targets}
        for node in reversed(order):
            if id(node) in wanted:
                wanted.update(id(d) for d in deps.get(node, []) if d is not None)
    shapes = _sample_shapes(model_repr, order)
    param = next((p for node in order for p in node.parameters()), None)
    device = param.device if param is not None else torch.device("cpu")
    shifted = set(stamped) | {id(n) for n in given}
    out: dict = {}
    for source, value in given.items():
        try:
            found = _carry(order, deps, shapes, device, source, value, shifted, wanted)
        except ReceivedShiftRefusal as exc:
            if refused is None:
                raise
            refused[source] = str(exc)
            continue
        for node, field in found:
            entry = out.setdefault(id(node), (node, {}, {}))
            entry[1][_name(source)] = field
            entry[2][_name(source)] = torch.as_tensor(value)
    return out

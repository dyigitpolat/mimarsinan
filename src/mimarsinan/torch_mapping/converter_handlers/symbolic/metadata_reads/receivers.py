"""Which metadata reads a stand-in serves, and how its receiver follows a cast of the model (the package docstring)."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Set, cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import map_arg

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import RECEIVER_FACTORIES
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import attribute_at

__all__ = ["FOLLOWS", "PINNED", "MetadataReadDtypeRefusal", "metadata_read", "receiver_dtype_rule",
           "receiver_in_program", "stand_in"]

FOLLOWS, PINNED = "follows", "pinned"
_FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class MetadataReadDtypeRefusal(ToolchainRefusal, TypeError):
    """A metadata read whose receiver's dtype neither follows every floating cast of the program nor stays its traced
    dtype under every one (or cannot be decided): no stand-in computes the native value after every cast."""


def metadata_read(node: fx.Node) -> bool:
    """``node`` is a receiver factory whose every operand but the receiver is a literal (the package docstring)."""
    return node.op == "call_method" and node.target in RECEIVER_FACTORIES and bool(node.args) and isinstance(
        node.args[0], fx.Node) and node.all_input_nodes == [node.args[0]] and hasattr(
        node.args[0].meta.get("tensor_meta"), "dtype")


def _traced_dtype(node: fx.Node) -> torch.dtype:
    """The dtype the metadata read ``node``'s receiver was traced with."""
    return cast(fx.Node, node.args[0]).meta["tensor_meta"].dtype


def stand_in(node: fx.Node, device: torch.device) -> torch.Tensor:
    """A zero-element tensor of the dtype the metadata read ``node``'s receiver was traced with, on ``device``."""
    return torch.empty(0, dtype=_traced_dtype(node), device=device)


def receiver_in_program(graph: fx.Graph, node: fx.Node, buffer: str, rule: str) -> fx.Node:
    """The metadata read ``node``'s receiver in a value's program ``graph``: the stand-in buffer at ``buffer`` when its
    rule FOLLOWS, else (PINNED) a zero-element tensor made at every call at the traced dtype on the buffer's device."""
    held = graph.get_attr(buffer)
    if rule == FOLLOWS:
        return held
    device = graph.call_function(getattr, (held, "device"))
    return graph.call_function(torch.empty, ((0,),), {"dtype": _traced_dtype(node), "device": device})


def _on_meta(value: Any, dtype: Optional[torch.dtype]) -> Any:
    """A meta tensor of ``value``'s shape, at ``dtype`` when ``value`` is floating and ``dtype`` is given (a value a
    cast of the model casts), else at its own dtype; anything else as it is."""
    if isinstance(value, torch.Tensor):
        cast_to = dtype if dtype is not None and value.is_floating_point() else value.dtype
        return torch.empty(value.shape, dtype=cast_to, device="meta")
    return value


def _cast_with_the_model(gm: fx.GraphModule, target: str) -> bool:
    """Whether a cast of the model casts the value at ``target``: a registered parameter or buffer of its module. A
    tensor constant the trace attached as a plain attribute is computed natively at its own dtype at every call."""
    *path, last = target.split(".")
    owner = attribute_at(gm, ".".join(path)) if path else gm
    return isinstance(owner, nn.Module) and (last in owner._parameters or last in owner._buffers)


def _upstream(nodes: Sequence[fx.Node]) -> Set[fx.Node]:
    """``nodes`` and every node upstream of them, inputs included."""
    found: Set[fx.Node] = set()
    stack = list(nodes)
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(current.all_input_nodes)
    return found


def _meta_dtypes(gm: fx.GraphModule, receivers: Sequence[fx.Node], dtype: torch.dtype) -> Dict[fx.Node, Any]:
    """The dtype of every ``receiver`` when the program's floating state and inputs are ``dtype``: the receivers'
    computations evaluated once on meta tensors, a module called with meta copies of its parameters and buffers, a
    traced tensor constant at its own dtype (``_cast_with_the_model``). A computation meta cannot run raises."""
    cone, modules = _upstream(receivers), dict(gm.named_modules())
    env: Dict[fx.Node, Any] = {}
    for node in gm.graph.nodes:
        if node not in cone:
            continue
        if node.op == "placeholder":
            meta = node.meta.get("tensor_meta")
            if meta is None or not hasattr(meta, "dtype"):
                raise TypeError(f"input {node.name} carries no traced dtype")
            env[node] = torch.empty(tuple(meta.shape), device="meta",
                                    dtype=dtype if meta.dtype.is_floating_point else meta.dtype)
            continue
        if node.op == "get_attr":
            target = str(node.target)
            env[node] = _on_meta(attribute_at(gm, target), dtype if _cast_with_the_model(gm, target) else None)
            continue
        args, kwargs = map_arg((node.args, node.kwargs), env.__getitem__)
        if node.op == "call_module":
            module = modules[str(node.target)]
            state = {name: _on_meta(value, dtype)
                     for name, value in itertools.chain(module.named_parameters(), module.named_buffers())}
            env[node] = torch.func.functional_call(module, state, tuple(args), dict(kwargs))
        elif node.op == "call_method":
            env[node] = getattr(args[0], str(node.target))(*args[1:], **kwargs)
        else:
            env[node] = cast(Callable[..., Any], node.target)(*args, **kwargs)
    return {r: getattr(env[r], "dtype", None) for r in receivers}


def receiver_dtype_rule(gm: fx.GraphModule, reads: Sequence[fx.Node]) -> Dict[fx.Node, str]:
    """FOLLOWS or PINNED for every metadata read of ``reads`` (the package docstring); refused by name otherwise."""
    floating = [n for n in reads if _traced_dtype(n).is_floating_point]
    rule = {n: PINNED for n in reads if n not in floating}
    if not floating:
        return rule
    receivers = list(dict.fromkeys(cast(fx.Node, n.args[0]) for n in floating))
    seen: Dict[torch.dtype, Dict[fx.Node, Any]] = {}
    with torch.no_grad():
        for dtype in _FLOATING:
            try:
                seen[dtype] = _meta_dtypes(gm, receivers, dtype)
            except Exception as exc:  # a computation meta cannot run is undecided: refused by name
                raise MetadataReadDtypeRefusal(
                    f"metadata reads {', '.join(n.name for n in floating)}: their receivers' dtypes under a {dtype} "
                    f"cast of the program cannot be decided (meta evaluation: {type(exc).__name__}: "
                    f"{str(exc)[:200]})") from exc
    for node in floating:
        receiver, traced = cast(fx.Node, node.args[0]), _traced_dtype(node)
        got = {dtype: seen[dtype][receiver] for dtype in _FLOATING}
        if all(got[d] == d for d in _FLOATING):
            rule[node] = FOLLOWS
        elif all(got[d] == traced for d in _FLOATING):
            rule[node] = PINNED
        else:
            raise MetadataReadDtypeRefusal(
                f"metadata read {node.name}: its receiver {receiver.name} is {traced} as traced and, with the program "
                f"cast to {', '.join(str(d) for d in _FLOATING)}, "
                f"{', '.join(str(got[d]) for d in _FLOATING)}: it follows some casts only, so no stand-in computes "
                "the native value after every cast")
    return rule

"""Leaf-call inlining: an opaque call of a Python function or a bound method of one (an FX leaf function,
``torch.fx.wrap``) whose operands other than tensors are literals is replaced by its body as TorchDynamo captures it
(``body_capture``), spliced into
the program; the call stays opaque, its reason recorded (``OPAQUE_BODY``, which ``declared_operators.opaque_reason``
appends), when the capture does not hold for every batch (fix round 2 of the coverage fixes: round 1 traced the body
with FX on Proxies, and a Python decision FX made without recording a node was certified and wrong from batch 9).

The helper's Python never runs on Proxies or FakeTensors: Dynamo interprets its bytecode, and the helper runs only
natively, on real tensors, in the seeded runs that give its operands' metadata (``operand_samples``, batches 1, 2 and
3), the shape propagation and the probe, as the native model runs it. Calls are decided in traced order on a trial
copy of the program, then the operands' extents of every inlined call are proved in the converter's batch domain
(``leaf_operands``); a call whose operands are not is kept opaque and the trial runs again; the decisions of the
trial whose every inlined call is proved are applied to the program. A call stays opaque:

- when Dynamo's capture does not hold for every batch B >= 1 (``body_capture`` rules (a) to (e): not one graph, a
  guard on the batch, another path at batch 1 or where a symbol is 0 or 1, a write in place, a node outside the
  declared set, an output the splice does not rebuild), or the capture fails, its private API included;
- when an operand's extents are not the batch domain's affine forms (``leaf_operands``, review round 2: ``h[:8]``);
- when an operand is computed from a value outside the declared operator set (a call that stays opaque, a user
  function or module);
- when a traced operand is not a tensor at every trace batch (a shape, a number read from one), changes its dtype,
  device or rank between them, is a tensor of another type than ``torch.Tensor`` or ``nn.Parameter``, or is a
  distinct tensor of another operand's storage; when its operands were not reached by the seeded runs;
- when its target's binding is not a Python function's code (a ``functools.partial``, a callable object, a builtin);
  the call is bound as Python binds it, by the signature of a fresh function of the code, defaults, keyword defaults
  and closure (``operand_samples.code_call``, a bound method's ``__self__`` first), never by one the target reports
  (``__signature__``, ``__wrapped__``, ``__text_signature__``; fix round 3);
- when an operand is neither a traced value nor a declared literal, that signature is variadic or does not bind, or
  the call runs under a torch-function mode.

The splice: each graph input reads its operand, the operand's extent (``size``) or a tensor constant of the body (a
buffer under a fresh FX constant name); every other node is copied without Dynamo's metadata; the call's users read
the body's return value (a literal index of a returned tuple, list or dict its item).
"""

from __future__ import annotations

import inspect
import operator
from typing import Any, Dict, Tuple, Union, cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import map_arg

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    OPAQUE_BODY,
    is_conversion_time_operator,
    is_tensor_operator,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import declared_code_change
from mimarsinan.torch_mapping.fx_rewrites.body_capture import capture_body
from mimarsinan.torch_mapping.fx_rewrites.body_output import CapturedBody, GraphOutput, OperandOutput
from mimarsinan.torch_mapping.fx_rewrites.leaf_operands import (
    computed_from,
    impure_nodes,
    operand_reason,
    tensor_reason,
    unproved_operands,
)
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import (
    TRACE_BATCHES,
    CodeCall,
    OperandSamples,
    code_call,
    example_dtype,
    sample_operands,
)

__all__ = ["inline_leaf_calls", "trace_body"]


def trace_body(function: Any, arguments: Dict[str, Any], samples: Dict[str, Any]):
    """``function``'s body as Dynamo captures it for the call bound to ``arguments`` (``body_capture.capture_body``),
    or the reason the call stays opaque."""
    return capture_body(function, arguments, samples)


def _opaque_leaf_call(node: fx.Node) -> bool:
    """A call of a target outside the declared operator set (``opaque_reason``'s rule), not yet decided."""
    target = node.target
    return (node.op == "call_function" and target is not getattr and OPAQUE_BODY not in node.meta
            and not (is_conversion_time_operator(target) or is_tensor_operator(target)))


def _plain(value: Any) -> Any:
    """``value`` as the native call received it: FX holds a list or dict operand as its immutable form."""
    if type(value) in (list, immutable_list):
        return [_plain(item) for item in value]
    if type(value) in (dict, immutable_dict):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_plain(item) for item in value)
    return value


def _constant(owner: nn.Module, value: torch.Tensor) -> str:
    """A body constant registered on ``owner`` under a fresh FX constant name, as the root trace registers its own."""
    index = 0
    while hasattr(owner, f"_tensor_constant{index}"):
        index += 1
    name = f"_tensor_constant{index}"
    if isinstance(value, nn.Parameter):
        setattr(owner, name, value)
    else:
        owner.register_buffer(name, value)
    return name


def _substitute(call: fx.Node, value: Any) -> None:
    """Every use of ``call`` reads ``value`` (the body's output); a literal index of a returned tuple, list or dict
    reads its item."""
    if isinstance(value, fx.Node):
        call.replace_all_uses_with(value)
        return
    for user in list(call.users):
        index = user.args[1] if user.op == "call_function" and user.target is operator.getitem else None
        held = (type(index) is int and isinstance(value, (tuple, list)) and -len(value) <= index < len(value)) or (
            type(index) in (int, str) and isinstance(value, dict) and index in value)
        if held and user.args[0] is call and not user.kwargs:
            _substitute(user, cast(Any, value)[index])
            user.graph.erase_node(user)
            continue
        user.args = map_arg(user.args, lambda node: value if node is call else node)
        user.kwargs = map_arg(user.kwargs, lambda node: value if node is call else node)


def _splice(graph: fx.Graph, owner: nn.Module, call: fx.Node, body: CapturedBody, operands: Dict[str, fx.Node]):
    """``body`` spliced into ``graph`` before ``call``, its constants on ``owner``; ``call`` erased."""
    env: Dict[fx.Node, Any] = {}

    def value_of(tree: Any, outputs: list) -> Any:
        if isinstance(tree, GraphOutput):
            return outputs[tree.index]
        if isinstance(tree, OperandOutput):
            return operands[tree.name]
        if isinstance(tree, torch.Tensor):
            return graph.get_attr(_constant(owner, tree))
        if type(tree) in (tuple, list):
            return type(tree)(value_of(item, outputs) for item in tree)
        if type(tree) is dict:
            return {key: value_of(item, outputs) for key, item in tree.items()}
        return tree

    with graph.inserting_before(call):
        for node, binding in zip(body.placeholders(), body.bindings):
            kind = binding[0]
            if not node.users:
                continue  # an input the body does not read (an operand's extent Dynamo passes along)
            env[node] = (operands[binding[1]] if kind == "operand" else
                         graph.get_attr(_constant(owner, binding[1])) if kind == "constant" else
                         graph.call_method("storage_offset", (operands[binding[1]],)) if kind == "offset" else
                         graph.call_method(kind, (operands[binding[1]], binding[2])))
        for node in body.gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op == "get_attr":
                found: Any = body.gm
                for part in str(node.target).split("."):
                    found = getattr(found, part)
                env[node] = graph.get_attr(_constant(owner, found))
                continue
            env[node] = graph.node_copy(node, env.__getitem__)
            env[node].meta = {}
        value = value_of(body.output, [env[o] if isinstance(o, fx.Node) else o for o in body.outputs()])
    _substitute(call, value)
    graph.erase_node(call)


def _bound(call: fx.Node) -> Union[Tuple[CodeCall, inspect.BoundArguments], str]:
    """``call``'s target's code binding (its ``function`` is the body captured) and its operands bound as Python binds
    them, by that code (``code_call``: never a signature the target reports; fix round 3, a wrapper's ``__signature__``
    copied off the wrapped function was bound, and review round 2, ``functools.wraps``), defaults applied; else why
    not."""
    code = code_call(call.target)
    if isinstance(code, str):
        return code
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in code.signature.parameters.values()):
        return "its signature takes variadic parameters"
    try:
        bound = code.bind(call.args, call.kwargs)
    except TypeError:
        return "its operands do not bind to its signature"
    bound.apply_defaults()
    return code, bound


def _decide(call: fx.Node, rows: list, impure: Dict[fx.Node, str], captures: Dict[fx.Node, Any], key: fx.Node
            ) -> Union[Tuple[CapturedBody, Dict[str, fx.Node], Dict[str, tuple]], str]:
    """``call``'s captured body (cached under ``key``, the program's call), its tensor operands and their samples; else
    the reason it stays opaque."""
    found = _bound(call)
    if isinstance(found, str):
        return found
    code, bound = found
    if len(rows) < len(TRACE_BATCHES):
        return "its operands were not reached by the seeded runs at batches 1, 2 and 3"
    arguments: Dict[str, Any] = {}
    operands: Dict[str, fx.Node] = {}
    tensors: Dict[str, tuple] = {}
    for name, value in bound.arguments.items():
        if isinstance(value, fx.Node):
            found = computed_from(value, impure)
            if found is not None:
                return f"its operand {name!r} is computed from {found}"
            reason = tensor_reason(name, [row.get(name) for row in rows])
            if reason is not None:
                return reason
            arguments[name], operands[name], tensors[name] = None, value, tuple(row[name] for row in rows)
            continue
        reason = operand_reason(name, value)
        if reason is not None:
            return reason
        arguments[name] = _plain(value)
    if key not in captures:
        captures[key] = trace_body(code.function, arguments, tensors)
    body = captures[key]
    return body if isinstance(body, str) else (body, operands, tensors)


def _trial(gm: fx.GraphModule, calls: list, samples: OperandSamples, impure: Dict[fx.Node, str],
           forced: Dict[fx.Node, str], captures: Dict[fx.Node, Any], input_shape: tuple, device: Any):
    """Every call decided in traced order on a copy of ``gm`` (``forced``: the calls kept opaque), the inlined ones
    spliced there; then their operands' extents proved (``leaf_operands``). Returns the bodies and the reasons."""
    graph, env = fx.Graph(), {}
    graph.output(graph.graph_copy(gm.graph, env))
    trial = fx.GraphModule(gm, graph)
    trial_impure = {env[node]: reason for node, reason in impure.items()}
    bodies: Dict[fx.Node, CapturedBody] = {}
    reasons: Dict[fx.Node, str] = {}
    operands: Dict[fx.Node, Dict[str, fx.Node]] = {}
    tensors: Dict[fx.Node, Dict[str, tuple]] = {}
    for call in calls:
        found = forced.get(call) or _decide(env[call], samples.operands.get(call, []), trial_impure, captures, call)
        if isinstance(found, str):
            reasons[call], trial_impure[env[call]] = found, "a call that stays opaque"
            continue
        bodies[call], operands[call], tensors[call] = found
        _splice(graph, trial, env[call], found[0], found[1])
    return bodies, reasons, unproved_operands(trial, operands, tensors, input_shape, example_dtype(gm), device)


def inline_leaf_calls(gm: fx.GraphModule, tracer: fx.Tracer, input_shape: tuple, device: Any = "cpu"
                      ) -> fx.GraphModule:
    """Inlines every opaque leaf call of ``gm`` (traced by ``tracer`` for inputs of per-sample ``input_shape`` on
    ``device``) whose captured body holds for every batch and whose operands' extents the batch domain proves, in
    traced order; each call that stays records why (``OPAQUE_BODY``) and is outside the declared set for every later
    decision. Returns ``gm``."""
    under_mode = {node for node, _ in getattr(getattr(tracer, "modes", None), "recorded", ())}
    calls = []
    for node in [node for node in gm.graph.nodes if _opaque_leaf_call(node)]:
        code = code_call(node.target)  # a target whose binding is not its code's stays opaque by name, unsampled
        if isinstance(code, str):
            node.meta[OPAQUE_BODY] = code
        else:
            calls.append(node)
    if not calls:
        return gm
    rebound = declared_code_change()
    if rebound is not None:  # Dynamo would interpret the body with the rebound code: nothing is inlined
        for call in calls:
            call.meta[OPAQUE_BODY] = f"the declared operators are not torch's own code as imported: {rebound}"
        return gm
    samples = sample_operands(gm, calls, tuple(input_shape), device)
    impure = impure_nodes(gm, samples, set(calls))
    forced = {call: "its call runs under a torch-function mode the traced code enters" for call in calls
              if call in under_mode}
    captures: Dict[fx.Node, Any] = {}
    while True:  # each trial keeps at least one more call opaque, or is the last
        bodies, reasons, unproved = _trial(gm, calls, samples, impure, forced, captures, tuple(input_shape), device)
        if not unproved:
            break
        forced.update(unproved)
    for call in calls:
        if call in bodies:
            _, bound = cast(Tuple[Any, inspect.BoundArguments], _bound(call))
            _splice(gm.graph, gm, call, bodies[call], {
                name: value for name, value in bound.arguments.items() if isinstance(value, fx.Node)})
        else:
            call.meta[OPAQUE_BODY] = reasons[call]
    if bodies:
        gm.graph.lint()
        gm.recompile()
    return gm

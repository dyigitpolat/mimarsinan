"""A captured body ready to splice (``body_capture``): Dynamo's graph, normalized; how the root program reads each
graph input (``bindings``); the body's return value with each graph output a ``GraphOutput`` and each operand
returned as it is an ``OperandOutput`` (``output``); and the checks of rules (d) and (e).

A graph input is an operand (``("operand", name)``), an extent, stride or storage offset of one (``("size", name,
axis)``, ``("stride", name, axis)``, ``("offset", name)``), or a tensor the body reads from its globals, its closure or
a literal operand (``("constant", tensor)``: a buffer of the program, as FX holds a tensor root code reads); a body
that reads a stride or an offset stays opaque, so the splice binds operands, extents (``size``) and constants only. A
value is read without running user code: exact containers by index, attributes by ``inspect.getattr_static``.

The return value is rebuilt by Dynamo's residual code (``GraphRuntimeEnv.forward_callable``), with every graph
output a ``GraphOutput`` and every tensor operand an inert stand-in whose extent, stride and offset read as an inert
value. A profile hook refuses any call the residual code makes itself other than Dynamo's helpers and the graph's
stand-in, before it runs: a user class (a namedtuple, a dataclass) is never constructed. An attribute read of an
operand (``t.shape``) or an inert value in the result refuses too.
"""

from __future__ import annotations

import functools
import inspect
import operator
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import source as dynamo_source
from torch._dynamo import utils as dynamo_utils

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    foreign_value,
    is_conversion_time_operator,
    opaque_reason,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    declared_writes,
    in_place_form,
    schema_declares_no_alias,
)
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import code_signature

__all__ = ["BodyRefusal", "CapturedBody", "GraphOutput", "OperandOutput", "extract", "first_line"]


class BodyRefusal(Exception):
    """A capture whose body stays opaque; the message is the reason."""


def first_line(exc: BaseException) -> str:
    """``Type: first non-empty line`` of ``exc``'s message (300 characters at most)."""
    text = next((line for line in str(exc).splitlines() if line and not line.isspace()), "")
    return f"{type(exc).__name__}: {text[:300]}"


@dataclass(frozen=True)
class GraphOutput:
    index: int


@dataclass(frozen=True)
class OperandOutput:
    name: str


@dataclass
class CapturedBody:
    gm: fx.GraphModule
    bindings: List[Tuple[Any, ...]]
    output: Any

    def placeholders(self) -> List[fx.Node]:
        return [node for node in self.gm.graph.nodes if node.op == "placeholder"]

    def outputs(self) -> List[Any]:
        node = next(node for node in self.gm.graph.nodes if node.op == "output")
        return list(node.args[0]) if isinstance(node.args[0], (tuple, list)) else [node.args[0]]

    def check_declared(self) -> None:
        """(e) every node a declared operator; (d) nothing written in place: a write the body makes stays the opaque
        call's effect, which the probe observes (T3), and is never spliced into the program's own writes."""
        for node in self.gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op == "call_module":
                raise BodyRefusal(f"its body calls a module ({node.target}), which the splice does not bind")
            if node.op == "get_attr":
                found = foreign_value(functools.reduce(getattr, str(node.target).split("."), self.gm))
                if found is not None:
                    raise BodyRefusal(f"its body reads a constant that is {found}")
                continue
            reason = opaque_reason(node, _receiver_type(node), {})
            if reason is not None:
                raise BodyRefusal(f"its body calls {_name(node.target)} ({reason})")
            if _writes(node):
                raise BodyRefusal(f"its body writes in place ({node.name}: {_name(node.target)})")


def _writes(node: fx.Node) -> bool:
    """A write in place torch declares (its schema, ``inplace=True``) or names (``__imul__``, ``operator.iadd``)."""
    return bool(declared_writes(node, {})) or in_place_form(node) is not None


def _name(target: Any) -> str:
    return target if isinstance(target, str) else str(getattr(target, "__name__", target))


def _receiver_type(node: fx.Node) -> Any:
    receiver = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
    value = None if receiver is None else receiver.meta.get("example_value")
    return torch.Tensor if isinstance(value, torch.Tensor) else type(value)


# ----------------------------------------------------------------------------- normalization
def _operator_of(name: str) -> Tuple[Any, bool]:
    """The value function of ``operator`` a dunder method computes, and whether its operands are swapped."""
    if not (name.startswith("__") and name.endswith("__")):
        return None, False
    found = getattr(operator, name, None)
    if found is not None and is_conversion_time_operator(found):
        return found, False
    swapped = getattr(operator, "__" + name[3:], None) if name.startswith("__r") else None
    return (swapped, True) if swapped is not None and is_conversion_time_operator(swapped) else (None, False)


def _normalize(gm: fx.GraphModule) -> None:
    """A dunder method on a tensor as its ``operator`` function; a call with no users dropped when it has no node
    operand (trace-time Python Dynamo records), or when its aten schema is known and annotates no write and no alias
    (a value never read); any other call stays, so rules (d) and (e) see it (review round 2: a dead ``__imul__``, which
    has no schema, was dropped with its write)."""
    graph = gm.graph
    for node in list(graph.nodes):
        function, swapped = _operator_of(node.target) if node.op == "call_method" and isinstance(
            node.target, str) else (None, False)
        if function is None or len(node.args) != 2 or node.kwargs:
            continue
        with graph.inserting_before(node):
            new = graph.call_function(function, tuple(reversed(node.args)) if swapped else node.args)
        new.meta = dict(node.meta)
        node.replace_all_uses_with(new)
        graph.erase_node(node)
    for node in reversed(list(graph.nodes)):
        if node.op not in ("call_function", "call_method") or node.users:
            continue
        if not node.all_input_nodes or (not _writes(node) and schema_declares_no_alias(node)):
            graph.erase_node(node)


# ----------------------------------------------------------------------------- graph inputs
def _value_of(source: Any, target: types.FunctionType, arguments: Dict[str, Any]) -> Any:
    """The object a Dynamo source names, read without running user code; ``BodyRefusal`` when it is none of those."""
    if isinstance(source, dynamo_source.LocalSource):
        if source.local_name in arguments:
            return arguments[source.local_name]
        names = target.__code__.co_freevars
        if source.local_name in names and target.__closure__:
            return target.__closure__[names.index(source.local_name)].cell_contents
    elif isinstance(source, dynamo_source.GlobalSource) and source.global_name in target.__globals__:
        return target.__globals__[source.global_name]
    elif isinstance(source, (dynamo_source.GetItemSource, dynamo_source.DictGetItemSource)):
        base = _value_of(source.base, target, arguments)
        index = source.unpack_slice() if isinstance(source, dynamo_source.GetItemSource) and (
            source.index_is_slice) else source.index
        if type(base) in (list, tuple, dict) and not isinstance(index, fx.Node):
            return base[index]
    elif isinstance(source, dynamo_source.AttrSource):
        base = _value_of(source.base, target, arguments)
        tables = [vars(base).get(t, {}) for t in ("_parameters", "_buffers")] if isinstance(base, nn.Module) else []
        for table in tables:
            if source.member in table:
                return table[source.member]
        found = inspect.getattr_static(base, source.member, None)
        if found is not None and not hasattr(type(found), "__get__"):
            return found
    raise BodyRefusal(f"its body reads {source.name}, which the splice does not bind")


def _binding(source: Any, target: types.FunctionType, arguments: Dict[str, Any], operands) -> Tuple[Any, ...]:
    base = getattr(source, "base", None)
    if isinstance(source, dynamo_source.TensorPropertySource) and isinstance(base, dynamo_source.LocalSource) and (
            base.local_name in operands):
        kind = {dynamo_source.TensorProperty.SIZE: "size", dynamo_source.TensorProperty.STRIDE: "stride"}.get(
            source.prop, "offset")
        return (kind, base.local_name) if kind == "offset" else (kind, base.local_name, source.idx)
    if isinstance(source, dynamo_source.LocalSource) and source.local_name in operands:
        return ("operand", source.local_name)
    value = _value_of(source, target, arguments)
    if type(value) in (torch.Tensor, nn.Parameter):
        return ("constant", value)
    raise BodyRefusal(f"its body reads {source.name}, a value of type {type(value).__name__} as a graph input")


# ----------------------------------------------------------------------------- the return value
class _InertValue:
    """An operand's extent, stride or offset in the residual code: it only reaches the graph's stand-in."""


_INERT = _InertValue()


class _Inert:
    """A tensor operand in the residual code: its extent, stride and offset read as ``_INERT``, anything else
    refuses."""

    def __init__(self, name: str):
        self.__dict__["name"] = name

    def size(self, *_args: Any) -> _InertValue:
        return _INERT

    stride = storage_offset = size

    def __getattr__(self, attribute: str) -> Any:
        raise BodyRefusal(f"its output is rebuilt from its operand's {attribute!r}")


def _allowed_codes(stand_in: Any) -> set:
    codes = {stand_in.__code__, _Inert.__getattr__.__code__}  # an operand's attribute refuses by its own name
    for name in ("call_size", "call_stride", "call_storage_offset", "record_pregraph_bytecode_enter",
                 "record_pregraph_bytecode_exit"):
        function = getattr(dynamo_utils, name, None)
        while function is not None:
            codes.add(getattr(function, "__code__", None))
            function = getattr(function, "__wrapped__", None)
    return codes


def _run_residual(function: Any, stand_in: Any, args: tuple, kwargs: dict) -> Any:
    """``function`` (the residual code) with a profile hook refusing any call it makes other than ``stand_in`` and
    Dynamo's helpers, before the callee runs."""
    code, allowed = function.__code__, _allowed_codes(stand_in)

    def hook(frame, event, arg):
        if event == "call" and frame.f_back is not None and frame.f_back.f_code is code and frame.f_code not in allowed:
            raise BodyRefusal(f"its output is rebuilt by a call of {frame.f_code.co_name}")
        if event == "c_call" and frame.f_code is code:
            raise BodyRefusal(f"its output is rebuilt by a call of {getattr(arg, '__qualname__', arg)}")

    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        return function(*args, **kwargs)
    finally:
        sys.setprofile(previous)


def _leaf(value: Any) -> Any:
    if isinstance(value, GraphOutput):
        return value
    if isinstance(value, _Inert):
        return OperandOutput(value.__dict__["name"])
    if type(value) in (tuple, list):
        return type(value)(_leaf(item) for item in value)
    if type(value) is dict and all(foreign_value(key) is None for key in value):
        return {key: _leaf(item) for key, item in value.items()}
    if type(value) in (torch.Tensor, nn.Parameter):
        return value
    if not callable(value) and type(value) not in (dict, slice) and foreign_value(value) is None:
        return value
    raise BodyRefusal(f"its body returns a value of type {type(value).__name__}, which the splice does not rebuild")


def _rebuild(captured: Any, function: types.FunctionType, arguments: Dict[str, Any], operands, count: int) -> Any:
    runtime = captured.graph_capture_output.get_runtime_env()
    outputs = tuple(GraphOutput(i) for i in range(count))

    def stand_in(*_args: Any) -> tuple:
        return outputs

    backend = captured.backend_input.backend_id if captured.backend_input is not None else "_mimarsinan_no_graph"
    residual = runtime.forward_callable(backend, stand_in)
    bound = inspect.BoundArguments(code_signature(function), {  # type: ignore[arg-type]
        name: _Inert(name) if name in operands else value for name, value in arguments.items()})
    return _leaf(_run_residual(residual, stand_in, tuple(bound.args), dict(bound.kwargs)))


def extract(captured: Any, target: types.FunctionType, function: types.FunctionType, arguments: Dict[str, Any],
            operands) -> CapturedBody:
    """The body of one capture: its normalized graph, the bindings of its inputs and its return value."""
    if captured.backend_input is not None:
        gm = captured.backend_input.graph_module
    else:
        gm = fx.GraphModule(nn.Module(), fx.Graph())
        gm.graph.output(())
    sources = captured.graph_capture_output.output_graph.export_metadata.graph_input_idx_to_local_source
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(sources):
        raise BodyRefusal(f"its capture has {len(placeholders)} inputs and {len(sources)} sources")
    bindings = [_binding(sources[i], target, arguments, operands) for i in sorted(sources)]
    _normalize(gm)
    for node, binding in zip(placeholders, bindings):
        if binding[0] in ("stride", "offset") and node.users:
            raise BodyRefusal(f"its body reads the {binding[0]} of its operand {binding[1]!r}, which the converted "
                              "program does not compute")
    body = CapturedBody(gm, bindings, None)
    body.output = _rebuild(captured, function, arguments, operands, len(body.outputs()))
    return body

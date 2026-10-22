"""FX-based graph extraction with shape propagation for native PyTorch models."""

from __future__ import annotations

import collections
import threading
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.immutable_collections import immutable_dict, immutable_list

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.binding import canonical_graph_keywords
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import declare_leaf_module
from mimarsinan.torch_mapping.converter_handlers.symbolic.entered_modes import (
    EnteredModes,
    mode_change,
    run_under_mode,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import call_code_change, call_reason
from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import (
    WriteRecorder,
    check_record,
    define_writes,
    settle_writes,
)


# ChannelsLastBatchNorm1d must stay a call_module node (not traced through) so
# the absorption plan can fold Linear -> BN -> act into one Perceptron. Its forward is
# torch's own BatchNorm on a permuted view, so it is declared to the symbolic pass.
_LEAF_MODULES: Tuple[type, ...] = (
    nn.MultiheadAttention,
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    declare_leaf_module(ChannelsLastBatchNorm1d),
)


_KEPT_CONTAINERS = (tuple, list, dict, torch.Size, immutable_list, immutable_dict)
"""Container types FX records as they are (or as an equivalent: ``torch.Size`` as a tuple of its entries)."""

_REFERENCE_CLASS = collections.namedtuple("_Reference", "field")  # held: a collected class's dict is cleared
_REFERENCE_NAMEDTUPLE = vars(_REFERENCE_CLASS)
_NAMEDTUPLE_CODE = ("_make", "_replace", "__repr__", "_asdict", "__getnewargs__")
"""The methods ``collections.namedtuple`` writes, the same code for every class it makes."""
_NAMEDTUPLE_DATA = frozenset({"__doc__", "__slots__", "_fields", "_field_defaults", "__match_args__", "__module__",
                              "__qualname__", "__annotations__", "__orig_bases__", "__slotnames__"})
"""The data a namedtuple class holds (``typing.NamedTuple`` adds its annotations and original bases; ``copyreg``
caches ``__slotnames__`` on the first copy of an instance)."""


def _code(value: Any) -> Any:
    return getattr(getattr(value, "__func__", value), "__code__", None)


def _plain_namedtuple(cls: type) -> bool:
    """``cls`` is a namedtuple class as ``collections.namedtuple`` (or ``typing.NamedTuple``) makes it, unmodified
    (review round 13): a direct ``tuple`` subclass whose class dict holds the generated names only, its ``__new__``
    the namedtuple template's and its methods the namedtuple code. FX records such a literal as a call of the class
    on its iterated fields, which rebuilds the same tuple; a class with its own ``__new__``, ``__iter__`` or
    ``__getitem__`` would build another."""
    names = vars(cls)
    fields = names.get("_fields")
    if cls.__bases__ != (tuple,) or type(cls) is not type or type(fields) is not tuple:
        return False
    for name, value in names.items():
        if name in fields:
            plain = type(value) is type(_REFERENCE_NAMEDTUPLE["field"])
        elif name in _NAMEDTUPLE_CODE:
            plain = _code(value) is not None and _code(value) is _code(_REFERENCE_NAMEDTUPLE[name])
        elif name == "__new__":
            template = getattr(getattr(value, "__func__", None), "__globals__", {})
            plain = template.get("_tuple_new") is tuple.__new__ and str(template.get("__name__")).startswith(
                "namedtuple_")
        else:
            plain = name in _NAMEDTUPLE_DATA
        if not plain:
            return False
    return True


def _retyped_literal(value: Any) -> Optional[Any]:
    """The first literal container of ``value`` FX would re-type: a strict subclass of ``tuple``, ``list`` or ``dict``
    other than ``torch.Size`` and an unmodified namedtuple class (FX records it as the plain container, dropping its
    own methods, or a tuple subclass with ``_fields`` as a call of its class on its iterated fields)."""
    kind = type(value)
    if isinstance(value, (tuple, list, dict)) and kind not in _KEPT_CONTAINERS and not _plain_namedtuple(kind):
        return value
    if isinstance(value, dict):
        items = [*value.keys(), *value.values()]
    elif isinstance(value, (tuple, list)):
        items = list(value)
    elif kind is slice:
        items = [value.start, value.stop, value.step]
    else:
        return None
    return next((found for found in map(_retyped_literal, items) if found is not None), None)


_UNDISPATCHED = ("placeholder", "get_attr", "output")
"""The FX node kinds that record no call: a mode does not see an input, an attribute read or the return."""


class _MimarsinanTracer(fx.Tracer):
    """Custom tracer that treats certain modules as leaves, refuses a literal operand FX would re-type, and records
    the calls it makes on Proxies under a torch-function mode entered inside the traced code (``entered_modes.py``;
    a dispatch mode entered there is refused). A torch function called under such a mode is dispatched through it,
    and the mode's code runs with the mode popped: FX records that code as it runs natively."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, _LEAF_MODULES):
            return True
        return super().is_leaf_module(m, module_qualified_name)

    def trace(self, root, concrete_args=None):
        self.modes = EnteredModes()  # the model's environment: the impurity rule's (``purity.py``)
        return super().trace(root, concrete_args)

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        operator = getattr(target, "__name__", None) or str(target)
        found = _retyped_literal((args, kwargs))
        if found is not None:
            base = next(b.__name__ for b in (tuple, list, dict) if isinstance(found, b))
            kind_name = type(found).__name__
            how = (f"{base} other than a namedtuple class as collections.namedtuple makes it; FX records it as a call "
                   f"of {kind_name} on its iterated fields, so the traced graph would rebuild it with {kind_name}'s "
                   "own methods" if isinstance(found, tuple) and hasattr(found, "_fields") else
                   f"{base}; FX records it as a plain {base}, so the traced graph would call {base}'s own methods in "
                   f"place of {kind_name}'s")
            raise TracingError(
                f"{operator} (the FX node being traced, {kind}): a literal operand of type {kind_name}, a subclass of "
                f"{how} and differ from the native model, so the model is refused"
            )
        modes = getattr(self, "modes", None)
        entered = modes.entered() if modes is not None and kind not in _UNDISPATCHED else None
        if entered is not None and EnteredModes.dispatching(entered):
            raise TracingError(
                f"{operator} (the FX node being traced, {kind}): a dispatch mode of type {type(entered).__name__} is "
                "active while FX traces it, entered inside the traced code; it works on aten operators, below what "
                "FX records, so the traced graph would differ from the native model, so the model is refused"
            )
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        if entered is not None and modes is not None:
            modes.recorded.append((proxy.node, entered))
        return proxy


def _check_entered_modes(graph_module: fx.GraphModule, modes: EnteredModes) -> bool:
    """Every call recorded under a torch-function mode entered in the traced code computes what it computes without
    the mode (``entered_modes.mode_change``); else a tensor method or operator runs with the mode entered at run time
    (``entered_modes.run_under_mode``) and a module's call is refused by name. True when a node was rewritten."""
    rewritten = False
    for node, mode in modes.recorded:
        change = mode_change(graph_module, node, mode)
        if change is None:
            continue
        if run_under_mode(graph_module, node, mode):
            rewritten = True
            continue
        operator = getattr(node.target, "__name__", None) or str(node.target)
        raise TracingError(
            f"{operator} (FX node {node.name!r}, {node.op}): a mode of type {type(mode).__name__} is active while FX "
            f"traces it, entered inside the traced code, and it {change}; FX records calls on Proxies without "
            "dispatching them through the mode, so the traced graph would differ from the native model, so the model "
            "is refused"
        )
    return rewritten


class TracingError(ToolchainRefusal):
    """Raised when a model cannot be symbolically traced."""


class RootCallError(TracingError):
    """The model's native call is not the program FX traces (review rounds 10 and 11): FX traces
    ``getattr(type(root), 'forward')`` and never runs the root's call, so user code there (a hook on the root, a global
    module hook, an instance ``forward``, a class's own ``__call__``, a name the call reads bound by the class, a global
    it reads rebound) would run natively only (``symbolic/module_call.py``); and the converted model takes one input, so
    a forward parameter after the first that the program reads (natively its default) has no counterpart.
    ``operator``: the root class, or ``placeholder``; ``node_name``: the placeholder's FX node, ``None`` for the root
    call (no FX node)."""

    def __init__(self, operator: str, node_name: Optional[str], reason: str):
        self.operator, self.node_name, self.reason = operator, node_name, reason
        where = f"FX node {node_name!r}" if node_name is not None else (
            "the root module, no FX node: its call is outside the traced program")
        consequence = ("the converted model takes one input" if node_name is not None else
                       f"FX traces {operator}.forward and never runs the module's call: the traced program would "
                       "differ from the native model")
        super().__init__(f"{operator} ({where}): {reason}; {consequence}, so the model is refused")


# FX patches nn.Module.__call__/__getattr__ non-atomically; capture the true originals at import time and serialize tracing so a stale cross-thread wrapper cannot leak.
_TRACE_LOCK = threading.Lock()
_ORIG_MODULE_CALL = nn.Module.__call__
_ORIG_MODULE_GETATTR = nn.Module.__getattr__


def _root_call_reason(root: nn.Module) -> Optional[str]:
    """Why ``root``'s native call runs code its class's ``forward`` does not (``module_call``: torch's call code and
    the globals it reads as imported, then every name it reads on ``root``); ``None`` when it runs that alone."""
    return call_code_change() or call_reason(root)


def _read_placeholder(graph: fx.Graph) -> Optional[RootCallError]:
    """A forward parameter after the first that the program reads: natively its default, an input of its own here."""
    for node in [n for n in graph.nodes if n.op == "placeholder"][1:]:
        if node.users:
            default = f"whose default is a value of type {type(node.args[0]).__name__}" if node.args else (
                "with no default")
            return RootCallError("placeholder", node.name,
                                 f"a forward parameter after the first that the program reads, {default}")
    return None


def _ensure_module_call_unpatched() -> None:
    """If a stale FX wrapper is still installed on nn.Module, restore originals."""
    current_call = nn.Module.__call__
    if getattr(current_call, "__fx_already_patched", False):
        nn.Module.__call__ = _ORIG_MODULE_CALL
    current_getattr = nn.Module.__getattr__
    if getattr(current_getattr, "__fx_already_patched", False):
        nn.Module.__getattr__ = _ORIG_MODULE_GETATTR


def _propagate(graph_module: fx.GraphModule, example_input: torch.Tensor):
    """The shape propagation (``ShapeProp``) at batch 1, recording the writes of the run (``operand_writes``)."""
    recorder = WriteRecorder(graph_module)
    recorder.propagate(example_input)
    return recorder.record


def trace_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device | str = "cpu",
) -> fx.GraphModule:
    """Trace a native PyTorch model and annotate node shapes via ShapeProp; a write in place is a definition of its
    operand (``operand_writes``, review round 15).

    ``input_shape`` excludes the batch dim, e.g. ``(3, 32, 32)``.
    Raises ``TracingError`` if the model is not symbolically traceable, ``RootCallError`` if its native call is not
    the traced program (user code in the root's call, a read forward parameter after the first), ``OperandWriteError``
    for a write in place the conversion cannot take as a definition.
    """
    model = model.eval().to(device)

    with _TRACE_LOCK:
        _ensure_module_call_unpatched()
        try:
            try:
                reason = _root_call_reason(model)
                if reason is not None:
                    raise RootCallError(type(model).__name__, None, reason)
                tracer = _MimarsinanTracer()
                graph = tracer.trace(model)
                unread = _read_placeholder(graph)
                if unread is not None:
                    raise unread
                graph_module = fx.GraphModule(model, graph)
                writes = define_writes(graph_module)
            except TracingError:
                raise
            except Exception as exc:
                raise TracingError(
                    f"Failed to symbolically trace the model ({type(model).__name__}): {exc}\n"
                    f"Models with data-dependent control flow (if/for on tensor values) cannot be traced.  Consider "
                    f"refactoring or wrapping dynamic sections in leaf modules."
                ) from exc

            # ShapeProp runs a real forward, so the example input carries the MODEL's parameter dtype (fp64/fp16 too).
            param = next(model.parameters(), None)
            example_dtype = param.dtype if param is not None else torch.get_default_dtype()
            example_input = torch.randn(1, *input_shape, device=device, dtype=example_dtype)
            try:
                record = _propagate(graph_module, example_input)
            except Exception as exc:
                raise TracingError(
                    f"Shape propagation failed for {type(model).__name__}: {exc}\n"
                    f"Ensure the model accepts input of shape (1, {', '.join(str(d) for d in input_shape)})."
                ) from exc
            if _check_entered_modes(graph_module, tracer.modes):
                record = _propagate(graph_module, example_input)  # the rewritten calls run under their modes
            canonical_graph_keywords(graph_module)  # torch's numpy names read as its own (``axis=`` as ``dim=``)
            writes, routed = settle_writes(graph_module, writes, record)
            if routed:
                record = _propagate(graph_module, example_input)  # a reader after an observed write was routed
            check_record(graph_module, writes, record)
        finally:
            _ensure_module_call_unpatched()

    return graph_module

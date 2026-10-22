"""Modes entered inside the traced code (review rounds 13 and 14).

A torch-function mode entered with ``with`` inside ``forward`` is active while FX traces the code under it. A torch
function called there dispatches through the mode as natively, and FX records the mode's own code. FX records three
kinds of call directly on Proxies, without that dispatch: a tensor method (``y.size(0)``), a Python operator on a
tensor (``y * b``) and a leaf module's call, whose torch functions dispatch through the mode natively. The tracer
records each such node with the mode it ran under (``EnteredModes``). After shape propagation, every recorded call
whose native form dispatches through the mode (a method or an operator of a tensor, a module's call) is traced again,
alone, once with the mode entered and once without it (``mode_change``):

- the same graph: the mode changes nothing the call computes (a counting or logging mode), and the recorded node is
  the native call;
- a tensor method or operator whose graph under the mode is another one, or one FX cannot trace (the mode's code reads
  a traced value, ``min(size, 8)``, or calls the method on the Proxy, which only a tensor takes): the node runs at run
  time with the mode entered (``UnderMode``), as natively; it is a call outside the declared operator set, so the
  program is impure (no static shape, stage 2 refused by name);
- a module's call whose graph under the mode is another one, or one FX cannot trace: refused by name.

A dispatch mode works on aten operators, below what FX records: one entered inside the traced code is refused by name
as the tracer meets it. Python arithmetic on sizes (``b + 1``) dispatches through no mode natively, and is kept.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import map_arg
from torch.utils._python_dispatch import TorchDispatchMode

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.imported_code import user_modes

__all__ = ["EnteredModes", "UnderMode", "mode_change", "run_under_mode"]

_REFLECTED = {operator.lt: "__gt__", operator.le: "__ge__", operator.gt: "__lt__", operator.ge: "__le__",
              operator.eq: "__eq__", operator.ne: "__ne__"}
_UNARY = (operator.neg, operator.pos, operator.invert, operator.abs)


class EnteredModes:
    """The modes a trace began under (the model's environment, the impurity rule's) and the nodes recorded under a
    torch-function mode entered inside the traced code."""

    def __init__(self) -> None:
        self.ambient: Tuple[Any, ...] = tuple(user_modes())
        self.recorded: List[Tuple[fx.Node, Any]] = []

    def entered(self) -> Optional[Any]:
        """The first active mode that is not torch's own and was not active when the trace began."""
        return next((mode for mode in user_modes() if not any(mode is held for held in self.ambient)), None)

    @staticmethod
    def dispatching(mode: Any) -> bool:
        return isinstance(mode, TorchDispatchMode)


@dataclass(frozen=True)
class _Slot:
    """The position of an FX node among a recorded call's operands."""

    index: int


class UnderMode:
    """A tensor method or operator FX recorded under a mode entered in the traced code: run with that mode entered.
    It holds the mode the trace created; a mode whose answers depend on its own state across calls is outside what it
    reproduces."""

    def __init__(self, mode: Any, call: Callable[..., Any], name: str):
        self.mode, self.call, self.__name__ = mode, call, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.mode:
            return self.call(*args, **kwargs)


def _method(name: str, receiver: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(receiver, name)(*args, **kwargs)


def run_under_mode(gm: fx.GraphModule, node: fx.Node, mode: Any) -> bool:
    """Rewrite a recorded tensor method or operator ``node`` into a call of ``UnderMode``; False for a module's call."""
    if node.op == "call_method":
        call: Callable[..., Any] = functools.partial(_method, str(node.target))
    elif node.op == "call_function":
        call = cast(Callable[..., Any], node.target)
    else:
        return False
    node.op, node.target = "call_function", UnderMode(mode, call, getattr(node.target, "__name__", str(node.target)))
    gm.recompile()
    return True


def _tensor(arg: Any) -> bool:
    kind = arg.meta.get("type") if isinstance(arg, fx.Node) else type(arg)
    return isinstance(kind, type) and issubclass(kind, torch.Tensor)


def _native_call(gm: fx.GraphModule, node: fx.Node) -> Optional[Tuple[Callable[..., Any], Any]]:
    """The call ``node`` makes natively through torch's function dispatch, and the root to trace it from (a module, or
    ``None`` for a function of the operands); ``None`` when the native call dispatches through no mode."""
    if node.op == "call_module":
        module = gm.get_submodule(str(node.target))
        return module, module
    if node.op == "call_method" and node.args and _tensor(node.args[0]):
        method = getattr(torch.Tensor, str(node.target), None)
        return ((lambda tensor, *rest, **named: torch.overrides.handle_torch_function(
            method, (tensor,), tensor, *rest, **named)), None) if method is not None else None
    if node.op != "call_function" or not any(_tensor(arg) for arg in node.args):
        return None
    if node.target is getattr:  # a tensor attribute dispatches natively; FX reads it on the Proxy
        descriptor = getattr(torch.Tensor, str(node.args[1]), None)
        getter = getattr(descriptor, "__get__", None)
        return ((lambda tensor, _name: torch.overrides.handle_torch_function(getter, (tensor,), tensor)), None) \
            if getter is not None else None
    name = getattr(node.target, "__name__", "").strip("_")
    if node.target in _UNARY or (len(node.args) == 2 and _tensor(node.args[0])):
        method = getattr(torch.Tensor, f"__{name}__", None)
        return (method, None) if method is not None else None
    reflected = _REFLECTED.get(node.target, f"__r{name}__")
    method = getattr(torch.Tensor, reflected, None)
    return ((lambda first, second: method(second, first)), None) if method is not None else None


class _Inside(fx.Tracer):
    """Traces every module's code: a leaf module's torch functions dispatch through a mode natively."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return False


def _canonical(graph: fx.Graph) -> List[Tuple[Any, ...]]:
    index = {node: i for i, node in enumerate(graph.nodes)}

    def arg(value: Any) -> Any:
        return map_arg(value, lambda n: ("node", index[n]))

    return [(n.op, n.target if isinstance(n.target, str) else id(n.target), arg(n.args), arg(n.kwargs))
            for n in graph.nodes]


def _subtrace(call: Callable[..., Any], root: Optional[nn.Module], node: fx.Node, mode: Any):
    """The canonical graph of ``node``'s call traced alone, under ``mode`` when given; or the exception."""
    slots: List[fx.Node] = []

    def slot(n: fx.Node) -> Any:
        slots.append(n)
        return _Slot(len(slots) - 1)

    template = map_arg((node.args, node.kwargs), slot)

    def run(*proxies: Any) -> Any:
        args, kwargs = _filled(template, proxies)
        return call(*args, **kwargs)

    def trace() -> fx.Graph:
        if root is not None:
            return _Inside().trace(root)
        names = ", ".join(f"p{i}" for i in range(len(slots)))
        fn = eval(f"lambda {names}: run({names})", {"run": run})  # noqa: S307 (a fixed-arity wrapper of ``run``)
        return _Inside().trace(fn)

    if mode is None:
        graph, exc = guarded(trace)
    else:
        with mode:
            graph, exc = guarded(trace)
    return exc if exc is not None or graph is None else _canonical(graph)


def _filled(template: Any, proxies: Sequence[Any]) -> Any:
    if isinstance(template, _Slot):
        return proxies[template.index]
    if isinstance(template, (tuple, list)):
        return type(template)(_filled(item, proxies) for item in template) if type(template) in (tuple, list) \
            else tuple(_filled(item, proxies) for item in template)
    if isinstance(template, dict):
        return {key: _filled(item, proxies) for key, item in template.items()}
    return template


def mode_change(gm: fx.GraphModule, node: fx.Node, mode: Any) -> Optional[str]:
    """Why the mode ``node`` was recorded under changes what the call computes; ``None`` when the call dispatches
    through no mode natively, or when its graph is the same with the mode entered and without it."""
    native = _native_call(gm, node)
    if native is None:
        return None
    call, root = native
    found, plain = _subtrace(call, root, node, mode), _subtrace(call, root, node, None)
    if isinstance(found, BaseException):
        first = str(found).splitlines()[0] if str(found) else ""
        return f"runs code on the call that FX cannot trace ({type(found).__name__}: {first[:160]})"
    if isinstance(plain, BaseException) or found != plain:
        return "changes what the call computes (its graph traced under the mode differs from the one without it)"
    return None

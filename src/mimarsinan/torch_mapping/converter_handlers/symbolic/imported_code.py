"""Torch's declared code as imported: the snapshot ``torch_rebinding`` compares with (review rounds 9 to 11).

The declared operators (``declared_operators.py``) are torch's own code only while the code they run is the code
imported with mimarsinan. Four kinds of state hold that code; each is snapshotted when it is declared
(``remember_namespaces``, ``remember_classes``, ``remember_code_of``):

- every binding of a declared namespace (``TENSOR_NAMESPACES``, ``operator``, ``math``), of each declared class
  (``torch.Tensor``, ``nn.Parameter``, ``nn.Module``, each leaf module class) and of every torch class in its method
  resolution order, by identity: a rebound name changes what a declared call runs (``torch.Tensor.size = f``,
  ``F.linear = f``); so does a name of any kind added to such a class that differs from the binding it inherits (a
  method; a property, ``torch.Tensor.width = property(...)``; data that torch's code reads with a default), except the
  caches Python itself writes on a class (``copyreg``'s ``__slotnames__``, ``_PYTHON_CACHES``);
- the code bindings (callables, descriptors, modules) among the globals of every torch module that defines declared
  code (a declared function's ``__globals__``, the module of each class above), by identity: a rebound one changes
  what that code calls (``torch.nn.modules.conv.F``, the global ``_ConvNd._conv_forward`` reads), and so does a global
  added there that shadows a builtin. Plain data among those globals is torch's own state, which torch rebinds itself
  (``torch.random`` flips a warn-once flag), and is not compared;
- the mutable state of every Python function the bindings above hold (review round 11): a function mutated in place
  keeps its identity, so its ``__code__``, ``__defaults__``, ``__kwdefaults__`` (and its entries) and each closure
  cell's contents are held as imported (``_FUNCTIONS``), through a ``staticmethod``, a ``classmethod``, a bound method,
  a ``property``'s accessors and an ``lru_cache`` wrapper, and through the functions in closure cells
  (``F.max_pool2d``'s ``if_false``);
- torch's own torch-function and dispatch mode classes, by identity, as loaded (``torch.device(...)``'s
  ``DeviceContext`` among them): any other active mode is a user mode, whatever module its class claims.

``rebinding_of`` is the same comparison restricted to given classes (``module_call.py``: ``nn.Module``'s call). A patch
installed before mimarsinan is imported is part of the snapshot and is not detected.
"""

from __future__ import annotations

import builtins
import functools
import sys
import types
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from torch.overrides import TorchFunctionMode
from torch.overrides import _get_current_function_mode_stack as _function_modes
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack as _dispatch_modes
from torch.utils._device import DeviceContext

__all__ = ["is_code", "rebinding_of", "remember_classes", "remember_code_of", "remember_namespaces",
           "torch_rebinding", "user_modes"]

_NAMESPACES: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
"""id -> (a declared namespace or class, its bindings: the objects themselves, held so no id is reused)."""

_GLOBALS: Dict[int, Tuple[Dict[str, Any], str, Dict[str, Any]]] = {}
"""id of a module's globals -> (the globals, the module's name, its code bindings: the objects themselves)."""

_FUNCTIONS: Dict[int, Tuple[types.FunctionType, str, Tuple[Any, ...], Tuple[int, ...]]] = {}
"""id -> (a Python function a binding above runs, where it is bound, its state as imported (``_state``), the ids of
the functions its closure cells held)."""

_BUILTINS = frozenset(vars(builtins))

_PYTHON_CACHES = frozenset({"__slotnames__"})
"""Names Python's runtime itself writes on a class, per class: ``copyreg`` caches ``__slotnames__`` on the first copy
or pickle of an instance (``torch.Tensor``'s, then ``nn.Parameter``'s, shadowing it)."""

_EMPTY = object()
"""The contents of an empty closure cell."""

_CACHE_WRAPPER = type(functools.lru_cache(maxsize=None)(lambda: None))
"""``functools.lru_cache``'s wrapper: it calls the function it wraps (its ``__wrapped__``) itself."""


def _subclasses(root: type) -> FrozenSet[type]:
    found, stack = set(), [root]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                stack.append(sub)
    return frozenset(found)


_TORCH_MODES = _subclasses(TorchFunctionMode) | _subclasses(TorchDispatchMode) | {DeviceContext}
"""Torch's own mode classes as loaded with mimarsinan, by identity (``torch.device(...)``'s ``DeviceContext``, which
torch loads lazily, imported here)."""


def is_code(value: Any) -> bool:
    """A callable, a descriptor (a property: its ``__get__`` runs on attribute access) or a module."""
    return callable(value) or isinstance(value, types.ModuleType) or hasattr(type(value), "__get__")


def _cell(cell: Any) -> Any:
    try:
        return cell.cell_contents
    except ValueError:  # an empty cell has no contents
        return _EMPTY


def _state(fn: types.FunctionType) -> Tuple[Any, ...]:
    keywords = fn.__kwdefaults__
    return (fn.__code__, fn.__defaults__, keywords, tuple(keywords.items()) if keywords else (),
            tuple(map(_cell, fn.__closure__ or ())))


def _runs(value: Any) -> List[Tuple[str, Any]]:
    """The callables a binding runs, each with its suffix: itself, a static, class or bound method's function, a
    property's accessors, the function a cache wrapper calls."""
    if isinstance(value, (staticmethod, classmethod, types.MethodType)):
        return [("", value.__func__)]
    if type(value) is _CACHE_WRAPPER:
        return [(".__wrapped__", value.__wrapped__)]
    if isinstance(value, property):
        return [(f".{name}", getattr(value, name)) for name in ("fget", "fset", "fdel") if getattr(value, name)]
    return [("", value)]


def _remember_function(value: Any, where: str) -> List[int]:
    """Snapshot the state of every Python function ``value`` runs (through its closure cells); their ids."""
    found = []
    for suffix, fn in _runs(value):
        if not isinstance(fn, types.FunctionType):
            continue
        found.append(id(fn))
        if id(fn) not in _FUNCTIONS:
            state = _state(fn)
            _FUNCTIONS[id(fn)] = (fn, where + suffix, state, ())
            held = [i for name, content in zip(fn.__code__.co_freevars, state[4])
                    for i in _remember_function(content, f"{where}{suffix} (closure cell {name!r})")]
            _FUNCTIONS[id(fn)] = (fn, where + suffix, state, tuple(held))
    return found


def _remember_globals(namespace: Dict[str, Any], name: str) -> None:
    if id(namespace) not in _GLOBALS:
        code = {key: value for key, value in list(namespace.items()) if is_code(value)}
        _GLOBALS[id(namespace)] = (namespace, name, code)
        for key, value in code.items():
            _remember_function(value, f"{name}.{key}")


def _qualified(namespace: Any, name: str) -> str:
    owner = f"{namespace.__module__}.{namespace.__qualname__}" if isinstance(namespace, type) else namespace.__name__
    return f"{owner}.{name}"


def remember_namespaces(*namespaces: Any) -> None:
    """Snapshot every binding of each namespace (a module or a class with a ``__dict__``)."""
    for namespace in namespaces:
        if hasattr(namespace, "__dict__") and id(namespace) not in _NAMESPACES:
            _NAMESPACES[id(namespace)] = (namespace, dict(vars(namespace)))
            for name, value in _NAMESPACES[id(namespace)][1].items():
                _remember_function(value, _qualified(namespace, name))


def _torchs(module_name: Any) -> bool:
    return isinstance(module_name, str) and (module_name == "torch" or module_name.startswith("torch."))


def remember_classes(*classes: type) -> None:
    """Snapshot each class and every torch class in its method resolution order, with the code bindings of their
    modules (``typing.Generic``, ``object`` and other bases outside torch are not torch's declared code)."""
    for cls in classes:
        for base in cls.__mro__[:-1]:
            if base is not cls and not _torchs(base.__module__):
                continue
            remember_namespaces(base)
            module = sys.modules.get(base.__module__)
            if module is not None:
                _remember_globals(vars(module), base.__module__)


def remember_code_of(functions: Iterable[Any]) -> None:
    """Snapshot the code bindings of the torch module each declared function is defined in (its ``__globals__``)."""
    for fn in functions:
        namespace = getattr(fn, "__globals__", None)
        if isinstance(namespace, dict) and _torchs(namespace.get("__name__")):
            _remember_globals(namespace, str(namespace.get("__name__")))


def _inherited(cls: type, name: str) -> Any:
    """What ``cls`` would bind ``name`` to without its own binding (its first base's binding along the MRO)."""
    return next((vars(base)[name] for base in cls.__mro__[1:] if name in vars(base)), None)


def _namespace_change(namespace: Any, imported: Dict[str, Any]) -> Optional[str]:
    now = vars(namespace)
    for name, value in imported.items():
        if name in now and now[name] is not value:
            return f"{_qualified(namespace, name)} is rebound since import"
    if isinstance(namespace, type):
        added = sorted(name for name in now.keys() - imported.keys() - _PYTHON_CACHES
                       if now[name] is not _inherited(namespace, name))
        if added:
            return f"{_qualified(namespace, added[0])} is added since import"
    return None


def _globals_change(namespace: Dict[str, Any], name: str, imported: Dict[str, Any]) -> Optional[str]:
    for key, value in imported.items():
        if key in namespace and namespace[key] is not value:
            return f"{name}.{key} is rebound since import"
    added = namespace.keys() - imported.keys()
    shadowing = sorted(key for key in added if key in _BUILTINS and is_code(namespace[key]))
    return f"{name}.{shadowing[0]} is added since import, shadowing a builtin" if shadowing else None


def _function_change(fn: types.FunctionType, where: str, imported: Tuple[Any, ...]) -> Optional[str]:
    code, defaults, keywords, entries, cells = imported
    now = fn.__kwdefaults__
    if fn.__code__ is not code or fn.__defaults__ is not defaults:
        return f"{where}.{'__code__' if fn.__code__ is not code else '__defaults__'} is replaced since import"
    if now is not keywords or (now is not None and (len(now) != len(entries) or any(
            now.get(key, _EMPTY) is not value for key, value in entries))):
        return f"{where}.__kwdefaults__ is replaced since import"
    for name, cell, content in zip(code.co_freevars, fn.__closure__ or (), cells):
        if _cell(cell) is not content:
            return f"{where}'s closure cell {name!r} is replaced since import"
    return None


def _functions_change(ids: Iterable[int]) -> Optional[str]:
    """The first change of a remembered function among ``ids`` and the functions their closure cells held."""
    todo, seen = list(ids), set()
    while todo:
        key = todo.pop()
        if key in seen or key not in _FUNCTIONS:
            continue
        seen.add(key)
        fn, where, imported, held = _FUNCTIONS[key]
        found = _function_change(fn, where, imported)
        if found is not None:
            return found
        todo.extend(held)
    return None


def _ids(values: Iterable[Any]) -> List[int]:
    return [id(fn) for value in values for _, fn in _runs(value) if isinstance(fn, types.FunctionType)]


def rebinding_of(*classes: type) -> Optional[str]:
    """``torch_rebinding`` restricted to the remembered classes given and the torch classes in their MROs: their
    bindings, their modules' code bindings and the state of the functions these hold; ``None`` when unchanged."""
    for cls in classes:
        for base in cls.__mro__[:-1]:
            entry, module = _NAMESPACES.get(id(base)), sys.modules.get(base.__module__)
            held = _GLOBALS.get(id(vars(module))) if module is not None else None
            found = None if entry is None else _namespace_change(*entry) or _functions_change(_ids(entry[1].values()))
            if found is None and held is not None:
                found = _globals_change(*held) or _functions_change(_ids(held[2].values()))
            if found is not None:
                return found
    return None


def torch_rebinding() -> Optional[str]:
    """Why the declared operators are not torch's own code as imported: a declared namespace's or class's binding
    rebound, a name added to a declared class, a code binding among the globals of a module defining declared code
    rebound (or a builtin shadowed there), a remembered function's code, defaults or closure cells replaced, or an
    active torch-function or dispatch mode of a class that is not torch's own; ``None`` when there is none."""
    for namespace, imported in list(_NAMESPACES.values()):
        found = _namespace_change(namespace, imported)
        if found is not None:
            return found
    for namespace, name, imported in list(_GLOBALS.values()):
        found = _globals_change(namespace, name, imported)
        if found is not None:
            return found
    for fn, where, imported, _ in list(_FUNCTIONS.values()):
        found = _function_change(fn, where, imported)
        if found is not None:
            return found
    foreign = user_modes()
    return f"a mode of type {type(foreign[0]).__name__} is active" if foreign else None


def user_modes() -> List[Any]:
    """The active torch-function and dispatch modes whose class is not torch's own (``_TORCH_MODES``)."""
    return [mode for mode in (*_function_modes(), *_dispatch_modes()) if type(mode) not in _TORCH_MODES]

"""A module's native call as ``nn.Module`` runs it (review round 11).

``module(x)`` runs ``type(module).__call__``: ``nn.Module``'s (``_wrapped_call_impl``), or on a GraphModule class FX's
wrapper of it. The call reads names on the module through Python's attribute lookup and globals of
``torch.nn.modules.module``; FX traces ``getattr(type(module), 'forward')`` and never runs the call. The names and
globals are taken from torch's call code itself (``CALL_READS``, ``CALL_GLOBALS``: the attribute loads on ``self`` and
the global loads of ``__call__`` and of every ``nn.Module`` function it reads, transitively), so a torch whose call
reads another name is covered.

The call runs its class's ``forward`` alone (``call_reason`` is ``None``) when:

- attribute lookup is Python's own: no class of the module's MRO but ``object`` defines ``__getattribute__`` or a
  ``__dict__`` of its own, and no class of its metaclass's MRO but ``type`` and ``object`` defines
  ``__getattribute__``, ``__getattr__`` or ``forward``;
- ``type(module).__call__`` is ``nn.Module``'s as imported, or FX's GraphModule wrapper of it (``_wrapped_call`` the
  class's own ``_WrappedCall`` of no class call);
- every name the call reads resolves statically (``inspect.getattr_static``) to ``nn.Module``'s own class binding as
  imported or to the instance state ``nn.Module`` keeps: a hook table of the exact type it installs, the forward tables
  empty (``dict.__len__``), its flags plain, no compiled call; ``forward`` resolves to a plain function of the class,
  the one FX traces (``getattr(type(module), 'forward')``);
- no global forward hook table holds a hook (``forward_hooks``).

Backward hooks do not change a forward value; they may be registered. ``call_code_change``: the call's code is torch's
as imported (``nn.Module``, the torch classes its globals name and FX's ``_WrappedCall``: ``rebinding_of``) and every
global it reads is the object imported, the hook tables of their exact type, plain data of a plain type.
"""

from __future__ import annotations

import builtins
import dis
import inspect
import types
from collections import OrderedDict
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, cast

import torch.fx as fx
import torch.nn as nn
from torch.fx.graph_module import _WrappedCall
from torch.nn.modules import module as _module_globals

from mimarsinan.torch_mapping.converter_handlers.symbolic.imported_code import is_code, rebinding_of, remember_classes

__all__ = ["CALL_GLOBALS", "CALL_READS", "HOOK_REASONS", "call_code_change", "call_reason", "forward_hooks"]

_MISSING = object()
_MODULE = dict(vars(nn.Module))
"""``nn.Module``'s class bindings as imported."""

_STATE = dict(vars(nn.Module()))
"""The instance state ``nn.Module`` keeps (hook tables, flags), by name."""

_TABLE_TYPES = (dict, OrderedDict)
_PLAIN = (type(None), bool, int, float, str)


def _codes(code: types.CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _codes(constant)


def _loads(code: types.CodeType) -> Tuple[Set[str], Set[Tuple[str, ...]]]:
    """The attribute names loaded on ``self`` and the global loads (with the attributes read on them) of ``code``."""
    names: Set[str] = set()
    chains: Set[Tuple[str, ...]] = set()
    for inner in _codes(code):
        instructions = list(dis.get_instructions(inner))
        for index, instruction in enumerate(instructions):
            attributes = []
            for following in instructions[index + 1:]:
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                attributes.append(following.argval)
            if instruction.opname in ("LOAD_FAST", "LOAD_DEREF") and instruction.argval == "self" and attributes:
                names.add(attributes[0])
            elif instruction.opname == "LOAD_GLOBAL":
                chains.add((instruction.argval, *attributes))
    return names, chains


def _call_reads() -> Tuple[FrozenSet[str], FrozenSet[Tuple[str, ...]]]:
    names: Set[str] = set()
    chains: Set[Tuple[str, ...]] = set()
    todo, seen = ["__call__"], set()
    while todo:
        name = todo.pop()
        function = _MODULE.get(name)
        if name in seen or name == "forward" or not isinstance(function, types.FunctionType):
            continue
        seen.add(name)
        reads, loads = _loads(function.__code__)
        names |= reads
        chains |= loads
        todo.extend(sorted(reads))
    return frozenset(names), frozenset(chains)


CALL_READS, CALL_GLOBALS = _call_reads()
"""The names ``nn.Module``'s call reads on the module, and its global loads (each with the attributes it reads)."""


def _resolve(chain: Tuple[str, ...]) -> Tuple[Any, str]:
    """A global load's value, read through code (modules, callables) and stopping at data, and its name."""
    namespace = vars(_module_globals) if chain[0] in vars(_module_globals) else vars(builtins)
    value = namespace.get(chain[0], _MISSING)
    owner = _module_globals.__name__ if namespace is vars(_module_globals) else "builtins"
    where = [value.__name__] if isinstance(value, types.ModuleType) else [owner, chain[0]]
    for attribute in chain[1:]:
        if value is _MISSING or not is_code(value) or isinstance(value, type):
            break
        value = getattr(value, attribute, _MISSING)
        where.append(attribute)
    return value, ".".join(where)


_IMPORTED = {where: (chain, value) for chain in sorted(CALL_GLOBALS) for value, where in [_resolve(chain)]}
"""Each global the call reads (by its name), as imported: the load and the object."""

_REACHED = (_WrappedCall, *(value for _, value in _IMPORTED.values()
                            if type(value) is type and str(value.__module__).startswith("torch")))
"""FX's ``_WrappedCall`` and the Python classes of torch the call's globals name: ``BackwardHook``, ``_WrappedHook``."""

remember_classes(*_REACHED)

_FORWARD_TABLES = {"_forward_pre_hooks": "a forward pre-hook registered on it",
                   "_forward_hooks": "a forward hook registered on it"}
_GLOBAL_FORWARD_TABLES = {
    "_global_forward_pre_hooks": "a global forward pre-hook (register_module_forward_pre_hook)",
    "_global_forward_hooks": "a global forward hook (register_module_forward_hook)",
}
HOOK_REASONS = frozenset({*_FORWARD_TABLES.values(), *_GLOBAL_FORWARD_TABLES.values()})
"""The reasons that name a forward hook the call runs (the module's own or a global one)."""

_BOUND = {"forward": "a forward bound on the instance", "_compiled_call_impl": "a compiled call (Module.compile)"}


def _graph_module_call_code() -> types.CodeType:
    graph = fx.Graph()
    graph.output(None)
    return type(fx.GraphModule(nn.Module(), graph)).__call__.__code__


_GRAPH_MODULE_CALL = _graph_module_call_code()
"""The code of the ``__call__`` FX installs on a GraphModule's class (``call_wrapped``): ``nn.Module.__call__``
through the class's ``_WrappedCall``, which only decorates an exception's message."""


def call_code_change() -> Optional[str]:
    """Why ``nn.Module``'s call is not torch's code as imported: a binding or function of ``nn.Module``, of a torch
    class its globals name or of FX's ``_WrappedCall`` changed, or a global it reads rebound (a hook table also when not
    of its type, plain data when not of a plain type); ``None`` when it is."""
    found = rebinding_of(nn.Module, *_REACHED)
    if found is not None:
        return found
    for where, (chain, imported) in _IMPORTED.items():
        now, _ = _resolve(chain)
        if is_code(imported) or isinstance(imported, dict):
            if now is not imported or type(now) is not type(imported):
                return f"{where} is rebound since import"
        elif type(now) not in _PLAIN:
            return f"{where} is a value of type {type(now).__name__}"
    return None


def _global_forward_hooks() -> Optional[str]:
    for name, reason in _GLOBAL_FORWARD_TABLES.items():
        where = f"{_module_globals.__name__}.{name}"
        table = vars(_module_globals).get(name)
        if table is not _IMPORTED[where][1] or type(table) is not OrderedDict:
            return f"{where} is rebound since import"
        if dict.__len__(table):
            return reason
    return None


def forward_hooks(module: nn.Module) -> Optional[str]:
    """A forward hook ``module``'s call runs (its own, read from its instance dict, or a global one), or a global
    forward hook table rebound since import; ``None`` when there is none."""
    own = vars(module)
    for name, reason in _FORWARD_TABLES.items():
        table = own.get(name)
        if type(table) in _TABLE_TYPES and dict.__len__(cast(dict, table)):
            return reason
    return _global_forward_hooks()


_MRO, _DICT = vars(type)["__mro__"], vars(type)["__dict__"]
"""``type``'s own descriptors of a class's MRO and dict, as ``inspect.getattr_static`` reads them: a metaclass
attribute named ``__mro__`` or ``__dict__`` does not change them."""


def _mro(cls: type) -> Tuple[type, ...]:
    return _MRO.__get__(cls)


def _dict(cls: type) -> Any:
    return _DICT.__get__(cls)


def _lookup_reason(cls: type) -> Optional[str]:
    standard = _MODULE["__dict__"]
    for base in _mro(cls)[:-1]:
        if "__getattribute__" in _dict(base):
            return f"its class {base.__name__} defines __getattribute__"
        if type(_dict(base).get("__dict__", standard)) is not type(standard):
            return f"its class {base.__name__} defines __dict__"
    for meta in _mro(type(cls)):
        found = [name for name in ("__getattribute__", "__getattr__", "forward") if name in _dict(meta)]
        if meta not in (type, object) and found:
            return f"its metaclass {meta.__name__} defines {found[0]}"
    return None


def _static(classes: Tuple[type, ...], name: str) -> Any:
    return next((_dict(base)[name] for base in classes if name in _dict(base)), _MISSING)


def _call_reason(module: nn.Module, cls: type) -> Optional[str]:
    call = _static(_mro(cls), "__call__")
    if call is _MODULE["__call__"]:
        return None
    if not (isinstance(call, types.FunctionType) and call.__code__ is _GRAPH_MODULE_CALL):
        return "its class overrides nn.Module.__call__"
    wrapped = inspect.getattr_static(module, "_wrapped_call", _MISSING)
    fx_own = (type(wrapped) is _WrappedCall and wrapped is _dict(cls).get("_wrapped_call") and wrapped.cls is cls
              and wrapped.cls_call is None and _static(_mro(cls)[1:], "__call__") is _MODULE["__call__"])
    return None if fx_own else "its GraphModule call is not FX's wrapper of nn.Module's"


def _instance_reason(name: str, value: Any) -> Optional[str]:
    """Why an instance entry the call reads is not state ``nn.Module`` keeps there."""
    if isinstance(_STATE.get(name), dict):
        if type(value) not in _TABLE_TYPES:
            return f"a {name} table that is a user container"
        return _FORWARD_TABLES.get(name) if dict.__len__(value) else None
    if name in _STATE:
        return None if type(value) in _PLAIN else f"a {name} of type {type(value).__name__} on it"
    if value is _MODULE.get(name, _MISSING) and not is_code(value):
        return None
    return _BOUND.get(name, f"a {name} bound on the instance")


def _read_reason(module: nn.Module, cls: type, own: Dict[str, Any], name: str) -> Optional[str]:
    found = inspect.getattr_static(module, name, _MISSING)
    if found is _MISSING:
        return f"no {name} on it, which nn.Module's call reads"
    if name in own and found is own[name]:
        return _instance_reason(name, found)
    if name == "forward":
        traced = isinstance(found, types.FunctionType) and getattr(cls, "forward", None) is found
        return None if traced else f"its class's forward is a {type(found).__name__}, not the plain function FX traces"
    if name in _STATE:
        return f"its class binds {name}, which nn.Module keeps on the instance"
    return None if found is _MODULE.get(name) else f"its class overrides nn.Module.{name}"


def call_reason(module: nn.Module) -> Optional[str]:
    """Why ``module``'s native call runs code its class's ``forward`` does not; ``None`` when it runs that alone
    (given ``call_code_change`` is ``None``)."""
    cls = type(module)
    found = _lookup_reason(cls) or _call_reason(module, cls)
    if found is not None:
        return found
    own = vars(module)
    if type(own) is not dict:  # a subclass's own methods would answer for the dict attribute lookup reads
        return "its instance dict is a user container"
    for name in sorted(CALL_READS):
        found = _read_reason(module, cls, own, name)
        if found is not None:
            return found
    return _global_forward_hooks()

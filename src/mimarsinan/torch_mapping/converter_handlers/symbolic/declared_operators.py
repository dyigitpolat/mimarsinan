"""The declared operator set and the declared value types: what a PURE program is made of (``purity.py``).

Membership is by identity (functions) or exact type (modules, values), so a user callable's ``__eq__``/``__hash__``, a
user subclass's ``forward`` and the dunders of a user subclass of a builtin or tensor type are never consulted. A call
is declared when its target is in one of three parts and every literal operand has a declared value type:

- **Scalar operators** (``CONVERSION_TIME_OPERATORS``): the value functions of ``operator`` and ``math``, torch's
  symbolic-size functions (``torch.sym_*``) and ``torch._assert``. They are the only calls a static shape value is
  computed with at conversion time (on native values and types), and they run on symbolic sizes in the symbolic-batch
  pass.
- **Tensor operators**: the public callables of torch's own namespaces (``TENSOR_NAMESPACES``), aten operators, and the
  methods and attributes of a value whose run-time type is exactly ``torch.Tensor``, ``torch.nn.Parameter`` or
  ``torch.Size`` (``call_method``, ``getattr``). They run on FakeTensors in the symbolic-batch pass, never at
  conversion time.
- **Leaf modules with a declared shape rule**: every module class defined in ``torch.nn`` and its submodules (FX's
  default leaves; its ``forward`` is torch's own), by exact type, and every class declared through
  ``declare_leaf_module`` (the tracer declares the leaves it adds), without hooks, whose state and attributes have
  declared value types (``foreign_module``).

The declared value types (``foreign_value``): ``int``, ``float``, ``bool``, ``str``, ``None``, ``Ellipsis``,
``torch.dtype``, ``torch.device``, ``torch.layout``, ``torch.memory_format`` (``DECLARED_LITERAL_TYPES``) and the
tensors ``torch.Tensor`` and ``torch.nn.Parameter`` (``DECLARED_TENSOR_TYPES``: no subclass, so no user
``__torch_function__``), all by exact type, through plain ``tuple``, ``list``, ``dict``, ``slice`` and ``torch.Size``
(FX's ``immutable_list`` and ``immutable_dict`` hold a node's lists and dicts); a declared callable counts as one.
Anything else (``self.k = Odd(1)``, an ``int`` subclass whose ``__rmul__`` answers a ``SymInt``; a tensor subclass
buffer) would run its own code wherever torch reads it.

Every other call is OPAQUE (``opaque_reason``): a ``torch.fx.wrap`` helper, any user function, method or module, a
declared function given a callable outside the set (``torch.vmap``), a module with hooks or foreign state. A program
with one is impure (``purity.py``), and the symbolic pass never runs for it. So is a program converted while the
declared code is not torch's own as imported (``torch_rebinding``, ``imported_code.py``: ``torch.Tensor.size = f``, a
rebound ``torch.nn.modules.conv.F``, a user torch function mode). The set is extended in one place: a leaf module
class whose ``forward`` is made of declared operators is declared with ``declare_leaf_module``; a function is
declared by adding its namespace here.
"""
from __future__ import annotations

import math
import operator
import sys
import weakref
from collections import OrderedDict
from typing import Any, Mapping, Optional, Set, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.immutable_collections import immutable_dict, immutable_list

from mimarsinan.torch_mapping.converter_handlers.symbolic.imported_code import (
    remember_classes,
    remember_code_of,
    remember_namespaces,
    torch_rebinding,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import HOOK_REASONS, forward_hooks

__all__ = [
    "CONVERSION_TIME_OPERATORS", "DECLARED_LITERAL_TYPES", "DECLARED_TENSOR_TYPES", "TENSOR_NAMESPACES",
    "declare_leaf_module", "foreign_module", "foreign_value", "is_conversion_time_operator", "is_declared_module",
    "is_tensor_operator", "opaque_reason", "torch_rebinding",
]


def _public_functions(namespace, names) -> set:
    found = [getattr(namespace, name) for name in names]
    return {value for value in found if callable(value) and not isinstance(value, type)}


def _not_a_value_function(name: str) -> bool:
    """``call`` (it calls its argument), ``setitem``, ``delitem``, the in-place forms (``iadd``, ``iand``) and the
    identity tests ``is_`` and ``is_not`` (an object's identity is not a function of its value: ``n is m`` holds for
    two equal sizes up to 256 in CPython and not above)."""
    in_place = name[:1] == "i" and (hasattr(operator, name[1:]) or hasattr(operator, name[1:] + "_"))
    return in_place or name in ("call", "setitem", "delitem", "is_", "is_not")


CONVERSION_TIME_OPERATORS = frozenset(
    _public_functions(operator, [n for n in dir(operator) if not n.startswith("_") and not _not_a_value_function(n)])
    | _public_functions(math, [n for n in dir(math) if not n.startswith("_")])
    | _public_functions(torch, [n for n in dir(torch) if n.startswith("sym_")])
    | {torch._assert}
)
"""The scalar operators: the closed set a static shape value may be computed with at conversion time; their result
is a function of the values, and the values carry their native types. Any other callable keeps its host op."""

_CONVERSION_TIME_IDS = frozenset(id(fn) for fn in CONVERSION_TIME_OPERATORS)

TENSOR_NAMESPACES = (
    torch, F, torch.special, torch.linalg, torch.fft, getattr(torch._C, "_nn"), getattr(torch._C, "_VariableFunctions"),
    torch.Tensor,
)
"""Torch's own namespaces: their callables (functions, builtins, method descriptors; classes excluded) are tensor
operators."""



def _namespace_callables(namespace) -> set:
    """A module's own attributes (``vars``: a lazily imported submodule is not loaded), or a class's (``dir``)."""
    names = list(vars(namespace)) if isinstance(namespace, type(math)) else dir(namespace)
    return _public_functions(namespace, [n for n in names if not n.startswith("__")])


_TENSOR_FUNCTIONS = [value for namespace in TENSOR_NAMESPACES for value in _namespace_callables(namespace)]
_TENSOR_IDS = frozenset(map(id, _TENSOR_FUNCTIONS))

DECLARED_TENSOR_TYPES = frozenset({torch.Tensor, nn.Parameter})
"""The exact tensor types of a declared value: a subclass (its own ``__torch_function__``) is not one."""

_RECEIVER_TYPES = (*DECLARED_TENSOR_TYPES, torch.Size)
"""The exact run-time types whose methods and attributes (``call_method``, ``getattr``) are tensor operators."""

def _torch_nn_module_classes() -> Set[Type[nn.Module]]:
    """Every module class defined in ``torch.nn`` and its loaded submodules (FX's default leaves), by identity."""
    found: Set[Type[nn.Module]] = set()
    for name, namespace in list(sys.modules.items()):
        if namespace is None or not (name == "torch.nn" or name.startswith("torch.nn.")):
            continue
        for value in list(vars(namespace).values()):
            if isinstance(value, type) and issubclass(value, nn.Module) and value.__module__ == name:
                found.add(value)
    return found


_LEAF_MODULES: Set[Type[nn.Module]] = _torch_nn_module_classes()


remember_namespaces(*TENSOR_NAMESPACES, operator, math)
remember_classes(nn.Module, *DECLARED_TENSOR_TYPES, *_LEAF_MODULES)
remember_code_of([*CONVERSION_TIME_OPERATORS, *_TENSOR_FUNCTIONS])


def declare_leaf_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """Declare a leaf module class whose ``forward`` is made of declared operators: it runs in the symbolic pass.
    Exact type only: a subclass is declared on its own or stays opaque."""
    _LEAF_MODULES.add(cls)
    remember_classes(cls)
    return cls


def is_conversion_time_operator(fn: Any) -> bool:
    """``fn`` is a scalar operator: a static shape value computed with it is folded at conversion time."""
    return id(fn) in _CONVERSION_TIME_IDS


def _is_aten(fn: Any) -> bool:
    if isinstance(fn, torch._ops.OpOverload):
        return fn.namespace == "aten"
    return isinstance(fn, torch._ops.OpOverloadPacket) and fn._qualified_op_name.startswith("aten::")


def _declared_function(fn: Any) -> bool:
    return id(fn) in _CONVERSION_TIME_IDS or id(fn) in _TENSOR_IDS or _is_aten(fn)


def is_tensor_operator(fn: Any) -> bool:
    """``fn`` is a tensor operator: a callable of torch's own namespaces or an aten operator (``operand_writes``
    reads its schema)."""
    return id(fn) in _TENSOR_IDS or _is_aten(fn)


def is_declared_module(module: nn.Module) -> bool:
    """``module``'s class is a declared leaf (a ``torch.nn`` class, or one ``declare_leaf_module`` declared)."""
    return type(module) in _LEAF_MODULES


def _hook_reason(module: nn.Module) -> Optional[str]:
    """A forward hook ``module``'s call runs, read as the call reads it (``module_call.forward_hooks``: its instance
    tables and the global tables torch installed at import, counted with ``dict.__len__``)."""
    found = forward_hooks(module)
    if found is None:
        return None
    hooked = found in HOOK_REASONS
    return "a module with a forward hook" if hooked else f"a module whose call is not torch's as imported ({found})"


DECLARED_LITERAL_TYPES = frozenset({
    int, float, bool, str, type(None), type(Ellipsis), torch.dtype, torch.device, torch.layout, torch.memory_format,
})
"""The exact types of a declared literal leaf; a subclass (an ``int`` with its own ``__rmul__``) is not one."""

_LITERAL_CONTAINERS = frozenset({tuple, list, dict, slice, torch.Size, immutable_list, immutable_dict})
"""The exact container types a declared value is held in: the plain ones, and FX's immutable list and dict (the form
of a node's list and dict arguments); a namedtuple or a user subclass of a container is none of them."""

_MODULE_BOOKKEEPING = frozenset(vars(nn.Module()))
"""``nn.Module``'s own instance state (hooks, ``training``, the parameter, buffer and submodule tables), read
separately: the tables by their type and entries, the forward hooks by ``_hook_reason``."""

_TABLES = tuple(name for name, value in vars(nn.Module()).items() if isinstance(value, dict))
"""``nn.Module``'s tables (parameters, buffers, submodules, hooks): ``dict`` or ``OrderedDict`` as it installs them; a
user subclass runs its own code in every lookup (``_parameters.__getitem__``, which ``nn.Module.__getattr__`` calls)."""

_TABLE_ROLES = {"_parameters": "parameter table", "_buffers": "buffer table", "_modules": "submodule table"}


def _user_table(module: nn.Module) -> Optional[str]:
    """The first table of ``module`` that is not of a type ``nn.Module`` installs, named; ``None`` when none is."""
    own = vars(module)
    if type(own) is not dict:  # attribute lookup reads the dict itself, a subclass's methods answer the reads here
        return "instance dict"
    found = next((name for name in _TABLES if type(own.get(name)) not in (dict, OrderedDict)), None)
    return None if found is None else _TABLE_ROLES.get(found, f"table {found}")


def _foreign_leaf(value: Any) -> Optional[Tuple[Any]]:
    """``(leaf,)``: the first leaf of ``value`` outside the declared value types (a declared callable is one, a weak
    reference is its referent), walking the declared containers; ``None`` when there is none. Reads types and
    structure only."""
    kind = type(value)
    if isinstance(value, fx.Node) or kind in DECLARED_LITERAL_TYPES or kind in DECLARED_TENSOR_TYPES:
        return None
    if kind is weakref.ref:
        return _foreign_leaf(value())
    if kind not in _LITERAL_CONTAINERS:
        declared = callable(value) and not isinstance(value, type) and _declared_function(value)
        return None if declared else (value,)
    if kind in (dict, immutable_dict):
        items = [*value.keys(), *value.values()]
    else:
        items = [value.start, value.stop, value.step] if kind is slice else list(value)
    return next((found for found in map(_foreign_leaf, items) if found is not None), None)


def foreign_value(value: Any) -> Optional[str]:
    """Why ``value`` (a literal operand, a get_attr value, a module's state) is outside the declared value types;
    ``None`` when it is not."""
    found = _foreign_leaf(value)
    if found is None:
        return None
    if callable(found[0]) and not isinstance(found[0], (type, torch.Tensor)):
        return "a callable outside the declared operator set"
    return f"a value of type {type(found[0]).__name__} outside the declared value types"


def foreign_module(module: nn.Module) -> Optional[str]:
    """Why a leaf module's ``forward`` would run code outside the set: a submodule (itself included) of an undeclared
    class, with a table of a user type (a ``dict`` subclass as ``_parameters``) or with a hook, or a parameter, buffer
    or attribute outside the declared value types (a tensor subclass weight, ``nn.GroupNorm(Spy(2), 6)``,
    ``TransformerEncoderLayer(activation=fn)``)."""
    for sub in module.modules():  # each visited before its submodule table is read
        if type(sub) not in _LEAF_MODULES:
            return "a module class without a declared shape rule"
        table = _user_table(sub)
        if table is not None:
            return f"a module whose {table} is a user container"
        hooked = _hook_reason(sub)
        if hooked is not None:
            return hooked
        state = [("parameter", key, value) for key, value in dict.items(vars(sub)["_parameters"])]
        state += [("buffer", key, value) for key, value in dict.items(vars(sub)["_buffers"])]
        state += [("attribute", key, value) for key, value in vars(sub).items() if key not in _MODULE_BOOKKEEPING]
        for role, key, value in state:
            reason = foreign_value(value)
            if reason is not None:
                return f"a module whose {role} {key!r} is {reason}"
    return None


def _literal_reason(node: fx.Node) -> Optional[str]:
    """Why a literal operand of ``node`` would run code outside the set: a callable outside it (a user function given
    to a declared higher-order function), or a value of a type outside the declared value types."""
    found = _foreign_leaf((node.args, node.kwargs))
    if found is None:
        return None
    if callable(found[0]) and not isinstance(found[0], type):
        return "a declared call given a callable outside the declared operator set"
    kind = type(found[0]).__name__
    return f"a declared call given a literal operand of type {kind} outside the declared literal types"


def opaque_reason(node: fx.Node, receiver_type: Optional[type], modules: Mapping[str, nn.Module]) -> Optional[str]:
    """Why ``node``'s call is outside the declared operator set; ``None`` when it is declared. ``receiver_type`` is the
    run-time type of its first argument (the value a method or attribute is read on)."""
    if node.op == "call_module":
        module = modules.get(str(node.target))
        reason = "a module class without a declared shape rule" if module is None else foreign_module(module)
        return reason or _literal_reason(node)
    if node.op == "call_method":
        if receiver_type not in _RECEIVER_TYPES or not hasattr(receiver_type, str(node.target)):
            return "a method of a value that is not a tensor or a shape"
    elif node.op == "call_function":
        if node.target is getattr:
            name = node.args[1] if len(node.args) > 1 else None
            if (receiver_type not in _RECEIVER_TYPES or not isinstance(name, str)
                    or not hasattr(receiver_type, name)):
                return "an attribute of a value that is not a tensor or a shape"
        elif not _declared_function(node.target):
            return "a call outside the declared operator set"
    else:
        return None
    return _literal_reason(node)

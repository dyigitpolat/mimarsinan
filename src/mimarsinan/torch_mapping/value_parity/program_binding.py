"""The binding of the probe's certificate to the returned program (review rounds 13 and 14).

The probe's witness program is derived from the returned conversion's own traced graph (``witness.fp64_program``,
review round 14): FX's deep copy of the traced ``GraphModule``, cast to fp64, never a copy of the user's root object.
The only copies in it are the deep copies of the modules and tensors the graph calls and reads, so the binding checks
that they hold the same program: the two traced programs are compared node by node (``program_record``,
``program_difference``): each node's op, target and every literal argument, recursively; each called module's class,
configuration (its attributes other than nn.Module's tables), forward hooks and the shapes of its tensors; each
``get_attr`` value, an FX tensor constant included, by value. And each tensor a node reads is compared between the
returned program and the witness before the probe draws any statistic or parameter (``state_differences``: the
witness's fp64 values against the returned program's cast to fp64). A difference is named by its first FX node
(``WitnessProgramMismatchError.of_program``).
"""

from __future__ import annotations

import functools
import math
import types
import weakref
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.refusals import node_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import attribute_at

__all__ = ["NodeRecord", "program_difference", "program_record", "state_differences"]

_TABLES = frozenset(name for name, value in vars(nn.Module()).items() if isinstance(value, dict))
"""nn.Module's tables (parameters, buffers, submodules, hooks), compared through their entries."""

_FORWARD_HOOKS = ("_forward_pre_hooks", "_forward_hooks", "_forward_pre_hooks_with_kwargs",
                  "_forward_hooks_with_kwargs", "_forward_hooks_always_called")
_DEPTH = 4
"""How deep a plain object's attributes are compared (a configuration object holding others)."""


@dataclass(frozen=True)
class _Ref:
    """An FX node an argument refers to, by name."""

    name: str


@dataclass(frozen=True)
class NodeRecord:
    """One traced node: what it computes (``call``: op, target, arguments; ``module``: a called module's configuration
    per submodule) and the qualified names of the tensors it reads (``reads``)."""

    name: str
    operator: str
    call: Tuple[Any, ...]
    module: Tuple[Tuple[str, Any], ...]
    reads: Tuple[str, ...]


def _detached(value: Any) -> Any:
    """``value`` with every FX node its name (``_Ref``), through tuples, lists, dicts and slices."""
    if isinstance(value, fx.Node):
        return _Ref(value.name)
    if isinstance(value, (tuple, list)):
        return type(value)(map(_detached, value)) if type(value) in (tuple, list) else tuple(map(_detached, value))
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if type(value) is slice:
        return slice(_detached(value.start), _detached(value.stop), _detached(value.step))
    return value


def _tensors(module: nn.Module, prefix: str) -> Dict[str, torch.Tensor]:
    """Every tensor ``module`` holds by qualified name: parameters, buffers and plain tensor attributes."""
    found: Dict[str, torch.Tensor] = {}
    for name, sub in module.named_modules(prefix=prefix, remove_duplicate=False):
        own = {**dict(sub._parameters), **dict(sub._buffers),
               **{key: value for key, value in vars(sub).items() if isinstance(value, torch.Tensor)}}
        found.update({f"{name}.{key}" if name else key: value for key, value in own.items() if value is not None})
    return found


def _configuration(module: nn.Module, prefix: str) -> Tuple[Tuple[str, Any], ...]:
    """Per submodule: its class, attributes (tensors as their shapes) and forward hooks (functions unwrapped)."""
    found: List[Tuple[str, Any]] = []
    for name, sub in module.named_modules(prefix=prefix, remove_duplicate=False):
        attributes = {key: value for key, value in vars(sub).items() if key not in _TABLES}
        hooks = tuple((table, tuple(getattr(sub, table, {}).values())) for table in _FORWARD_HOOKS)
        state = tuple((key, tuple(value.shape)) for key, value in [*sub._parameters.items(), *sub._buffers.items()]
                      if value is not None)
        found.append((name, (type(sub), attributes, hooks, state)))
    return tuple(found)


def program_record(gm: fx.GraphModule) -> Tuple[NodeRecord, ...]:
    """The traced program of ``gm`` as ``program_difference`` compares it (taken before any rewrite of the graph)."""
    modules = dict(gm.named_modules())
    records = []
    for node in gm.graph.nodes:
        target = str(node.target)
        module: Tuple[Tuple[str, Any], ...] = ()
        reads: Tuple[str, ...] = ()
        if node.op == "call_module" and target in modules:
            module, reads = _configuration(modules[target], target), tuple(_tensors(modules[target], target))
        elif node.op == "get_attr":  # a parameter or buffer by shape (its value: ``state_differences``), an FX tensor
            value = guarded(lambda: attribute_at(gm, target))[0]  # constant or other attribute by value (round 14)
            if _registered(gm, target) and isinstance(value, torch.Tensor):
                value = tuple(value.shape)
            module, reads = ((target, ("attribute", value)),), (target,)
        call = (node.op, node.target, _detached(node.args), _detached(node.kwargs))
        records.append(NodeRecord(node.name, node_operator(node, modules), call, module, reads))
    return tuple(records)


def _registered(root: nn.Module, target: str) -> bool:
    """``target`` names a parameter or buffer of its module (the probe draws some; ``state_differences`` compares
    them before it does)."""
    owner_path, _, name = target.rpartition(".")
    owner, exc = guarded(lambda: root.get_submodule(owner_path))
    return exc is None and isinstance(owner, nn.Module) and (name in owner._parameters or name in owner._buffers)


def _leaf_difference(a: Any, b: Any) -> bool:
    if isinstance(a, float):
        return not (a == b or (math.isnan(a) and math.isnan(b))) or math.copysign(1, a) != math.copysign(1, b)
    if isinstance(a, np.ndarray):
        return not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True))
    equal, exc = guarded(lambda: bool(a == b))
    return exc is not None or not equal


def difference(a: Any, b: Any, depth: int = 0) -> Optional[Tuple[Any, Any]]:
    """The first pair of differing leaves of two values walked together, ``None`` when they are the same: functions,
    classes and modules by identity, a bound method by its function, a tensor by its shape (its values are compared
    through ``state_differences``), a plain object by its attributes."""
    if type(a) is not type(b):
        return a, b
    if isinstance(a, (types.FunctionType, types.BuiltinFunctionType, type, types.ModuleType)):
        return None if a is b else (a, b)
    if isinstance(a, torch.Tensor):  # by value, cast to the witness's dtype (review round 14)
        return None if _same_tensor(a, b) else (_shown(a), _shown(b))
    if isinstance(a, nn.Module):
        return None  # compared through the node that calls it
    if isinstance(a, weakref.ref):
        return difference(a(), b(), depth + 1)
    if isinstance(a, types.MethodType):
        return None if a.__func__ is b.__func__ and type(a.__self__) is type(b.__self__) else (a, b)
    if isinstance(a, functools.partial):
        return difference((a.func, a.args, a.keywords), (b.func, b.args, b.keywords), depth)
    if isinstance(a, (tuple, list)) or type(a) is slice:
        a_items, b_items = ((a.start, a.stop, a.step), (b.start, b.stop, b.step)) if type(a) is slice else (a, b)
        if len(a_items) != len(b_items):
            return a, b
        return next((found for found in map(difference, a_items, b_items, [depth] * len(a_items)) if found), None)
    if isinstance(a, dict):
        if list(a) != list(b):
            return a, b
        return next((found for found in (difference(a[key], b[key], depth) for key in a) if found), None)
    if hasattr(a, "__dict__") and depth < _DEPTH:
        return difference(vars(a), vars(b), depth + 1)
    return (a, b) if _leaf_difference(a, b) else None


def _shown(tensor: torch.Tensor) -> str:
    values = tensor.detach().reshape(-1)[:4].tolist()
    return f"a tensor of shape {tuple(tensor.shape)} holding {values}{' ...' if tensor.numel() > 4 else ''}"


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.is_floating_point() != b.is_floating_point():
        return False
    cast = a.detach().to("cpu", b.dtype)
    if not b.is_floating_point():
        return torch.equal(cast, b.detach())
    other = b.detach()
    return bool(((cast == other) | (cast.isnan() & other.isnan())).all())


def state_differences(model: nn.Module, snapshot: nn.Module) -> FrozenSet[str]:
    """The qualified names of the tensors ``model`` (the returned conversion's traced program) holds that its fp64
    witness ``snapshot`` does not hold alike (shape, value cast to the witness's dtype), or holds and the model does
    not, FX tensor constants included; taken before the probe draws any."""
    mine, theirs = _tensors(model, ""), _tensors(snapshot, "")
    differing = {name for name in set(mine) ^ set(theirs)}
    differing |= {name for name in set(mine) & set(theirs) if not _same_tensor(mine[name], theirs[name])}
    return frozenset(differing)


def _described(value: Any) -> str:
    return getattr(value, "__name__", None) or repr(value) if callable(value) else repr(value)


def _call_difference(mine: NodeRecord, theirs: NodeRecord) -> Optional[str]:
    if mine.name != theirs.name or difference(mine.call[:2], theirs.call[:2]) is not None:
        return (f"the call {mine.call[0]} {_described(mine.call[1])} in the returned conversion, {theirs.call[0]} "
                f"{_described(theirs.call[1])} (FX node {theirs.name!r}) in the witness")
    found = difference(mine.call[2:], theirs.call[2:])
    if found is not None:
        return f"the literal {_described(found[0])} in the returned conversion, {_described(found[1])} in the witness"
    if [name for name, _ in mine.module] != [name for name, _ in theirs.module]:
        return "other submodules in the returned conversion than in the witness"
    for (name, config), (_, other) in zip(mine.module, theirs.module):
        what = _module_difference(config, other)
        if what is not None:
            return f"the module {name or mine.name}'s {what[0]} {_described(what[1])} in the returned conversion, " \
                   f"{_described(what[2])} in the witness"
    return None


def _module_difference(config: Any, other: Any) -> Optional[Tuple[str, Any, Any]]:
    """``(part, returned, witness)`` of the first part of a called module's configuration that differs."""
    if config[0] == "attribute":  # a get_attr node's value (a tensor's shape)
        return None if difference(config, other) is None else ("value", config[1], other[1])
    kind, attributes, hooks, state = config
    if kind is not other[0]:
        return "class", kind, other[0]
    for key in dict.fromkeys([*attributes, *other[1]]):
        if key not in attributes or key not in other[1] or difference(attributes[key], other[1][key]) is not None:
            return key, attributes.get(key, "no value"), other[1].get(key, "no value")
    if difference(hooks, other[2]) is not None:
        return "forward hooks", hooks, other[2]
    return None if difference(state, other[3]) is None else ("tensor shapes", state, other[3])


def program_difference(returned: Iterable[NodeRecord], witness: Iterable[NodeRecord],
                       altered: FrozenSet[str]) -> Optional[Tuple[str, str, str]]:
    """``(FX node, operator, difference)`` of the first node at which the returned and the witness programs differ,
    in what they compute or in a tensor they read that the copy holds otherwise (``altered``:
    ``state_differences``); ``None`` when they are the same program."""
    mine, theirs = list(returned), list(witness)
    for record, other in zip(mine, theirs):
        found = _call_difference(record, other)
        if found is not None:
            return record.name, record.operator, found
    if len(mine) != len(theirs):
        extra = (mine if len(mine) > len(theirs) else theirs)[min(len(mine), len(theirs))]
        return extra.name, extra.operator, "a node of one program only"
    for record in mine:
        read = next((name for name in record.reads if name in altered), None)
        if read is not None:
            return record.name, record.operator, (f"it reads {read}, another tensor in the traced program than in its "
                                                  "fp64 witness program")
    return None

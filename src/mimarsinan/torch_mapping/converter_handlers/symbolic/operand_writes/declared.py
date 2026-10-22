"""The operands a declared call writes, and the operands its result may share storage with, as torch declares them
(review round 15; ``__init__.py`` states the rule and its lemma).

A call WRITES a tensor operand in place when:

- its aten schema annotates the operand ``Tensor(a!)``: an aten operator (``torch.ops.aten.resize_.default``), or a
  tensor function or method of the same name as an aten operator (``torch.resize_as_``, ``torch.clamp_``,
  ``y.relu_()``), every overload of the name counted; a keyword-only written argument (``out``) only where the call
  passes it;
- it is given ``inplace=True`` (a function of torch's namespaces with an ``inplace`` parameter, ``F.relu(y, True)``);
- it calls a declared module whose ``inplace`` flag is set (``nn.ReLU(inplace=True)``).

Such a write RETURNS its operand when the schema's return is the written alias (``Tensor(a!) self -> Tensor(a!)``), and
for ``inplace``: torch's in-place convention, which the traced program's run checks (``observed.py``).

A call's result MAY SHARE an operand's storage by declaration when the schema annotates the operand with an alias set
(``Tensor(a) self``: a view, ``contiguous``, ``to``), when it writes and returns it, for a tensor attribute that is a
view (``.T``, ``.mT``, ``.data``), an index of a tensor (``y[:, :3]``), and ``Tensor.sum_to_size`` (its operand itself
when the target is its own shape, with no annotation; review round 16). Every other sharing is what the traced
program's run shows (``observed.py``). A call that MAY share and that the run shows copying (``reshape`` of a permuted
view, ``contiguous`` of a transposed one) is taken as a copy, guarded at run time (``rewrite.py``, review round 16).
"""

from __future__ import annotations

import functools
import inspect
import operator
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, cast

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    is_declared_module,
    is_tensor_operator,
)

__all__ = ["Write", "declared_aliases", "declared_writes", "in_place_form", "observable_write",
           "schema_declares_no_alias", "view_aliases"]

_VIEW_ATTRIBUTES = frozenset({"T", "mT", "H", "mH", "data", "real", "imag"})


@dataclass(frozen=True)
class Write:
    """``node`` writes ``operand`` in place; ``returned``: its result is the written operand."""

    node: fx.Node
    operand: fx.Node
    returned: bool


def _schemas(node: fx.Node) -> Sequence[Any]:
    """The aten schemas of ``node``'s call: an aten operator's own, or every overload of the aten operator of the name
    of a declared tensor function or method; none for any other call."""
    target = node.target
    if node.op == "call_function" and isinstance(target, torch._ops.OpOverload):
        return [target._schema]
    if node.op == "call_function" and isinstance(target, torch._ops.OpOverloadPacket):
        return [getattr(target, name)._schema for name in target.overloads()]
    if node.op == "call_function" and is_tensor_operator(target):
        name = getattr(target, "__name__", None)
    elif node.op == "call_method":
        name = target
    else:
        return []
    return _named_schemas(name) if isinstance(name, str) else []


@functools.lru_cache(maxsize=None)
def _named_schemas(name: str) -> List[Any]:
    """Every overload's schema of the aten operator ``name`` (none when there is no such operator)."""
    found, _ = guarded(lambda: torch._C._jit_get_schemas_for_operator(f"aten::{name}"))
    return list(found or [])


def _operand(node: fx.Node, index: int, name: str) -> Optional[fx.Node]:
    """The FX node passed as the schema's argument ``index`` (``name``): positional, or by its keyword (``input`` for
    the receiver ``self``)."""
    if index < len(node.args) and name not in node.kwargs:
        found = node.args[index]
    else:
        found = node.kwargs.get(name, node.kwargs.get("input") if index == 0 else None)
    return found if isinstance(found, fx.Node) else None


def _tensor(arg: Any) -> bool:
    """A schema argument of tensor type (``Tensor``, ``Tensor?``): TorchScript's overloads of a list (``sort.Tensor``,
    ``Tensor[](a!) self``, which sorts a list in place) are no tensor's (review round 15)."""
    kind = arg.type
    if isinstance(kind, torch._C.OptionalType):
        kind = kind.getElementType()
    return isinstance(kind, torch._C.TensorType)


def _applies(schema: Any, node: fx.Node) -> bool:
    """The overload takes this call: its first argument is a tensor, and every keyword-only written argument
    (``out``) it has is passed."""
    return bool(schema.arguments) and _tensor(schema.arguments[0]) and all(
        arg.name in node.kwargs for arg in schema.arguments
        if arg.kwarg_only and arg.alias_info is not None and arg.alias_info.is_write)


def _schema_writes(node: fx.Node) -> List[Write]:
    found: List[Write] = []
    for schema in _schemas(node):
        if not _applies(schema, node):
            continue
        returned = {frozenset(r.alias_info.before_set) for r in schema.returns
                    if r.alias_info is not None and r.alias_info.is_write}
        for index, arg in enumerate(schema.arguments):
            operand = _operand(node, index, arg.name)
            if arg.alias_info is not None and arg.alias_info.is_write and operand is not None and _tensor(arg):
                found.append(Write(node, operand, frozenset(arg.alias_info.before_set) in returned))
    return found


def _inplace_argument(node: fx.Node) -> bool:
    """A function of torch's namespaces called with ``inplace=True``, positional or by keyword."""
    if node.op != "call_function" or not is_tensor_operator(node.target) or not callable(node.target):
        return False
    signature, _ = guarded(lambda: inspect.signature(cast(Callable[..., Any], node.target)))
    if signature is None or "inplace" not in signature.parameters:
        return False
    bound, _ = guarded(lambda: signature.bind(*node.args, **node.kwargs))
    return bound is not None and bound.arguments.get("inplace") is True


def _first_operand(node: fx.Node) -> Optional[fx.Node]:
    found = node.args[0] if node.args else next(iter(node.kwargs.values()), None)
    return found if isinstance(found, fx.Node) else None


def declared_writes(node: fx.Node, modules: Mapping[str, nn.Module]) -> List[Write]:
    """Every operand ``node``'s call writes in place as torch declares it (the module docstring)."""
    if node.op == "call_module":
        module = modules.get(str(node.target))
        operand = _first_operand(node)
        inplace = module is not None and is_declared_module(module) and getattr(module, "inplace", None) is True
        return [Write(node, operand, True)] if inplace and operand is not None else []
    found = _schema_writes(node)
    if _inplace_argument(node):
        operand = _first_operand(node)
        found += [Write(node, operand, True)] if operand is not None else []
    unique = {(write.operand, write.returned): write for write in found}
    return list(unique.values())


def in_place_form(node: fx.Node) -> Optional[fx.Node]:
    """The operand a call writes by the name of an in-place operator (``t.__imul__(2.0)``, ``Tensor.__idiv__``: a
    dunder ``__i<op>__`` of an operator ``__<op>__``; ``operator.iadd``: ``operator``'s in-place form), whether or not
    an aten schema declares the write (``__imul__`` has none): its first operand; ``None`` for a call of any other
    name."""
    name = node.target if node.op == "call_method" else getattr(node.target, "__name__", None) \
        if node.op == "call_function" else None
    if not isinstance(name, str) or name[:1] != "i" and not name.startswith("__i"):
        return None
    if name.startswith("__") and name.endswith("__"):
        named = any(hasattr(owner, f"__{name[3:-2]}__") for owner in (operator, torch.Tensor)) and len(name) > 5
    else:
        named = getattr(operator, name, None) is node.target and (
            hasattr(operator, name[1:]) or hasattr(operator, name[1:] + "_"))
    return _first_operand(node) if named else None


def schema_declares_no_alias(node: fx.Node) -> bool:
    """``node``'s call has an aten schema that applies to it, and no applicable overload annotates an argument or a
    result with an alias set (it neither writes nor returns a view), nor is it given ``inplace=True``."""
    applicable = [schema for schema in _schemas(node) if _applies(schema, node)]
    return bool(applicable) and not _inplace_argument(node) and all(
        arg.alias_info is None for schema in applicable for arg in (*schema.arguments, *schema.returns))


def declared_aliases(node: fx.Node, modules: Mapping[str, nn.Module]) -> List[fx.Node]:
    """The operands ``node``'s result may share storage with, as torch declares it (the module docstring): the written
    operands it returns, and its ``view_aliases``."""
    found = [write.operand for write in declared_writes(node, modules) if write.returned]
    return list(dict.fromkeys([*found, *view_aliases(node)]))


_MAY_RETURN_OPERAND = frozenset({"sum_to_size"})
"""Tensor methods whose result is their operand at some shapes and a new tensor at others, with no alias annotation in
their schema (``aten::sum_to_size(Tensor self, SymInt[] size) -> Tensor``, review round 16)."""


def view_aliases(node: fx.Node) -> List[fx.Node]:
    """The operands ``node``'s result may share storage with as a view or a value that may be one (an operand annotated
    ``Tensor(a)``, a view attribute, an index of a tensor, ``_MAY_RETURN_OPERAND``): torch's declaration that it MAY;
    the traced program's run shows where it does (``rewrite._alias_edges``)."""
    if node.op == "call_function" and node.target is getattr and len(node.args) == 2 \
            and node.args[1] in _VIEW_ATTRIBUTES and isinstance(node.args[0], fx.Node):
        return [node.args[0]]
    found: List[fx.Node] = []
    if node.op == "call_function" and node.target is operator.getitem and isinstance(node.args[0], fx.Node):
        found.append(node.args[0])  # an index of a tensor (a view), or an element of a tuple of tensors
    if node.op == "call_method" and node.target in _MAY_RETURN_OPERAND and isinstance(node.args[0], fx.Node):
        found.append(node.args[0])
    for schema in _schemas(node):
        if not _applies(schema, node):
            continue
        for index, arg in enumerate(schema.arguments):
            operand = _operand(node, index, arg.name)
            if arg.alias_info is not None and not arg.alias_info.is_write and operand is not None and _tensor(arg):
                found.append(operand)
    return list(dict.fromkeys(found))


def observable_write(node: fx.Node, modules: Mapping[str, nn.Module], order: Mapping[fx.Node, int]) -> bool:
    """``node`` writes an operand that a node after it reads (dead-code elimination keeps it; after
    ``define_writes`` no such reader is left but through the write's own result)."""
    return any(order.get(user, -1) > order[node] for write in declared_writes(node, modules)
               for user in write.operand.users)

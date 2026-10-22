"""Shape queries as values: static ones are conversion-time literals, every other one runs on the host.

A queried shape is the value's batch layout (``batch_layout.py``): an axis is static only where the symbolic-batch
pass proves it, and ``BATCH_SIZE`` marks every other axis; a value no proof covers has no layout and keeps its
w0-base host ops. A static whole shape is a ``torch.Size``, as natively. A shape value is evaluated at conversion time
only when its whole argument tree is static (``is_static_tree``) and its operator is a scalar operator of the declared
operator set (``CONVERSION_TIME_OPERATORS``, ``symbolic/declared_operators.py``: the ``operator`` module, ``math``,
``torch.sym_*``, ``torch._assert``): a tree holding an unproven extent never reaches a Python function (CPython
compares containers by identity first, so one stand-in for two different run-time extents would compare equal), and
any other callable (a user's ``torch.fx.wrap`` helper, a tensor factory) is never called at conversion; such a value
keeps its host op, which evaluates it at run time. The one operation on a tree with an unproven extent is a projection (``project``), which
picks entries of a recorded shape and reads none of them.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import map_arg

from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE
from mimarsinan.torch_mapping.converter_handlers.converter_contract import ConverterContract
from mimarsinan.torch_mapping.converter_handlers.refusals import node_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    CONVERSION_TIME_OPERATORS,
    is_conversion_time_operator,
)

__all__ = [
    "BATCH_SIZE", "CONVERSION_TIME_OPERATORS", "ShapeValueMixin", "depends_on_batch", "depends_on_input",
    "evaluate_shape_value", "is_static_tree", "project",
]


# Tensor methods and attributes that query metadata, never values.
SHAPE_QUERY_METHODS = frozenset({"size", "dim"})
SHAPE_QUERY_ATTRIBUTES = frozenset({"shape", "ndim"})

STATIC_LEAVES = (int, float, bool, str, type(None))
"""The exact types of a static leaf: a subclass (a user ``int`` with its own ``__eq__``), a tensor, a symbolic size,
a dtype or any other object is not one."""

STATIC_SEQUENCES = (tuple, list, torch.Size, immutable_list)
"""The exact sequence types of a static tree: the plain ones, ``torch.Size`` and FX's immutable list; a namedtuple or
a user subclass of a container (its own ``__getitem__``) is not one."""

STATIC_MAPPINGS = (dict, immutable_dict)


def is_static_tree(value: Any) -> bool:
    """Every leaf of ``value`` is a concrete Python scalar (``STATIC_LEAVES``), through containers of the exact types
    ``STATIC_SEQUENCES`` and ``STATIC_MAPPINGS`` and slices; reads only types and container structure."""
    if type(value) in STATIC_LEAVES:
        return True
    if type(value) in STATIC_SEQUENCES:
        return all(is_static_tree(item) for item in value)
    if type(value) in STATIC_MAPPINGS:
        return all(is_static_tree(key) and is_static_tree(item) for key, item in value.items())
    if type(value) is slice:
        return is_static_tree((value.start, value.stop, value.step))
    return False


def depends_on_batch(value: Any) -> bool:
    """Known only at run time: an unproven extent (the batch size among them) or any non-static leaf in ``value``."""
    return not is_static_tree(value)


def _native(shape: Any) -> Any:
    """A recorded shape as natively: a ``torch.Size`` when static; one holding an unproven extent stays a tuple."""
    return torch.Size(shape) if type(shape) is tuple and is_static_tree(shape) else shape


def project(fn, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Optional[Tuple[Any]]:
    """``(entries,)`` when ``fn`` projects a recorded shape (a tuple or ``torch.Size``): its entry or slice at a static
    integer index (a slice is a shape), or its length. Indexing and length read no entry, so an unproven entry is
    carried, never observed."""
    if kwargs or not args or type(args[0]) not in (tuple, torch.Size):
        return None
    shape = args[0]
    if fn is operator.getitem and len(args) == 2 and type(args[1]) in (int, slice) and is_static_tree(args[1]):
        return (_native(shape[args[1]]) if type(args[1]) is slice else shape[args[1]],)
    if fn is len and len(args) == 1:
        return (len(shape),)
    return None


def evaluate_shape_value(fn, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Tuple[bool, Any]:
    """``fn(*args, **kwargs)`` as a shape value: ``(True, value)``, or ``(False, None)`` when it is not one.

    A projection is taken as is. A call on a tree holding an unproven extent is ``(True, BATCH_SIZE)`` without calling
    ``fn`` (a run-time value: its host op evaluates it). A call on a static tree is evaluated only when ``fn`` is in
    ``CONVERSION_TIME_OPERATORS``; any other ``fn`` is ``(False, None)``, never called (its host op evaluates it at run
    time, on the values the native model passes it). An exception of a closed-set operator propagates as itself,
    never as a fold: the operator receives the values, with their native types, that the native model passes it.
    """
    projected = project(fn, args, kwargs)
    if projected is not None:
        return True, projected[0]
    if not is_static_tree((tuple(args), dict(kwargs))):
        return True, BATCH_SIZE
    if not is_conversion_time_operator(fn):
        return False, None
    value = fn(*args, **kwargs)
    if isinstance(value, torch.Tensor):
        return False, None
    return True, value


def depends_on_input(node: fx.Node) -> bool:
    """A placeholder lies upstream of ``node``: it is an activation, not a parameter or a constant."""
    seen, stack = set(), [node]
    while stack:
        current = stack.pop()
        if current.op == "placeholder":
            return True
        if current not in seen:
            seen.add(current)
            stack.extend(current.all_input_nodes)
    return False


def _query(shape: Tuple[Any, ...], method: str, args: Sequence[Any]) -> Any:
    """A query of a recorded shape: its rank (proved with the layout), the shape (a ``torch.Size`` when static) or a
    projection at a literal index."""
    if method == "dim" or method == "ndim":
        return len(shape)
    if method == "shape" or not args:
        return _native(shape)
    return shape[int(args[0])]


class ShapeValueMixin(ConverterContract):
    """Records the value of every shape query and of the arithmetic built only from such values."""

    def _shape_query(self, node: fx.Node) -> Tuple[bool, Any]:
        """The value of a ``size``/``dim`` call or a ``shape``/``ndim`` attribute read on a traced tensor."""
        if node.op == "call_method" and node.target in SHAPE_QUERY_METHODS:
            method, rest = str(node.target), node.args[1:] or tuple(node.kwargs.values())
        elif (node.op == "call_function" and node.target is getattr and len(node.args) == 2
              and node.args[1] in SHAPE_QUERY_ATTRIBUTES):
            method, rest = str(node.args[1]), ()
        else:
            return False, None
        tensor = node.args[0] if node.args else None
        shape = self._batch_layouts.layout(tensor) if isinstance(tensor, fx.Node) else None
        if not shape or any(isinstance(a, fx.Node) for a in rest) or set(node.kwargs) - {"dim"}:
            return False, None
        return True, _query(shape, method, rest)

    def _shape_arithmetic(self, node: fx.Node) -> Tuple[bool, Any]:
        """A call_function whose every FX argument is a recorded shape value (``evaluate_shape_value``)."""
        if node.op != "call_function":
            return False, None
        found: list[fx.Node] = []
        map_arg((node.args, node.kwargs), found.append)
        if not found or any(arg not in self._shape_values for arg in found):
            return False, None
        args, kwargs = map_arg((node.args, node.kwargs), self._shape_values.__getitem__)
        return evaluate_shape_value(node.target, args, kwargs)

    def _record_shape_value(self, node: fx.Node) -> bool:
        """Record ``node``'s shape value; True when it is static (no host op emitted, a literal for its users)."""
        known, value = self._shape_query(node)
        if not known:
            known, value = self._shape_arithmetic(node)
        if not known:
            return False
        self._shape_values[node] = value
        if not is_static_tree(value):
            return False
        self._node_to_mapper[node] = None
        return True

    def _static_shape_value(self, node: fx.Node) -> Tuple[bool, Any]:
        value = self._shape_values.get(node, BATCH_SIZE)
        return (True, value) if is_static_tree(value) else (False, None)

    def _static_dim(self, node: fx.Node, value: Any, refusal: Callable[[fx.Node, str], Exception]) -> Any:
        """A layout operator's dimension argument known at conversion time: a literal, or a static shape value
        (``y.dim() - 1``), through tuples and lists; ``refusal(node, reason)`` is raised for any other."""
        if isinstance(value, (tuple, list)):
            return tuple(self._static_dim(node, item, refusal) for item in value)
        if isinstance(value, fx.Node):
            static, resolved = self._static_shape_value(value)
            if not static:
                raise refusal(node, f"dim {value.name!r} is not known at conversion time")
            return resolved
        return value

    def shape_literals(self) -> Dict[str, Tuple[str, str]]:
        """Every shape value folded into a conversion-time literal: FX node name -> (operator, ``repr`` of value)."""
        return {node.name: (node_operator(node), repr(value)) for node, value in self._shape_values.items()
                if is_static_tree(value)}

    def shape_opaque(self) -> Optional[Tuple[str, str, str]]:
        """The program's first opaque batch value (FX node, operator, why), ``None`` when there is none."""
        found = self._batch_layouts.opaque()
        return None if found is None else (found.node.name, found.operator, found.detail)

    def shape_proof(self) -> Optional[str]:
        """What the program's static shapes are proved by (``static shape proof: ...`` or ``no static shape proof:
        ...``, ``SymbolicPass.proof_statement``); ``None`` when the conversion asked for no shape."""
        return self._batch_layouts.proof_statement()

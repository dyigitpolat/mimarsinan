"""Positional call templates: an FX call's argument order kept through source, bound and literal slots."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.fx as fx

# Callables whose two operands swap with bit-identical results (IEEE 754 addition
# and multiplication are commutative); keyword arguments such as ``alpha`` void it.
COMMUTATIVE_BINARY_FNS: frozenset = frozenset({operator.add, operator.mul, torch.add, torch.mul})


@dataclass(frozen=True)
class SourceSlot:
    """The ``index``-th activation input of the op."""

    index: int


@dataclass(frozen=True)
class BoundSlot:
    """The ``index``-th bound tensor; ``shape`` restores its native shape when storage stripped it."""

    index: int
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LiteralSlot:
    """The ``index``-th top-level literal argument (``ComputeAdapter.extra_args``)."""

    index: int


_SLOT_TYPES = (SourceSlot, BoundSlot, LiteralSlot)


def _map_structure(obj: Any, is_leaf: Callable[[Any], bool], leaf: Callable[[Any], Any]) -> Any:
    """Rebuild tuples, lists, dicts and slices with every ``is_leaf`` element replaced by ``leaf``."""
    if is_leaf(obj):
        return leaf(obj)
    if isinstance(obj, tuple):
        items = [_map_structure(item, is_leaf, leaf) for item in obj]
        return type(obj)(*items) if hasattr(obj, "_fields") else type(obj)(items)
    if isinstance(obj, list):
        return [_map_structure(item, is_leaf, leaf) for item in obj]
    if isinstance(obj, dict):
        return {key: _map_structure(value, is_leaf, leaf) for key, value in obj.items()}
    if isinstance(obj, slice):
        return slice(*(_map_structure(part, is_leaf, leaf) for part in (obj.start, obj.stop, obj.step)))
    return obj


def _is_node(obj: Any) -> bool:
    return isinstance(obj, fx.Node)


def _is_slot(obj: Any) -> bool:
    return isinstance(obj, _SLOT_TYPES)


def _contains_node(obj: Any) -> bool:
    found: list[Any] = []
    _map_structure(obj, _is_node, found.append)
    return bool(found)


@dataclass(frozen=True)
class ArgTemplate:
    """An FX call's arguments in their traced order, every value replaced by its slot."""

    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()

    def resolve(
        self,
        inputs: Sequence[Any],
        bound: Sequence[torch.Tensor],
        literals: Sequence[Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        def value(slot):
            if isinstance(slot, SourceSlot):
                return inputs[slot.index]
            if isinstance(slot, BoundSlot):
                tensor = bound[slot.index]
                return tensor if slot.shape is None else tensor.reshape(slot.shape)
            return literals[slot.index]

        args = tuple(_map_structure(arg, _is_slot, value) for arg in self.args)
        kwargs = {key: _map_structure(arg, _is_slot, value) for key, arg in self.kwargs}
        return args, kwargs


@dataclass(frozen=True)
class SourceArg:
    """Classification of an FX argument as an activation input (``payload`` is the caller's handle)."""

    payload: Any


@dataclass(frozen=True)
class BoundArg:
    """Classification of an FX argument as a bound tensor stored as ``tensor``, natively ``native_shape``."""

    tensor: torch.Tensor
    native_shape: tuple[int, ...]


@dataclass(frozen=True)
class LiteralArg:
    """Classification of an FX argument whose value is known at conversion time; it is a literal, as if traced as one."""

    value: Any


@dataclass
class CallPartition:
    """An FX call split into sources, bound tensors and literals, plus the order that reassembles it.

    ``template`` is ``None`` for the canonical layout ``fn(*sources, *bound, *literals)``.
    """

    sources: list[Any] = field(default_factory=list)
    bound: list[torch.Tensor] = field(default_factory=list)
    bound_native_shapes: list[tuple[int, ...]] = field(default_factory=list)
    literals: tuple[Any, ...] = ()
    literal_kwargs: dict[str, Any] = field(default_factory=dict)
    template: ArgTemplate | None = None
    unresolved: list[fx.Node] = field(default_factory=list)
    positional: tuple[Any, ...] = ()
    slot_kwargs: tuple[tuple[str, Any], ...] = ()

    @property
    def strips_bound_shapes(self) -> bool:
        return any(
            tuple(t.shape) != native for t, native in zip(self.bound, self.bound_native_shapes)
        )

    def explicit_template(self, *, restore_native_shapes: bool = False, every_bound: bool = False) -> ArgTemplate:
        """The call's template, spelled out even for the canonical layout; ``every_bound``: every bound tensor is
        reshaped to its native shape, stored stripped or not (a single-sample axis the adapter adds is undone too)."""

        def restore(slot):
            if not ((restore_native_shapes or every_bound) and isinstance(slot, BoundSlot)):
                return slot
            native = self.bound_native_shapes[slot.index]
            stored = tuple(self.bound[slot.index].shape)
            return slot if native == stored and not every_bound else BoundSlot(slot.index, native)

        return ArgTemplate(
            tuple(_map_structure(arg, _is_slot, restore) for arg in self.positional),
            tuple((key, _map_structure(arg, _is_slot, restore)) for key, arg in self.slot_kwargs),
        )


_CANONICAL_ORDER = {SourceSlot: 0, BoundSlot: 1, LiteralSlot: 2}


def _is_canonical(positional: Sequence[Any]) -> bool:
    if not all(isinstance(arg, _SLOT_TYPES) for arg in positional):
        return False
    ranks = [_CANONICAL_ORDER[type(arg)] for arg in positional]
    return ranks == sorted(ranks)


def partition_call(
    fn: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    classify: Callable[[fx.Node], SourceArg | BoundArg | LiteralArg | None],
) -> CallPartition:
    """Split ``fn(*args, **kwargs)`` by ``classify`` into slots that keep the traced argument order."""
    part = CallPartition()
    literals: list[Any] = []
    kinds: dict[fx.Node, Any] = {}

    def kind_of(node: fx.Node):
        if node not in kinds:
            kinds[node] = classify(node)
        return kinds[node]

    def resolve_literals(obj: Any) -> Any:
        return _map_structure(
            obj, lambda o: _is_node(o) and isinstance(kind_of(o), LiteralArg), lambda o: kind_of(o).value,
        )

    args = [resolve_literals(arg) for arg in args]
    kwargs = {key: resolve_literals(value) for key, value in kwargs.items()}

    def slot_for(node: fx.Node):
        kind = kind_of(node)
        if isinstance(kind, SourceArg):
            part.sources.append(kind.payload)
            return SourceSlot(len(part.sources) - 1)
        if isinstance(kind, BoundArg):
            part.bound.append(kind.tensor)
            part.bound_native_shapes.append(tuple(kind.native_shape))
            return BoundSlot(len(part.bound) - 1)
        part.unresolved.append(node)
        return None

    positional = []
    for arg in args:
        if _contains_node(arg):
            positional.append(_map_structure(arg, _is_node, slot_for))
        else:
            literals.append(arg)
            positional.append(LiteralSlot(len(literals) - 1))
    slot_kwargs = []
    for key, value in kwargs.items():
        if _contains_node(value):
            slot_kwargs.append((key, _map_structure(value, _is_node, slot_for)))
        else:
            part.literal_kwargs[key] = value

    part.literals = tuple(literals)
    part.positional = tuple(positional)
    part.slot_kwargs = tuple(slot_kwargs)
    if slot_kwargs or not _is_canonical(positional):
        swappable = (
            any(fn is op for op in COMMUTATIVE_BINARY_FNS) and len(positional) == 2 and not kwargs
            and _is_canonical(positional[::-1])
        )
        if not swappable:
            part.template = ArgTemplate(part.positional, part.slot_kwargs)
    return part

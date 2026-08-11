"""Strict JSON serde shared by every schema fragment: unknown fields always raise."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def strict_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """``dict(data)`` after rejecting fields the dataclass does not declare."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__} has unknown fields {sorted(unknown)} — the format "
            f"may have drifted; migrate explicitly, never tolerate silently"
        )
    return dict(data)


def tuple_of(
    from_item: Callable[[Any], T], items: Optional[Sequence[Any]]
) -> Tuple[T, ...]:
    """A tuple built by mapping ``from_item`` over a JSON sequence (``None`` = empty)."""
    return tuple(from_item(item) for item in (items or ()))


def optional(from_value: Callable[[Any], T], value: Any) -> Optional[T]:
    """``from_value(value)`` unless ``value`` is None."""
    return None if value is None else from_value(value)


def int_pair(value: Sequence[Any]) -> Tuple[int, int]:
    """A JSON 2-sequence as an ``(int, int)`` tuple; wrong arity raises."""
    first, second = value
    return (int(first), int(second))


def float_pair(value: Sequence[Any]) -> Tuple[float, float]:
    """A JSON 2-sequence as a ``(float, float)`` tuple; wrong arity raises."""
    first, second = value
    return (float(first), float(second))


def require_choice(cls_name: str, field_name: str, value: str, allowed: frozenset[str]) -> None:
    """Fail loud when an enumerated string field carries an unknown value."""
    if value not in allowed:
        raise ValueError(
            f"{cls_name}.{field_name} must be one of {sorted(allowed)}, got {value!r}"
        )

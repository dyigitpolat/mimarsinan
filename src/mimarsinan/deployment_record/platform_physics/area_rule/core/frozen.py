"""[U-AREA] A frozen object of the unit (the fourth delta round): the read-only mapping and list a block holds its
declared objects in, so that a frozen block is frozen through and through and the door's memo keys it by its identity
(core/door.py), never by its pickled text.

Each is its plain form's subclass: it equals the plain form, writes the same text (json), pickles and copies, and
``dict(x)``, ``{**x}``, ``list(x)`` and slices give plain forms. A change in place is a TypeError, as on any immutable
value: a changed block is a new block (``dataclasses.replace``), a new input the door judges again. A value nested
deeper than the domain's depth (units.NESTING) is left as it is: the door refuses it by name.
"""

from __future__ import annotations

from typing import Any, Iterable, NoReturn

from mimarsinan.deployment_record.platform_physics.area_rule import units as u


def _refused(self: Any, *_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError(f"a frozen {type(self).__name__} of the area rule does not change in place: a changed block is a "
                    "new block (dataclasses.replace)")


class FrozenDict(dict):
    """A read-only mapping whose values are frozen."""

    __slots__ = ()

    def __init__(self, data: Any = (), _depth: int = u.TOP) -> None:
        super().__init__((key, freeze(value, _depth + u.LEVEL)) for key, value in dict(data).items())

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refused

    def __reduce__(self) -> Any:
        return FrozenDict, (dict(self),)


class FrozenList(list):
    """A read-only list whose elements are frozen."""

    __slots__ = ()

    def __init__(self, data: Iterable[Any] = (), _depth: int = u.TOP) -> None:
        super().__init__(freeze(value, _depth + u.LEVEL) for value in data)

    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = extend = insert = pop = remove = clear = sort = \
        reverse = _refused

    def __reduce__(self) -> Any:
        return FrozenList, (list(self),)


def freeze(value: Any, depth: int = u.TOP) -> Any:
    """``value`` frozen: a dict a FrozenDict, a list a FrozenList, a tuple of frozen values; anything else, and a value
    nested deeper than the domain's depth, as it is (the door judges it)."""
    kind = type(value)
    if depth > u.NESTING or kind in (FrozenDict, FrozenList):
        return value
    if kind is dict:
        return FrozenDict(value, depth)
    if kind is list:
        return FrozenList(value, depth)
    if kind is tuple:
        return tuple(freeze(item, depth + u.LEVEL) for item in value)
    return value


__all__ = ["FrozenDict", "FrozenList", "freeze"]

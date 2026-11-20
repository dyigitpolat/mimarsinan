"""[U-AREA] The door's memo (the fourth delta round; moved out of core/door.py for the module budget, which re-exports
its names): a judgement, or a value derived from a frozen input alone, made once per frozen input, keyed without
pickling (the objects by their identity, the values read off what is no frozen object as a digest)."""

from __future__ import annotations

import copy
import weakref
from typing import Any, Callable, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

_MEMO = "door"
_REF = weakref.ref


def _alive(refs: Tuple[Any, ...], held: Tuple[Any, ...]) -> bool:
    """Whether each remembered object is still the one held (a weak reference read, or the entry's own object)."""
    for ref, obj in zip(refs, held):
        if (ref() if type(ref) is _REF else ref) is not obj:
            return False
    return True


def remembered(registry: Any, what: str, held: Tuple[Any, ...], make: Callable[[], Any], *digest: Any) -> Any:
    """``make()`` (a judgement, or a value derived from a frozen input alone: ``what`` names it) once per frozen input:
    the objects ``held`` by their identity (each kept by a weak reference where it takes one, else by the entry
    itself), the values ``digest`` read off what is no frozen object of the unit, and the registry's exclusions; its
    outcome kept in the registry's memo: its value, or its named refusal raised again as a fresh copy. No input is
    pickled; ``forget`` drops every outcome."""
    key = (_MEMO, what, tuple(getattr(registry, "excl", ())), tuple(map(id, held)), digest)
    memo = registry._memo
    try:
        entry = memo.get(key)
    except TypeError:  # a digest value no key holds (a list): made afresh
        return make()
    if entry is None or not _alive(entry[u.FIRST], held):
        try:
            outcome, refused = make(), None
        except AreaRuleRefusal as caught:
            outcome, refused = None, caught.with_traceback(None)
        refs = []
        for obj in held:
            try:
                refs.append(_REF(obj, lambda _ref, key=key: memo.pop(key, None)))
            except TypeError:  # a tuple or a mapping takes no weak reference: the entry keeps it
                refs.append(obj)
        entry = memo[key] = (tuple(refs), refused, outcome)
    _refs, refused, outcome = entry
    if refused is not None:
        raise copy.copy(refused)
    return outcome


def forget(registry: Any) -> None:
    """Drop the door's outcomes from the registry's memo (a test's fresh start)."""
    for key in [k for k in registry._memo if k and k[u.FIRST] == _MEMO]:
        del registry._memo[key]


__all__ = ["forget", "remembered"]

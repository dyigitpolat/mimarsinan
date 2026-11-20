"""[U-AREA] The one judge of a quantity the area rule reads (the second delta round, MA-D1) and its kinds, each an
interval with both ends stated (the third delta round: every kind closed), its values of the declared domain's exact
number types (the fourth delta round). A value of its kind is returned as it is; any other value is a named refusal
naming the field and the value. A band is three such sizes in order (mi-K1).
"""

from __future__ import annotations

import math
import sys
from typing import Any, Mapping, NamedTuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal


class Kind(NamedTuple):
    """A declared quantity's kind: its words and its interval, both ends stated (``each``: a mapping of terms to values
    of the kind; ``whole``: a whole number)."""

    words: str
    low: float
    low_closed: bool
    high: float
    each: bool = False
    whole: bool = False


#: The largest count a float holds exactly (so a count's products never leave the floats), and the largest float.
EXACT = u.BINARY ** sys.float_info.mant_dig
LARGEST = sys.float_info.max
COUNT = Kind("a whole count above 0 (at most 2^53)", u.NONE, False, EXACT, whole=True)
COUNT_OR_ZERO = Kind("a whole count of at least 0 (at most 2^53)", u.NONE, True, EXACT, whole=True)
WEIGHTS = Kind("a finite weight of at least 0 (at most 2^53)", u.NONE, True, EXACT, each=True)
SHARE = Kind("a share above 0 and at most 1", u.NONE, False, u.UNITY)
POSITIVE = Kind("a finite number above 0", u.NONE, False, LARGEST)
NONNEGATIVE = Kind("a finite number of at least 0", u.NONE, True, LARGEST)
FINITE = Kind("a finite number", -LARGEST, True, LARGEST)
CELL = Kind("a cell's area above 0", u.NONE, False, LARGEST)  # finite: at most the largest float


def power(base: float, exponent: float) -> float:
    """base ** exponent in the floats, infinity where it leaves them: the one form of a power the rule computes, so the
    quantity it makes is judged where it is made (a named refusal), never an OverflowError (MA-K2)."""
    try:
        return base ** exponent
    except OverflowError:
        return math.inf


def judged(field: str, value: Any, kind: Kind, *words: Any) -> Any:
    """The one judge of a quantity read from a declaration or an argument: ``value`` where it is of its ``kind`` (a
    number of the declared domain's exact types, Python's int or float (the fourth delta round, MA-C1: never by the
    number tower, never a bool), finite, in the kind's interval (an int compared exactly), whole for a count), returned
    as it came, a Python int or float; else a named refusal naming the field (``field.format(*words)``, made only on
    a refusal) and the value."""
    exact = type(value)
    if not kind.each:
        if (exact is float or exact is int) and (kind.low <= value if kind.low_closed else kind.low < value) and \
                value <= kind.high and not (kind.whole and exact is float and value != math.floor(value)):
            return value  # a NaN fails every comparison; an int is compared exactly with the ends
    elif isinstance(value, Mapping):
        place = field.format(*words) if words else field
        for term, item in value.items():
            judged("{}.{}", item, kind._replace(each=False), place, term)
        return value
    place = field.format(*words) if words else field
    if kind.each:
        raise AreaRuleRefusal(f"{place} {value!r} is not a mapping of terms to {kind.words}")
    raise AreaRuleRefusal(f"{place} {value!r} is not {kind.words}")


def judged_band(field: str, low: Any, nominal: Any, high: Any) -> None:
    """A band where it is made: each edge a finite number above 0, and in order, low <= nominal <= high (mi-K1)."""
    band = (low, nominal, high)
    for edge in band:
        judged("{} {!r}: an edge", edge, POSITIVE, field, band)
    if not low <= nominal <= high:
        raise AreaRuleRefusal(f"{field} {band!r} is not ordered low <= nominal <= high")


__all__ = ["CELL", "COUNT", "COUNT_OR_ZERO", "EXACT", "FINITE", "Kind", "LARGEST", "NONNEGATIVE", "POSITIVE", "SHARE",
           "WEIGHTS", "judged", "judged_band", "power"]

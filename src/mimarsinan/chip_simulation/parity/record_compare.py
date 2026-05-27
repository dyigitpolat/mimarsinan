"""Generic field-wise record comparison for parity harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence


@dataclass
class FieldDiff:
    path: str
    expected: Any
    actual: Any


def compare_segment_records(
    ref: Any,
    actual: Any,
    fields: Sequence[tuple[str, Callable[[Any], Any]]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> List[FieldDiff]:
    """Compare ``fields`` on two segment-like records; return mismatches."""
    diffs: List[FieldDiff] = []
    for name, getter in fields:
        a = getter(ref)
        b = getter(actual)
        if isinstance(a, float) or isinstance(b, float):
            if not _float_close(float(a), float(b), rtol=rtol, atol=atol):
                diffs.append(FieldDiff(name, a, b))
        elif a != b:
            diffs.append(FieldDiff(name, a, b))
    return diffs


def _float_close(a: float, b: float, *, rtol: float, atol: float) -> bool:
    if a == b:
        return True
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

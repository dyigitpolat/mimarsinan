"""Generic field-wise record comparison for parity harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from mimarsinan.common.measurement import MetricTolerance


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
    tolerance = MetricTolerance(abs_tol=atol, rel_tol=rtol)
    diffs: List[FieldDiff] = []
    for name, getter in fields:
        a = getter(ref)
        b = getter(actual)
        if isinstance(a, float) or isinstance(b, float):
            if not tolerance.matches(float(a), float(b)):
                diffs.append(FieldDiff(name, a, b))
        elif a != b:
            diffs.append(FieldDiff(name, a, b))
    return diffs

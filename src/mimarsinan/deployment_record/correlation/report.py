"""The correlation table: predicted against published, per device, with the caveats."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from mimarsinan.deployment_record.correlation.run import CaseCorrelation, worst_error

_HEADER = (
    f"{'case':<30}{'basis':<11}{'axis':<24}"
    f"{'predicted':>13}{'published':>13}{'error':>10}  evidence"
)


def _row(result: CaseCorrelation, axis) -> str:
    predicted = "refused" if axis.predicted is None else f"{axis.predicted:.5g}"
    error = "  —" if axis.predicted is None else f"{axis.error_pct:+.1f}%"
    marks = [] if result.case.is_independent else ["self_consistency"]
    if result.overrides_applied:
        marks.append("override:" + ",".join(sorted(result.overrides_applied)))
    if axis.band_contains_published:
        marks.append("in-band")
    return (
        f"{result.case.name:<30}{result.measurement_kind:<11}{axis.name:<24}"
        f"{predicted:>13}{axis.published:>13.5g}{error:>10}  {' '.join(marks)}"
    )


def correlation_payload(results: Sequence[CaseCorrelation]) -> Dict[str, Any]:
    """The whole suite as JSON-safe data — the shape the checked-in golden pins.

    The BAND rides along with the nominal: a constant whose band moved while its
    nominal held is still a changed prediction, and a golden that could not see
    that would pass a physics edit it was written to catch.
    """
    return {
        "cases": [
            {
                "name": result.case.name,
                "profile": result.case.profile,
                "measurement_kind": result.measurement_kind,
                "independence": result.case.independence,
                "citation": result.case.citation,
                "overrides_applied": dict(result.overrides_applied),
                "passed": result.passed,
                "axes": [
                    {
                        "axis": axis.name,
                        "predicted": axis.predicted,
                        "published": axis.published,
                        "band": None if axis.band is None else list(axis.band),
                        "error_pct": None if axis.predicted is None else axis.error_pct,
                        "refusal": axis.refusal,
                    }
                    for axis in result.axes
                ],
            }
            for result in results
        ]
    }


def correlation_payload_json(results: Sequence[CaseCorrelation]) -> str:
    """Exactly the bytes ``scripts/silicon_correlation.py --json`` writes."""
    return json.dumps(correlation_payload(results), indent=2, sort_keys=True) + "\n"


def render_correlation(results: Sequence[CaseCorrelation]) -> str:
    """The whole suite as text, worst error per device called out at the end."""
    lines = [_HEADER, "-" * len(_HEADER)]
    for result in results:
        for axis in result.axes:
            lines.append(_row(result, axis))
    lines.append("")
    lines.append("worst absolute error per target")
    for profile, error in sorted(worst_error(results).items()):
        lines.append(f"  {profile:<28}{error:6.1f}%")
    refused = [
        (r.case.name, a.name, a.refusal)
        for r in results for a in r.axes if a.refusal is not None
    ]
    if refused:
        lines.append("")
        lines.append("axes the physics refused (scored as a miss, never as agreement)")
        for name, axis, reason in refused:
            lines.append(f"  {name:<28}{axis:<24}{reason[:60]}")
    circular = [r.case.name for r in results if not r.case.is_independent]
    if circular:
        lines.append("")
        lines.append(
            "NOTE: self_consistency rows re-derive the operating point their own "
            "constant was divided out of. They check arithmetic, not generalization: "
            + ", ".join(circular)
        )
    return "\n".join(lines)

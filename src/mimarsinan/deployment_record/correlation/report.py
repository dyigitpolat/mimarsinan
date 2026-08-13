"""The correlation table: predicted against published, per device, with the caveats."""

from __future__ import annotations

from typing import Sequence

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

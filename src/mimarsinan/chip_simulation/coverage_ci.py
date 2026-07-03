"""CI guards that fail loud on each way the coverage instrument could lie."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mimarsinan.chip_simulation.coverage_ledger import (
    AXES,
    CoverageReport,
    HypervolumeAxis,
    ScreeningStatus,
)


__all__ = [
    "CoverageGuardError",
    "DEFAULT_FLAG_AGE_DAYS",
    "assert_axes_screening_sound",
    "assert_no_merged_valid_tiers",
    "assert_no_aged_unowned_flags",
    "audit_coverage_instrument",
]


DEFAULT_FLAG_AGE_DAYS = 90


_MERGED_VALID_TIER_KEYS = (
    "valid_total",
    "covered_valid_total",
    "valid_covered_total",
    "valid_or_flagged",
    "valid_plus_flagged",
)


class CoverageGuardError(AssertionError):
    """A coverage self-audit invariant was violated (CI must fail)."""


def assert_axes_screening_sound(
    axes: Sequence[HypervolumeAxis] = AXES,
) -> None:
    """FAIL if any axis is ``SCREENED_COLLAPSED`` without a ``screening_artifact`` (catches post-construction tampering)."""
    offenders = [
        a.name
        for a in axes
        if a.screening_status is ScreeningStatus.SCREENED_COLLAPSED
        and not (a.screening_artifact and a.screening_artifact.strip())
    ]
    if offenders:
        raise CoverageGuardError(
            f"axes {offenders} are SCREENED_COLLAPSED with no screening_artifact — "
            f"a collapse REQUIRES a linked artifact (doc/test/result ref); "
            f"collapse-on-a-hunch is forbidden"
        )


def assert_no_merged_valid_tiers(report_dict: Mapping[str, Any]) -> None:
    """FAIL if a report (or its dict) carries a headline key that merges VALID + VALID_FLAGGED."""
    data = report_dict.to_dict() if isinstance(report_dict, CoverageReport) else report_dict
    merged = [key for key in _MERGED_VALID_TIER_KEYS if key in data]
    if merged:
        raise CoverageGuardError(
            f"report headline merges the two valid tiers via {merged} — VALID and "
            f"VALID_FLAGGED must ALWAYS be separate counts (a flagged cell owes a "
            f"research gap; never report it as plainly valid)"
        )


def assert_no_aged_unowned_flags(
    report: CoverageReport,
    max_age_days: int = DEFAULT_FLAG_AGE_DAYS,
) -> None:
    """FAIL if any flag is older than ``max_age_days`` AND has no owner."""
    offenders = [
        m
        for m in report.flag_metadata
        if m.is_unowned and m.age_days is not None and m.age_days > max_age_days
    ]
    if offenders:
        names = [(m.cell_key, m.age_days) for m in offenders]
        raise CoverageGuardError(
            f"{len(offenders)} flag(s) aged > {max_age_days} days with NO owner: "
            f"{names} — assign an owner or resolve the flag (it must not rot silently)"
        )


def audit_coverage_instrument(
    axes: Sequence[HypervolumeAxis],
    report: CoverageReport,
    max_age_days: int = DEFAULT_FLAG_AGE_DAYS,
) -> None:
    """Run every self-audit guard over the live axes + a coverage report."""
    assert_axes_screening_sound(axes)
    assert_no_merged_valid_tiers(report.to_dict())
    assert_no_aged_unowned_flags(report, max_age_days=max_age_days)

"""Frontier P1 CI guard — the self-auditing checks that keep the instrument honest.

The coverage instrument is only trustworthy if its INVARIANTS are mechanically
enforced. These guards FAIL LOUD on each way the instrument could lie:

* :func:`assert_axes_screening_sound` — every ``SCREENED_COLLAPSED`` axis carries a
  non-empty ``screening_artifact`` (collapse-on-a-hunch is forbidden);
* :func:`assert_no_merged_valid_tiers` — a report headline NEVER fuses VALID +
  VALID_FLAGGED into one covered total (a flagged cell is never claimed plainly valid);
* :func:`assert_no_aged_unowned_flags` — a flag aged past the threshold MUST have an
  owner (a flag does not rot silently).

:func:`audit_coverage_instrument` runs all three over the live axes + a report.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

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


# The default age (days) past which an UNOWNED flag is a CI violation.
DEFAULT_FLAG_AGE_DAYS = 90


# Headline keys that would FUSE the two valid tiers into one covered total — forbidden,
# because a VALID_FLAGGED cell owes a research gap and must never be reported as plainly
# valid. The report's ``to_dict`` deliberately emits NONE of these.
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
    """FAIL if any axis is ``SCREENED_COLLAPSED`` without a ``screening_artifact``.

    The dataclass already raises at construction; this guard catches a status tampered
    AFTER construction (the defense-in-depth CI check) so a collapse can never reach the
    denominator without a linked artifact.
    """
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
    """FAIL if a report dict carries a headline that merges VALID + VALID_FLAGGED.

    Accepts either a :class:`CoverageReport` (its ``to_dict`` is taken) or a raw dict.
    A merged-tier key (``valid_total`` / ``covered_valid_total`` / …) is forbidden: the
    two valid tiers must always be reported separately so a flagged cell is never
    claimed as plainly valid.
    """
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
    """FAIL if any flag is older than ``max_age_days`` AND has no owner.

    A fresh unowned flag is fine (someone may pick it up); an OWNED aged flag is fine
    (it has a driver); but an UNOWNED flag aged past the threshold is rot — the guard
    fails so the flag gets an owner or gets resolved.
    """
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

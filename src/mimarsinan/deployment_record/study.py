"""Cross-platform comparison: one workload census, priced by many declared targets.

The question a chip designer actually asks — "what would this deployment cost on
THAT chip?" — is one census against many physics declarations. Because the pricer is
one pure function of (quantities, physics), the comparison is a fan-out over
profiles rather than a second cost model.

Two disciplines the artifact enforces rather than documents: an axis a target cannot
back is ABSENT with its reason (never zero, which on a minimized axis would read as
the best possible result), and every row discloses its evidence — measurement kind
and how many of its constants are estimated — so a comparison resting on projected
numbers says so in the table itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.report import ABSOLUTE_TERM_NAMES
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.serde import strict_kwargs, tuple_of


@dataclass(frozen=True)
class ComparisonRow:
    """One target's answer for the shared census, with its evidence."""

    profile: str
    display_name: str
    measurement_kind: str
    values: Mapping[str, float]
    bands: Mapping[str, Tuple[float, float]]
    unavailable: Mapping[str, str]
    evidence_counts: Mapping[str, int]

    @property
    def available_axes(self) -> Tuple[str, ...]:
        return tuple(axis for axis in ABSOLUTE_TERM_NAMES if axis in self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "display_name": self.display_name,
            "measurement_kind": self.measurement_kind,
            "values": dict(self.values),
            "bands": {k: list(v) for k, v in self.bands.items()},
            "unavailable": dict(self.unavailable),
            "evidence_counts": dict(self.evidence_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComparisonRow":
        kwargs = strict_kwargs(cls, data)
        kwargs["bands"] = {
            k: (float(v[0]), float(v[1])) for k, v in (kwargs["bands"] or {}).items()
        }
        return cls(**kwargs)


@dataclass(frozen=True)
class CrossPlatformComparison:
    """The whole table, plus what a reader must be told about how to read it."""

    rows: Tuple[ComparisonRow, ...]

    @property
    def mixes_measurement_kinds(self) -> bool:
        """Whether this table compares chips established on different bases.

        A silicon measurement beside a pre-silicon simulation is a legitimate
        comparison and an illegitimate one to report silently.
        """
        return len({row.measurement_kind for row in self.rows}) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": [row.to_dict() for row in self.rows]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CrossPlatformComparison":
        kwargs = strict_kwargs(cls, data)
        kwargs["rows"] = tuple_of(ComparisonRow.from_dict, kwargs["rows"])
        return cls(**kwargs)


def compare_platforms(
    census: Quantities, profiles: Sequence[str]
) -> CrossPlatformComparison:
    """Price one census with each named target."""
    if not profiles:
        raise ValueError(
            "a comparison needs at least one target profile; comparing nothing "
            "produces a table that says nothing"
        )
    rows = []
    for name in profiles:
        physics = get_platform_physics(name)
        pricing = price_absolute(census, physics)
        terms = {term.name: term for term in pricing.terms}
        evidence: Dict[str, int] = {}
        for value in physics.constants.values():
            evidence[value.evidence_kind] = evidence.get(value.evidence_kind, 0) + 1
        rows.append(ComparisonRow(
            profile=name,
            display_name=physics.display_name,
            measurement_kind=physics.validity.measurement_kind,
            values={
                axis: terms[axis].value
                for axis in ABSOLUTE_TERM_NAMES if axis in terms
            },
            bands=_bands_of(terms),
            unavailable={
                refusal.name: refusal.reason
                for refusal in pricing.refusals
                if refusal.name in ABSOLUTE_TERM_NAMES
            },
            evidence_counts=evidence,
        ))
    return CrossPlatformComparison(rows=tuple(rows))


def _bands_of(terms: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    """The banded axes' (low, high); a measured term carries no band by contract."""
    bands: Dict[str, Tuple[float, float]] = {}
    for axis in ABSOLUTE_TERM_NAMES:
        term = terms.get(axis)
        if term is None or term.band is None:
            continue
        bands[axis] = (float(term.band.low), float(term.band.high))
    return bands


def _cell(row: ComparisonRow, axis: str, width: int) -> str:
    if axis in row.values:
        return f"{row.values[axis]:>{width}.4g}"
    return f"{'—':>{width}}"


def render_comparison(comparison: CrossPlatformComparison) -> str:
    """The table as text, with the evidence disclosure a reader needs."""
    # The header carries the axes' OWN KEYS: they are the names a user selects in
    # the objective picker, and a prettified header would make them match by hand.
    width = max(len(axis) for axis in ABSOLUTE_TERM_NAMES) + 2
    lines = [
        f"{'target':<26}{'basis':<12}"
        + "".join(f"{axis:>{width}}" for axis in ABSOLUTE_TERM_NAMES),
        "-" * (38 + width * len(ABSOLUTE_TERM_NAMES)),
    ]
    for row in comparison.rows:
        lines.append(
            f"{row.profile:<26}{row.measurement_kind:<12}"
            + "".join(_cell(row, axis, width) for axis in ABSOLUTE_TERM_NAMES)
        )
    lines.append("")
    lines.append("evidence")
    for row in comparison.rows:
        counts = ", ".join(
            f"{kind} {count}" for kind, count in sorted(row.evidence_counts.items())
        )
        lines.append(f"  {row.profile:<24}{row.measurement_kind:<12}{counts}")
    unavailable = [
        (row.profile, axis, reason)
        for row in comparison.rows for axis, reason in sorted(row.unavailable.items())
    ]
    if unavailable:
        lines.append("")
        lines.append("axes no target could back (absent, never zero)")
        for profile, axis, reason in unavailable:
            lines.append(f"  {profile:<24}{axis:<26}{reason[:80]}")
    if comparison.mixes_measurement_kinds:
        lines.append("")
        lines.append(
            "NOTE: this table compares targets established on DIFFERENT bases "
            "(see the basis column); a silicon measurement and a pre-silicon "
            "simulation are not the same kind of claim."
        )
    return "\n".join(lines)

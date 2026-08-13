"""Run a reference case: price its census with its profile, compare to the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.correlation.case import ReferenceCase
from mimarsinan.deployment_record.correlation.derived import DERIVED_AXES
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


@dataclass(frozen=True)
class AxisCorrelation:
    """One published axis: what the model said, what the paper said, the gap."""

    name: str
    predicted: Optional[float]
    published: float
    band: Optional[Tuple[float, float]]
    refusal: Optional[str]

    @property
    def error_pct(self) -> float:
        """Signed relative error; an axis the physics refused is a total miss."""
        if self.predicted is None:
            return float("inf")
        return 100.0 * (self.predicted - self.published) / self.published

    @property
    def band_contains_published(self) -> bool:
        if self.band is None:
            return False
        return self.band[0] <= self.published <= self.band[1]


@dataclass(frozen=True)
class CaseCorrelation:
    """A whole case's verdict, with the evidence a reader needs to weigh it."""

    case: ReferenceCase
    measurement_kind: str
    axes: Tuple[AxisCorrelation, ...]
    overrides_applied: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return bool(self.axes) and all(
            abs(axis.error_pct) <= self.case.tolerance_pct for axis in self.axes
        )


def _with_overrides(
    physics: PlatformPhysics, overrides: Mapping[str, float], case: str
) -> PlatformPhysics:
    """The profile at this case's operating point, refusing unknown constants.

    An override may only retune a constant the profile already declares: inventing
    one here would smuggle an undeclared constant past the profile's evidence rules.
    """
    if not overrides:
        return physics
    payload = physics.to_dict()
    for key, value in overrides.items():
        if key not in payload["constants"]:
            raise KeyError(
                f"{case}: cannot override {key!r} — the {physics.name} profile does "
                f"not declare it, and a case may restate an operating point but "
                f"never introduce a constant"
            )
        declared = dict(payload["constants"][key])
        for corner in ("low", "nominal", "high"):
            if corner in declared:
                declared[corner] = value
        payload["constants"][key] = declared
    return PlatformPhysics.from_dict(payload)


def _axis(
    name: str,
    published: float,
    terms: Mapping[str, CostTerm],
    refusals: Mapping[str, str],
    case: str,
) -> AxisCorrelation:
    """One published axis, priced directly or derived from two priced terms."""
    if name in DERIVED_AXES:
        value, band, basis = DERIVED_AXES[name](terms)
        return AxisCorrelation(
            name=name, predicted=value, published=published,
            band=None if band is None else (band.low, band.high),
            refusal=None if value is not None else f"{basis} ({name} is derived)",
        )
    if name not in terms and name not in refusals:
        raise KeyError(
            f"{case}: {name!r} is neither a term this pricer produces nor a derived "
            f"axis; the priced terms are {sorted(terms)}, the derived axes are "
            f"{sorted(DERIVED_AXES)}"
        )
    term = terms.get(name)
    band = None
    if term is not None and term.band is not None:
        band = (term.band.low, term.band.high)
    return AxisCorrelation(
        name=name,
        predicted=None if term is None else term.value,
        published=published,
        band=band,
        refusal=refusals.get(name),
    )


def correlate(case: ReferenceCase) -> CaseCorrelation:
    """Price the case's census with its target and score every published axis."""
    physics = _with_overrides(
        get_platform_physics(case.profile), case.overrides, case.name
    )
    quantities = Quantities({
        key: QuantityValue(value, "measured") for key, value in case.census.items()
    })
    pricing = price_absolute(quantities, physics)
    terms = {term.name: term for term in pricing.terms}
    refusals = {refusal.name: refusal.reason for refusal in pricing.refusals}
    axes = [
        _axis(name, case.published[name].value, terms, refusals, case.name)
        for name in case.axes
    ]
    return CaseCorrelation(
        case=case,
        measurement_kind=physics.validity.measurement_kind,
        axes=tuple(axes),
        overrides_applied=dict(case.overrides),
    )


def worst_error(results: Sequence[CaseCorrelation]) -> Dict[str, float]:
    """The largest absolute error per profile — the number a reader remembers."""
    worst: Dict[str, float] = {}
    for result in results:
        for axis in result.axes:
            profile = result.case.profile
            worst[profile] = max(worst.get(profile, 0.0), abs(axis.error_pct))
    return worst

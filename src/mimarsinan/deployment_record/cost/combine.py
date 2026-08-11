"""Term construction and band algebra shared by every cost surface.

The three constructors are the epistemic gate: a measured value can only be
built without a band, a modeled value only with one, and a derived value says
which of the two it combined. The band algebra is corner-wise — legitimate
because every coefficient band in this model is monotone in its corner, so the
corners never cross (:class:`Band` re-validates the order anyway).
"""

from __future__ import annotations

from typing import Optional, Sequence

from mimarsinan.deployment_record.cost.coefficients import banded, corner_value
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.schema import Band


def measured(name: str, unit: str, value: float, source: str) -> CostTerm:
    """A record quantity passed through untouched — never banded."""
    return CostTerm(
        name=name, unit=unit, value=value, band=None, kind="measured", source=source
    )


def modeled(name: str, unit: str, band: Band, source: str) -> CostTerm:
    """A coefficient-applied quantity: the nominal, carrying its band."""
    return CostTerm(
        name=name, unit=unit, value=band.nominal, band=band, kind="modeled",
        source=source,
    )


def derived(
    name: str, unit: str, value: float, source: str, band: Optional[Band] = None
) -> CostTerm:
    """A quantity computed from record values (banded when a model term feeds it)."""
    return CostTerm(
        name=name, unit=unit, value=value, band=band, kind="derived", source=source
    )


def band_of(term: CostTerm) -> Band:
    """The term's band, loudly: only banded terms can be combined band-wise."""
    if term.band is None:
        raise ValueError(f"term {term.name!r} carries no band to combine")
    return term.band


def sum_bands(bands: Sequence[Band], *, basis: str) -> Band:
    """Corner-wise sum of monotone bands."""
    return banded(
        lambda corner: sum(corner_value(band, corner) for band in bands), basis=basis
    )


def flat_band(value: float, basis: str) -> Band:
    """A measured value as a degenerate band, so corner-wise sums stay uniform."""
    return Band(low=value, nominal=value, high=value, basis=basis)

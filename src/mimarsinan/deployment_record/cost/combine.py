"""Term construction and band algebra shared by every cost surface.

The three constructors are the epistemic gate: a measured value can only be
built without a band, a modeled value only with one, and a derived value says
which of the two it combined. The band algebra is corner-wise — legitimate
because every coefficient band in this model is monotone in its corner, so the
corners never cross (:class:`Band` re-validates the order anyway).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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


def host_ops_term(decomposition: Any) -> Optional[CostTerm]:
    """The per-pass host-op latency term, or ``None`` when nothing was timed.

    ``compute_sim_time_s`` is a per-sample census while the raw host wall covers
    the whole run, so only the per-pass normalization may join the sum; a timed
    wall with no invocation count raises rather than guess a divisor.
    """
    if decomposition.host_ops_s is None:
        return None
    per_pass = decomposition.host_ops_s_per_pass
    if per_pass is None:
        raise ValueError(
            "measured host-op walls carry no per pass normalization "
            "(timing.latency.host_ops_s_per_pass is None because some timed op "
            "has no invocation count); the whole-run total cannot join a "
            "per-sample decomposition and a guessed divisor would be a proxy "
            "presented as a measurement"
        )
    return measured(
        "host_ops_s", "s", float(per_pass), "timing.latency.host_ops_s_per_pass"
    )

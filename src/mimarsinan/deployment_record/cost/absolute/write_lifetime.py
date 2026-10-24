"""The write law's lifetime: endurance over the most-written cell's wear, in inferences and hours, and the lifetime
floor as a separate admissibility verdict (G3, CH-A1); and its join over a chip's write technologies (N2b): the
chip lives as long as its shortest-lived technology, and its verdict is the worst technology's.

``WritePricing`` (what the write law priced, refused and judged) lives here so that both the one-law pricer
(``writes.py``) and the join read one type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.context import AdmissibilityVerdict, PricingRefusal
from mimarsinan.deployment_record.cost.absolute.formulas import scale_band
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.write_law import CellWear, WritePolicy, cell_wear
from mimarsinan.deployment_record.schema.provenance import Band

LIFETIME_INFERENCES = "lifetime_inferences"
LIFETIME_HOURS = "lifetime_hours"
LIFETIME_VERDICT = "lifetime_admissible"
_S_PER_H = 3600.0
#: The evidence-class marker of a SECONDARY row (a published row read in a secondary source).
SECONDARY_MARKERS = ("SECONDARY", "secondary")


@dataclass
class WritePricing:
    """What the write law priced, refused and judged; the two bands join the headlines."""

    terms: List[CostTerm] = field(default_factory=list)
    refusals: List[PricingRefusal] = field(default_factory=list)
    verdicts: List[AdmissibilityVerdict] = field(default_factory=list)
    energy: Optional[Band] = None
    exposed: Optional[Band] = None

    def term(self, name: str, unit: str, band: Band) -> Band:
        self.terms.append(CostTerm(name=name, unit=unit, value=band.nominal, band=band,
                                   kind="modeled", source=band.basis))
        return band

    def refuse(self, name: str, reason: str) -> None:
        self.refusals.append(PricingRefusal(name=name, reason=reason))


def _life_band(endurance: Band, wear: CellWear, basis: str) -> Band:
    """endurance / wear corner-wise: the low lifetime at the high-wear end, the high at the low."""
    if wear.high is None or wear.exact:
        return scale_band(endurance, 1.0 / wear.low, basis)
    return Band(endurance.low / wear.high, endurance.nominal / wear.high, endurance.high / wear.low, basis)


def price_lifetime(out: WritePricing, census: Any, physics: PlatformPhysics, policy: WritePolicy,
                   most_written: Optional[int], producer_slots: Optional[Sequence[Tuple[int, int]]] = None) -> None:
    """The lifetime terms and the floor verdict of one census under one physics (G3's rule, moved from
    ``writes.py`` unchanged)."""
    wear, unknown = cell_wear(census, policy, most_written, producer_slots)
    floor = policy.min_lifetime_hours
    if wear is not None and wear.high == 0.0:
        if floor is not None:
            out.verdicts.append(AdmissibilityVerdict(LIFETIME_VERDICT, "admitted",
                                                     "no cell is written per inference", "no wear"))
        return
    if not physics.has("endurance_cycles"):
        unknown = "cells are written per inference but endurance_cycles is undeclared"
    if unknown is not None or wear is None:
        reason = unknown or "the most-written cell is unknown"
        out.refuse(LIFETIME_INFERENCES, reason)
        if floor is not None:
            out.refuse(LIFETIME_VERDICT, f"a lifetime floor is declared but the lifetime is refused: {reason}")
        return
    endurance = physics.constants["endurance_cycles"]
    life = out.term(LIFETIME_INFERENCES, "inferences", _life_band(
        endurance.band, wear, f"endurance_cycles / max_writes_per_cell_per_inference ({wear.basis})"))
    rate = policy.inference_rate_hz
    if rate is None:
        if floor is not None:
            out.refuse(LIFETIME_VERDICT, "min_lifetime_hours is declared without inference_rate_hz, "
                                         "so the lifetime has no hours to compare")
        return
    hours = out.term(LIFETIME_HOURS, "h", scale_band(life, 1.0 / (rate * _S_PER_H),
                                                     f"lifetime_inferences / inference_rate_hz ({wear.basis})"))
    if floor is None:
        return
    # Sound at both ends: inadmissible only when the longest lifetime fails the floor, admitted only
    # when the shortest clears it; a lower-bound-only wear can never admit.
    verdict = ("inadmissible" if hours.high < floor else "admitted" if hours.low >= floor else "straddles")
    one_sided = wear.high is None
    if one_sided and verdict != "inadmissible":
        out.refuse(LIFETIME_VERDICT, f"the lifetime is an upper bound ({wear.basis}), which can prove a program "
                                     "inadmissible but never admit one")
        return
    kind = " (upper bound)" if one_sided else "" if wear.exact else " (a wear band)"
    secondary = "; a secondary source" if any(m in endurance.note for m in SECONDARY_MARKERS) else ""
    out.verdicts.append(AdmissibilityVerdict(
        LIFETIME_VERDICT, verdict,
        f"lifetime {hours.low:.4g}-{hours.high:.4g} h{kind} against min_lifetime_hours {floor:g}",
        f"endurance_cycles [{endurance.basis}{secondary}]"))


def _bands(parts: Sequence[Tuple[str, WritePricing]], name: str) -> List[Tuple[str, str, Band]]:
    return [(label, term.unit, term.band) for label, priced in parts for term in priced.terms
            if term.name == name and term.band is not None]


def join_lifetimes(out: WritePricing, parts: Sequence[Tuple[str, WritePricing]], floor: Optional[float]) -> None:
    """The chip's lifetime over its write technologies (each priced by ``price_lifetime`` on its own share): the
    shortest-lived technology's, corner by corner; refused, by the technology's name, where any written technology's
    is. The floor verdict is the worst: inadmissible where any technology is, else refused where any refuses, else
    straddling where any straddles, else admitted."""
    refused = [f"{label}: {r.reason}" for label, priced in parts for r in priced.refusals
               if r.name == LIFETIME_INFERENCES]
    if refused:
        out.refuse(LIFETIME_INFERENCES, "; ".join(refused))
    else:
        for name in (LIFETIME_INFERENCES, LIFETIME_HOURS):
            bands = _bands(parts, name)
            if bands:
                out.term(name, bands[0][1], Band(
                    min(b.low for _, _, b in bands), min(b.nominal for _, _, b in bands),
                    min(b.high for _, _, b in bands),
                    "the shortest-lived write technology: " + "; ".join(f"{label}: {b.basis}"
                                                                       for label, _, b in bands)))
    if floor is None:
        return
    verdicts = [(label, v) for label, priced in parts for v in priced.verdicts if v.name == LIFETIME_VERDICT]
    unjudged = [f"{label}: {r.reason}" for label, priced in parts for r in priced.refusals
                if r.name == LIFETIME_VERDICT]
    worst = [(label, v) for label, v in verdicts if v.verdict == "inadmissible"]
    if not verdicts and not unjudged:
        return
    if not worst and unjudged:
        out.refuse(LIFETIME_VERDICT, "; ".join(unjudged))
        return
    verdict = ("inadmissible" if worst else
               "straddles" if any(v.verdict == "straddles" for _, v in verdicts) else "admitted")
    shown = worst or verdicts
    out.verdicts.append(AdmissibilityVerdict(LIFETIME_VERDICT, verdict,
                                             "; ".join(f"{label}: {v.reason}" for label, v in shown),
                                             "; ".join(f"{label}: {v.evidence}" for label, v in shown)))

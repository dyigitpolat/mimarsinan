"""[CL0] Energy with and without the host: every energy verdict read twice, a flip labelled host-bound.

The host is not the chip. A host-carrying energy headline is chip energy plus ``energy_host_mj`` (p_host over
the host wall), and where the host's share dominates, a verdict about the chip can be a verdict about the host.
``host_energy_split`` reads both energies off one priced report, end by end; ``dual_verdict`` evaluates any
verdict over a set of points on each energy, and the verdicts that differ are ``HOST_BOUND``.

Bands are corner-aligned: a band's low end prices every constant at its low end, so one shared constant (the one
host block of every priced point) sits at the same end of every band. ``corner_order`` compares two points end
by end under that reading, so a host energy two points share cancels in it; ``band_order`` is the interval rule,
which assumes nothing about shared constants.

[U-SH] The program without its host, and without its boundary too: one share rule for every plane (the scaling
ladder's candidate plane and the design point's record plane). ``price_shares`` prices a program's quantities in the
two share variants (``SHARE_VARIANTS``); ``shares`` reads a term's compute-only share (the dynamic energy's MAC
share) and boundary share (its carry share) off them, naming each absent one by the compute-absence rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Tuple, TypeVar

from mimarsinan.deployment_record.cost.absolute.energy import ENERGY_HEADLINE
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.host_terms import HOST_SIDE_QUANTITIES
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema.provenance import Band

ENERGY_HOST = "energy_host_mj"
HOST_BOUND = "host-bound"
CORNERS: Tuple[str, ...] = ("low", "nominal", "high")

K = TypeVar("K", bound=Hashable)


def _ends(band: Band) -> Tuple[float, float, float]:
    return band.low, band.nominal, band.high


@dataclass(frozen=True)
class HostEnergySplit:
    """One priced point's energy per inference with and without the host's share (mJ)."""

    with_host: Band
    without_host: Band
    #: None when the program carries no priced host energy (no host work).
    host: Optional[Band] = None

    def host_share(self) -> Tuple[float, float, float]:
        """The host's share of the headline at each end (low, nominal, high)."""
        if self.host is None:
            return 0.0, 0.0, 0.0
        low, nominal, high = (h / w if w else 0.0 for h, w in zip(_ends(self.host), _ends(self.with_host)))
        return low, nominal, high


def host_energy_split(terms: Iterable[CostTerm]) -> Optional[HostEnergySplit]:
    """Both energies of one priced report, or None when its energy headline is refused.

    Without the host is the headline less ``energy_host_mj`` at each end (the headline sums its components
    end by end). The chip energy can be a small difference of two large numbers when the host dominates, so
    the three ends are re-sorted: one rounding step must not unorder the band.
    """
    by_name = {term.name: term for term in terms}
    headline = by_name.get(ENERGY_HEADLINE)
    if headline is None or headline.band is None:
        return None
    host = by_name.get(ENERGY_HOST)
    if host is None or host.band is None:
        return HostEnergySplit(with_host=headline.band, without_host=headline.band)
    chip = sorted(w - h for w, h in zip(_ends(headline.band), _ends(host.band)))
    without = Band(chip[0], chip[1], chip[2], f"{ENERGY_HEADLINE} - {ENERGY_HOST}, end by end")
    return HostEnergySplit(with_host=headline.band, without_host=without, host=host.band)


def corner_order(a: Band, b: Band) -> str:
    """``below`` or ``above`` when a lies on that side of b at every end, ``equal`` when equal at every
    end, else ``straddles``."""
    signs = {(x > y) - (x < y) for x, y in zip(_ends(a), _ends(b))}
    if signs == {-1}:
        return "below"
    if signs == {1}:
        return "above"
    return "equal" if signs == {0} else "straddles"


def band_order(a: Band, b: Band) -> str:
    """The interval rule: ``below`` when a's high end lies under b's low end, ``above`` the converse,
    else ``overlaps``."""
    if a.high < b.low:
        return "below"
    if a.low > b.high:
        return "above"
    return "overlaps"


def lowest(bands: Mapping[K, Band], order: Callable[[Band, Band], str] = corner_order) -> Optional[K]:
    """The point that lies ``below`` every other under ``order``, or None when no point does."""
    for key, band in bands.items():
        if all(order(band, other) == "below" for name, other in bands.items() if name != key):
            return key
    return None


def cap_verdict(band: Band, cap: float) -> str:
    """``admitted`` when every end is at most ``cap``, ``refused`` when every end exceeds it, else
    ``straddles``."""
    if band.high <= cap:
        return "admitted"
    if band.low > cap:
        return "refused"
    return "straddles"


@dataclass(frozen=True)
class DualVerdict:
    """One verdict read on the energies with and without the host."""

    with_host: Hashable
    without_host: Hashable

    @property
    def host_bound(self) -> bool:
        return self.with_host != self.without_host

    @property
    def label(self) -> str:
        return HOST_BOUND if self.host_bound else ""

    def line(self, name: str) -> str:
        text = f"{name}: with host {self.with_host}; without host {self.without_host}"
        return f"{text}; {HOST_BOUND}" if self.host_bound else text


def dual_verdict(
    verdict: Callable[[Mapping[K, Band]], Hashable], splits: Mapping[K, HostEnergySplit],
) -> DualVerdict:
    """``verdict`` over the points' energies, once with and once without the host's."""
    return DualVerdict(
        with_host=verdict({key: split.with_host for key, split in splits.items()}),
        without_host=verdict({key: split.without_host for key, split in splits.items()}),
    )


#: [U-SH] The quantities a share variant drops, both with host_macs 0: compute-only (the MAC share) prices no carry,
#: programming, sync, op count or host-side quantity; no-host (the MAC and carry shares) no op count or host-side one.
CARRY_QUANTITIES = ("carried_raster_bytes", "carry_peak_live_bytes", "carry_out_bytes", "carry_in_bytes")
PROGRAMMING_QUANTITIES = ("reprogrammed_bytes", "reprogrammed_cores", "reprogram_passes")
SHARE_VARIANTS: Mapping[str, Tuple[str, ...]] = {
    "compute_only": (CARRY_QUANTITIES + PROGRAMMING_QUANTITIES + ("sync_count", "compute_op_count")
                     + HOST_SIDE_QUANTITIES),
    "no_host": ("compute_op_count",) + HOST_SIDE_QUANTITIES,
}


def price_shares(
    quantities: Quantities, physics: PlatformPhysics, *, compute: Any = None, readout_bits: Optional[int] = None,
    geometry: Any = None,
) -> Tuple[Dict[str, Dict[str, Band]], Dict[str, Dict[str, str]]]:
    """[U-SH] Each share variant of ``quantities`` (``SHARE_VARIANTS``) priced on ``physics`` with the plane's compute
    split, readout bits and geometry, as its headline is priced: (the priced bands by term, the refusals by name),
    each keyed by variant."""
    terms: Dict[str, Dict[str, Band]] = {}
    refusals: Dict[str, Dict[str, str]] = {}
    for name, drop in SHARE_VARIANTS.items():
        values = {k: quantities.get(k) for k in quantities.keys() if k not in drop}
        values["host_macs"] = QuantityValue(value=0.0, provenance="static")
        pricing = price_absolute(Quantities(values), physics, compute=compute, readout_bits=readout_bits,
                                 geometry=geometry)
        terms[name] = {t.name: t.band for t in pricing.terms if t.band is not None}
        refusals[name] = {r.name: r.reason for r in pricing.refusals}
    return terms, refusals


#: [U-CR C-2] The reason a row prices no write law where it ran no pricing: its target declares no physics profile.
NO_PHYSICS = "no physics declared"


def sub_corners(total: Band, part: Band, basis: str) -> Tuple[float, float, float]:
    """Corner-wise total - part, for a total that is part plus non-negative components (no re-sorting)."""
    out = []
    for a, b in zip(_ends(total), _ends(part)):
        d = a - b
        if d < 0.0:
            if -d > 1e-12 * max(abs(a), abs(b)):
                raise ValueError(f"{basis}: the total {a!r} is below its part {b!r}")
            d = 0.0
        out.append(d)
    return (out[0], out[1], out[2])


def share_absence(terms: Mapping[str, Any], refusals: Mapping[str, Mapping[str, str]], term: str, headline: str,
                  pricing: str = "compute_only") -> str:
    """[U-CR C-2] Why an axis's compute-only term is absent: the compute-only pricing's refusal of the headline (a
    pricing gate names every headline it suppresses) or of the term; 'no physics declared' only where no pricing ran
    (no profile: no pricing variant at all); a declared profile's gap no refusal names, by the term and the pricing's
    refusals. [U-SH] The same rule on the no-host pricing (``pricing``) names an absent boundary."""
    if pricing not in terms and pricing not in refusals:
        return NO_PHYSICS
    named = dict(refusals.get(pricing) or {})
    return named.get(headline) or named.get(term) or (
        f"{term} is not priced and no refusal names {headline}: the {pricing.replace('_', '-')} pricing refuses "
        f"{sorted(named) or 'nothing'}")


def shares(terms: Mapping[str, Mapping[str, Band]], refusals: Mapping[str, Mapping[str, str]], term: str,
           headline: str) -> Dict[str, Any]:
    """[U-SH] The one share rule of every plane, on the share pricings (``price_shares``): ``term``'s compute-only
    share and its boundary share (carry and sync: the no-host pricing less the compute-only, corner by corner), each
    absent one named by ``share_absence`` on the pricing that lacks it: ``absent`` where the compute-only share is
    (the boundary with it), ``boundary_absent`` where the boundary alone is."""
    compute, no_host = (terms.get(name, {}).get(term) for name in SHARE_VARIANTS)
    if compute is None:
        return {"compute_only": None, "boundary": None, "absent": share_absence(terms, refusals, term, headline)}
    if no_host is None:
        return {"compute_only": compute, "boundary": None, "absent": None,
                "boundary_absent": share_absence(terms, refusals, term, headline, "no_host")}
    return {"compute_only": compute, "boundary": sub_corners(no_host, compute, "boundary"), "absent": None}


def share_column(split: Mapping[str, Any]) -> Dict[str, Any]:
    """[U-SH] ``shares``'s reading in a row's column form, as the ladder's axis columns and the record plane print it
    (``absent_shares`` reads it back): the two terms as corner lists, and each absent one's name."""
    compute, boundary = split["compute_only"], split["boundary"]
    return {"terms": {"compute_only": None if compute is None else [compute.low, compute.nominal, compute.high],
                      "boundary": None if boundary is None else list(boundary)},
            **{key: value for key, value in split.items() if key.endswith("absent")}}


def absent_shares(column: Mapping[str, Any]) -> Dict[str, str]:
    """[U-SH] The name of each absent share of a column (``share_column``) by its term: ``absent`` names both where
    the compute-only term is absent, ``boundary_absent`` the boundary alone; a priced share names none."""
    terms = column.get("terms") or {}
    names = ({"compute_only": column.get("absent"), "boundary": column.get("absent")}
             if terms.get("compute_only") is None else
             {"boundary": column.get("boundary_absent")} if terms.get("boundary") is None else {})
    return {share: str(name) for share, name in names.items() if name is not None}

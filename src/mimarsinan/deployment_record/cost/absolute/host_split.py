"""[CL0] Energy with and without the host: every energy verdict read twice, a flip labelled host-bound.

The host is not the chip. A host-carrying energy headline is chip energy plus ``energy_host_mj`` (p_host over
the host wall), and where the host's share dominates, a verdict about the chip can be a verdict about the host.
``host_energy_split`` reads both energies off one priced report, end by end; ``dual_verdict`` evaluates any
verdict over a set of points on each energy, and the verdicts that differ are ``HOST_BOUND``.

Bands are corner-aligned: a band's low end prices every constant at its low end, so one shared constant (the one
host block of every priced point) sits at the same end of every band. ``corner_order`` compares two points end
by end under that reading, so a host energy two points share cancels in it; ``band_order`` is the interval rule,
which assumes nothing about shared constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Optional, Tuple, TypeVar

from mimarsinan.deployment_record.cost.absolute.energy import ENERGY_HEADLINE
from mimarsinan.deployment_record.cost.terms import CostTerm
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

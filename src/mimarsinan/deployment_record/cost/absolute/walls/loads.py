"""[U-ENERGY classes 1 and 2] The one load law.

Every load of a deployment is priced by ONE law over its census: the write law (``writes.price_write_inputs``, the
pricer of every write term, the connectivity entries of the cores its census writes included) at K = 1, serial,
producer writes left out (they are per inference), over

  - the every-core census for the deployment's first load: ``latency_load_s``, ``energy_load_mj``;
  - the period census for the reprogramming per round: ``latency_programming_s``, ``energy_programming_mj``;

plus what the write law leaves to the load: a host-owned store's transfer of the load's bytes (the link and
host-memory rows, each link row with the ends of the link it leaves unpriced), and the chip's static power over the
load's own wall, exactly once (``static_power.StaticSum``, a required argument), printed apart as well
(``energy_static_load_mj``, ``energy_static_programming_mj``). The host's idle power over a load's wall is named, never
priced (its whole-system line is per inference). Each pass's wall is priced once per program for both loads
(``write_walls.PassWalls``). Where the law refuses, the load's terms refuse by the law's reason; a plane that states
no pass structure has no load census, and its load terms refuse by that reason. [V-M1] The leg is the one leg form's
(``static_power.static_leg``): where a declared static part is refused, or the chip declares static power and the
load's wall is refused, the load's energy and its leg refuse with the leg's reason, the rule of every energy figure
over a wall (the static leg it cannot price is the load energy's dominant part wherever the wall prices). [m-2] What
the census reads in place of the plane's own totals (``LoadCensus.notes``) is named in the wall's and the energy's
basis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, programming_byte_energies, scale_band
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC, StaticSum, static_leg
from mimarsinan.deployment_record.cost.absolute.walls.write_walls import PassWalls, entry_bytes
from mimarsinan.deployment_record.cost.absolute.writes import (
    ENERGY_WRITE, LATENCY_WRITE, TRANSFER, WriteInputs, price_write_inputs)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import link_end
from mimarsinan.deployment_record.platform_physics.hosts.store import host_owned
from mimarsinan.deployment_record.quantities.planes.loads import LoadCensus, census_entries
from mimarsinan.deployment_record.schema.provenance import Band

_J_TO_MJ = 1e3
#: The first load's whole energy: the ladder's column key of a load's energy (``scaling_ladder_reading.load_column``).
LOAD_ENERGY = "energy_load_mj"
#: (census, wall term, energy term, its static leg's term) of every load figure, one law for each.
LOAD_FIGURES: Tuple[Tuple[str, str, str, str], ...] = (
    ("first", "latency_load_s", LOAD_ENERGY, "energy_static_load_mj"),
    ("period", "latency_programming_s", "energy_programming_mj", "energy_static_programming_mj"),
)
#: Why a pricing call that passes no load census prices no load.
NO_LOAD_CENSUS = ("the pricer was given no load census (the view states no pass structure), so no load is priced: "
                  "the load law prices a census, never a scalar")
_PER_LOAD = "; per program load, amortized separately from the per-inference headline"


@dataclass(frozen=True)
class LoadPricing:
    """One load priced by the one law: its wall, energy and static leg (or each refusal's reason), the parts, the
    names, and the load's payload and written cores."""

    wall: Optional[Band]
    energy: Optional[Band]
    write_energy: Optional[Band] = None
    static_leg: Optional[Band] = None
    wall_refused: str = ""
    energy_refused: str = ""
    leg_refused: str = ""
    named: Tuple[str, ...] = ()
    payload_bytes: Optional[int] = None
    programmed_cores: Optional[int] = None


def _refused(reason: str) -> LoadPricing:
    return LoadPricing(wall=None, energy=None, wall_refused=reason, energy_refused=reason, leg_refused=reason)


def _transfer(physics: Any, size: float, priced: Callable[[str], bool]) -> Tuple[List[Band], List[str], List[str]]:
    """A host-owned store's transfer of the load's bytes (``size``): (bands, evidence, named)."""
    bands: List[Band] = []
    evidence: List[str] = []
    named: List[str] = []
    for constant, label in programming_byte_energies(physics):
        if label not in TRANSFER:
            continue
        if priced(constant):
            bands.append(scale_band(physics.band(constant), size * _J_TO_MJ,
                                    f"{constant} x the load's bytes (host-owned store)"))
            evidence.append(f"store transfer ({label}): {constant}{link_end(physics, constant)}")
        else:
            named.append(f"{constant}({label}; the host-owned store's transfer of the load)")
    return bands, evidence, named


def price_load(census: Optional[LoadCensus], physics: Any, static: StaticSum, *, wall_name: str,
               priced: Optional[Callable[[str], bool]] = None, walls: Optional[PassWalls] = None) -> LoadPricing:
    """One load by the one law (module docstring), at every corner. ``static`` is required: the pricing's one
    resolved sum, or ``static_power.NO_STATIC`` stated by a caller that prices no static leg."""
    if census is None:
        return _refused(NO_LOAD_CENSUS)
    if census.writes is None or census.writes.census is None:
        return _refused(census.unavailable or (census.writes.unavailable if census.writes is not None else "")
                        or NO_LOAD_CENSUS)
    law = price_write_inputs(replace(WriteInputs.of(census.writes), entries_absent=census.entries_absent), physics,
                             lifetime=False, walls=walls)
    reasons = {r.name: r.reason for r in law.refusals}
    passes = census.writes.census.census.passes
    payload = int(sum(p.payload_bytes for p in passes))
    cores = int(sum(p.cores - p.resident_cores for p in passes))
    size = payload + entry_bytes(census_entries(census.writes.census.census), physics)
    extra, evidence, named = (_transfer(physics, size, priced or physics.has) if host_owned(physics) and size
                              else ([], [], []))
    if physics.has("p_host_idle"):  # [U-ENERGY T-E1d] the host waits through the load: named, never priced
        named.append(f"p_host_idle(the host's idle power over {wall_name}; not priced: the whole-system line is "
                     "per inference)")
    wall = law.exposed
    wall_refused = "" if wall is not None else reasons.get(LATENCY_WRITE, "the write law prices no write time")
    if wall is not None and census.notes:  # [m-2] what the census read in place of the plane's totals, named
        wall = Band(wall.low, wall.nominal, wall.high, f"{wall.basis}; {'; '.join(census.notes)}")
    form = static_leg(static, wall, wall_name, wall_refused)  # [V-M1] the one leg form
    leg = form.band
    leg_refused = form.refused or ("" if leg is not None else static.leg_refusal(wall_name))
    write = law.energy
    if write is None or form.refused:
        # the law's own refusal, or a static leg the pricing refuses (a declared part, or the wall under a priced
        # sum), which would be the energy's dominant part
        refused = form.refused if write is not None else reasons.get(ENERGY_WRITE,
                                                                     "the write law prices no write energy")
        return LoadPricing(wall=wall, energy=None, write_energy=write, static_leg=leg, wall_refused=wall_refused,
                           energy_refused=refused, leg_refused=leg_refused, payload_bytes=payload,
                           programmed_cores=cores)
    leading = [] if leg is None else [
        f"static over {wall_name}: {form.power_basis} x {wall_name}"]  # the leg first: it leads where it prices
    named.extend(form.named)
    source = "; ".join(leading + [f"write law (K = 1, serial): {write.basis}"] + evidence + list(census.notes)) + (
        _PER_LOAD + (f"; unpriced: {', '.join(named)}" if named else ""))
    return LoadPricing(wall=wall, energy=add_bands([write] + extra + ([] if leg is None else [leg]), source),
                       write_energy=write, static_leg=leg, wall_refused=wall_refused, leg_refused=leg_refused,
                       named=tuple(named), payload_bytes=payload, programmed_cores=cores)


def _put(ctx: Any, name: str, unit: str, band: Optional[Band], refused: str) -> None:
    if band is not None:
        ctx.term(name, unit, band, band.basis)
    else:
        ctx.refuse(name, refused)


def price_loads(ctx: Any, loads: Optional[Any]) -> Dict[str, LoadPricing]:
    """Every load figure of ``LOAD_FIGURES`` on a pricing context, each a term or a refusal by name; the two loads
    share one ``PassWalls``, so each pass's wall is priced once for both."""
    out: Dict[str, LoadPricing] = {}
    walls = PassWalls()
    for kind, wall_name, energy_name, leg_name in LOAD_FIGURES:
        census = None if loads is None else getattr(loads, kind)
        load = price_load(census, ctx.physics, ctx.static, wall_name=wall_name, priced=ctx.priced, walls=walls)
        _put(ctx, wall_name, "s", load.wall, load.wall_refused)
        _put(ctx, energy_name, "mJ", load.energy, load.energy_refused)
        _put(ctx, leg_name, "mJ", load.static_leg, load.leg_refused)
        out[kind] = load
    return out


def load_refusals(loads: Optional[Any], physics: Any) -> List[Tuple[str, str]]:
    """Every load figure refused where the description's external gate refuses the pricing: each by the law's own
    gate (or by the absence of its census), its static leg with its energy."""
    out: List[Tuple[str, str]] = []
    for kind, wall_name, energy_name, leg_name in LOAD_FIGURES:
        load = price_load(None if loads is None else getattr(loads, kind), physics, NO_STATIC, wall_name=wall_name)
        out += [(wall_name, load.wall_refused), (energy_name, load.energy_refused), (leg_name, load.energy_refused)]
    return out

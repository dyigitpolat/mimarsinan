"""Lifetime and the minimum batch K* of an executed program, read off the write law's own verdicts.

Lifetime at a batch K is G3's write pricer (``price_write_inputs``: per write technology where the core types
declare several, N2b) on the executed program's period census at K. K*
per endurance corner is the smallest K whose priced lifetime at that corner reaches ``min_lifetime_hours``, found on
the pricer itself (doubling, then bisection; the wear M / K + p falls with K), so the boundary is the toolchain's
verdict, never a closed form read at one point: below K* the corner is inadmissible and from K* on it is admitted,
both priced. A program whose operand cells are written every inference (p >= 1) may admit no K at a corner; that is
printed by name. The closed form K = M r T / E (static writes only) and the two-ceiling minimum batch
(``min_batch``) ride beside where the census states them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute.writes import (
    LIFETIME_HOURS, LIFETIME_VERDICT, WriteInputs, WritePricing, price_write_inputs,
)
from mimarsinan.deployment_record.cost.write_lifetime import price_lifetime
from mimarsinan.deployment_record.design_point.wear import binding
from mimarsinan.deployment_record.platform_physics.resolve import technology_physics
from mimarsinan.deployment_record.design_point.timeline import ProgramWrites
from mimarsinan.deployment_record.introspection.scaling_ladder_reading import min_batch
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.deployment_record.quantities.write_law import cell_wear, write_policy_of
from mimarsinan.deployment_record.quantities.write_technology import TechnologySplit
from mimarsinan.deployment_record.schema.provenance import Band

#: Endurance band corners in the pricer's band order (low endurance gives the shortest lifetime).
CORNERS = ("low", "nominal", "high")
#: The largest batch the K* search prices before it states that no K admits the corner.
K_SEARCH_MAX = 2 ** 40


class EnduranceUnavailable(ToolchainRefusal, ValueError):
    """The executed program states no period census, so no batch can be priced."""


def priced_at(writes: ProgramWrites, physics: PlatformPhysics, platform: Mapping[str, Any], k: int) -> WritePricing:
    """The write law on the executed census at batch ``k`` (the platform's other declarations unchanged): [E-6,
    targeted pass] the plane's one entry (``price_write_inputs``), so a chip of several write technologies prices
    each technology's share under its own physics and joins them, and the operand regions' placement
    (``producer_slots``) states the most-written cell's wear exactly, as on the record plane."""
    census = writes.census
    if census is None:
        raise EnduranceUnavailable(writes.unavailable or "the program states no period census")
    policy = write_policy_of({**platform, "inference_batch_K": int(k)})
    assert policy is not None
    return price_write_inputs(WriteInputs(
        census=census.census, policy=policy, written_extents=census.written_extents,
        compute_sealed=census.compute_sealed, most_written=census.most_written_core,
        # a census that places no operand core states no slot: its producer cells bound the wear from below only
        pass_core_types=census.pass_core_types, producer_slots=census.producer_slots or None,
        by_technology=split_at(writes.by_technology, int(k))), physics)


def split_at(split: Optional[TechnologySplit], k: int) -> Optional[TechnologySplit]:
    """The census split by write technology at batch ``k``: each technology's policy at ``k`` (the split's census
    is the period's, which no batch changes)."""
    if split is None:
        return None
    return replace(split, parts=tuple(replace(part, policy=replace(part.policy, batch_k=int(k))) for part in split.parts))


def _band(pricing: WritePricing, name: str) -> Optional[Band]:
    return next((t.band for t in pricing.terms if t.name == name and t.band is not None), None)


def _triple(band: Optional[Band]) -> Optional[List[float]]:
    return None if band is None else [band.low, band.nominal, band.high]


def _verdict(pricing: WritePricing) -> Any:
    return next((v for v in pricing.verdicts if v.name == LIFETIME_VERDICT), None)


def _corner_states(pricing: WritePricing, floor: float) -> Dict[str, str]:
    """Per endurance corner, admitted or inadmissible at this batch; every corner admitted where no cell wears;
    empty where the lifetime is refused."""
    hours = _band(pricing, LIFETIME_HOURS)
    if hours is not None:
        values = (hours.low, hours.nominal, hours.high)
        return {c: "admitted" if h >= floor else "inadmissible" for c, h in zip(CORNERS, values)}
    verdict = _verdict(pricing)
    if verdict is not None and verdict.verdict == "admitted":
        return {c: "admitted" for c in CORNERS}
    return {}


def endurance_of(physics: PlatformPhysics) -> Optional[Dict[str, Any]]:
    """The declared endurance band with its evidence, or None when undeclared; [U-C C-1] with the technology's pulses
    per write where it declares them (the wear of one write in endurance cycles)."""
    if not physics.has("endurance_cycles"):
        return None
    constant = physics.constants["endurance_cycles"]
    pulses = physics.constants.get("pulses_per_write")
    return {"cycles": [constant.low, constant.nominal, constant.high], "evidence_kind": constant.evidence_kind,
            "source": "; ".join(s for s in (constant.citation, constant.note) if s),
            **({} if pulses is None else {"pulses_per_write": [pulses.low, pulses.nominal, pulses.high]})}


def lifetime_at(writes: ProgramWrites, physics: PlatformPhysics, platform: Mapping[str, Any], k: int
                ) -> Dict[str, Any]:
    """The priced lifetime, its verdict against the floor and the pricer's refusals at batch ``k``."""
    pricing = priced_at(writes, physics, platform, k)
    verdict = _verdict(pricing)
    return {"at_K": int(k), "hours": _triple(_band(pricing, LIFETIME_HOURS)),
            "verdict": None if verdict is None else verdict.verdict,
            "reason": "" if verdict is None else verdict.reason, "endurance": endurance_of(physics),
            "refusals": [{"name": r.name, "reason": r.reason} for r in pricing.refusals]}


def _smallest_admitting(writes, physics, platform, corner: str, floor: float) -> Tuple[Optional[int], str]:
    """(the smallest admitting batch at ``corner``, or None with why)."""
    def state(k: int) -> Optional[str]:
        return _corner_states(priced_at(writes, physics, platform, k), floor).get(corner)

    first = state(1)
    if first is None:
        return None, "the write law refuses the lifetime"
    if first == "admitted":
        return 1, ""
    high = 2
    while state(high) != "admitted":
        if high >= K_SEARCH_MAX:
            return None, f"no batch up to {K_SEARCH_MAX} admits"
        high *= 2
    low = high // 2  # inadmissible at low, admitted at high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if state(middle) == "admitted" else (middle, high)
    return high, ""


def k_star(writes: ProgramWrites, physics: PlatformPhysics, platform: Mapping[str, Any], k_grid: Sequence[int], *,
           x_fluid: Optional[float] = None, x_packed: Optional[float] = None) -> Dict[str, Any]:
    """K* per endurance corner on the pricer, the priced verdicts at the grid and at each boundary, and the closed
    forms beside."""
    floor = float(platform["min_lifetime_hours"])
    rate = float(platform["inference_rate_hz"])
    census = writes.census
    assert census is not None
    producer = any(p.written_cells_producer for p in census.census.passes)
    found = {c: _smallest_admitting(writes, physics, platform, c, floor) for c in CORNERS}
    per_corner = {c: value for c, (value, _why) in found.items()}
    ks = set(int(k) for k in k_grid)
    for value in per_corner.values():
        if value is not None:
            ks.update(k for k in (value - 1, value) if k >= 1)
    verdicts = []
    for k in sorted(ks):
        pricing = priced_at(writes, physics, platform, k)
        hours = _band(pricing, LIFETIME_HOURS)
        verdict = _verdict(pricing)
        verdicts.append({"K": k, "verdict": "refused" if verdict is None else verdict.verdict,
                         "hours": _triple(hours), "per_corner": _corner_states(pricing, floor)})
    checked = all(
        next(v for v in verdicts if v["K"] == value)["per_corner"].get(c) == "admitted" and (
            value == 1 or next(v for v in verdicts if v["K"] == value - 1)["per_corner"].get(c) == "inadmissible")
        for c, value in per_corner.items() if value is not None)
    none = [c for c, v in per_corner.items() if v is None]
    most = census.most_written_core
    shares = "" if writes.by_technology is None or not writes.by_technology.parts else (
        "; per write technology (each under its own endurance, the row's lifetime the shortest): " + ", ".join(
            f"{part.label}: M = {part.most_written}" for part in writes.by_technology.parts))
    out: Dict[str, Any] = {
        "per_corner": per_corner, "checked": checked, "verdicts_by_K": verdicts,
        "basis": (f"K* per endurance corner: the smallest batch whose lifetime priced by price_write_terms on the "
                  f"executed period census reaches {floor:g} h at {rate:g} inferences/s; the wear is "
                  f"M / K{' + operand writes per inference' if producer else ''} with M = {most} writes of the "
                  f"most-written core per period{shares}"),
        "boundary": "; ".join(
            (f"{c} endurance: admitted from K = 1" if v == 1 else
             f"{c} endurance: inadmissible below K = {v}, admitted from K = {v}") if v is not None
            else f"{c} endurance: {found[c][1]}" for c, v in per_corner.items()),
        "no_K_admits": None if not none else (
            f"no batch admits the {', '.join(none)} endurance corner ({found[none[0]][1]})"
            + (": operand cells are written every inference, which no batch amortizes" if producer else "")),
    }
    if not producer and most is not None and x_fluid is not None and x_packed is not None:
        declared = write_policy_of(platform)
        exposed = _band(priced_at(writes, physics, platform, 1), "latency_write_exposed_s")
        endurance = endurance_of(physics)
        out["min_batch"] = min_batch(
            x_fluid=x_fluid, x_packed=x_packed, max_writes=float(most), rate_hz=rate, lifetime_hours=floor,
            endurance=None if endurance is None else endurance["cycles"],
            pulses_per_write=None if endurance is None else endurance.get("pulses_per_write"),
            write_time_per_period=_triple(exposed),
            write_time_source="the write law's exposed write time at K = 1 (serial: the write time of one period)",
            k=None if declared is None else int(declared.batch_k))
    return out


def technology_wear(writes: ProgramWrites, physics: PlatformPhysics, k: int,
                    roles: Mapping[str, str]) -> Optional[Dict[str, Any]]:
    """[E-6, targeted pass] The per-core-type blocks of a row of several write technologies at batch ``k``: per
    technology's share of the executed census (``ProgramWrites.by_technology``), its role (``roles``: write physics
    name to the row's role; else operand where the share's core types hold operand regions, static otherwise), its
    static cells per period and operand cells per inference ([U-CR C-3] each written weight's cells), its most-written
    cell's wear (``cell_wear``, the slot rule), its endurance and the lifetime the write law prices under its own
    physics (``price_lifetime`` on the share, the rule ``price_write_inputs`` joins), with the binding type named. None
    where the census is not split (one technology), or the split refuses (the write terms then refuse by its reason)."""
    split = split_at(writes.by_technology, k)
    if split is None or split.refused is not None or not split.parts:
        return None
    entries_writes: List[Dict[str, Any]] = []
    entries_life: List[Dict[str, Any]] = []
    for part in split.parts:
        name = part.technology.write_physics
        role = roles.get(str(name)) or ("operand" if any(t.operand for t in part.group) else "static")
        own_physics = technology_physics(physics, part.technology)
        wear, _unknown = cell_wear(part.census, part.policy, part.most_written, part.producer_slots)
        per_cell = None if wear is None or wear.high is None else float(wear.high)
        own = WritePricing()
        price_lifetime(own, part.census, own_physics, part.policy, part.most_written, part.producer_slots)
        verdict = next((v.verdict for v in own.verdicts if v.name == LIFETIME_VERDICT), None)
        cells = int(part.policy.cells_per_weight)  # each written weight's cells, as the row's summary counts them
        static = int(sum(p.written_cells_backing_store for p in part.census.passes)) * cells
        operand = int(sum(p.written_cells_producer for p in part.census.passes)) * cells
        entries_writes.append({
            "role": role, "write_technology": name, "static_written_cells_per_period": static,
            "operand_written_cells_per_inference": operand, "max_writes_per_cell_per_inference": per_cell})
        entries_life.append({"role": role, "write_technology": name, "max_writes_per_cell_per_inference": per_cell,
                             "endurance": endurance_of(own_physics), "hours": _triple(_band(own, LIFETIME_HOURS)),
                             "verdict": verdict})
    return {"writes": entries_writes, "lifetime": entries_life, "binding": binding(entries_life)}

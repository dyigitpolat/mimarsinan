"""[U-J] The host's terms beyond its compute wall, read by the absolute pricer (``cost/absolute``).

An edge host block (``platform_physics/hosts``) declares what a server's block never priced, and each term prices
only where the resolved physics declares its constant, so on every other physics nothing here prices:

- the host-chip link and host memory: every byte that crosses the boundary per inference, in either direction (the
  pass carry ``carry_in_bytes`` / ``carry_out_bytes``, the segment I/O ``segment_in_bytes`` / ``segment_out_bytes``
  and the write law's payload from the host, ``write_link_bytes``), at ``e_link_per_byte`` plus
  ``e_host_mem_per_byte`` (the byte is written to or read from host memory once), and in time at the slower of
  ``t_link_per_byte`` and ``t_host_mem_per_byte`` per byte (the pipelined bound of a two-stage transfer) plus
  ``t_link_transfer`` per transfer (``host_link_transfers``); the write payload's time is the write law's, at the
  programming channel the run declares (``hosts.description.host_attach_block`` composes the link into it);
- the memory-bound host operators: their bytes (``host_memory_bound_bytes``) at ``t_host_mem_per_byte``, part of the
  host's own time (their energy is inside ``p_host`` over that time);
- the host's idle power: ``p_host_idle`` over the inference's latency less the host's own time (the host waits on
  the chip and on the link), beside ``p_host`` over the host's time.

[U-J2] Whose energy each term is, the host's block states per energy row (``hosts.description.ENERGY_PLANE_FIELD``,
carried as ``PlatformPhysics.system_energy_rows``): attributable rows price the deployment's work and join the
headline as ``energy_host_mj``; system-level rows price the host's own draw, printed only in the whole-system line
``energy_system_mj`` beside the headline, never in it. A system-level idle power is a floor drawn in every state: the
host's work is charged above it (``energy_host_work_mj``, (p_host - p_host_idle) x host time) and the floor under the
work (``energy_host_floor_mj``) and over the wait (``energy_host_idle_mj``) are system-level. The wait is the
latency's parts the host does not execute, summed directly (``PricingContext.host_wait``). The link and host memory
are each their own term (``energy_host_link_mj``, ``energy_host_mem_mj``).
[U-J2 targeted M1] The work's power is a difference of two bands, so it takes the cost ends of the difference
(``hosts.description.power_above``: the high end the power's high less the floor's low, the low end never below
zero); the system line keeps the whole host at its own corners: ``energy_host_system_mj`` is the whole host (p_host
over the host's time, the floor inside it, p_host_idle over the wait, the link and host memory), and
``energy_system_mj`` the headline's chip side (``PricingContext.energy_chip_side``) plus it, composed from parts at
every end. At the nominal (one setup) the whole host is the attributable host plus the floor and the idle.
[U-J3] The store traffic (the pass carry, the write law's backing-store payload) crosses the link only to a host-owned
store (``platform_physics.hosts.store``); on a chip-owned one the chip's rows price it; beside the link a chip that
states no store refuses its store traffic by name (``store_refusal``), as a host-owned store without the link does.
[U-J4] A crossing byte pays the link and host memory here and none of the chip's store rows (``hosts.store``'s rule);
a row of the two the physics does not declare beside the link is named unpriced on the headline; the link's energy
names the chip's own end of the link unpriced (``link_end``, a boundary), and the latency names a host-sourced
payload's link time unpriced (no term times its transfer).

A part whose quantity the view cannot size is named unpriced with the reason, never dropped."""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.hosts.description import power_above
from mimarsinan.deployment_record.platform_physics.hosts.store import (  # [U-J3] the link's rows and the store's rule
    CHIP, HOST, LINK_ROWS as LINK_ROWS, PAYLOAD_TIME_UNPRICED, chip_store_table as chip_store_table,
    link_end as link_end, owner, prices_host_link as prices_host_link, slower_port as slower_port,
    store_refusal as store_refusal, unpriced_link_row,
)
from mimarsinan.deployment_record.schema.provenance import Band

#: Every byte that crosses the host-chip boundary per inference, both directions: the program's own (the segment I/O)
#: and [U-J3] the store traffic (the pass carry), which crosses to a store the chip does not own (``link_bytes``).
PROGRAM_BYTES = (("segment_in_bytes", "segment I/O"), ("segment_out_bytes", "segment I/O"))
STORE_BYTES = (("carry_in_bytes", "pass carry"), ("carry_out_bytes", "pass carry"))
MEMORY_BOUND, TRANSFERS = "host_memory_bound_bytes", "host_link_transfers"
#: The quantities the host-side census adds: a host-free or compute-only variant drops them with the host.
HOST_SIDE_QUANTITIES = ("segment_in_bytes", "segment_out_bytes", TRANSFERS, MEMORY_BOUND)
ENERGY_IDLE, ENERGY_LINK, LATENCY_LINK = "energy_host_idle_mj", "energy_host_link_mj", "latency_host_link_s"
#: [U-J2] The host's work above its floor, the floor under the work, host memory per crossing byte; the whole
#: host at its corners (the system plane) and the whole-system line beside the headline.
ENERGY_WORK, ENERGY_FLOOR, ENERGY_MEM = "energy_host_work_mj", "energy_host_floor_mj", "energy_host_mem_mj"
ENERGY_HOST_SYSTEM, ENERGY_SYSTEM = "energy_host_system_mj", "energy_system_mj"
_HEADLINE = "energy_per_inference_mj"
#: The host's breakdown of the headlines it joins (its time and link; its work, link, memory and system energy).
HOST_DECOMPOSITION = ("latency_host_s", LATENCY_LINK, ENERGY_WORK, ENERGY_LINK, ENERGY_MEM, ENERGY_FLOOR, ENERGY_IDLE,
                      ENERGY_HOST_SYSTEM, ENERGY_SYSTEM)
_J_TO_MJ = 1e3
_NO_CENSUS = "the view states no host-chip segment census"


def _ends(fn, *bands: Band) -> Tuple[float, float, float]:
    """``fn`` of the bands' values at each end (corner-aligned, as every band of one physics is)."""
    low, nominal, high = (fn(*(getattr(b, end) for b in bands)) for end in ("low", "nominal", "high"))
    return low, nominal, high


def scale_band(band: Band, factor: float, basis: str) -> Band:
    """``band`` x a non-negative quantity (``cost.absolute.formulas.scale_band``'s rule; this module sits below the
    pricer, which imports it, so it keeps its own two band operations)."""
    return Band(*_ends(lambda v: v * factor, band), basis)


def add_bands(bands: List[Band], basis: str) -> Band:
    return Band(*_ends(lambda *v: sum(v), *bands), basis)


def link_bytes(ctx: Any, writes: bool = True) -> Tuple[float, List[str]]:
    """The bytes that cross the boundary (both directions) and, per kind the view cannot size, its reason. [U-J3] The
    pass carry and the write law's store payload only to a store the chip does not own; the program's own always; the
    write law's bytes (``writes``) for the energy only, their time being the write law's."""
    total, unknown, routed = 0.0, {}, owner(ctx.physics) != CHIP
    for key, kind in PROGRAM_BYTES + (STORE_BYTES if routed else ()):
        if ctx.quantities.has(key):
            total += ctx.quantities.get(key).value
        else:
            unknown.setdefault(kind, ctx.quantities.withheld_reason(key) or _NO_CENSUS)
    if writes and (crossing := getattr(ctx, "write_link_bytes", None)) is not None:
        total += crossing.program + (crossing.store if routed else 0.0)
    return total, [f"{kind}; {reason}" for kind, reason in unknown.items()]


def _declared(physics: Any, keys: Tuple[str, ...]) -> List[str]:
    return [key for key in keys if physics.has(key)]


def host_memory_time(ctx: Any) -> Tuple[Optional[Band], List[str]]:
    """The MAC-free host operators' bytes at ``t_host_mem_per_byte`` (a part of the host's time), or its note."""
    if not ctx.physics.has("t_host_mem_per_byte"):
        return None, []
    q = ctx.quantities
    if q.has(MEMORY_BOUND):
        size = q.get(MEMORY_BOUND).value
        return (scale_band(ctx.physics.band("t_host_mem_per_byte"), size,
                           "t_host_mem_per_byte x host_memory_bound_bytes") if size else None), []
    if q.has("compute_op_count") and not q.get("compute_op_count").value:
        return None, []  # no host operator
    return None, [f"t_host_mem_per_byte(memory-bound host ops; {q.withheld_reason(MEMORY_BOUND) or _NO_CENSUS})"]


def host_link_time(ctx: Any) -> Tuple[Optional[Band], List[str]]:
    """The link's time per inference: every crossing byte at the slower stage, one latency per transfer."""
    physics, q = ctx.physics, ctx.quantities
    if not prices_host_link(physics):
        return None, []
    parts: List[Band] = []
    unpriced: List[str] = []
    stages = _declared(physics, ("t_link_per_byte", "t_host_mem_per_byte"))
    if stages:
        total, unknown = link_bytes(ctx, writes=False)
        bands = [physics.band(key) for key in stages]
        per = Band(*_ends(max, *bands), f"the slower of {' and '.join(stages)} per byte")
        if total:
            parts.append(scale_band(per, total, f"{per.basis} x host_link_bytes"))
        unpriced += [f"{stages[0]}({item})" for item in unknown]
    if (crossing := getattr(ctx, "write_link_bytes", None)) is not None and crossing.program:  # [U-J4] named, not timed
        unpriced.append(PAYLOAD_TIME_UNPRICED)
    if physics.has("t_link_transfer"):
        if q.has(TRANSFERS):
            if q.get(TRANSFERS).value:
                parts.append(scale_band(physics.band("t_link_transfer"), q.get(TRANSFERS).value,
                                        "t_link_transfer x host_link_transfers"))
        else:
            unpriced.append(f"t_link_transfer(host-chip transfers; {q.withheld_reason(TRANSFERS) or _NO_CENSUS})")
    if not parts:
        return None, unpriced
    return (parts[0] if len(parts) == 1 else add_bands(parts, " + ".join(p.basis for p in parts))), unpriced


class WriteCrossing(NamedTuple):
    """[U-J3] The write law's bytes per inference toward the chip: the store's (the period's backing-store payload over
    K) and the program's own (host-sourced producer payloads, which cross the link on either store)."""

    store: float
    program: float


def write_link_bytes(writes: Any) -> Optional[WriteCrossing]:
    """The write law's bytes per inference that may cross the host-chip link (``cost.absolute.writes``' transfer
    terms); None without a period census."""
    census = getattr(writes, "census", None)
    if census is None:
        return None
    policy, bits = writes.policy, writes.policy.payload_bits
    producer = sum(p.written_cells_producer for p in census.passes)
    host = producer and policy.producer_write_source == "host" and bits is not None
    return WriteCrossing(float(sum(p.payload_bytes for p in census.passes) / int(policy.batch_k)),
                         float(producer * int(bits) / 8.0) if host and bits is not None else 0.0)


def _times(power: Band, time: Band, basis: str) -> Band:
    return Band(*_ends(lambda p, t: p * t * _J_TO_MJ, power, time), basis)


def system_rows(physics: Any) -> Tuple[str, ...]:
    """The host energy rows ``physics`` states system-level (``PlatformPhysics.system_energy_rows``)."""
    return tuple(getattr(physics, "system_energy_rows", ()) or ())


class _Planes:
    """The host's energy terms and unpriced parts, each on the plane its row states: attributable (the headline's
    ``energy_host_mj``) or system-level; and [targeted M1] the whole host's parts at its corners (``whole``: every
    term, the attributable power whole over its time in place of the work above the floor and the floor under it)."""

    def __init__(self, system: Tuple[str, ...]) -> None:
        self.system = set(system)
        self.parts: Dict[str, List[Band]] = {"main": [], "system": []}
        self.unpriced: Dict[str, List[str]] = {"main": [], "system": []}
        self.whole: List[Band] = []

    def plane(self, row: str) -> str:
        return "system" if row in self.system else "main"

    def add(self, ctx: Any, row: str, name: str, band: Band, whole: bool = True) -> None:
        ctx.term(name, "mJ", band, band.basis)
        self.parts[self.plane(row)].append(band)
        if whole:
            self.whole.append(band)


def host_side_energy(ctx: Any, active: Optional[Band], busy: Optional[Band],
                     e2e: Optional[Band]) -> Tuple[Optional[Band], List[str], Optional[str]]:
    """(the host's attributable energy, its unpriced parts, a refusal by name): ``active`` (p_host over the host's
    time) as before off an edge host; on one, each term on the plane its row states (module docstring), the
    system-level ones summed as ``energy_host_system_mj`` (or refused by name where the wait is not priced)."""
    physics = ctx.physics
    if refused := store_refusal(ctx):  # [U-J3] store traffic with no store to price it on, refused by name
        return None, [], refused
    if not (physics.has("p_host_idle") or prices_host_link(physics)):
        return active, [], None
    planes = _Planes(system_rows(physics))
    if active is not None and busy is not None:
        work = active
        if "p_host_idle" in planes.system:  # a system-level idle floor: the work is charged above it
            floor = physics.band("p_host_idle")
            power, why = power_above(physics.band("p_host"), floor,
                                     "p_host - p_host_idle (above the idle floor, the cost ends of the difference)")
            if power is None:
                return None, [], f"p_host above its system-level floor p_host_idle {why}"
            work = _times(power, busy, f"(p_host - p_host_idle) [{physics.band('p_host').basis}; the cost ends of "
                                       "the difference] x host time")
            planes.add(ctx, "p_host_idle", ENERGY_FLOOR,
                       _times(floor, busy, "p_host_idle x host time (the idle floor under the host's work)"), whole=False)
            planes.whole.append(active)  # the whole host draws p_host over its time, the floor inside it
        planes.add(ctx, "p_host", ENERGY_WORK, work, whole=work is active)
    if physics.has("p_host_idle"):
        wait = getattr(ctx, "host_wait", None) if e2e is not None else None
        if wait is None:
            planes.unpriced[planes.plane("p_host_idle")].append("p_host_idle(the latency it multiplies is not priced)")
        else:
            planes.add(ctx, "p_host_idle", ENERGY_IDLE, _times(
                physics.band("p_host_idle"), wait,
                f"p_host_idle [{physics.band('p_host_idle').basis}] x (e2e_latency_s - host time)"))
    total, unknown = link_bytes(ctx)  # [U-J3] the store traffic to a host-owned store only
    route = "; the store traffic too, to the host-owned store" if owner(physics) == HOST else ""
    for key, name in (("e_link_per_byte", ENERGY_LINK), ("e_host_mem_per_byte", ENERGY_MEM)):
        if physics.has(key):
            if total:
                planes.add(ctx, key, name, scale_band(physics.band(key), total * _J_TO_MJ,
                                                      f"{key} x host_link_bytes{route}{link_end(key)}"))
            planes.unpriced[planes.plane(key)].extend(f"{key}({item})" for item in unknown)
        elif (total or unknown) and prices_host_link(physics):  # [U-J4] a crossing byte's row undeclared: named
            planes.unpriced["main"].append(unpriced_link_row(physics, key, "the host-chip crossing bytes"))
    if planes.system:
        _system_host(ctx, planes.whole, planes.unpriced["system"], planes.unpriced["main"])
    main = planes.parts["main"]
    if not main:
        return None, planes.unpriced["main"], None
    band = main[0] if len(main) == 1 else add_bands(main, " + ".join(p.basis for p in main))
    return band, planes.unpriced["main"], None


def _system_host(ctx: Any, parts: List[Band], unpriced: List[str], also_unpriced: List[str]) -> None:
    """[targeted M1] The whole host at its corners (``parts``: every term, the attributable power whole over its time),
    or its refusal by name where the idle draw's wait is not priced (a system line without the host's idle over the
    inference would read as the whole system and is not one)."""
    if missing := [item for item in unpriced if item.startswith("p_host_idle")]:
        ctx.refuse(ENERGY_HOST_SYSTEM, f"{missing[0]}: the system-level host draw is not priced without it")
        return
    if parts:
        notes = unpriced + also_unpriced
        source = ("the whole host at its corners: " + " + ".join(p.basis for p in parts)
                  + (f"; unpriced: {', '.join(notes)}" if notes else ""))
        ctx.term(ENERGY_HOST_SYSTEM, "mJ", add_bands(parts, source), source)


def price_system_energy(ctx: Any) -> None:
    """[U-J2] The whole-system line, printed beside the headline and labelled, never the headline; refused by name
    where either it or the whole host is. Only where the host states a system-level row. [targeted M1] The headline's
    chip side (the chip and its writes, ``PricingContext.energy_chip_side``) plus the whole host at its corners, so
    every end is the whole system's at the host's own corners (the headline's host takes the work's cost ends)."""
    rows = system_rows(ctx.physics)
    if not rows:
        return
    terms = {term.name: term for term in ctx.terms}
    refused = {refusal.name: refusal.reason for refusal in ctx.refusals}
    for name in (_HEADLINE, ENERGY_HOST_SYSTEM):
        if name not in terms:
            ctx.refuse(ENERGY_SYSTEM, f"the whole-system energy needs {name}, refused: "
                                      f"{refused.get(name, 'it was not priced')}")
            return
    chip = getattr(ctx, "energy_chip_side", None)
    if chip is None:
        ctx.refuse(ENERGY_SYSTEM, f"the whole-system energy needs the chip side of {_HEADLINE}, which was not recorded")
        return
    host = terms[ENERGY_HOST_SYSTEM].band
    source = (f"whole-system energy (system line, not the headline): the chip side of {_HEADLINE} [{chip.basis}] + "
              f"{ENERGY_HOST_SYSTEM} (the whole host at its corners, its system-level draw {', '.join(rows)} "
              f"included) [{host.basis}]")
    ctx.term(ENERGY_SYSTEM, "mJ", add_bands([chip, host], source), source)

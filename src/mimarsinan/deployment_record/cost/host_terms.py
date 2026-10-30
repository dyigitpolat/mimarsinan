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

A part whose quantity the view cannot size is named unpriced with the reason, never dropped."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from mimarsinan.deployment_record.schema.provenance import Band

#: The edge host's link and host-memory rows (the vocabulary's HOST group).
LINK_ROWS = ("e_link_per_byte", "t_link_per_byte", "t_link_transfer", "e_host_mem_per_byte", "t_host_mem_per_byte")
#: Every byte that crosses the host-chip boundary per inference, both directions: the pass carry, the segment I/O.
BOUNDARY_BYTES = (("carry_in_bytes", "pass carry"), ("carry_out_bytes", "pass carry"),
                  ("segment_in_bytes", "segment I/O"), ("segment_out_bytes", "segment I/O"))
MEMORY_BOUND, TRANSFERS = "host_memory_bound_bytes", "host_link_transfers"
#: The quantities the host-side census adds: a host-free or compute-only variant drops them with the host.
HOST_SIDE_QUANTITIES = ("segment_in_bytes", "segment_out_bytes", TRANSFERS, MEMORY_BOUND)
ENERGY_IDLE, ENERGY_LINK, LATENCY_LINK = "energy_host_idle_mj", "energy_host_link_mj", "latency_host_link_s"
#: The host's breakdown of the headlines it joins (its time and link; its idle and link energy).
HOST_DECOMPOSITION = ("latency_host_s", LATENCY_LINK, ENERGY_IDLE, ENERGY_LINK)
_J_TO_MJ = 1e3
_NO_CENSUS = "the view states no host-chip segment census"


def prices_host_link(physics: Any) -> bool:
    """Whether ``physics`` declares the host-chip link or host memory (an edge host block)."""
    return physics is not None and any(physics.has(key) for key in LINK_ROWS)


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


def _crossing(quantities: Any) -> Tuple[float, List[str]]:
    """The bytes that cross the boundary (both directions) and, per kind the view cannot size, its reason."""
    total, unknown = 0.0, {}
    for key, kind in BOUNDARY_BYTES:
        if quantities.has(key):
            total += quantities.get(key).value
        else:
            unknown.setdefault(kind, quantities.withheld_reason(key) or _NO_CENSUS)
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
        total, unknown = _crossing(q)
        bands = [physics.band(key) for key in stages]
        per = Band(*_ends(max, *bands), f"the slower of {' and '.join(stages)} per byte")
        if total:
            parts.append(scale_band(per, total, f"{per.basis} x host_link_bytes"))
        unpriced += [f"{stages[0]}({item})" for item in unknown]
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


def write_link_bytes(writes: Any) -> Optional[float]:
    """The write law's bytes per inference over the host-chip link: the period's backing-store payload over K and
    the host-sourced producer payloads (``cost.absolute.writes``' transfer terms); None without a period census."""
    census = getattr(writes, "census", None)
    if census is None:
        return None
    policy = writes.policy
    payload = sum(p.payload_bytes for p in census.passes) / int(policy.batch_k)
    producer = sum(p.written_cells_producer for p in census.passes)
    if producer and policy.producer_write_source == "host" and policy.payload_bits is not None:
        payload += producer * int(policy.payload_bits) / 8.0
    return float(payload)


def host_side_energy(ctx: Any, active: Optional[Band], busy: Optional[Band],
                     e2e: Optional[Band]) -> Tuple[Optional[Band], List[str]]:
    """The host's energy: ``active`` (p_host over the host's time) as before, and on an edge host its idle power over
    the wait and the link and host memory per crossing byte, each emitted as its own term."""
    physics = ctx.physics
    if not (physics.has("p_host_idle") or prices_host_link(physics)):
        return active, []
    parts: List[Band] = [] if active is None else [active]
    unpriced: List[str] = []
    if physics.has("p_host_idle"):
        if e2e is None:
            unpriced.append("p_host_idle(the latency it multiplies is not priced)")
        else:
            host = busy if busy is not None else Band(0.0, 0.0, 0.0, "no host time")
            wait = Band(*_ends(lambda t, h: t - h, e2e, host), "e2e_latency_s - host time")
            idle = Band(*_ends(lambda p, w: p * w * _J_TO_MJ, physics.band("p_host_idle"), wait),
                        f"p_host_idle [{physics.band('p_host_idle').basis}] x (e2e_latency_s - host time)")
            ctx.term(ENERGY_IDLE, "mJ", idle, idle.basis)
            parts.append(idle)
    energies = _declared(physics, ("e_link_per_byte", "e_host_mem_per_byte"))
    if energies:
        total, unknown = _crossing(ctx.quantities)
        total += getattr(ctx, "write_link_bytes", None) or 0.0  # the write law's payload, over the link too
        if total:
            per = Band(*_ends(lambda *v: sum(v), *(physics.band(key) for key in energies)), " + ".join(energies))
            link = scale_band(per, total * _J_TO_MJ, f"({per.basis}) x host_link_bytes")
            ctx.term(ENERGY_LINK, "mJ", link, link.basis)
            parts.append(link)
        unpriced += [f"{energies[0]}({item})" for item in unknown]
    if not parts:
        return None, unpriced
    return (parts[0] if len(parts) == 1 else add_bands(parts, " + ".join(p.basis for p in parts))), unpriced

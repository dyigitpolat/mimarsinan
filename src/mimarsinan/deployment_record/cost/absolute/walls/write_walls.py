"""The write law's walls: each pass's period write wall and their exposure per inference (serial or overlapped).

[U-ENERGY] A pass's wall carries the connectivity entries of the cores it writes (``PassCensus.connectivity_entries``,
at the declared wire width and the store's per-byte port), so the period load and the writes per inference at a
declared batch read one census, entries included. ``PassWalls`` prices each pass's wall once per program: a pass whose
written content a census already priced under the same physics and policy is read back, never priced again (the
every-core census's walls, read again for the period's passes that write the same cores)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, mul_bands, scale_band
from mimarsinan.deployment_record.cost.compute_time import (
    exposed_beyond, pass_compute_walls, payload_link_note, plus_core_walls, producer_walls)
from mimarsinan.deployment_record.platform_physics.hosts.store import slower_port
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, pass_events, pass_units
from mimarsinan.deployment_record.quantities.write_technology import shadow_room_refusal
from mimarsinan.deployment_record.schema.provenance import Band

WIDTH = "bytes_per_connectivity_entry"
#: Why a census states no entries where its plane gives no reason of its own.
NO_ENTRIES = "the census states no connectivity entries per core"
Walls = Tuple[Optional[List[Band]], Optional[Band], str]


def zero(basis: str) -> Band:
    return Band(0.0, 0.0, 0.0, basis)


def iterations(physics: PlatformPhysics) -> Optional[Band]:
    return physics.band("write_verify_iterations") if physics.has("write_verify_iterations") else None


def entry_bytes(entries: Optional[int], physics: PlatformPhysics) -> float:
    """The bytes of ``entries`` connectivity entries at the declared wire width (nominal), 0 where either is absent."""
    return float(entries or 0) * physics.band(WIDTH).nominal if entries and physics.has(WIDTH) else 0.0


def entries_named(census: Any, physics: PlatformPhysics, rows: Sequence[str], absent: str = "") -> List[str]:
    """What the census's connectivity entries leave unpriced, by name: their width where they are counted and it is
    undeclared, each of ``rows`` the description leaves undeclared, the width with the plane's reason where the
    census states no entries per core and the description declares it. A census that writes nothing names nothing."""
    entries = [p.connectivity_entries for p in census.passes if p.payload_bytes or p.written_cells_backing_store]
    if any(e is None for e in entries):
        stated = f"{WIDTH}(connectivity; not produced on this plane: {absent or NO_ENTRIES})"
        return [stated] if physics.has(WIDTH) else []
    if not sum(entries):
        return []
    if not physics.has(WIDTH):
        return [f"{WIDTH}(connectivity entries; undeclared, so their bytes are unpriced)"]
    return [f"{row}(connectivity entries)" for row in rows if not physics.has(row)]


def _plus_entries(walls: List[Band], census: Any, physics: PlatformPhysics) -> List[Band]:
    """Each pass's wall plus its written cores' connectivity entries at the store's per-byte port."""
    if not physics.has("t_program_per_byte") or not physics.has(WIDTH):
        return walls
    port = slower_port(physics, physics.band("t_program_per_byte"))
    out = []
    for wall, p in zip(walls, census.passes):
        size = entry_bytes(p.connectivity_entries, physics)
        out.append(wall if not size else add_bands(
            [wall, scale_band(port, size, f"t_program_per_byte [{port.basis}] x connectivity entries at the declared "
                                          "wire width")], "write wall + connectivity entries"))
    return out


def pass_write_times(census: Any, physics: PlatformPhysics, policy: WritePolicy, extents: Any) -> Walls:
    """(per-pass period write walls, the per-inference producer wall, or None with the reason)."""
    passes = census.passes
    producer = sum(p.written_cells_producer for p in passes)
    if not any(p.payload_bytes or p.written_cells_backing_store for p in passes) and not producer:
        return [zero("nothing is written") for _ in passes], zero("no producer write"), ""
    if physics.has("t_write_per_unit"):
        units = pass_units(census, policy, extents)
        if units is None:
            return None, None, (policy.mixed_core_writes or
                                f"t_write_per_unit needs the write granularity's units: granularity "
                                f"{policy.write_granularity!r} is undeclared or this plane states no "
                                "row/column extents of its written cores")
        if (count := iterations(physics)) is None:
            return None, None, "t_write_per_unit is declared but write_verify_iterations is not"
        step = mul_bands(physics.band("t_write_per_unit"), count, "t_write_per_unit x iterations")
        if physics.has("t_write_setup_per_event"):
            step = add_bands([step, physics.band("t_write_setup_per_event")], "write step + setup")
        walls = plus_core_walls([scale_band(step, events, "write_events x write step")  # [U-C2 G-1] + t_core_program
                                 for events in pass_events(units, policy)], census, physics)
        if producer and policy.write_granularity != "cell":
            return None, None, "producer writes are counted in cells; their units at another granularity are unstated"
        steps = pass_events([p.written_cells_producer * policy.cells_per_weight for p in passes], policy)
        return (_plus_entries(walls, census, physics),
                producer_walls(step, steps, census, physics, policy), "")  # [U-J5 L-5] beside the link too
    if ((byte := physics.has("t_program_per_byte")) or physics.has("t_core_program")) and not producer:
        port = physics.band("t_program_per_byte") if byte else None
        walls = [scale_band(port, p.payload_bytes, "t_program_per_byte x payload_bytes")
                 for p in passes] if port is not None else None
        return _plus_entries(plus_core_walls(walls, census, physics), census, physics), zero("no producer write"), ""
    if producer:
        return None, None, ("producer cells are written per inference and need t_write_per_unit; t_program_per_byte "
                            "prices backing-store bytes only")
    return None, None, "cells are written but the target declares no write time (t_write_per_unit or t_program_per_byte)"


@dataclass(frozen=True)
class _Passes:
    """A census's passes read as a census of their own (the per-pass wall rules read ``passes`` only)."""

    passes: Tuple[Any, ...]


def _key(p: Any, extent: Any) -> Tuple[Any, ...]:
    return (p.payload_bytes, p.written_cells_backing_store, p.written_cells_producer, p.cores, p.resident_cores,
            p.operand_only_cores, p.connectivity_entries, extent)


class PassWalls:
    """Each pass's write wall priced once per program (module docstring), [m-6] under each physics and policy it is
    asked for (a chip of several write technologies prices each share under its own); ``priced`` counts the passes
    priced."""

    def __init__(self) -> None:
        self._bound: List[Tuple[Any, Any, Dict[Tuple[Any, ...], Band]]] = []
        self.priced = 0

    def _cache(self, physics: Any, policy: Any) -> Dict[Tuple[Any, ...], Band]:
        for bound, bound_policy, cache in self._bound:
            if (bound is physics or bound == physics) and bound_policy == policy:
                return cache
        self._bound.append((physics, policy, {}))
        return self._bound[-1][2]

    def walls(self, census: Any, physics: PlatformPhysics, policy: WritePolicy, extents: Any) -> Walls:
        """``pass_write_times`` over ``census``, each pass read back where its content was priced already (a
        producer's per-inference writes are never read back)."""
        if any(p.written_cells_producer for p in census.passes):
            return pass_write_times(census, physics, policy, extents)
        cached = self._cache(physics, policy)
        keys = [_key(p, None if extents is None else extents[q]) for q, p in enumerate(census.passes)]
        missing = [q for q, key in enumerate(keys) if key not in cached]
        if missing:
            whole = len(missing) == len(keys)
            walls, _producer, reason = pass_write_times(
                census if whole else _Passes(tuple(census.passes[q] for q in missing)), physics, policy,
                extents if whole or extents is None else [extents[q] for q in missing])
            if walls is None:
                return None, None, reason
            cached.update((keys[q], wall) for q, wall in zip(missing, walls))
            self.priced += len(missing)
        return [cached[key] for key in keys], zero("no producer write"), ""


def exposed_write(census: Any, physics: PlatformPhysics, policy: WritePolicy, extents: Any, compute_sealed: bool,
                  pass_types: Any = None, pass_sops: Any = None, walls: Optional[PassWalls] = None,
                  named: Sequence[str] = ()) -> Tuple[Optional[Band], str]:
    """The exposed write time per inference over the census's per-pass walls (``walls``: read back where priced),
    with ``named`` (what the walls leave unpriced) in its basis."""
    priced = pass_write_times(census, physics, policy, extents) if walls is None else walls.walls(
        census, physics, policy, extents)
    pass_walls, producer, reason = priced
    if pass_walls is None or producer is None:
        return None, reason
    exposed, reason = expose(pass_walls, producer, census, physics, policy, compute_sealed, pass_types, pass_sops)
    if exposed is None or not named:
        return exposed, reason
    return Band(exposed.low, exposed.nominal, exposed.high, f"{exposed.basis}; unpriced: {', '.join(named)}"), reason


def expose(walls: Sequence[Band], producer: Band, census: Any, physics: PlatformPhysics, policy: WritePolicy,
           compute_sealed: bool, pass_types: Any, pass_sops: Any = None) -> Tuple[Optional[Band], str]:
    """The exposure rule over per-pass write walls and the per-inference producer wall (serial or overlapped)."""
    k, tail = policy.batch_k, payload_link_note(census, physics, policy)  # [U-J5 L-5] the payload over the link
    if all(w.high == 0.0 for w in walls) and producer.high == 0.0:
        return zero("nothing is written: no write time is exposed"), ""
    if policy.overlap_mode == "serial" or not walls:  # [U-ENERGY] at K = 1 (a load) each wall is its own share
        terms = (list(walls) if k == 1 else [scale_band(w, 1.0 / k, "t_w / K") for w in walls]) + [producer]
        return add_bands(terms, "serial: sum_q t_w(q) / K + producer writes per inference" + tail), ""
    room = shadow_room_refusal(policy.core_capacity, policy.shadow_capacity_fraction, pass_types,
                               [p.cores for p in census.passes])
    if room is not None:
        return None, room
    if not compute_sealed:
        return None, "overlapped writes hide behind compute, and this census states no compute steps"
    computes, no_compute = pass_compute_walls(physics, census, pass_sops)  # [U-C2 G-2] per step or per event
    if computes is None:
        return None, no_compute
    q = len(walls)
    terms = [exposed_beyond(walls[(i + 1) % q], k, computes[i]) for i in range(q)] + [producer]
    return add_bands(terms, "overlapped: sum_q max(0, t_w(q+1) / K - t_c(q)) + producer writes" + tail), ""

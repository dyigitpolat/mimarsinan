"""The write law priced over a period census: write energy and exposed write time per inference at
batch K, lifetime against endurance, and the lifetime floor as a separate admissibility verdict.

One pricer for every census producer (the record and candidate planes, the residency plan steps,
the capability-level census): it reads the S0 ``PassPeriodCensus`` and the declared policy only.
On a chip of several write technologies (N2b, ``quantities/write_technology.py``) the same law prices each
technology's share under its physics, and the shares join: energies and per-pass walls add, the lifetime and
verdict are ``write_lifetime.join_lifetimes``'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, mul_bands, scale_band
from mimarsinan.deployment_record.cost.absolute.write_lifetime import (
    LIFETIME_HOURS, LIFETIME_INFERENCES, LIFETIME_VERDICT, SECONDARY_MARKERS, WritePricing, join_lifetimes,
    price_lifetime,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import technology_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, pass_events, pass_units
from mimarsinan.deployment_record.quantities.write_technology import TechnologySplit, shadow_room_refusal
from mimarsinan.deployment_record.schema.provenance import Band

__all__ = ["ENERGY_WRITE", "LATENCY_WRITE", "LIFETIME_HOURS", "LIFETIME_INFERENCES", "LIFETIME_VERDICT",
           "SECONDARY_MARKERS", "WriteInputs", "WritePricing", "price_write_inputs", "price_write_terms"]

ENERGY_WRITE = "energy_write_mj"
LATENCY_WRITE = "latency_write_exposed_s"
_J_TO_MJ = 1e3


@dataclass(frozen=True)
class WriteInputs:
    """A plane's period census with the run's write policy (what ``price_absolute`` adds)."""

    census: Any
    policy: WritePolicy
    written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None
    compute_sealed: bool = True
    #: Why the plane states no period census (``census`` None): the write terms refuse with it.
    unavailable: Optional[str] = None
    #: The period writes of the most-written physical core, where the plane states them.
    most_written: Optional[int] = None
    #: Per pass, the cores of each type it computes on (``WriteCensus.pass_core_types``).
    pass_core_types: Optional[Sequence[Sequence[Tuple[Tuple[int, int], int]]]] = None
    #: N2: (static writes per period, producer writes per inference) of each core holding operand regions
    #: (``WriteCensus.producer_slots``).
    producer_slots: Optional[Sequence[Tuple[int, int]]] = None
    #: N2b: the census split by write technology (``PlaneWrites.by_technology``), where the policy states several.
    by_technology: Optional[TechnologySplit] = None

    @classmethod
    def of(cls, writes: Any) -> "WriteInputs":
        """The pricer's inputs from a plane's ``write_law.PlaneWrites``."""
        census = writes.census
        return cls(census=None if census is None else census.census, policy=writes.policy,
                   written_extents=None if census is None else census.written_extents,
                   compute_sealed=census is not None and census.compute_sealed,
                   unavailable=writes.unavailable,
                   most_written=None if census is None else census.most_written_core,
                   pass_core_types=None if census is None else census.pass_core_types,
                   producer_slots=None if census is None else census.producer_slots,
                   by_technology=getattr(writes, "by_technology", None))


def _zero(basis: str) -> Band:
    return Band(0.0, 0.0, 0.0, basis)


def _iterations(physics: PlatformPhysics) -> Optional[Band]:
    return physics.band("write_verify_iterations") if physics.has("write_verify_iterations") else None


def _write_energy(census: Any, physics: PlatformPhysics, policy: WritePolicy) -> Tuple[Optional[Band], str]:
    """(per-inference write energy in mJ, or None with the refusal reason)."""
    k = policy.batch_k
    payload = sum(p.payload_bytes for p in census.passes)
    cells = sum(p.written_cells_backing_store for p in census.passes)
    producer = sum(p.written_cells_producer for p in census.passes)
    if not (payload or cells or producer):
        return _zero("nothing is rewritten per period: the write term is 0"), ""
    parts: List[Band] = []
    notes: List[str] = []
    if physics.has("e_dma_per_byte"):
        parts.append(scale_band(physics.band("e_dma_per_byte"), payload / k * _J_TO_MJ,
                                "e_dma_per_byte x period_rewrite_bytes / K"))
    elif payload:
        notes.append("unpriced: e_dma_per_byte(write transfer)")
    if producer and policy.producer_write_source == "host" and policy.payload_bits is not None:
        moved = producer * int(policy.payload_bits) / 8.0  # host-sourced payloads cross into the chip per inference
        if physics.has("e_dma_per_byte"):
            parts.append(scale_band(physics.band("e_dma_per_byte"), moved * _J_TO_MJ,
                                    "e_dma_per_byte x host-sourced payload bytes per inference (transfer)"))
        else:
            notes.append("unpriced: e_dma_per_byte(host-sourced payload transfer)")
    if physics.has("e_write_per_cell"):
        iterations = _iterations(physics)
        if iterations is None:
            return None, ("e_write_per_cell is declared but write_verify_iterations is not, so the "
                          "pulses per written cell are unknown")
        per_cell = physics.band("e_write_per_cell")
        if physics.has("e_verify_read_per_cell"):
            per_cell = add_bands([per_cell, physics.band("e_verify_read_per_cell")],
                                 "e_write_per_cell + e_verify_read_per_cell")
        parts.append(scale_band(mul_bands(per_cell, iterations, "per-cell write-verify energy"),
                                (cells / k + producer) * _J_TO_MJ,
                                "e_write_per_cell x runtime_written_cells x write_verify_iterations"))
    elif physics.has("e_program_per_byte") and not producer:
        parts.append(scale_band(physics.band("e_program_per_byte"), payload / k * _J_TO_MJ,
                                "e_program_per_byte x period_rewrite_bytes / K"))
    else:
        return None, ("cells are written per inference but the target declares no commit energy "
                      "(e_write_per_cell, or e_program_per_byte for backing-store bytes)")
    basis = " + ".join(p.basis for p in parts) + (f"; {'; '.join(notes)}" if notes else "")
    return add_bands(parts, basis), ""


def _pass_write_times(census: Any, physics: PlatformPhysics, policy: WritePolicy,
                      extents: Any) -> Tuple[Optional[List[Band]], Optional[Band], str]:
    """(per-pass period write walls, the per-inference producer wall, or None with the reason)."""
    passes = census.passes
    producer = sum(p.written_cells_producer for p in passes)
    if not any(p.payload_bytes or p.written_cells_backing_store for p in passes) and not producer:
        return [_zero("nothing is written") for _ in passes], _zero("no producer write"), ""
    if physics.has("t_write_per_unit"):
        units = pass_units(census, policy, extents)
        if units is None:
            return None, None, (policy.mixed_core_writes or
                                f"t_write_per_unit needs the write granularity's units: granularity "
                                f"{policy.write_granularity!r} is undeclared or this plane states no "
                                "row/column extents of its written cores")
        iterations = _iterations(physics)
        if iterations is None:
            return None, None, "t_write_per_unit is declared but write_verify_iterations is not"
        step = mul_bands(physics.band("t_write_per_unit"), iterations, "t_write_per_unit x iterations")
        if physics.has("t_write_setup_per_event"):
            step = add_bands([step, physics.band("t_write_setup_per_event")], "write step + setup")
        walls = [scale_band(step, events, "write_events x write step") for events in pass_events(units, policy)]
        if producer and policy.write_granularity != "cell":
            return None, None, "producer writes are counted in cells; their units at another granularity are unstated"
        steps = sum(pass_events([p.written_cells_producer for p in passes], policy))  # no step spans two passes
        return walls, scale_band(step, steps, "producer write steps, per pass"), ""
    if physics.has("t_program_per_byte") and not producer:
        per_byte = physics.band("t_program_per_byte")
        return [scale_band(per_byte, p.payload_bytes, "t_program_per_byte x payload_bytes")
                for p in passes], _zero("no producer write"), ""
    if producer:
        return None, None, ("producer cells are written per inference and need t_write_per_unit; t_program_per_byte "
                            "prices backing-store bytes only")
    return None, None, "cells are written but the target declares no write time (t_write_per_unit or t_program_per_byte)"


def _exposed(write: Band, k: int, compute: Band) -> Band:
    """max(0, t_w / K - t_c) corner-wise (monotone up in the write, down in the compute)."""
    return Band(max(0.0, write.low / k - compute.high), max(0.0, write.nominal / k - compute.nominal),
                max(0.0, write.high / k - compute.low), "max(0, t_w(q+1) / K - t_c(q))")


def _exposed_write(census: Any, physics: PlatformPhysics, policy: WritePolicy, extents: Any,
                   compute_sealed: bool, pass_types: Any = None) -> Tuple[Optional[Band], str]:
    walls, producer, reason = _pass_write_times(census, physics, policy, extents)
    if walls is None or producer is None:
        return None, reason
    return _expose(walls, producer, census, physics, policy, compute_sealed, pass_types)


def _expose(walls: Sequence[Band], producer: Band, census: Any, physics: PlatformPhysics, policy: WritePolicy,
            compute_sealed: bool, pass_types: Any) -> Tuple[Optional[Band], str]:
    """The exposure rule over per-pass write walls and the per-inference producer wall (serial or overlapped)."""
    k = policy.batch_k
    if all(w.high == 0.0 for w in walls) and producer.high == 0.0:
        return _zero("nothing is written: no write time is exposed"), ""
    if policy.overlap_mode == "serial" or not walls:
        terms = [scale_band(w, 1.0 / k, "t_w / K") for w in walls] + [producer]
        return add_bands(terms, "serial: sum_q t_w(q) / K + producer writes per inference"), ""
    room = shadow_room_refusal(policy.core_capacity, policy.shadow_capacity_fraction, pass_types,
                               [p.cores for p in census.passes])
    if room is not None:
        return None, room
    if not compute_sealed:
        return None, "overlapped writes hide behind compute, and this census states no compute steps"
    if not physics.has("t_cycle"):
        return None, "overlapped writes hide behind compute, which needs t_cycle"
    cycle = physics.band("t_cycle")
    q = len(walls)
    terms = [_exposed(walls[(i + 1) % q], k, scale_band(cycle, census.passes[i].compute_steps, "t_c"))
             for i in range(q)] + [producer]
    return add_bands(terms, "overlapped: sum_q max(0, t_w(q+1) / K - t_c(q)) + producer writes"), ""


def _refuse_all(policy: WritePolicy, reason: str) -> WritePricing:
    out = WritePricing()
    for name in (ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES) + (
            (LIFETIME_VERDICT,) if policy.min_lifetime_hours is not None else ()):
        out.refuse(name, reason)
    return out


#: Why the one-law pricer refuses a policy of several write technologies: their shares need the census split.
_SEVERAL = ("the chip declares several write technologies (core types' write_physics or write declarations), and "
            "this census is not split by write technology, so one law would price every cell alike")


def price_write_inputs(writes: WriteInputs, physics: PlatformPhysics) -> WritePricing:
    """``price_write_terms`` over a plane's inputs, per write technology where the chip declares several; a plane
    without a census, or without its split, refuses by name."""
    if writes.census is None:
        return _refuse_all(writes.policy, writes.unavailable or "the plane states no period census")
    if writes.policy.core_technologies is not None:
        split = writes.by_technology
        if split is None or split.refused is not None:
            return _refuse_all(writes.policy, _SEVERAL if split is None else split.refused or _SEVERAL)
        return _price_technologies(writes, split, physics)
    return price_write_terms(writes.census, physics, writes.policy,
                             written_extents=writes.written_extents, compute_sealed=writes.compute_sealed,
                             most_written=writes.most_written, pass_core_types=writes.pass_core_types,
                             producer_slots=writes.producer_slots)


def _price_technologies(writes: WriteInputs, split: TechnologySplit, chip: PlatformPhysics) -> WritePricing:
    """G3's law on each write technology's share, under its physics; one technology is the one law at its physics."""
    shares = [(part, technology_physics(chip, part.technology)) for part in split.parts]
    if len(shares) == 1:
        part, physics = shares[0]
        return price_write_terms(part.census, physics, part.policy, written_extents=part.written_extents,
                                 compute_sealed=writes.compute_sealed, most_written=part.most_written,
                                 pass_core_types=part.pass_core_types, producer_slots=part.producer_slots)
    out = WritePricing()
    energies = [(part.label, _write_energy(part.census, physics, part.policy)) for part, physics in shares]
    refused = [f"{label}: {reason}" for label, (band, reason) in energies if band is None]
    if refused:
        out.refuse(ENERGY_WRITE, "; ".join(refused))
    else:
        out.energy = out.term(ENERGY_WRITE, "mJ", add_bands(
            [band for _, (band, _) in energies if band is not None],
            "per write technology: " + "; ".join(f"{label}: {band.basis}" for label, (band, _) in energies
                                                 if band is not None)))
    timed = [(part.label, _pass_write_times(part.census, physics, part.policy, part.written_extents))
             for part, physics in shares]
    refused = [f"{label}: {reason}" for label, (walls, producer, reason) in timed if walls is None or producer is None]
    if refused:
        out.refuse(LATENCY_WRITE, "; ".join(refused))
    else:
        walls = [add_bands([w[q] for _, (w, _, _) in timed if w is not None], "the technologies' walls, shared ports")
                 for q in range(len(writes.census.passes))]
        producer = add_bands([p for _, (_, p, _) in timed if p is not None], "every technology's producer writes")
        exposed, reason = _expose(walls, producer, writes.census, chip, writes.policy, writes.compute_sealed,
                                  writes.pass_core_types)
        if exposed is None:
            out.refuse(LATENCY_WRITE, reason)
        else:
            out.exposed = out.term(LATENCY_WRITE, "s", exposed)
    lives = []
    for part, physics in shares:
        own = WritePricing()
        price_lifetime(own, part.census, physics, part.policy, part.most_written, part.producer_slots)
        lives.append((part.label, own))
    join_lifetimes(out, lives, writes.policy.min_lifetime_hours)
    return out


def price_write_terms(census: Any, physics: PlatformPhysics, policy: WritePolicy, *,
                      written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                      compute_sealed: bool = True, most_written: Optional[int] = None,
                      pass_core_types: Any = None,
                      producer_slots: Optional[Sequence[Tuple[int, int]]] = None) -> WritePricing:
    """Write energy, exposed write time and lifetime of one period census at the declared K.

    ``most_written`` is the period writes of the most-written physical core (``WriteCensus.
    most_written_core``). A producer that places no core leaves it None only where the census
    provably rewrites; the lifetime is then an upper bound that can only prove inadmissibility.
    ``pass_core_types`` (``WriteCensus.pass_core_types``) gives the overlapped mode its per-type room;
    ``producer_slots`` (``WriteCensus.producer_slots``) the operand regions' placement by the same slot rule. A
    policy of several write technologies refuses by name (``price_write_inputs`` prices their shares)."""
    if policy.core_technologies is not None:
        return _refuse_all(policy, _SEVERAL)
    out = WritePricing()
    energy, reason = _write_energy(census, physics, policy)
    if energy is None:
        out.refuse(ENERGY_WRITE, reason)
    else:
        out.energy = out.term(ENERGY_WRITE, "mJ", energy)
    exposed, reason = _exposed_write(census, physics, policy, written_extents, compute_sealed, pass_core_types)
    if exposed is None:
        out.refuse(LATENCY_WRITE, reason)
    else:
        out.exposed = out.term(LATENCY_WRITE, "s", exposed)
    price_lifetime(out, census, physics, policy, most_written, producer_slots)
    return out

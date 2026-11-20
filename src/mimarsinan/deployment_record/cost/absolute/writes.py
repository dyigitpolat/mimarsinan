"""The write law priced over a period census: write energy and exposed write time per inference at batch K, lifetime
against endurance, and the lifetime floor as a separate admissibility verdict.

One pricer for every census producer (the record and candidate planes, the residency plan steps, the capability-level
census): it reads the S0 ``PassPeriodCensus`` and the declared policy only. On a chip of several write technologies
(N2b, ``quantities/write_technology.py``) the same law prices each technology's share under its physics, and the
shares join: energies and per-pass walls add, the lifetime and verdict are ``write_lifetime.join_lifetimes``'.
[consts, U-C C-6] A rewritten core costs its fetch and commit and, if it holds a static program, ``e_core_program``.
[U-ENERGY] It also costs the connectivity entries of the cores the census writes, at the declared wire width; the
per-pass walls and their exposure are ``walls/write_walls.py``'s."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.context import external_refusals
from mimarsinan.deployment_record.cost.absolute.formulas import (
    add_bands, mul_bands, programming_byte_energies, scale_band)
from mimarsinan.deployment_record.cost.absolute.walls.write_walls import (
    PassWalls, entries_named, entry_bytes, expose, exposed_write, iterations, pass_write_times, zero)
from mimarsinan.deployment_record.cost.write_lifetime import (
    LIFETIME_HOURS, LIFETIME_INFERENCES, LIFETIME_VERDICT, SECONDARY_MARKERS, WritePricing, join_lifetimes,
    price_lifetime)
from mimarsinan.deployment_record.platform_physics.hosts.transfer import BUFFER_BASIS, chip_buffer
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import technology_physics
from mimarsinan.deployment_record.quantities.planes.loads import census_entries
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, programmed_basis, programmed_cores
from mimarsinan.deployment_record.quantities.write_technology import TechnologySplit, cells_basis
from mimarsinan.deployment_record.schema.provenance import Band

__all__ = ["ENERGY_WRITE", "LATENCY_WRITE", "LIFETIME_HOURS", "LIFETIME_INFERENCES", "LIFETIME_VERDICT",
           "SECONDARY_MARKERS", "WriteInputs", "WritePricing", "price_write_inputs", "price_write_terms"]

ENERGY_WRITE = "energy_write_mj"
LATENCY_WRITE = "latency_write_exposed_s"
_J_TO_MJ = 1e3
#: The transfer rows of a host-owned store (``formulas.programming_byte_energies``' labels): the load prices them.
TRANSFER = ("link", "host_memory")


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
    pass_sops: Optional[Tuple[float, ...]] = None  #: [targeted M2] per census pass, its busiest core's events
    #: [U-ENERGY] Why the plane states no connectivity entries per core, where it gives a reason of its own.
    entries_absent: str = ""

    @classmethod
    def of(cls, writes: Any) -> "WriteInputs":
        """The pricer's inputs from a plane's ``write_law.PlaneWrites``."""
        census = writes.census
        return cls(census=None if census is None else census.census, policy=writes.policy,
                   written_extents=None if census is None else census.written_extents,
                   compute_sealed=census is not None and census.compute_sealed, unavailable=writes.unavailable,
                   most_written=None if census is None else census.most_written_core,
                   pass_core_types=None if census is None else census.pass_core_types,
                   producer_slots=None if census is None else census.producer_slots,
                   by_technology=getattr(writes, "by_technology", None),
                   pass_sops=getattr(writes, "pass_sops", None))


def chip_byte_rows(physics: PlatformPhysics) -> Tuple[Tuple[str, str], ...]:
    """The chip's own per-byte rows a written byte pays (a host-owned store's transfer is the load's to price)."""
    return tuple(row for row in programming_byte_energies(physics) if row[1] not in TRANSFER)


def _write_energy(census: Any, physics: PlatformPhysics, policy: WritePolicy,
                  entries_absent: str = "") -> Tuple[Optional[Band], str]:
    """(per-inference write energy in mJ, or None with the refusal reason)."""
    k = policy.batch_k
    payload = sum(p.payload_bytes for p in census.passes)
    cells = sum(p.written_cells_backing_store for p in census.passes)
    producer = sum(p.written_cells_producer for p in census.passes)
    if not (payload or cells or producer):
        return zero("nothing is rewritten per period: the write term is 0"), ""
    parts: List[Band] = []
    notes: List[str] = []
    routed = physics.store_transfer_note()  # [U-J3] a host-owned store's transfer: the host link's, not the chip's
    dma, buffer = physics.has("e_dma_per_byte"), chip_buffer(physics, "e_dma_per_byte")  # [U-J5 L-1] its place
    if dma and (routed is None or (buffer and payload)):  # a chip-buffer row: every arriving byte pays it
        parts.append(scale_band(physics.band("e_dma_per_byte"), payload / k * _J_TO_MJ,
                                "e_dma_per_byte x period_rewrite_bytes / K" + (BUFFER_BASIS if routed else "")))
    if payload and (routed is not None or not dma):
        notes.append(routed or "unpriced: e_dma_per_byte(write transfer)")
    if producer and policy.producer_write_source == "host" and policy.payload_bits is not None:
        crossed, moved = physics.payload_link_note(), producer * int(policy.payload_bits) / 8.0  # per inference
        if dma and (crossed is None or buffer):  # no link rows: the chip's one channel row prices it, as on RC1
            parts.append(scale_band(physics.band("e_dma_per_byte"), moved * _J_TO_MJ, "e_dma_per_byte x host-sourced "
                                    "payload bytes per inference (transfer)" + (BUFFER_BASIS if crossed else "")))
        if crossed is not None or not dma:
            notes.append(crossed or "unpriced: e_dma_per_byte(host-sourced payload transfer)")
    if physics.has("e_write_per_cell"):
        if (count := iterations(physics)) is None:
            return None, ("e_write_per_cell is declared but write_verify_iterations is not, so the "
                          "pulses per written cell are unknown")
        per_cell = physics.band("e_write_per_cell")
        if physics.has("e_verify_read_per_cell"):
            per_cell = add_bands([per_cell, physics.band("e_verify_read_per_cell")],
                                 "e_write_per_cell + e_verify_read_per_cell")
        parts.append(scale_band(mul_bands(per_cell, count, "per-cell write-verify energy"),  # [U-C4] each cell
                                (cells / k + producer) * policy.cells_per_weight * _J_TO_MJ, cells_basis(policy)))
    elif physics.has("e_program_per_byte") and not producer:
        parts.append(scale_band(physics.band("e_program_per_byte"), payload / k * _J_TO_MJ,
                                "e_program_per_byte x period_rewrite_bytes / K"))
    else:
        return None, ("cells are written per inference but the target declares no commit energy "
                      "(e_write_per_cell, or e_program_per_byte for backing-store bytes)")
    undeclared: List[str] = []
    if cores := programmed_cores(census):  # [U-C C-6] static programs only
        if physics.has("e_core_program"):
            core, which = physics.band("e_core_program"), programmed_basis(census)
            parts.append(scale_band(core, cores / k * _J_TO_MJ, f"e_core_program [{core.basis}] x {which} / K"))
        else:  # [U-ENERGY class 2] a per-core charge over programmed cores, undeclared: named, never dropped
            undeclared.append("e_core_program(core_program)")
    rows = chip_byte_rows(physics)
    if size := entry_bytes(census_entries(census), physics):  # [U-ENERGY] the written cores' connectivity entries
        parts += [scale_band(physics.band(row), size / k * _J_TO_MJ,
                             f"{row} x connectivity entries at the declared wire width / K")
                  for row, _label in rows if physics.has(row)]
    named = undeclared + entries_named(census, physics, [row for row, _label in rows], entries_absent)
    basis = " + ".join(p.basis for p in parts) + (f"; {'; '.join(notes)}" if notes else "") + (
        f"; unpriced: {', '.join(named)}" if named else "")
    return add_bands(parts, basis), ""


def _refuse_all(policy: WritePolicy, reason: str) -> WritePricing:
    out = WritePricing()
    for name in (ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES) + (
            (LIFETIME_VERDICT,) if policy.min_lifetime_hours is not None else ()):
        out.refuse(name, reason)
    return out


#: Why the one-law pricer refuses a policy of several write technologies: their shares need the census split.
_SEVERAL = ("the chip declares several write technologies (core types' write_physics or write declarations), and "
            "this census is not split by write technology, so one law would price every cell alike")


def price_write_inputs(writes: WriteInputs, physics: PlatformPhysics, *, lifetime: bool = True,
                       walls: Optional[PassWalls] = None) -> WritePricing:
    """``price_write_terms`` over a plane's inputs, per write technology where the chip declares several; a plane
    without a census, or without its split, refuses by name. [U-ENERGY] ``lifetime`` False: a load figure, which has
    no lifetime; ``walls``: each pass's wall read back where a census of the same program priced it."""
    if writes.census is None:
        return _refuse_all(writes.policy, writes.unavailable or "the plane states no period census")
    if gate := external_refusals(physics):  # [fix_c] the one gate, before a chip's technology shares too
        return _refuse_all(writes.policy, "; ".join(r.reason for r in gate))
    if writes.policy.core_technologies is not None:
        split = writes.by_technology
        if split is None or split.refused is not None:
            return _refuse_all(writes.policy, _SEVERAL if split is None else split.refused or _SEVERAL)
        return _price_technologies(writes, split, physics, lifetime=lifetime, walls=walls)
    return price_write_terms(writes.census, physics, writes.policy,
                             written_extents=writes.written_extents, compute_sealed=writes.compute_sealed,
                             most_written=writes.most_written, pass_core_types=writes.pass_core_types,
                             producer_slots=writes.producer_slots, pass_sops=writes.pass_sops, lifetime=lifetime,
                             walls=walls, entries_absent=writes.entries_absent)


def _price_technologies(writes: WriteInputs, split: TechnologySplit, chip: PlatformPhysics, *,
                        lifetime: bool = True, walls: Optional[PassWalls] = None) -> WritePricing:
    """G3's law on each write technology's share, under its physics; one technology is the one law at its physics.
    [m-6] Each share names the plane's own reason for the entries it does not state and reads each pass's wall back
    where a census of the same program priced it (``walls``)."""
    shares = [(part, technology_physics(chip, part.technology)) for part in split.parts]
    if len(shares) == 1:
        part, physics = shares[0]
        return price_write_terms(part.census, physics, part.policy, written_extents=part.written_extents,
                                 compute_sealed=writes.compute_sealed, most_written=part.most_written,
                                 pass_core_types=part.pass_core_types, producer_slots=part.producer_slots,
                                 pass_sops=writes.pass_sops, lifetime=lifetime, walls=walls,
                                 entries_absent=writes.entries_absent)
    out = WritePricing()
    energies = [(part.label, _write_energy(part.census, physics, part.policy, writes.entries_absent))
                for part, physics in shares]
    if refused := [f"{label}: {reason}" for label, (band, reason) in energies if band is None]:
        out.refuse(ENERGY_WRITE, "; ".join(refused))
    else:
        out.energy = out.term(ENERGY_WRITE, "mJ", add_bands(
            [band for _, (band, _) in energies if band is not None],
            "per write technology: " + "; ".join(f"{label}: {band.basis}" for label, (band, _) in energies
                                                 if band is not None)))
    timed = [(part.label, pass_write_times(part.census, physics, part.policy, part.written_extents) if walls is None
              else walls.walls(part.census, physics, part.policy, part.written_extents)) for part, physics in shares]
    refused = [f"{label}: {reason}" for label, (wall, producer, reason) in timed if wall is None or producer is None]
    if refused:
        out.refuse(LATENCY_WRITE, "; ".join(refused))
    else:
        joined = [add_bands([w[q] for _, (w, _, _) in timed if w is not None], "the technologies' walls, shared ports")
                  for q in range(len(writes.census.passes))]
        producer = add_bands([p for _, (_, p, _) in timed if p is not None], "every technology's producer writes")
        exposed, reason = expose(joined, producer, writes.census, chip, writes.policy, writes.compute_sealed,
                                 writes.pass_core_types, writes.pass_sops)
        if exposed is None:
            out.refuse(LATENCY_WRITE, reason)
        else:
            out.exposed = out.term(LATENCY_WRITE, "s", exposed)
    if not lifetime:
        return out
    lives = []
    for part, physics in shares:
        own = WritePricing()
        price_lifetime(own, part.census, physics, part.policy, part.most_written, part.producer_slots)
        lives.append((part.label, own))
    join_lifetimes(out, lives, writes.policy.min_lifetime_hours)
    return out


def price_write_terms(census: Any, physics: PlatformPhysics, policy: WritePolicy, *,
                      written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                      compute_sealed: bool = True, most_written: Optional[int] = None, pass_core_types: Any = None,
                      producer_slots: Optional[Sequence[Tuple[int, int]]] = None, pass_sops: Any = None,
                      lifetime: bool = True, walls: Optional[PassWalls] = None, entries_absent: str = ""
                      ) -> WritePricing:
    """Write energy, exposed write time and lifetime of one period census at the declared K.

    ``most_written`` is the period writes of the most-written physical core (``WriteCensus.most_written_core``). A
    producer that places no core leaves it None only where the census provably rewrites; the lifetime is then an
    upper bound that can only prove inadmissibility. ``pass_core_types`` (``WriteCensus.pass_core_types``) gives the
    overlapped mode its per-type room; ``producer_slots`` (``WriteCensus.producer_slots``) the operand regions'
    placement by the same slot rule. A policy of several write technologies refuses by name (``price_write_inputs``
    prices their shares). [fix_c] A profile whose external component the run has not declared (an SRAM-CIM macro
    without its backing store) prices no write term and refuses each by name (``external_refusals``, the gate
    ``price_absolute`` reads), so the executed and the exploration plane read one rule."""
    if gate := external_refusals(physics):
        return _refuse_all(policy, "; ".join(r.reason for r in gate))
    if policy.core_technologies is not None:
        return _refuse_all(policy, _SEVERAL)
    out = WritePricing()
    energy, reason = _write_energy(census, physics, policy, entries_absent)
    if energy is None:
        out.refuse(ENERGY_WRITE, reason)
    else:
        out.energy = out.term(ENERGY_WRITE, "mJ", energy)
    named = entries_named(census, physics, ("t_program_per_byte",), entries_absent)
    exposed, reason = exposed_write(census, physics, policy, written_extents, compute_sealed, pass_core_types,
                                    pass_sops, walls, named)
    if exposed is None:
        out.refuse(LATENCY_WRITE, reason)
    else:
        out.exposed = out.term(LATENCY_WRITE, "s", exposed)
    if lifetime:  # [U-ENERGY] a load figure has no lifetime: the load law skips it
        price_lifetime(out, census, physics, policy, most_written, producer_slots)
    return out

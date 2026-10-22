"""The write law priced over a period census: write energy and exposed write time per inference at
batch K, lifetime against endurance, and the lifetime floor as a separate admissibility verdict.

One pricer for every census producer (the record and candidate planes, the residency plan steps,
the capability-level census): it reads the S0 ``PassPeriodCensus`` and the declared policy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.build.write_census import shadow_room_refusal
from mimarsinan.deployment_record.cost.absolute.context import AdmissibilityVerdict, PricingRefusal
from mimarsinan.deployment_record.cost.absolute.formulas import add_bands, mul_bands, scale_band
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.write_law import (
    CellWear, WritePolicy, cell_wear, pass_events, pass_units,
)
from mimarsinan.deployment_record.schema.provenance import Band

ENERGY_WRITE = "energy_write_mj"
LATENCY_WRITE = "latency_write_exposed_s"
LIFETIME_INFERENCES = "lifetime_inferences"
LIFETIME_HOURS = "lifetime_hours"
LIFETIME_VERDICT = "lifetime_admissible"
_J_TO_MJ = 1e3
_S_PER_H = 3600.0
#: The evidence-class marker of a SECONDARY row (a published row read in a secondary source).
SECONDARY_MARKERS = ("SECONDARY", "secondary")


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

    @classmethod
    def of(cls, writes: Any) -> "WriteInputs":
        """The pricer's inputs from a plane's ``write_law.PlaneWrites``."""
        census = writes.census
        return cls(census=None if census is None else census.census, policy=writes.policy,
                   written_extents=None if census is None else census.written_extents,
                   compute_sealed=census is not None and census.compute_sealed,
                   unavailable=writes.unavailable,
                   most_written=None if census is None else census.most_written_core,
                   pass_core_types=None if census is None else census.pass_core_types)


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


def _zero(basis: str) -> Band:
    return Band(0.0, 0.0, 0.0, basis)


def _iterations(physics: PlatformPhysics) -> Optional[Band]:
    return physics.band("write_verify_iterations") if physics.has("write_verify_iterations") else None


def _write_energy(census: Any, physics: PlatformPhysics, k: int) -> Tuple[Optional[Band], str]:
    """(per-inference write energy in mJ, or None with the refusal reason)."""
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
        return walls, scale_band(step, pass_events([producer], policy)[0], "producer write steps"), ""
    if physics.has("t_program_per_byte") and not producer:
        per_byte = physics.band("t_program_per_byte")
        return [scale_band(per_byte, p.payload_bytes, "t_program_per_byte x payload_bytes")
                for p in passes], _zero("no producer write"), ""
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


def _life_band(endurance: Band, wear: CellWear, basis: str) -> Band:
    """endurance / wear corner-wise: the low lifetime at the high-wear end, the high at the low."""
    if wear.high is None or wear.exact:
        return scale_band(endurance, 1.0 / wear.low, basis)
    return Band(endurance.low / wear.high, endurance.nominal / wear.high, endurance.high / wear.low, basis)


def _lifetime(out: WritePricing, census: Any, physics: PlatformPhysics, policy: WritePolicy,
              most_written: Optional[int]) -> None:
    wear, unknown = cell_wear(census, policy, most_written)
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


def price_write_inputs(writes: WriteInputs, physics: PlatformPhysics) -> WritePricing:
    """``price_write_terms`` over a plane's inputs; a plane without a census refuses by name."""
    if writes.census is None:
        out = WritePricing()
        reason = writes.unavailable or "the plane states no period census"
        names = (ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES) + (
            (LIFETIME_VERDICT,) if writes.policy.min_lifetime_hours is not None else ())
        for name in names:
            out.refuse(name, reason)
        return out
    return price_write_terms(writes.census, physics, writes.policy,
                             written_extents=writes.written_extents, compute_sealed=writes.compute_sealed,
                             most_written=writes.most_written, pass_core_types=writes.pass_core_types)


def price_write_terms(census: Any, physics: PlatformPhysics, policy: WritePolicy, *,
                      written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                      compute_sealed: bool = True, most_written: Optional[int] = None,
                      pass_core_types: Any = None) -> WritePricing:
    """Write energy, exposed write time and lifetime of one period census at the declared K.

    ``most_written`` is the period writes of the most-written physical core (``WriteCensus.
    most_written_core``). A producer that places no core leaves it None only where the census
    provably rewrites; the lifetime is then an upper bound that can only prove inadmissibility.
    ``pass_core_types`` (``WriteCensus.pass_core_types``) gives the overlapped mode its per-type room."""
    out = WritePricing()
    energy, reason = _write_energy(census, physics, policy.batch_k)
    if energy is None:
        out.refuse(ENERGY_WRITE, reason)
    else:
        out.energy = out.term(ENERGY_WRITE, "mJ", energy)
    exposed, reason = _exposed_write(census, physics, policy, written_extents, compute_sealed, pass_core_types)
    if exposed is None:
        out.refuse(LATENCY_WRITE, reason)
    else:
        out.exposed = out.term(LATENCY_WRITE, "s", exposed)
    _lifetime(out, census, physics, policy, most_written)
    return out

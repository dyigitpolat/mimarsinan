"""The write policy a run declares (E192 keys) and the write-law quantities of a period census.

Every write quantity is presence-based on ``inference_batch_K``: a run that declares no batch has
no period, so none of them is emitted on either plane and every existing report is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.program_census import record_program_passes
from mimarsinan.deployment_record.build.write_census import (  # the core counts re-exported for their readers
    CoreType, ProgramPasses, WriteCensus, WriteCensusError, core_capacity_of, period_census,
    programmed_basis as programmed_basis, programmed_cores as programmed_cores, written_cores,
)
from mimarsinan.deployment_record.quantities.spec import QuantityValue
from mimarsinan.deployment_record.quantities import write_technology as wt
from mimarsinan.deployment_record.quantities.value_domain import domain_bytes_of, names_value_domain, value_window
from mimarsinan.mapping.support.period_census import WRITE_OVERLAP_MODES, natural

WRITE_KEYS = ("inference_batch_K", "write_overlap", "write_ports", "inference_rate_hz",
              "min_lifetime_hours")
#: A write unit read off a written core's used (rows, columns).
_EXTENT_UNITS = {"row": 0, "column": 1}


class WriteLawError(ToolchainRefusal, ValueError):
    """A write declaration the write law cannot price as declared."""


@dataclass(frozen=True)
class WritePolicy:
    """The run's write declarations, read off its resolved platform."""

    batch_k: int
    overlap_mode: str = "serial"
    shadow_capacity_fraction: float = 0.0
    write_ports: int = 1
    #: The granularity every core type declares (None: undeclared on every type).
    write_granularity: Optional[str] = None
    write_units_in_parallel: int = 1
    #: Why the core types' write declarations cannot be read as one, or None.
    mixed_core_writes: Optional[str] = None
    inference_rate_hz: Optional[float] = None
    min_lifetime_hours: Optional[float] = None
    #: The chip's declared cores per core type, sorted (the overlapped mode's shadow room is per type);
    #: None where a declared type states no geometry.
    core_capacity: Optional[Tuple[Tuple[CoreType, int], ...]] = None
    #: The chip's declared cores (every type); bounds the most-written core of a census that
    #: states no per-core writes.
    physical_cores: Optional[int] = None
    #: N2: where an operand core type's payloads come from (``host``: a transfer into the chip per inference,
    #: ``on_chip``: the producer's write-back) and their width; None without an operand core type.
    producer_write_source: Optional[str] = None
    payload_bits: Optional[int] = None
    #: N2b: per core geometry its write technology; None where every core type writes by one (the chip's one law).
    core_technologies: Optional[Tuple[wt.CoreTechnology, ...]] = None


def _common(cores: Sequence[Mapping[str, Any]], field: str, default: Any) -> Tuple[Any, Optional[str]]:
    declared = [ct.get(field, default) for ct in cores]
    values = set(declared)
    if len(values) > 1:
        return default, (f"the core types declare {field} {declared}: the chip states no one value, so its write "
                         "terms are priced per core type (write_technology.py)")
    return (values.pop() if values else default), None


def write_policy_of(platform: Mapping[str, Any]) -> Optional[WritePolicy]:
    """The declared write policy, or None when no ``inference_batch_K`` is declared."""
    batch = platform.get("inference_batch_K")
    if batch is None:
        return None
    overlap = dict(platform.get("write_overlap") or {})
    mode = str(overlap.get("mode", "serial"))
    if mode not in WRITE_OVERLAP_MODES:
        raise WriteLawError(f"write_overlap.mode {mode!r} is not one of {WRITE_OVERLAP_MODES}")
    cores = [dict(ct) for ct in platform.get("cores") or ()]
    granularity, mixed = _common(cores, "write_granularity", None)
    units, mixed_units = _common(cores, "write_units_in_parallel", 1)
    rate = platform.get("inference_rate_hz")
    floor = platform.get("min_lifetime_hours")
    capacity = core_capacity_of(platform)
    operand = next((dict(ct["operand"]) for ct in cores if ct.get("operand") is not None), None)
    return WritePolicy(
        batch_k=natural(batch, "inference_batch_K", error=WriteLawError),
        overlap_mode=mode,
        shadow_capacity_fraction=float(overlap.get("shadow_capacity_fraction", 0.0)),
        write_ports=natural(platform.get("write_ports", 1), "write_ports", error=WriteLawError),
        write_granularity=granularity,
        write_units_in_parallel=natural(units, "write_units_in_parallel", error=WriteLawError),
        mixed_core_writes=mixed or mixed_units,
        inference_rate_hz=None if rate is None else float(rate),
        min_lifetime_hours=None if floor is None else float(floor),
        core_capacity=None if capacity is None else tuple(sorted(capacity.items())),
        physical_cores=sum(int(ct.get("count", 1)) for ct in cores) if cores else None,
        producer_write_source=None if operand is None else str(operand.get("write_source", "host")),
        payload_bits=None if operand is None else int(operand["payload_bits"]),
        core_technologies=wt.core_technologies(platform),
    )


def pass_units(census: Any, policy: WritePolicy,
               written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]]) -> Optional[Tuple[int, ...]]:
    """Backing-store write units per pass at the declared granularity, or None when unstated."""
    granularity = policy.write_granularity
    if granularity is None or policy.mixed_core_writes is not None:
        return None
    if granularity == "cell":
        return tuple(p.written_cells_backing_store for p in census.passes)
    if granularity == "core":
        return tuple(p.cores - p.resident_cores for p in census.passes)
    if written_extents is None:
        return None
    axis = _EXTENT_UNITS[granularity]
    return tuple(sum(extent[axis] for extent in cores) for cores in written_extents)


def pass_events(units: Sequence[int], policy: WritePolicy) -> Tuple[int, ...]:
    """Write steps per pass: ``ceil(units / (write_units_in_parallel x write_ports))``."""
    lanes = policy.write_units_in_parallel * policy.write_ports
    return tuple(math.ceil(u / lanes) for u in units)


def placed_wear(batch_k: int, most_written: int, producer_slots: Sequence[Tuple[int, int]] = ()) -> float:
    """The most-written cell's writes per inference on a census that places every written core: the slot rule's
    max over physical cores of static writes per period / K plus producer writes per inference. Every written
    tile's used rectangle starts at its core's first row and column, so the core's most-written cell takes every
    write of the core."""
    static = natural(most_written, "most-written core writes", error=WriteLawError) / batch_k
    return max([static] + [s / batch_k + p for s, p in producer_slots])


def write_quantities(census: Any, policy: WritePolicy, *,
                     written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                     most_written: Optional[int] = None,
                     producer_slots: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, QuantityValue]:
    """The write-law quantities of a period census at the declared batch K (static facts); ``most_written`` and
    ``producer_slots`` are the slot rule's placement (``WriteCensus``), where the plane states it. The most-written
    cell's wear is the lifetime's (``cell_wear``), stated only where it has an upper end."""
    k = policy.batch_k
    producer = sum(p.written_cells_producer for p in census.passes)
    values = {
        "period_rewrite_bytes": float(sum(p.payload_bytes for p in census.passes)),
        "runtime_written_cells": sum(p.written_cells_backing_store for p in census.passes) / k + producer,
    }
    if most_written is not None:
        wear, _ = cell_wear(census, policy, most_written, producer_slots)
        if wear is not None and wear.high is not None:
            values["max_writes_per_cell_per_inference"] = wear.high
    units = pass_units(census, policy, written_extents)
    if units is not None and (producer == 0 or policy.write_granularity == "cell"):
        values["written_units"] = sum(units) / k + producer
        events = sum(pass_events(units, policy)) / k
        if producer:
            events += sum(pass_events([p.written_cells_producer for p in census.passes], policy))
        values["write_events"] = events
    return {key: QuantityValue(float(value), "static") for key, value in values.items()}


@dataclass(frozen=True)
class CellWear:
    """The most-written cell's writes per inference as a band: [low, high], high None when the
    census bounds it from below only."""

    low: float
    high: Optional[float]
    basis: str

    @property
    def exact(self) -> bool:
        return self.high == self.low


def cell_wear(census: Any, policy: WritePolicy, most_written: Optional[int],
              producer_slots: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[Optional[CellWear], Optional[str]]:
    """(the most-written cell's wear, or None with why it is unknown). A plane that places each rewritten core
    states its most-written core's period writes M, and with operand regions (N2) each core's producer writes per
    inference by the same slot rule (``producer_slots``): the wear is exact, ``placed_wear``. Producer cells on
    unplaced cores bound it from below only (one write per inference). A census that places no core (an untyped
    plane on a chip of several types whose tiles exceed every declared core, so it provably rewrites) bounds it
    below by pigeonhole, ceil(W / C) of W core writes on C declared cores: an upper-bound lifetime, which
    proves inadmissibility only."""
    k = policy.batch_k
    producer = any(p.written_cells_producer for p in census.passes)
    if most_written is not None and not producer:
        wear = placed_wear(k, most_written)
        return CellWear(wear, wear, f"most-written core: {most_written} writes per period / K"), None
    if most_written is not None and producer_slots is not None:
        wear = placed_wear(k, most_written, producer_slots)
        static, writes = max(producer_slots, key=lambda slot: slot[0] / k + slot[1])
        return CellWear(wear, wear, (f"one slot rule for static tiles and operand regions: the most-written "
                                     f"operand core takes {static} static writes per period / K and {writes} "
                                     f"producer writes per inference; the most-written core {most_written} writes "
                                     "per period / K")), None
    if most_written is not None:
        low = max(most_written / k, 1.0)
        return CellWear(low, None, (f"most-written core: {most_written} writes per period / K, and producer cells on "
                                    f"cores the census does not place: at least {low:g} writes per inference")), None
    period = written_cores(census)
    if period == 0:
        if not producer:
            return CellWear(0.0, 0.0, "no core is rewritten per period"), None
        return CellWear(1.0, None, ("no core is rewritten per period; producer cells on cores the census does not "
                                    "place are written at least once per inference")), None
    if not policy.physical_cores:
        return None, ("the census places no rewritten core on a physical core and the policy states "
                      "no physical core count, so the most-written cell is unknown")
    bound = math.ceil(period / policy.physical_cores)
    low = max(bound / k, 1.0 if producer else 0.0)
    basis = (f"upper bound: the census places no rewritten core, so the most-written core takes at least "
             f"ceil({period} / {policy.physical_cores}) = {bound} writes per period")
    return CellWear(low, None, basis + (", and at least one producer write per inference" if producer else "")), None


@dataclass(frozen=True)
class PlaneWrites:
    """A plane's write-law inputs: the declared policy and its period census, or why it has none."""

    policy: WritePolicy
    census: Optional[WriteCensus]
    unavailable: Optional[str] = None
    #: N2b: the census split by write technology, where the policy states several (``policy.core_technologies``).
    by_technology: Optional[wt.TechnologySplit] = None


def spans_match_carry(passes: ProgramPasses, carry: Optional[Mapping[str, Any]], timesteps: Optional[int]) -> bool:
    """Whether the per-boundary widths (one crossing per direction per carried wire) account for
    every train the plane's carry census counts; a census that counts more (a delayed slice's extra
    re-injection) states its totals, never a per-boundary split the spans cannot attribute."""
    if carry is None or timesteps is None:
        return True
    per = (int(timesteps) + 7) // 8
    out = sum(w for _, w, _ in passes.boundary_widths) * per
    inbound = sum(w for _, _, w in passes.boundary_widths) * per
    return (out, inbound) == (int(carry["boundary_out_bytes"]), int(carry["boundary_in_bytes"]))


def plane_write_census(passes: ProgramPasses, platform: Mapping[str, Any], *, weight_bits: Any,
                       timesteps: Optional[int], label: str, carry: Optional[Mapping[str, Any]] = None,
                       every_core: bool = False) -> Tuple[Optional[WriteCensus], Optional[str]]:
    """A plane's period census in its domain (the value rule's window under the value domain), or
    None with the reason it cannot be stated; ``every_core``: its deployment load (``period_census``)."""
    if weight_bits is None:
        return None, "the platform declares no weight_bits, so the rewritten payload has no size"
    rows = passes.rows
    if names_value_domain(platform):
        rows = tuple(replace(row, compute_steps=None if row.stage_latencies is None
                             else value_window(row.stage_latencies)) for row in rows)
    attributable = spans_match_carry(passes, carry, timesteps)
    try:
        return period_census(
            ProgramPasses(rows=rows, boundary_widths=passes.boundary_widths),
            weight_bits=int(weight_bits), capacity=core_capacity_of(platform),
            bytes_of=domain_bytes_of(platform, timesteps) if attributable else None,
            label=label if attributable else f"{label}; per-boundary carry withheld (the carry census "
                                             "counts trains the wire spans do not attribute)",
            every_core=every_core,
        ), None
    except WriteCensusError as refusal:
        return None, f"the {label} plane's period census refuses: {refusal}"


def plane_writes(passes: Optional[ProgramPasses], platform: Mapping[str, Any], *, weight_bits: Any,
                 timesteps: Optional[int], label: str, carry: Optional[Mapping[str, Any]] = None
                 ) -> Optional[PlaneWrites]:
    """A plane's write-law inputs, or None when the run declares no ``inference_batch_K``."""
    policy = write_policy_of(platform)
    if policy is None:
        return None
    if passes is None:
        return PlaneWrites(policy, None, f"the {label} plane states no pass structure, so it has no period census")
    if policy.payload_bits is not None and passes.operands_unstated is not None:
        return PlaneWrites(policy, None, f"the platform declares an operand core type and {passes.operands_unstated}"
                                         ", so its per-inference payload writes are unpriced: write terms refused")
    census, reason = plane_write_census(passes, platform, weight_bits=weight_bits, timesteps=timesteps,
                                        label=label, carry=carry)
    split = None if census is None or policy.core_technologies is None else wt.split_by_technology(
        passes, census, policy, weight_bits=int(weight_bits))
    return PlaneWrites(policy, census, reason, split)


def record_write_census(record: Any) -> Optional[PlaneWrites]:
    """The sealed record's write-law inputs, or None without a declared batch."""
    return plane_writes(record_program_passes(record), record.identity.platform,
                        weight_bits=record.utilization.crossbar.weight_bits,
                        timesteps=int(record.timing.s_global), label="record")

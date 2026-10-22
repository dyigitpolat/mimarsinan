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
from mimarsinan.deployment_record.build.write_census import (
    CoreType, ProgramPasses, WriteCensus, WriteCensusError, core_capacity_of, period_census,
)
from mimarsinan.deployment_record.quantities.spec import QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import (
    names_value_domain, value_bits, value_bytes, value_window,
)
from mimarsinan.mapping.support.period_census import WRITE_OVERLAP_MODES, natural
from mimarsinan.mapping.support.schedule.pass_cut import raster_bytes

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


def _common(cores: Sequence[Mapping[str, Any]], field: str, default: Any) -> Tuple[Any, Optional[str]]:
    declared = [ct.get(field, default) for ct in cores]
    values = set(declared)
    if len(values) > 1:
        return default, (f"the core types declare {field} {declared}; one write law per chip needs "
                         "one value (per-core-type write technology is Wave 2)")
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


def max_writes_per_cell(census: Any, batch_k: int, most_written: int) -> float:
    """The most-written cell's writes per inference: its physical core's period writes over K, plus
    one per inference where a producer writes cells. The census states no core for producer cells,
    so this is the high-wear end (they sit on the most-written core); the lifetime prices the band
    down to max(period writes / K, 1). Every written tile's used rectangle starts at its core's
    first row and column, so the core's most-written cell takes every write of the core."""
    producer = 1.0 if any(p.written_cells_producer for p in census.passes) else 0.0
    return natural(most_written, "most-written core writes", error=WriteLawError) / batch_k + producer


def written_cores(census: Any) -> int:
    """W: the core writes of one period (every rewritten core of every pass)."""
    return sum(p.cores - p.resident_cores for p in census.passes if p.written_cells_backing_store)


def write_quantities(census: Any, policy: WritePolicy, *,
                     written_extents: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                     most_written: Optional[int] = None) -> Dict[str, QuantityValue]:
    """The write-law quantities of a period census at the declared batch K (static facts);
    ``most_written`` is the period writes of the most-written physical core, where the plane states it."""
    k = policy.batch_k
    producer = sum(p.written_cells_producer for p in census.passes)
    values = {
        "period_rewrite_bytes": float(sum(p.payload_bytes for p in census.passes)),
        "runtime_written_cells": sum(p.written_cells_backing_store for p in census.passes) / k + producer,
    }
    if most_written is not None:
        values["max_writes_per_cell_per_inference"] = max_writes_per_cell(census, k, most_written)
    units = pass_units(census, policy, written_extents)
    if units is not None and (producer == 0 or policy.write_granularity == "cell"):
        values["written_units"] = sum(units) / k + producer
        events = sum(pass_events(units, policy)) / k
        if producer:
            events += sum(pass_events([producer], policy))
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


def cell_wear(census: Any, policy: WritePolicy,
              most_written: Optional[int]) -> Tuple[Optional[CellWear], Optional[str]]:
    """(the most-written cell's wear, or None with why it is unknown).

    A plane that places each rewritten core states its most-written core's period writes M. Producer
    cells sit on a core the census does not state, so with them the wear is a band from
    max(M / K, 1) (another core) to M / K + 1 (the most-written core). A census that places no core
    (an untyped plane on a chip of several types whose tiles exceed every declared core, so it
    provably rewrites) bounds the wear from below by pigeonhole, ceil(W / C) of W core writes on the
    chip's C declared cores: its lifetime is an upper bound, which proves inadmissibility only."""
    k = policy.batch_k
    producer = any(p.written_cells_producer for p in census.passes)
    if most_written is not None:
        high = max_writes_per_cell(census, k, most_written)
        if not producer:
            return CellWear(high, high, f"most-written core: {most_written} writes per period / K"), None
        low = max(most_written / k, 1.0)
        return CellWear(low, high, (f"most-written core: {most_written} writes per period / K, and producer "
                                    f"cells on an unstated core: a bound between {low:g} and {high:g} writes "
                                    "per inference")), None
    period = written_cores(census)
    if period == 0:
        wear = 1.0 if producer else 0.0
        return CellWear(wear, wear, "no core is rewritten per period" + (
            "; producer cells are written once per inference" if producer else "")), None
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


def domain_bytes_of(platform: Mapping[str, Any], timesteps: Optional[int]) -> Optional[Any]:
    """How a boundary's carried elements are sized in this run's domain, or None when unknown:
    values at the declared grid, else the dense spike raster at the window."""
    if names_value_domain(platform):
        bits = value_bits(platform)
        return None if bits is None else (lambda elements: value_bytes(elements, bits))
    return None if timesteps is None else (lambda elements: raster_bytes(elements, int(timesteps)))


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
                       timesteps: Optional[int], label: str, carry: Optional[Mapping[str, Any]] = None
                       ) -> Tuple[Optional[WriteCensus], Optional[str]]:
    """A plane's period census in its domain (the value rule's window under the value domain), or
    None with the reason it cannot be stated."""
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
    census, reason = plane_write_census(passes, platform, weight_bits=weight_bits, timesteps=timesteps,
                                        label=label, carry=carry)
    return PlaneWrites(policy, census, reason)


def record_write_census(record: Any) -> Optional[PlaneWrites]:
    """The sealed record's write-law inputs, or None without a declared batch."""
    return plane_writes(record_program_passes(record), record.identity.platform,
                        weight_bits=record.utilization.crossbar.weight_bits,
                        timesteps=int(record.timing.s_global), label="record")

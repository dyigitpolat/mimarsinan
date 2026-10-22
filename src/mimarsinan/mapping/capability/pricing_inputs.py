"""What a level's census hands the pricer: the ``Quantities`` field map and the per-pass period census."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from mimarsinan.mapping.capability.census import LevelCensus, level_census, value_bytes
from mimarsinan.mapping.capability.ladder import ON_CHIP_WRITE_BACK, CapabilityLevel, capability_level
from mimarsinan.mapping.capability.mac_split import MacCounters, mac_statement
from mimarsinan.mapping.capability.requirements import LANE_CLASSES
from mimarsinan.mapping.ir import IRGraph

#: Catalog names (``deployment_record/quantities/spec.py``) the field map emits; no deployment_record import here.
FIELD_MAP_FIELDS: Tuple[str, ...] = (
    "host_macs", "onchip_macs", "compute_op_count", "macs", "cells_used", "synaptic_events", "latency_steps",
    "segment_cores", "pass_count", "sync_count", "carry_in_bytes", "carry_out_bytes", "lane_ops", "lanes_per_tile",
)
#: Field-map names not yet in the catalog: none since integration, where G3's lane rows (plan §7.1) added
#: ``lane_ops`` and ``lanes_per_tile``; CL4 prices them.
PENDING_CATALOG_FIELDS: Tuple[str, ...] = ()


def _bytes(row: LevelCensus, elements: int) -> int:
    return value_bytes(elements, row.value_bits)


def level_quantities(
    row: LevelCensus, *, counters: Optional[MacCounters] = None, lanes_per_tile: Optional[int] = None,
) -> Dict[str, int]:
    """The level's quantities by catalog name; a field nobody produced, or one withheld, is absent, never 0.

    ``host_macs``/``onchip_macs`` need the member's MAC counters and appear only where ``mac_statement`` backs them
    (its ``withheld`` names the reason otherwise); ``lanes_per_tile`` is the declared tile's lane count, 0 at a
    level that declares no lane class. ``synaptic_events`` are value-domain MAC events: every committed cell once
    per inference plus the on-chip operators' MACs. Carry excludes the image entry and the logit readout, which
    every placement pays.
    """
    statement = mac_statement(row, counters)
    values = {
        "compute_op_count": row.host_ops,
        "macs": row.cells_committed,
        "cells_used": row.cells_committed,
        "latency_steps": row.latency_steps,
        "segment_cores": row.neural_cores,
        "pass_count": row.neural_segments,
        "sync_count": row.host_segments,
        "carry_in_bytes": _bytes(row, row.boundary.carry_in),
        "carry_out_bytes": _bytes(row, row.boundary.carry_out),
        "lane_ops": row.lane_ops,
    }
    for name in ("host_macs", "onchip_macs", "synaptic_events"):
        value = getattr(statement, name)
        if value is not None:
            values[name] = int(value)
    if lanes_per_tile is not None:
        declares_lanes = bool(capability_level(row.level).classes & LANE_CLASSES)
        values["lanes_per_tile"] = int(lanes_per_tile) if declares_lanes else 0
    return values


@dataclass(frozen=True)
class PassWrites:
    """One pass (an in-order neural segment) of the level, field for field S0's ``PassCensus``.

    S0's type (``mapping/support/period_census.py``) is not staged on this tree; at staging the adapter is
    ``PassCensus(**asdict(p))``. Every pass is resident (the unbounded-core identity program keeps its weights), so
    nothing is reloaded from the backing store and nothing is programmed per period.
    """

    position: int
    segment_index: int
    resident: bool
    cores: int
    resident_cores: int
    written_cells_backing_store: int
    written_cells_producer: int
    payload_bytes: int
    compute_steps: int


@dataclass(frozen=True)
class LevelPeriodCensus:
    """Every pass of one inference at a level, in program order, mirroring S0's ``PassPeriodCensus``.

    ``boundaries`` holds intra-segment pass boundaries only; a level's host-op boundaries cross segments, so their
    carry is in the field map (``carry_in_bytes``, ``carry_out_bytes``).
    """

    passes: Tuple[PassWrites, ...]
    boundaries: Tuple[object, ...] = ()
    label: str = ""
    value_bits: int = 8

    @property
    def written_cells_producer(self) -> int:
        return sum(p.written_cells_producer for p in self.passes)

    @property
    def written_cells_backing_store(self) -> int:
        return sum(p.written_cells_backing_store for p in self.passes)

    @property
    def producer_payload_bytes(self) -> int:
        """The producer-sourced writes per inference at the census value width."""
        return value_bytes(self.written_cells_producer, self.value_bits)


def period_census_of(row: LevelCensus) -> LevelPeriodCensus:
    """The period census of a placed level: one resident pass per in-order neural segment, D payloads as
    producer-sourced writes per inference sized from each on-chip dynamic product's stationary operand."""
    passes = tuple(
        PassWrites(
            position=index, segment_index=index, resident=True, cores=s.cores, resident_cores=s.cores,
            written_cells_backing_store=0, written_cells_producer=s.written_cells_producer, payload_bytes=0,
            compute_steps=s.latency_steps,
        )
        for index, s in enumerate(row.segments)
    )
    label = row.level if row.payload_write is None else f"{row.level} {row.payload_write}"
    return LevelPeriodCensus(passes=passes, label=label, value_bits=row.value_bits)


def level_period_census(
    member_ir: IRGraph, level: CapabilityLevel, *, value_bits: int, payload_write: str = ON_CHIP_WRITE_BACK,
) -> LevelPeriodCensus:
    """Plan §11.1's ``level_period_census(member_ir, level)``: the IR placed at ``level``, then its period census."""
    return period_census_of(level_census(member_ir, level, value_bits=value_bits, payload_write=payload_write))

"""What a level's census hands the pricer: the ``Quantities`` field map and the per-pass period census."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from mimarsinan.mapping.capability.census import LevelCensus, level_census, value_bytes
from mimarsinan.mapping.capability.ladder import ON_CHIP_WRITE_BACK, CapabilityLevel, capability_level
from mimarsinan.mapping.capability.mac_split import MacCounters, mac_statement
from mimarsinan.mapping.capability.requirements import LANE_CLASSES
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus, rotation_rewrites

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


#: The rule every level period census follows (finding 4 (a), ``period_census.rotation_rewrites``), in its label.
ROTATION_LABEL = (
    "executed rotation: each neural segment is one execution stage loaded from its core type's first core, so with "
    "two or more writing segments every core is rewritten once per period (no residency plan pins a level's "
    "requirement placement), and a lone writing segment is loaded once")


def period_census_of(row: LevelCensus, *, bytes_of_cells: Callable[[int], int]) -> PassPeriodCensus:
    """S0's period census of a placed level by the executed rotation: one pass per in-order neural segment (the
    unbounded-core identity program), every core of every pass rewritten from the backing store once per period when
    two or more passes write (a pass writes when it holds a core), nothing rewritten for a lone writing pass; D
    payloads are producer-sourced writes per inference sized from each on-chip dynamic product's stationary operand.
    ``bytes_of_cells`` prices one core's cells (the caller's weight width, ``params_bytes`` on the pricing side)."""
    rotates = rotation_rewrites(sum(1 for s in row.segments if s.cores))
    passes = tuple(
        PassCensus(
            position=index, segment_index=index, resident=not (rotates and s.cores), cores=s.cores,
            resident_cores=0 if rotates else s.cores,
            written_cells_backing_store=s.cells_committed if rotates else 0,
            written_cells_producer=s.written_cells_producer,
            payload_bytes=sum(int(bytes_of_cells(c)) for c in s.core_cells) if rotates else 0,
            compute_steps=s.latency_steps,
        )
        for index, s in enumerate(row.segments)
    )
    level = row.level if row.payload_write is None else f"{row.level} {row.payload_write}"
    return PassPeriodCensus(passes=passes, label=f"{level}; {ROTATION_LABEL}")


def level_period_census(
    member_ir: IRGraph, level: CapabilityLevel, *, value_bits: int, bytes_of_cells: Callable[[int], int],
    payload_write: str = ON_CHIP_WRITE_BACK,
) -> PassPeriodCensus:
    """Plan §11.1's ``level_period_census(member_ir, level)``: the IR placed at ``level``, then its period census by
    the executed rotation (``period_census_of``)."""
    return period_census_of(level_census(member_ir, level, value_bits=value_bits, payload_write=payload_write),
                            bytes_of_cells=bytes_of_cells)

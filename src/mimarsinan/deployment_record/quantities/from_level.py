"""Quantities of a capability level's requirement placement on a declared chip (CL4): the level plane.

The requirement census (``mapping/capability``) states a level's field map (``level_quantities``) and, on a core
budget, its schedule (``level_schedule``: passes, tiles, cut widths). This view joins them with the declared chip so
``price_absolute`` prices a level as it prices a candidate or a record:

- the period census comes from G3's one rule through the seam both other planes use (``write_law.plane_writes``,
  the executed rotation of finding 4 (a)): every tile of a pass is one physical core of the chip's one core type,
  and where the program fits its budget the residency floor marks every tile kept (programmed once, 0 B per
  inference); a dynamic product's stationary operand joins it as producer-sourced cells per inference, on its
  producer-written tiles (cores of no static cells), which N2's one slot rule places as on the other planes;
- a value-domain target reads the value rule: one evaluation per stage (``value_window``), MAC events = committed
  cells plus on-chip operator MACs, carry at the value grid (``activation_bits``);
- a spiking target's events need a measured record's switching activity, which no census member has, so
  ``synaptic_events`` is withheld by name and the level is priced on the time axis only, at its declared LIF window
  (``chip_simulation.stage_timesteps``: T plus the stage span plus input delivery, per stage).

The level places no core on the chip's mesh, so the traffic between its cores (value byte-hops, spike packets and
hops) is withheld by name: a term that prices it is named unpriced, never dropped. Above C0 every census field rests
on the requirement placement (no lowering yet), so its provenance is ``modeled``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.stage_timesteps import executed_stage_timesteps
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS, Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import (
    NO_VALUE_GRID,
    names_value_domain,
    refuse_spiking_activity,
    value_window,
)
from mimarsinan.deployment_record.quantities.write_law import (
    PlaneWrites, domain_bytes_of, plane_writes, summed_cells, write_quantities,
)
from mimarsinan.mapping.capability.census import LevelCensus
from mimarsinan.mapping.capability.level_schedule import LevelSchedule
from mimarsinan.mapping.capability.mac_split import MacCounters, mac_statement
from mimarsinan.mapping.capability.pricing_inputs import level_quantities
from mimarsinan.mapping.platform.weight_cells import array_extent
from mimarsinan.mapping.support.period_census import natural

#: Field-map quantities that hold as stated whatever the budget (``mapping/capability/pricing_inputs.py``).
CENSUS_FIELDS = ("host_macs", "onchip_macs", "compute_op_count", "macs", "cells_used", "synaptic_events", "lane_ops",
                 "lanes_per_tile")
#: Why a spiking target's level has no event census.
SPIKING_EVENTS_WITHHELD = (
    "a spiking target's synaptic events need a measured record's activity_factor, and no record of this census "
    "member exists: the level is priced on the time axis only")
#: Why the traffic between the level's cores has no size.
NO_LEVEL_MESH = "the level census places no core on the chip's mesh, so the traffic between its cores has no size"
_VALUE_MESH = ("noc_value_bytes",)
_SPIKE_MESH = ("noc_total_packets", "noc_intra_tile_packets", "noc_inter_tile_packets", "noc_total_hops")


class LevelViewError(ToolchainRefusal, ValueError):
    """A level view without the facts it prices: a chip of several core types, a schedule of another level, or a
    spiking target without a declared window."""


@dataclass(frozen=True)
class LevelView:
    """The priced level's quantities, its write-law inputs (None without ``inference_batch_K``) and its chip."""

    quantities: Quantities
    writes: Optional[PlaneWrites]
    chip: Mapping[str, Any]
    label: str


def level_chip(platform: Mapping[str, Any], budget: int) -> Dict[str, Any]:
    """The platform with ``budget`` cores of its one declared core type: the chip the level is priced on."""
    cores = [dict(core) for core in platform.get("cores") or ()]
    if len(cores) != 1:
        raise LevelViewError(f"a level is priced on one declared core type, the platform declares {len(cores)}")
    return {**platform, "cores": [{**cores[0], "count": natural(budget, "core budget", error=LevelViewError)}]}


def chip_facts(chip: Mapping[str, Any]) -> Dict[str, QuantityValue]:
    """The declared chip's static facts (one core per tile: a requirement placement has no floorplan)."""
    (core,) = chip["cores"]
    (axons, neurons), count = array_extent(core), int(core["count"])  # [U-C4] the array's cells, rows and columns
    facts = {"cores_physical": count, "tiles": count, "axons_physical": count * axons,
             "neurons_physical": count * neurons, "cells_physical": count * axons * neurons}
    if chip.get("weight_bits") is not None:
        facts["weight_bits"] = int(chip["weight_bits"])
    return {key: QuantityValue(float(value), "static") for key, value in facts.items()}


def _stage_steps(chip: Mapping[str, Any], timesteps: Optional[int]):
    if names_value_domain(chip):
        return lambda span: value_window([span])
    if timesteps is None:
        raise LevelViewError("a spiking target prices its time axis from a declared window (timesteps)")
    window = natural(timesteps, "timesteps", error=LevelViewError)
    return lambda span: executed_stage_timesteps(timesteps=window, max_latency=span, is_cycle=False,
                                                 is_cascade=False)


def level_program_passes(schedule: LevelSchedule, chip: Mapping[str, Any],
                         timesteps: Optional[int] = None) -> ProgramPasses:
    """The schedule as a plane's pass rows: one row per pass, one stage each, every tile kept where it is pinned. A
    pass's producer-written tiles (a dynamic product's stationary operand) follow its static tiles as cores of no
    static cells that hold its producer cells (N2's ``PassRow.producer_cells`` and ``producer_cores``), so the period
    census places them by the one slot rule, as on the mapping and candidate planes [integration of CL4 with N2]."""
    steps = _stage_steps(chip, timesteps)
    pinned = schedule.pinned  # once: the property sums every pass's tiles
    rows = tuple(PassRow(
        segment_index=p.segment_index, resident=False, core_cells=p.tile_cells + (0,) * p.producer_tiles,
        marks=frozenset(range(p.tiles)) if pinned else frozenset(),
        core_extents=None if p.tile_extents is None else p.tile_extents + ((0, 0),) * p.producer_tiles,
        compute_steps=steps(p.stage_latency), stage_latencies=(p.stage_latency,),
        producer_cells=p.producer_cells, producer_cores=frozenset(range(len(p.tile_cells), p.tiles)),
    ) for p in schedule.passes)
    return ProgramPasses(rows=rows, boundary_widths=schedule.boundary_widths)


def from_level(row: LevelCensus, schedule: LevelSchedule, platform: Mapping[str, Any], *,
               counters: Optional[MacCounters], lanes_per_tile: int, timesteps: Optional[int] = None) -> LevelView:
    """The level ``row`` on ``schedule.budget`` cores of ``platform``, its schedule ``schedule``.

    ``counters`` are the member's MAC counters (``mac_statement`` backs the MAC fields or names why not);
    ``lanes_per_tile`` is the declared tile's lane count (0 is stated at a level without a lane class);
    ``timesteps`` is a spiking target's declared LIF window, unused under the value rule."""
    if (schedule.level, schedule.payload_write) != (row.level, row.payload_write):
        raise LevelViewError(f"the schedule places {schedule.level} {schedule.payload_write}, the census row "
                             f"{row.level} {row.payload_write}")
    chip = level_chip(platform, schedule.budget)
    value_domain = names_value_domain(chip)
    if value_domain:
        refuse_spiking_activity(chip)
    provenance = "static" if row.label is None else "modeled"
    field_map = level_quantities(row, counters=counters, lanes_per_tile=lanes_per_tile)
    values: Dict[str, QuantityValue] = {key: QuantityValue(float(field_map[key]), provenance)
                                        for key in CENSUS_FIELDS if key in field_map}
    if "host_macs" in field_map and "onchip_macs" in field_map:
        values["total_macs"] = QuantityValue(float(field_map["host_macs"] + field_map["onchip_macs"]), provenance)
    values.update(chip_facts(chip))
    withheld = {key: reason for key, reason in mac_statement(row, counters).withheld.items() if key in QUANTITY_SPECS}
    passes = level_program_passes(schedule, chip, timesteps)
    weight_bits = int(chip["weight_bits"])
    stated: Dict[str, float] = {
        "pass_count": len(schedule.passes), "segment_cores": schedule.tiles,
        "cores_allocated": schedule.tiles, "sync_count": row.host_segments + len(schedule.boundary_widths),
        "reprogram_passes": sum(1 for p in schedule.passes if p.tile_cells),
        "reprogrammed_cores": sum(len(p.tile_cells) for p in schedule.passes),
        "reprogrammed_bytes": sum(params_bytes(c, weight_bits) for p in schedule.passes for c in p.tile_cells),
        "latency_steps": sum(int(r.compute_steps or 0) for r in passes.rows),
    }
    bytes_of = domain_bytes_of(chip, None if value_domain else timesteps)
    out = row.boundary.carry_out + sum(w for _, w, _ in schedule.boundary_widths)
    inbound = row.boundary.carry_in + sum(w for _, _, w in schedule.boundary_widths)
    for key, elements in (("carry_out_bytes", out), ("carry_in_bytes", inbound)):
        if bytes_of is not None:
            stated[key] = float(bytes_of(elements))
        elif elements:
            withheld[key] = NO_VALUE_GRID
        else:
            stated[key] = 0.0
    if value_domain:
        stated["timesteps"] = 1
        if schedule.tiles > 1:
            withheld.update(dict.fromkeys(_VALUE_MESH, NO_LEVEL_MESH))
    else:
        stated["timesteps"] = natural(timesteps, "timesteps", error=LevelViewError)
        values.pop("synaptic_events", None)
        withheld["synaptic_events"] = SPIKING_EVENTS_WITHHELD
        withheld.update(dict.fromkeys(_SPIKE_MESH, NO_LEVEL_MESH))
    values.update({key: QuantityValue(float(value), provenance) for key, value in stated.items()})
    label = f"level plane: {schedule.label}"
    writes = plane_writes(passes, chip, weight_bits=weight_bits, timesteps=None if value_domain else timesteps,
                          label=label)
    if writes is not None and writes.census is not None:  # [U-C4 targeted M2] the one rule (one core type here)
        quantities, held = summed_cells(writes, write_quantities(
            writes.census.census, writes.policy, written_extents=writes.census.written_extents,
            most_written=writes.census.most_written_core, producer_slots=writes.census.producer_slots))
        values.update(quantities)
        withheld.update(held)
    return LevelView(Quantities(values, withheld={k: v for k, v in withheld.items() if k not in values}),
                     writes, chip, label)


def lane_area_quantities(chip: Mapping[str, Any], lanes_per_tile: int) -> Quantities:
    """The chip's static facts with ``lanes_per_tile`` lanes on every tile: what its area reads."""
    values = chip_facts(chip)
    values["lanes_per_tile"] = QuantityValue(float(natural(lanes_per_tile, "lanes_per_tile")), "static")
    return Quantities(values)


def pass_rows(view: LevelView) -> Tuple[Any, ...]:
    """The period census's passes (empty without a write policy): what a report prints per pass."""
    if view.writes is None or view.writes.census is None:
        return ()
    return view.writes.census.census.passes

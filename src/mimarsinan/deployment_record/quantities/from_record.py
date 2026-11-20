"""Quantities of a sealed record: the measured plane, absent where nothing sealed."""

from __future__ import annotations

from typing import Dict, Optional

from mimarsinan.chip_simulation.hybrid_run.stage_timing import host_wall
from mimarsinan.chip_simulation.stage_timesteps import neuron_cycles_per_window
from mimarsinan.deployment_record.quantities.lanes import lane_block_quantities
from mimarsinan.deployment_record.quantities.planes.local_memory import record_local_memory
from mimarsinan.deployment_record.quantities.planes.states import RECORD_UNPRODUCED, declared_zeros, plane_view
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import (
    BUFFER_CAPACITY,
    ONCHIP_CARRY,
    ONCHIP_CENSUS,
    apply_value_rule,
    declared_buffer_bytes,
    names_value_domain,
    onchip_carry,
    record_carry_census,
    record_value_facts,
    refuse_spiking_activity,
)
from mimarsinan.deployment_record.quantities.write_law import (
    record_write_census,
    summed_cells,
    write_quantities,
)
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.deployment_record.build.program_census import declared_core_geometries
from mimarsinan.mapping.packing.residency_tiles import declared_tiles
from mimarsinan.mapping.platform.weight_cells import array_totals
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS


def _measured(value: float) -> QuantityValue:
    return QuantityValue(value=float(value), provenance="measured")


def _put(values: Dict[str, QuantityValue], key: str, value: Optional[float],
         provenance: str = "measured") -> None:
    """Attach ``key`` unless its source is absent — absence survives, never a zero."""
    if value is None:
        return
    values[key] = QuantityValue(value=float(value), provenance=provenance)


def _lane_quantities(values: Dict[str, QuantityValue], withheld: Dict[str, str], record: DeploymentRecord) -> None:
    """[W08] A chip declaring ``onchip_arithmetic`` prices its lane block (``lanes.lane_block_quantities``, the
    candidate plane's rule too): the lane work the deployed program executed (0 where its lanes do not run) and lanes
    per tile, or lanes per tile withheld by name where the lanes carry no lane count."""
    stage = record.output_stage
    if stage is None:
        return
    executed = stage.get("label") == "executed"
    work = sum(int(n) for n in dict(stage["deployed"]["lane_ops"]).values()) if executed else 0
    lane_values, lane_withheld = lane_block_quantities(
        classes=tuple(stage.get("classes") or ()), lanes_per_core=stage.get("lanes_per_core"),
        cores_per_tile=int(record.identity.platform.get("cores_per_tile_resolved") or 1), lane_ops=work,
        provenance="measured" if executed else "static")
    values.update(lane_values)
    withheld.update(lane_withheld)


def from_record(record: DeploymentRecord) -> Quantities:
    """Every multiplicand the sealed record carries, as measured quantities.

    Deliberately unclaimed: the ``adc_*`` pair (produced by C6 conversion
    models). ``synaptic_events`` is claimed from the sealed arrival census
    [H1]; records sealed before it stay honestly absent.
    """
    values: Dict[str, QuantityValue] = {}

    crossbar = record.utilization.crossbar
    # [U-F4] a wavefront head's dark cores are loaded, never evaluated: the occupancy and MAC census leave them out
    dark = [s.cores[i] for s in record.schedule.segments() for i in (s.dark_core_indices or ())]
    values["cells_used"] = _measured(crossbar.cells_used - sum(c.cells_used for c in dark))
    values["cores_allocated"] = _measured(crossbar.cores_allocated)
    values["neurons_used"] = _measured(crossbar.neurons_used - sum(c.neurons_used for c in dark))
    values["axons_used"] = _measured(crossbar.axons_used - sum(c.axons_used for c in dark))
    values["macs"] = _measured(crossbar.macs - sum(c.cells_used for c in dark))
    # The *_physical totals are the DECLARED chip's — area and static power price
    # the chip as declared, never just the cores this mapping happened to allocate.
    cores = list(record.identity.platform.get("cores") or ())
    # A declaration of the run, not a measurement of the execution. [WI-5g] On a chip whose core types carry
    # different widths no one width prices a cell: the quantity is withheld by name (each per-bit term says so).
    widths = sorted({int(core[CORE_WEIGHT_BITS]) for core in cores if core.get(CORE_WEIGHT_BITS) is not None})
    typed_withheld: Dict[str, str] = {}
    if len(widths) > 1:
        typed_withheld = {"weight_bits": f"the core types carry weight widths {widths}: no one width prices a cell "
                                         f"(per-type per-bit pricing is not built, WI-5g)"}
    else:
        _put(values, "weight_bits", crossbar.weight_bits, provenance="static")
    if cores:
        # [U-C4] The declared chip's array: its rows, columns and cells (a weight of several cells counts each).
        axons, neurons, cells = array_totals(cores)
        values["cores_physical"] = QuantityValue(float(sum(int(core.get("count", 1)) for core in cores)), "static")
        values["axons_physical"] = QuantityValue(float(axons), "static")
        values["neurons_physical"] = QuantityValue(float(neurons), "static")
        values["cells_physical"] = QuantityValue(float(cells), "static")

    partition = record.utilization.partition
    if partition is not None:
        values["onchip_params"] = _measured(partition.onchip_params)
        values["host_params"] = _measured(partition.host_params)
        values["total_params"] = _measured(partition.total_params)
        values["onchip_macs"] = _measured(partition.onchip_macs)
        values["host_macs"] = _measured(partition.host_macs)
        values["total_macs"] = _measured(partition.total_macs)

    schedule = record.schedule
    values["pass_count"] = _measured(schedule.pass_count)
    values["compute_op_count"] = _measured(schedule.compute_op_count)
    values["sync_count"] = _measured(schedule.sync_count)
    values["reprogram_passes"] = _measured(schedule.reprogram_passes)
    # [H2] The schedule is sealed either way, so "nothing crosses" is a KNOWN
    # zero (a structural fact), not an absence — the search minimizes these
    # axes, and a single-pass program must score 0, never refuse. [E3] The
    # directional figures are priced by different constants than the buffer
    # figures, which size what was held in between.
    carry = schedule.carry
    values["carried_raster_bytes"] = QuantityValue(
        float(0 if carry is None else carry.carried_bytes), "static")
    values["carry_peak_live_bytes"] = QuantityValue(
        float(0 if carry is None else carry.peak_live_bytes), "static")
    held, refusal = onchip_carry(record_carry_census(schedule), record.identity.platform)  # [U-F4]
    assert held is not None
    values["carry_out_bytes"] = QuantityValue(float(held["boundary_out_bytes"]), "static")
    values["carry_in_bytes"] = QuantityValue(float(held["boundary_in_bytes"]), "static")
    if refusal is not None:
        typed_withheld[ONCHIP_CARRY] = refusal
    elif ONCHIP_CENSUS in held:
        values[ONCHIP_CARRY] = QuantityValue(float(held[ONCHIP_CENSUS]), "static")
    _put(values, BUFFER_CAPACITY, declared_buffer_bytes(record.identity.platform), provenance="static")
    segments = list(schedule.segments())
    reprogrammed = [s for s in segments if s.programming == "reprogram"]
    counts_withheld = _physical_core_counts(values, record, segments, reprogrammed)
    values["reprogrammed_bytes"] = _measured(sum(s.params_bytes for s in reprogrammed))
    values["connectivity_entries"] = _measured(
        sum(s.connectivity_entries for s in reprogrammed)
    )

    # [R5-gate finding] The chip's tile count is a DECLARATION (the resolved
    # floorplan), not the placement fragment's — which under-describes
    # per-stage runs (it seals one stage's arch; the R3 note). The gate caught
    # the fork as exactly one tile's router+fixed area on the component-priced
    # profiles. Fall back to the fragment only when nothing was resolved.
    platform = record.identity.platform
    rows = int(platform.get("tile_grid_rows_resolved", 0) or 0)
    cols = int(platform.get("tile_grid_cols_resolved", 0) or 0)
    values["tiles"] = (
        QuantityValue(float(rows * cols), "static") if rows > 0 and cols > 0
        else _measured(len(record.placement.tiles))
    )

    values["timesteps"] = _measured(record.timing.s_global)
    values["neuron_cycles_per_window"] = _measured(neuron_cycles_per_window(  # [U-LIF2 W12] the run's window firing
        record.timing.s_global, getattr(record.identity, "window_firing", None), neurons=values["neurons_used"].value,
        window_arrivals=sum(int(s.window_arrivals or 0) for s in segments)))  # [U-LIF2 r6] its arrivals priced
    values["latency_steps"] = _measured(record.timing.latency.compute_steps)
    # [U-HOST] the one per-inference host wall (a diagnostic of the development machine: no pricer reads it)
    wall = host_wall(schedule.stages)
    _put(values, "host_ops_s", wall.per_inference_s)

    if record.energy is not None:
        _put(values, "total_spikes", record.energy.total_spikes)  # absent on an event-census-only fragment
        _put(values, "synaptic_events", record.energy.synaptic_events,
             provenance="measured")

    traffic = record.traffic
    if traffic is not None and traffic.boundaries is not None:
        values["boundary_events"] = _measured(
            sum(boundary.total_count for boundary in traffic.boundaries)
        )
    if traffic is not None and traffic.noc is not None:
        noc = traffic.noc
        values["noc_total_packets"] = _measured(noc.total_packets)
        values["noc_intra_tile_packets"] = _measured(noc.intra_tile_packets)
        values["noc_inter_tile_packets"] = _measured(noc.inter_tile_packets)
        # One packet crossing three links is three hops: the link loads ARE the
        # traversal census, so the total derives instead of being re-sealed.
        values["noc_total_hops"] = _measured(
            sum(link.packet_count for link in noc.link_loads)
        )

    withheld: Dict[str, str] = dict(typed_withheld, **({"host_ops_s": wall.refusal} if wall.refusal else {}))
    _lane_quantities(values, withheld, record)
    value_domain = names_value_domain(platform)
    if value_domain:
        refuse_spiking_activity(platform)
        withheld.update(apply_value_rule(values, record_value_facts(record)))
    withheld.update(record_local_memory(values, platform, value_domain))  # [U-ENERGY class 3]
    writes = record_write_census(record)
    if writes is not None and writes.census is not None:  # [U-C4 targeted M2] each core type's cells
        quantities, held = summed_cells(writes, write_quantities(
            writes.census.census, writes.policy, written_extents=writes.census.written_extents,
            most_written=writes.census.most_written_core, producer_slots=writes.census.producer_slots))
        values.update(quantities)
        withheld.update(held)
    zeros = declared_zeros(record.output_stage is not None, declared_buffer_bytes(platform) is not None, value_domain)
    return plane_view(values, {**counts_withheld, **withheld}, RECORD_UNPRODUCED, zeros)  # [U-ENERGY class 2]


_CORE_COUNTS = ("segment_cores", "reprogrammed_cores", "cores_allocated")


def _physical_core_counts(values: Dict[str, QuantityValue], record: DeploymentRecord, segments, reprogrammed
                          ) -> Dict[str, str]:
    """The core counts in physical tiles: a sealed core of a fused geometry (several declared cores of one type,
    its axons summed) counts its members, read against the declared core types. ``cores_allocated`` is the sealed
    allocated crossbars plus every fused core's further members. A geometry the declared types do not determine
    withholds the three counts by name; a platform that declares no core geometry counts each core once."""
    declared = declared_core_geometries(record.identity.platform)
    cores = [c for s in segments for c in s.cores]
    tiles = {id(c): 1 if not declared else len(declared_tiles(c.axons, c.neurons, declared) or ()) for c in cores}
    undetermined = sorted({(int(c.axons), int(c.neurons)) for c in cores if tiles[id(c)] == 0})
    if undetermined:
        values.pop("cores_allocated", None)
        reason = (f"sealed cores of {', '.join(f'{a} x {n}' for a, n in undetermined)} are of no declared core type "
                  f"and no unique fused core of one (declared: {', '.join(f'{a} x {n}' for a, n in declared)}): "
                  "their physical cores are not determined")
        return {key: reason for key in _CORE_COUNTS}
    dark = {id(s.cores[i]) for s in segments for i in (s.dark_core_indices or ())}  # [U-F4] loaded, never reset
    values["segment_cores"] = _measured(sum(tiles[id(c)] for c in cores if id(c) not in dark))
    values["cores_allocated"] = _measured(
        record.utilization.crossbar.cores_allocated + sum(tiles[id(c)] - 1 for c in cores))
    # [U-C C-6] e_core_program's multiplicand: the reprogrammed cores holding a static program (an operand-only core
    # holds none). On a chip with an operand core type a core sealed without its mark leaves them undetermined.
    programmed = [c for s in reprogrammed for c in s.cores]
    if any(c.get("operand") for c in record.identity.platform.get("cores") or ()) and any(
            c.operand_only is None for c in programmed):
        return {"reprogrammed_cores": "the record seals no operand-only mark on its cores (sealed before U-C C-6) on "
                                      "a chip with an operand core type: the cores holding a static program are not "
                                      "determined"}
    values["reprogrammed_cores"] = _measured(sum(tiles[id(c)] for c in programmed if not c.operand_only))
    return {}

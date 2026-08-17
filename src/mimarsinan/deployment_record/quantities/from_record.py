"""Quantities of a sealed record: the measured plane, absent where nothing sealed."""

from __future__ import annotations

from typing import Dict, Optional

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema import DeploymentRecord


def _measured(value: float) -> QuantityValue:
    return QuantityValue(value=float(value), provenance="measured")


def _put(values: Dict[str, QuantityValue], key: str, value: Optional[float],
         provenance: str = "measured") -> None:
    """Attach ``key`` unless its source is absent — absence survives, never a zero."""
    if value is None:
        return
    values[key] = QuantityValue(value=float(value), provenance=provenance)


def from_record(record: DeploymentRecord) -> Quantities:
    """Every multiplicand the sealed record carries, as measured quantities.

    Deliberately unclaimed: the ``adc_*`` pair (produced by C6 conversion
    models). ``synaptic_events`` is claimed from the sealed arrival census
    [H1]; records sealed before it stay honestly absent.
    """
    values: Dict[str, QuantityValue] = {}

    crossbar = record.utilization.crossbar
    values["cells_used"] = _measured(crossbar.cells_used)
    values["cores_allocated"] = _measured(crossbar.cores_allocated)
    values["neurons_used"] = _measured(crossbar.neurons_used)
    values["axons_used"] = _measured(crossbar.axons_used)
    values["macs"] = _measured(crossbar.macs)
    # A declaration of the run, not a measurement of the execution.
    _put(values, "weight_bits", crossbar.weight_bits, provenance="static")

    # The *_physical totals are the DECLARED chip's — area and static power price
    # the chip as declared, never just the cores this mapping happened to allocate.
    cores = list(record.identity.platform.get("cores") or ())
    if cores:
        counts = [int(core.get("count", 1)) for core in cores]
        axons = [int(core["max_axons"]) for core in cores]
        neurons = [int(core["max_neurons"]) for core in cores]
        values["cores_physical"] = QuantityValue(float(sum(counts)), "static")
        values["axons_physical"] = QuantityValue(
            float(sum(a * c for a, c in zip(axons, counts))), "static")
        values["neurons_physical"] = QuantityValue(
            float(sum(n * c for n, c in zip(neurons, counts))), "static")
        values["cells_physical"] = QuantityValue(
            float(sum(a * n * c for a, n, c in zip(axons, neurons, counts))), "static")

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
    values["carry_out_bytes"] = QuantityValue(
        float(0 if carry is None else carry.boundary_out_bytes), "static")
    values["carry_in_bytes"] = QuantityValue(
        float(0 if carry is None else carry.boundary_in_bytes), "static")
    segments = list(schedule.segments())
    reprogrammed = [s for s in segments if s.programming == "reprogram"]
    values["segment_cores"] = _measured(sum(len(s.cores) for s in segments))
    values["reprogrammed_cores"] = _measured(sum(len(s.cores) for s in reprogrammed))
    values["reprogrammed_bytes"] = _measured(sum(s.params_bytes for s in reprogrammed))
    values["connectivity_entries"] = _measured(
        sum(s.connectivity_entries for s in reprogrammed)
    )

    values["tiles"] = _measured(len(record.placement.tiles))

    values["timesteps"] = _measured(record.timing.s_global)
    values["latency_steps"] = _measured(record.timing.latency.compute_steps)
    _put(values, "host_ops_s", record.timing.latency.host_ops_s)

    if record.energy is not None:
        values["total_spikes"] = _measured(record.energy.total_spikes)
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

    return Quantities(values)

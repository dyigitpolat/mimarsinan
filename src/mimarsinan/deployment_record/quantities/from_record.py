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

    Deliberately unclaimed: ``synaptic_events`` (``total_spikes`` counts emissions,
    not synapse arrivals — no event census is sealed yet) and the ``adc_*`` pair
    (produced by C6 conversion models).
    """
    values: Dict[str, QuantityValue] = {}

    crossbar = record.utilization.crossbar
    values["cells_used"] = _measured(crossbar.cells_used)
    values["cells_physical"] = _measured(crossbar.cells_physical)
    values["cores_allocated"] = _measured(crossbar.cores_allocated)
    values["neurons_used"] = _measured(crossbar.neurons_used)
    values["neurons_physical"] = _measured(crossbar.neurons_physical)
    values["axons_used"] = _measured(crossbar.axons_used)
    values["axons_physical"] = _measured(crossbar.axons_physical)
    values["macs"] = _measured(crossbar.macs)
    # A declaration of the run, not a measurement of the execution.
    _put(values, "weight_bits", crossbar.weight_bits, provenance="static")

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

"""A sealed record yields its multiplicands — measured, and absent when unsealed."""

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import ComputePartitionRecord

from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_traffic,
    make_utilization,
)

# The fixture's census, restated once (record_fixtures is the source of these numbers).
_EXPECTED = {
    "cells_used": 15000.0,
    "cells_physical": 196608.0,
    "cores_allocated": 3.0,
    "macs": 15000.0,
    "neurons_used": 150.0,
    "neurons_physical": 768.0,
    "axons_used": 300.0,
    "axons_physical": 768.0,
    "tiles": 1.0,
    "weight_bits": 8.0,
    "pass_count": 2.0,
    "sync_count": 1.0,
    "reprogram_passes": 1.0,
    "reprogrammed_bytes": 100.0,
    "connectivity_entries": 24.0,
    "segment_cores": 3.0,
    "reprogrammed_cores": 2.0,
    "timesteps": 32.0,
    "latency_steps": 64.0,
    "total_spikes": 987.0,
    "boundary_events": 1234.0,
    "noc_total_packets": 1000.0,
    "noc_intra_tile_packets": 500.0,
    "noc_inter_tile_packets": 400.0,
    "noc_total_hops": 400.0,
}


def test_the_full_fixture_record_yields_its_census():
    quantities = from_record(make_full_record())
    for key, expected in _EXPECTED.items():
        assert quantities.has(key), key
        assert quantities.get(key).value == pytest.approx(expected), key


def test_reprogramming_quantities_count_reprogram_passes_only():
    """A resident pass sends nothing: its bytes/entries/cores never enter the
    programming multiplicands (the RESIDENT_PAYLOAD discipline)."""
    quantities = from_record(make_full_record())
    # Fixture: stage 1 reprogram (2 cores, 100 B, 24 entries), stage 2 resident.
    assert quantities.get("reprogrammed_bytes").value == 100.0
    assert quantities.get("connectivity_entries").value == 24.0
    assert quantities.get("reprogrammed_cores").value == 2.0
    assert quantities.get("segment_cores").value == 3.0, "init counts EVERY pass's cores"


def test_noc_total_hops_is_the_sum_of_link_traversals():
    quantities = from_record(make_full_record())
    record = make_full_record()
    assert record.traffic is not None and record.traffic.noc is not None
    assert quantities.get("noc_total_hops").value == sum(
        link.packet_count for link in record.traffic.noc.link_loads
    )


def test_record_quantities_are_measured_except_declarations():
    quantities = from_record(make_full_record())
    assert quantities.get("latency_steps").provenance == "measured"
    assert quantities.get("pass_count").provenance == "measured"
    assert quantities.get("weight_bits").provenance == "static"


def test_absent_fragments_yield_absent_quantities_never_zeros():
    record = replace(make_full_record(), traffic=None, energy=None)
    quantities = from_record(record)
    for key in ("noc_total_packets", "noc_total_hops", "boundary_events",
                "total_spikes"):
        assert not quantities.has(key), key


def test_boundaryless_traffic_keeps_noc_and_drops_boundary_events():
    record = replace(make_full_record(), traffic=make_traffic(with_boundaries=False))
    quantities = from_record(record)
    assert quantities.has("noc_total_packets")
    assert not quantities.has("boundary_events")


def test_an_untimed_host_yields_no_host_wall():
    """The fixture's host_ops_s is None: absence must survive, not become 0.0."""
    quantities = from_record(make_full_record())
    assert not quantities.has("host_ops_s")


def test_synaptic_events_are_not_claimed_at_record_time():
    """No event census is sealed yet (total_spikes counts EMISSIONS, not synapse
    arrivals); claiming one would be a silent model wearing a measurement's name."""
    quantities = from_record(make_full_record())
    assert not quantities.has("synaptic_events")


def test_the_partition_split_rides_in_when_sealed():
    record = make_full_record()
    partition = ComputePartitionRecord(
        onchip_params=90, host_params=10, total_params=100,
        onchip_macs=900, host_macs=100, total_macs=1000,
    )
    record = replace(
        record, utilization=replace(make_utilization(), partition=partition)
    )
    quantities = from_record(record)
    assert quantities.get("onchip_params").value == 90.0
    assert quantities.get("host_params").value == 10.0
    assert quantities.get("total_params").value == 100.0
    assert quantities.get("onchip_macs").value == 900.0
    assert quantities.get("host_macs").value == 100.0
    assert quantities.get("total_macs").value == 1000.0


def test_without_the_partition_the_split_is_absent():
    quantities = from_record(make_full_record())
    for key in ("onchip_params", "host_params", "total_params",
                "onchip_macs", "host_macs", "total_macs"):
        assert not quantities.has(key), key

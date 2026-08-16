"""Drift pins for the objectives catalogue and the capacity SSOT.

Every axis' direction/provenance/unit is part of the published optimization
contract: a flipped direction silently inverts a search. The legacy eight were
already pinned against literal data; these pin the record axes and the chip
capacity formula the candidate view scores against.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.deployment_record.objectives.views import chip_param_capacity

# key -> (direction, provenance, unit)
RECORD_AXES = {
    "deployed_accuracy": ("max", "measured", "fraction"),
    "mj_per_sample": ("min", "measured", "mJ/sample"),
    "latency_steps": ("min", "measured", "timesteps"),
    "host_op_wall_s": ("min", "measured", "s"),
    "total_spikes": ("min", "measured", "spikes"),
    "pass_count": ("min", "static", "passes"),
    "reprogram_passes": ("min", "static", "passes"),
    "reprogramming_bytes": ("min", "static", "bytes"),
    "params_reloaded": ("min", "static", "parameters"),
    "noc_inter_tile_packets": ("min", "measured", "packets"),
    "noc_total_packets": ("min", "measured", "packets"),
    "programming_energy_mj": ("min", "modeled", "mJ"),
    "sync_barrier_energy_mj": ("min", "modeled", "mJ"),
    "throughput_samples_per_s": ("max", "modeled", "samples/s"),
    # [B] The pass-buffer metrics of the sealed schedule (record-only).
    "carry_peak_live_bytes": ("min", "measured", "bytes"),
    "carried_raster_bytes": ("min", "measured", "bytes"),
}


@pytest.mark.parametrize("key,expected", sorted(RECORD_AXES.items()))
def test_record_axis_contract_is_pinned(key, expected):
    spec = OBJECTIVES.get(key)
    assert (spec.direction, spec.provenance, spec.unit) == expected


def test_the_catalogue_holds_exactly_the_published_axes():
    from mimarsinan.search.results import ALL_OBJECTIVES

    searchable = {spec.name for spec in ALL_OBJECTIVES}
    assert {s.key for s in OBJECTIVES.all()} == searchable | set(RECORD_AXES)


def test_chip_param_capacity_counts_every_core():
    # max_axons x max_neurons x count, summed over core types: dropping the
    # count would understate a 1000-core chip by three orders of magnitude.
    cores = [
        {"max_axons": 256, "max_neurons": 128, "count": 4},
        {"max_axons": 64, "max_neurons": 32, "count": 10},
    ]
    assert chip_param_capacity(cores) == 256 * 128 * 4 + 64 * 32 * 10
    assert chip_param_capacity([{"max_axons": 8, "max_neurons": 4, "count": 1}]) == 32
    assert chip_param_capacity([]) == 0

"""TTFS dual parity layer subsets (contract vs hardware trace)."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
)
from mimarsinan.chip_simulation.ttfs_recorder import (
    CoreTtfsActivations,
    SegmentTtfsRecord,
    TtfsRunRecord,
    compare_ttfs_records,
    format_first_ttfs_diff,
)


def _energy(total: float = 1.0) -> SanafeEnergyBreakdown:
    return SanafeEnergyBreakdown(
        synapse_j=0.4 * total,
        dendrite_j=0.1 * total,
        soma_j=0.3 * total,
        network_j=0.2 * total,
        total_j=total,
    )


def test_contract_vs_hardware_subset_sources():
    hw_core = SanafeCoreRecord(
        core_index=0,
        n_neurons=1,
        n_axons_used=1,
        core_latency=0,
        has_hardware_bias=False,
        n_always_on_axons=0,
        spikes_fired=0,
        input_spike_count=np.zeros(1, dtype=np.int64),
        output_spike_count=np.zeros(1, dtype=np.int64),
        energy=_energy(),
        output_activation=np.array([0.25], dtype=np.float64),
    )
    seg = SanafeSegmentRecord(
        stage_index=0,
        stage_name="s0",
        schedule_segment_index=None,
        schedule_pass_index=None,
        timesteps_executed=4,
        sim_time_s=0.0,
        energy=_energy(),
        spikes=0,
        packets_sent=0,
        neurons_updated=0,
        neurons_fired=0,
        seg_input_rates=np.zeros((1, 1), dtype=np.float32),
        seg_input_spike_count=np.zeros(1, dtype=np.int64),
        seg_output_spike_count=np.zeros(1, dtype=np.int64),
        per_core=[hw_core],
        contract_ttfs_cores=[
            CoreTtfsActivations(0, 1, np.array([0.5], dtype=np.float64)),
        ],
        contract_ttfs_seg_output=np.array([[0.5]], dtype=np.float64),
    )

    rec = SanafeRunRecord(
        arch_preset="loihi",
        arch_name="t",
        sample_index=0,
        T=4,
    )
    rec.segments[0] = seg

    ref = TtfsRunRecord(sample_index=0, simulation_length=4, spiking_mode="ttfs")
    ref.segments[0] = SegmentTtfsRecord(
        stage_index=0,
        stage_name="s0",
        schedule_segment_index=None,
        schedule_pass_index=None,
        seg_output=np.array([0.5]),
        cores=[CoreTtfsActivations(0, 1, np.array([0.5], dtype=np.float64))],
    )

    assert not compare_ttfs_records(ref, rec.to_ttfs_contract_subset())
    diffs = compare_ttfs_records(ref, rec.to_ttfs_hardware_subset())
    assert len(diffs) == 1
    assert "hardware" in format_first_ttfs_diff(diffs, layer="hardware")

"""Tests for on-disk snapshot rebuild (SANA-FE tab recovery)."""

from __future__ import annotations

import json
import pickle

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport
from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk


def _eb(total: float) -> SanafeEnergyBreakdown:
    return SanafeEnergyBreakdown(
        synapse_j=0.4 * total, dendrite_j=0.1 * total,
        soma_j=0.3 * total, network_j=0.2 * total, total_j=total,
    )


def _minimal_record() -> SanafeRunRecord:
    core = SanafeCoreRecord(
        core_index=0, n_neurons=2, n_axons_used=3,
        core_latency=0, has_hardware_bias=False, n_always_on_axons=0,
        spikes_fired=8,
        input_spike_count=np.arange(3, dtype=np.int64) + 1,
        output_spike_count=np.full(2, 4, dtype=np.int64),
        energy=_eb(1.0),
    )
    seg = SanafeSegmentRecord(
        stage_index=0, stage_name="stage0",
        schedule_segment_index=None, schedule_pass_index=None,
        timesteps_executed=8, sim_time_s=1.6e-6,
        energy=_eb(2.0), spikes=12, packets_sent=3,
        neurons_updated=16, neurons_fired=2,
        seg_input_rates=np.zeros((1, 3), dtype=np.float32),
        seg_input_spike_count=np.asarray([1, 1, 1], dtype=np.int64),
        seg_output_spike_count=np.asarray([2, 2], dtype=np.int64),
        per_core=[core],
        per_tile=[SanafeTileRecord(
            tile_index=0, cores=[0], energy=_eb(2.0), spikes_fired=8, packets_sent=3,
        )],
    )
    return SanafeRunRecord(
        arch_preset="loihi", arch_name="t", sample_index=0, T=8,
        segments={0: seg}, compute_outputs={},
        aggregate_energy=_eb(2.0), aggregate_sim_time_s=1.6e-6,
        total_spikes=12, total_packets=3,
    )


def test_rebuild_sanafe_snapshot_from_pickle(tmp_path):
    report = SanafeStepReport.from_records("loihi", [_minimal_record()])
    pickle_path = tmp_path / "SANA-FE Simulation.sanafe_simulation_results.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(report, f)

    rebuilt = rebuild_step_snapshot_from_disk(str(tmp_path), "SANA-FE Simulation")
    assert rebuilt is not None
    snapshot, kinds = rebuilt
    assert snapshot["sanafe_simulation"]["arch_preset"] == "loihi"
    assert kinds["sanafe_simulation"] == "new"
    json.dumps(snapshot)


def test_rebuild_returns_none_when_pickle_missing(tmp_path):
    assert rebuild_step_snapshot_from_disk(str(tmp_path), "SANA-FE Simulation") is None


def test_rebuild_returns_none_for_other_steps(tmp_path):
    assert rebuild_step_snapshot_from_disk(str(tmp_path), "Simulation") is None

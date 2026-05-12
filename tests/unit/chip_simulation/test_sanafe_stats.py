"""SanafeStepReport aggregation + GUI snapshot serialization.

``SanafeStepReport`` is what the pipeline step persists to cache and what
the GUI snapshot builder reads.  Its ``to_snapshot_dict()`` output must be
JSON-safe (no numpy scalars, no ndarrays at leaves) and stable in its
top-level key set so the frontend can rely on shape.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
    SanafeCoreRecord,
)
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport


# ---------------------------------------------------------------------------
# Builders (small, deterministic SanafeRunRecord fixtures)
# ---------------------------------------------------------------------------


def _eb(total: float) -> SanafeEnergyBreakdown:
    return SanafeEnergyBreakdown(
        synapse_j=0.4 * total, dendrite_j=0.1 * total,
        soma_j=0.3 * total, network_j=0.2 * total, total_j=total,
    )


def _core(core_index: int = 0) -> SanafeCoreRecord:
    return SanafeCoreRecord(
        core_index=core_index, n_neurons=2, n_axons_used=3,
        core_latency=5, has_hardware_bias=False, n_always_on_axons=0,
        spikes_fired=6,
        input_spike_count=np.asarray([1, 2, 3], dtype=np.int64),
        output_spike_count=np.asarray([3, 3], dtype=np.int64),
        energy=_eb(1.0),
    )


def _segment(stage_index: int = 0, *, spikes: int = 10,
             packets: int = 4, sim_time_s: float = 1.0e-6,
             energy_total: float = 2.0) -> SanafeSegmentRecord:
    return SanafeSegmentRecord(
        stage_index=stage_index,
        stage_name=f"stage{stage_index}",
        schedule_segment_index=None,
        schedule_pass_index=None,
        timesteps_executed=32,
        sim_time_s=sim_time_s,
        energy=_eb(energy_total),
        spikes=spikes,
        packets_sent=packets,
        neurons_updated=64,
        neurons_fired=spikes // 2,
        seg_input_rates=np.zeros((1, 3), dtype=np.float32),
        seg_input_spike_count=np.asarray([1, 1, 1], dtype=np.int64),
        seg_output_spike_count=np.asarray([2, 2], dtype=np.int64),
        per_core=[_core(0)],
        per_tile=[
            SanafeTileRecord(
                tile_index=0, cores=[0],
                energy=_eb(energy_total),
                spikes_fired=spikes, packets_sent=packets,
            )
        ],
    )


def _run(sample_index: int, *, segments: list[SanafeSegmentRecord]) -> SanafeRunRecord:
    seg_dict = {s.stage_index: s for s in segments}
    agg_e = SanafeEnergyBreakdown.zero()
    for s in segments:
        agg_e = agg_e.add(s.energy)
    return SanafeRunRecord(
        arch_preset="loihi", arch_name="test", sample_index=sample_index, T=32,
        segments=seg_dict, compute_outputs={},
        aggregate_energy=agg_e,
        aggregate_sim_time_s=max(s.sim_time_s for s in segments),
        total_spikes=sum(s.spikes for s in segments),
        total_packets=sum(s.packets_sent for s in segments),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_sanafe_step_report_from_records_populates_sample_indices():
    rec_a = _run(0, segments=[_segment(0)])
    rec_b = _run(7, segments=[_segment(0)])
    report = SanafeStepReport.from_records("loihi", [rec_a, rec_b])

    assert report.sample_indices == [0, 7]
    assert report.arch_preset == "loihi"
    assert report.per_sample == [rec_a, rec_b]


def test_sanafe_step_report_aggregate_sums_across_samples():
    rec_a = _run(0, segments=[_segment(0, spikes=10, packets=4, energy_total=2.0,
                                       sim_time_s=1.0e-6)])
    rec_b = _run(1, segments=[_segment(0, spikes=20, packets=6, energy_total=3.0,
                                       sim_time_s=2.0e-6)])
    report = SanafeStepReport.from_records("loihi", [rec_a, rec_b])

    agg = report.aggregate
    # Energy aggregates in joules (sum) AND surface in millijoules (sum/1e-3) for the UI.
    assert agg["total_energy_j"] == pytest.approx(5.0)
    assert agg["total_energy_mj"] == pytest.approx(5.0 * 1000.0)
    # Sim-time: max across all (segments and samples) — latency is wall-clock-ish.
    assert agg["max_sim_time_s"] == pytest.approx(2.0e-6)
    # Spikes / packets sum across samples.
    assert agg["total_spikes"] == 30
    assert agg["total_packets"] == 10
    # Sample count is plain.
    assert agg["sample_count"] == 2


def test_sanafe_step_report_aggregate_handles_empty_samples_list():
    """Edge case: a report built with no samples must still produce valid aggregates."""
    report = SanafeStepReport.from_records("loihi", [])
    agg = report.aggregate
    assert agg["sample_count"] == 0
    assert agg["total_energy_j"] == 0.0
    assert agg["total_energy_mj"] == 0.0
    assert agg["max_sim_time_s"] == 0.0
    assert agg["total_spikes"] == 0
    assert agg["total_packets"] == 0


def test_sanafe_step_report_aggregate_energy_breakdown_components():
    rec = _run(0, segments=[_segment(0, energy_total=10.0)])
    report = SanafeStepReport.from_records("loihi", [rec])
    eb = report.aggregate["energy_breakdown_j"]
    assert eb["synapse"] == pytest.approx(4.0)
    assert eb["dendrite"] == pytest.approx(1.0)
    assert eb["soma"] == pytest.approx(3.0)
    assert eb["network"] == pytest.approx(2.0)
    assert eb["total"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Snapshot dict (consumed by gui/snapshot/builders.py)
# ---------------------------------------------------------------------------


_EXPECTED_TOP_LEVEL_KEYS = {
    "arch_preset", "sample_indices", "aggregate", "per_sample",
}


def test_to_snapshot_dict_keys_stable():
    """Top-level snapshot keys are part of the GUI contract — pin them."""
    rec = _run(0, segments=[_segment(0)])
    snap = SanafeStepReport.from_records("loihi", [rec]).to_snapshot_dict()
    assert set(snap.keys()) >= _EXPECTED_TOP_LEVEL_KEYS


def test_to_snapshot_dict_is_json_safe():
    rec = _run(0, segments=[_segment(0), _segment(1, spikes=5)])
    snap = SanafeStepReport.from_records("loihi", [rec]).to_snapshot_dict()
    # Should not raise — json.dumps refuses ndarrays / numpy scalars / sets.
    serialized = json.dumps(snap)
    assert isinstance(serialized, str)
    assert "arch_preset" in serialized


def test_to_snapshot_dict_per_sample_shape():
    """Each entry in ``per_sample`` carries a compact summary, not the full record."""
    rec = _run(3, segments=[_segment(0, spikes=8, packets=2, energy_total=2.0)])
    snap = SanafeStepReport.from_records("loihi", [rec]).to_snapshot_dict()

    ps = snap["per_sample"]
    assert isinstance(ps, list) and len(ps) == 1
    entry = ps[0]
    assert entry["sample_index"] == 3
    assert entry["total_energy_j"] == pytest.approx(2.0)
    assert entry["total_spikes"] == 8
    assert entry["total_packets"] == 2
    # Each segment summary present.
    assert "segments" in entry
    assert isinstance(entry["segments"], list)


def test_to_snapshot_dict_per_segment_shape():
    """Each per-segment summary carries enough for the GUI bar/heatmap charts."""
    rec = _run(0, segments=[_segment(0, spikes=8, packets=2, energy_total=2.0)])
    snap = SanafeStepReport.from_records("loihi", [rec]).to_snapshot_dict()
    seg0 = snap["per_sample"][0]["segments"][0]

    assert seg0["stage_index"] == 0
    assert seg0["stage_name"] == "stage0"
    assert "energy_j" in seg0
    assert "sim_time_s" in seg0
    assert "spikes" in seg0
    assert "packets_sent" in seg0
    assert isinstance(seg0["per_tile"], list)
    assert isinstance(seg0["per_core"], list)


def test_to_snapshot_dict_per_core_shape():
    rec = _run(0, segments=[_segment(0)])
    snap = SanafeStepReport.from_records("loihi", [rec]).to_snapshot_dict()
    core0 = snap["per_sample"][0]["segments"][0]["per_core"][0]
    assert core0["core_index"] == 0
    assert core0["spikes_fired"] == 6
    assert "energy_j" in core0


def test_to_snapshot_dict_arch_preset_propagated():
    rec = _run(0, segments=[_segment(0)])
    snap = SanafeStepReport.from_records("truenorth", [rec]).to_snapshot_dict()
    assert snap["arch_preset"] == "truenorth"


def test_to_snapshot_dict_aggregate_keys_subset_of_aggregate_dict():
    """The aggregate dict surfaces verbatim — no key renames at the snapshot boundary."""
    rec = _run(0, segments=[_segment(0)])
    report = SanafeStepReport.from_records("loihi", [rec])
    snap = report.to_snapshot_dict()
    assert snap["aggregate"]["total_spikes"] == report.aggregate["total_spikes"]
    assert snap["aggregate"]["total_energy_mj"] == report.aggregate["total_energy_mj"]

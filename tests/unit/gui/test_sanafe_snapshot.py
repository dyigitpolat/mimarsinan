"""Snapshot builder for SanafeStepReport.

The builder is consumed by ``gui/snapshot/builders.py::build_step_snapshot``
when a step's cache entry resolves to the snapshot key ``sanafe_simulation``.
We pin:

* the cache-key → snapshot-key mapping (``helpers._CACHE_KEY_TO_SNAPSHOT_KEY``),
* the snapshot dict's top-level keys (frontend contract),
* the resource descriptors emitted per (sample, segment) — they back the
  lazy-loaded heatmap PNGs in the SANA-FE tab,
* JSON safety of the snapshot dict (no ndarrays at leaves).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.builders import snapshot_sanafe_simulation
from mimarsinan.gui.snapshot.helpers import _CACHE_KEY_TO_SNAPSHOT_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _eb(total: float) -> SanafeEnergyBreakdown:
    return SanafeEnergyBreakdown(
        synapse_j=0.4 * total, dendrite_j=0.1 * total,
        soma_j=0.3 * total, network_j=0.2 * total, total_j=total,
    )


def _core(idx: int = 0, neurons: int = 2, axons: int = 3) -> SanafeCoreRecord:
    return SanafeCoreRecord(
        core_index=idx, n_neurons=neurons, n_axons_used=axons,
        core_latency=0, has_hardware_bias=False, n_always_on_axons=0,
        spikes_fired=4 * neurons,
        input_spike_count=np.arange(axons, dtype=np.int64) + 1,
        output_spike_count=np.full(neurons, 4, dtype=np.int64),
        energy=_eb(1.0),
    )


def _segment(stage_index: int = 0, cores=None) -> SanafeSegmentRecord:
    cores = cores or [_core(0)]
    return SanafeSegmentRecord(
        stage_index=stage_index, stage_name=f"stage{stage_index}",
        schedule_segment_index=None, schedule_pass_index=None,
        timesteps_executed=8, sim_time_s=1.6e-6,
        energy=_eb(2.0), spikes=12, packets_sent=3,
        neurons_updated=16, neurons_fired=2,
        seg_input_rates=np.zeros((1, 3), dtype=np.float32),
        seg_input_spike_count=np.asarray([1, 1, 1], dtype=np.int64),
        seg_output_spike_count=np.asarray([2, 2], dtype=np.int64),
        per_core=cores,
        per_tile=[SanafeTileRecord(
            tile_index=0, cores=[c.core_index for c in cores],
            energy=_eb(2.0),
            spikes_fired=sum(c.spikes_fired for c in cores),
            packets_sent=3,
        )],
    )


def _record(sample_index: int = 0, segments=None) -> SanafeRunRecord:
    segs = segments if segments is not None else {0: _segment(0)}
    return SanafeRunRecord(
        arch_preset="loihi", arch_name="t", sample_index=sample_index, T=8,
        segments=segs, compute_outputs={},
        aggregate_energy=_eb(sum(s.energy.total_j for s in segs.values())),
        aggregate_sim_time_s=max(s.sim_time_s for s in segs.values()),
        total_spikes=sum(s.spikes for s in segs.values()),
        total_packets=sum(s.packets_sent for s in segs.values()),
    )


# ---------------------------------------------------------------------------
# Cache-key mapping
# ---------------------------------------------------------------------------


def test_cache_key_to_snapshot_key_maps_sanafe_simulation_results():
    assert _CACHE_KEY_TO_SNAPSHOT_KEY["sanafe_simulation_results"] == "sanafe_simulation"


# ---------------------------------------------------------------------------
# snapshot_sanafe_simulation
# ---------------------------------------------------------------------------


def test_snapshot_sanafe_simulation_returns_summary_dict_and_descriptors():
    report = SanafeStepReport.from_records("loihi", [_record(0)])
    snap, descriptors = snapshot_sanafe_simulation(report)

    assert isinstance(snap, dict)
    assert isinstance(descriptors, list)
    assert "arch_preset" in snap
    assert "aggregate" in snap
    assert "per_sample" in snap


def test_snapshot_sanafe_simulation_handles_empty_report():
    report = SanafeStepReport.from_records("loihi", [])
    snap, descriptors = snapshot_sanafe_simulation(report)
    assert snap["per_sample"] == []
    assert descriptors == []


def test_snapshot_sanafe_simulation_emits_descriptors_per_segment_axis():
    """Each (sample, segment) emits energy + spike heatmap descriptors."""
    rec = _record(
        sample_index=0,
        segments={
            0: _segment(0, cores=[_core(0), _core(1)]),
            1: _segment(1, cores=[_core(0)]),
        },
    )
    report = SanafeStepReport.from_records("loihi", [rec])
    snap, descriptors = snapshot_sanafe_simulation(report)

    kinds = sorted({d.kind for d in descriptors})
    # At minimum: per-tile energy strip + per-core spike strip per segment.
    assert "sanafe_tile_energy" in kinds
    assert "sanafe_core_spikes" in kinds

    rids = {d.rid for d in descriptors}
    # rid carries (sample, segment) so the frontend can request by axis.
    assert any("sample0" in r and "seg0" in r for r in rids)
    assert any("sample0" in r and "seg1" in r for r in rids)


def test_snapshot_sanafe_simulation_descriptors_have_png_media_type():
    report = SanafeStepReport.from_records("loihi", [_record(0)])
    _, descriptors = snapshot_sanafe_simulation(report)
    for d in descriptors:
        assert isinstance(d, ResourceDescriptor)
        assert d.media_type == "image/png"


def test_snapshot_sanafe_simulation_producer_returns_bytes():
    """Producer is lazily invoked; calling it returns PNG bytes."""
    report = SanafeStepReport.from_records("loihi", [_record(0)])
    _, descriptors = snapshot_sanafe_simulation(report)
    if descriptors:
        payload = descriptors[0].producer()
        assert isinstance(payload, (bytes, bytearray))
        # PNG signature on the first 8 bytes.
        assert bytes(payload[:8]) == b"\x89PNG\r\n\x1a\n"


def test_snapshot_sanafe_simulation_is_json_safe():
    report = SanafeStepReport.from_records(
        "loihi",
        [_record(0, segments={0: _segment(0, cores=[_core(0)])}),
         _record(1, segments={0: _segment(0, cores=[_core(0)])})],
    )
    snap, _ = snapshot_sanafe_simulation(report)
    json.dumps(snap)   # raises TypeError on numpy scalars / ndarrays.


def test_snapshot_sanafe_simulation_propagates_arch_preset():
    report = SanafeStepReport.from_records("truenorth", [_record(0)])
    snap, _ = snapshot_sanafe_simulation(report)
    assert snap["arch_preset"] == "truenorth"

"""SanafeRunRecord dataclass + projection-back-to-RunRecord behaviour.

SANA-FE's runner produces ``SanafeRunRecord`` — a richer shape than
``spike_recorder.RunRecord`` (it additionally carries per-tile and per-core
energy breakdowns, latency, NoC packet counts, and optional traces).

For the hard spike-parity gate against HCM we need a *lossless projection*
of the spike-count subset back to the existing ``RunRecord`` shape so we
can reuse ``compare_records()`` verbatim.  That projection is the
``to_hcm_subset()`` method.  The tests below pin its behaviour layer by
layer so a regression in any field will be caught here, not deep inside
the parity gate.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.recording.spike_recorder import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _energy(total: float = 10.0) -> SanafeEnergyBreakdown:
    """Construct an energy breakdown whose components sum to ``total``."""
    return SanafeEnergyBreakdown(
        synapse_j=0.4 * total,
        dendrite_j=0.1 * total,
        soma_j=0.3 * total,
        network_j=0.2 * total,
        total_j=total,
    )


def _core(
    *,
    core_index: int = 0,
    n_neurons: int = 2,
    n_axons_used: int = 3,
    core_latency: int = 5,
    has_hardware_bias: bool = False,
    n_always_on_axons: int = 0,
    input_spike_count: np.ndarray | None = None,
    output_spike_count: np.ndarray | None = None,
) -> SanafeCoreRecord:
    if input_spike_count is None:
        input_spike_count = np.arange(n_axons_used, dtype=np.int64) + 1
    if output_spike_count is None:
        output_spike_count = (np.arange(n_neurons, dtype=np.int64) + 1) * 2
    return SanafeCoreRecord(
        core_index=core_index,
        n_neurons=n_neurons,
        n_axons_used=n_axons_used,
        core_latency=core_latency,
        has_hardware_bias=has_hardware_bias,
        n_always_on_axons=n_always_on_axons,
        spikes_fired=int(output_spike_count.sum()),
        input_spike_count=input_spike_count,
        output_spike_count=output_spike_count,
        energy=_energy(total=1.0 + core_index),
    )


def _segment(
    *,
    stage_index: int = 0,
    stage_name: str = "stage0",
    seg_in_size: int = 4,
    seg_out_size: int = 2,
    cores: list[SanafeCoreRecord] | None = None,
) -> SanafeSegmentRecord:
    if cores is None:
        cores = [_core(core_index=0, n_axons_used=seg_in_size, n_neurons=seg_out_size)]
    return SanafeSegmentRecord(
        stage_index=stage_index,
        stage_name=stage_name,
        schedule_segment_index=None,
        schedule_pass_index=None,
        timesteps_executed=32,
        sim_time_s=1.6e-6,
        energy=_energy(total=5.0),
        spikes=42,
        packets_sent=17,
        neurons_updated=64,
        neurons_fired=10,
        seg_input_rates=np.linspace(0.0, 1.0, seg_in_size, dtype=np.float32).reshape(1, -1),
        seg_input_spike_count=np.arange(seg_in_size, dtype=np.int64) * 3,
        seg_output_spike_count=np.arange(seg_out_size, dtype=np.int64) + 7,
        per_core=cores,
        per_tile=[
            SanafeTileRecord(
                tile_index=0,
                cores=[c.core_index for c in cores],
                energy=_energy(total=5.0),
                spikes_fired=sum(c.spikes_fired for c in cores),
                packets_sent=17,
            )
        ],
        per_neuron_spike_counts=None,
        per_neuron_spike_trace=None,
        per_neuron_potential_trace=None,
        message_trace=None,
    )


def _run_record(
    *,
    sample_index: int = 0,
    T: int = 32,
    segments: dict[int, SanafeSegmentRecord] | None = None,
    compute_outputs: dict[int, np.ndarray] | None = None,
) -> SanafeRunRecord:
    if segments is None:
        segments = {0: _segment(stage_index=0)}
    if compute_outputs is None:
        compute_outputs = {}
    return SanafeRunRecord(
        arch_preset="loihi",
        arch_name="test_arch",
        sample_index=sample_index,
        T=T,
        segments=segments,
        compute_outputs=compute_outputs,
        aggregate_energy=_energy(total=sum(s.energy.total_j for s in segments.values())),
        aggregate_sim_time_s=max((s.sim_time_s for s in segments.values()), default=0.0),
        total_spikes=sum(s.spikes for s in segments.values()),
        total_packets=sum(s.packets_sent for s in segments.values()),
    )


# ---------------------------------------------------------------------------
# to_hcm_subset projection
# ---------------------------------------------------------------------------


def test_sanafe_run_record_to_hcm_subset_returns_run_record():
    """The projection returns a plain ``RunRecord``, not a ``SanafeRunRecord``."""
    sub = _run_record().to_hcm_subset()
    assert isinstance(sub, RunRecord)
    assert not isinstance(sub, SanafeRunRecord)


def test_to_hcm_subset_preserves_sample_index_and_T():
    rec = _run_record(sample_index=7, T=64)
    sub = rec.to_hcm_subset()
    assert sub.sample_index == 7
    assert sub.T == 64


def test_to_hcm_subset_preserves_seg_input_output_spike_counts():
    """Segment-level spike counts are the parity-layer-1 and parity-layer-4 surface."""
    seg_in = np.asarray([3, 5, 7, 11], dtype=np.int64)
    seg_out = np.asarray([2, 4], dtype=np.int64)
    seg = _segment(seg_in_size=4, seg_out_size=2)
    seg.seg_input_spike_count = seg_in
    seg.seg_output_spike_count = seg_out
    rec = _run_record(segments={0: seg})

    sub = rec.to_hcm_subset()

    assert isinstance(sub.segments[0], SegmentSpikeRecord)
    np.testing.assert_array_equal(sub.segments[0].seg_input_spike_count, seg_in)
    np.testing.assert_array_equal(sub.segments[0].seg_output_spike_count, seg_out)


def test_to_hcm_subset_preserves_per_core_input_output_spike_counts():
    """Per-core counts feed parity layers 2 (core_input) and 3 (core_output)."""
    in_a = np.asarray([1, 2, 3], dtype=np.int64)
    out_a = np.asarray([10, 20], dtype=np.int64)
    in_b = np.asarray([4, 5, 6], dtype=np.int64)
    out_b = np.asarray([30, 40], dtype=np.int64)
    core_a = _core(core_index=0, n_axons_used=3, n_neurons=2,
                   input_spike_count=in_a, output_spike_count=out_a)
    core_b = _core(core_index=1, n_axons_used=3, n_neurons=2,
                   input_spike_count=in_b, output_spike_count=out_b)
    seg = _segment(cores=[core_a, core_b])
    rec = _run_record(segments={0: seg})

    sub = rec.to_hcm_subset()
    cores = sub.segments[0].cores

    assert [c.core_index for c in cores] == [0, 1]
    np.testing.assert_array_equal(cores[0].input_spike_count, in_a)
    np.testing.assert_array_equal(cores[0].output_spike_count, out_a)
    np.testing.assert_array_equal(cores[1].input_spike_count, in_b)
    np.testing.assert_array_equal(cores[1].output_spike_count, out_b)


def test_to_hcm_subset_preserves_core_latency_hardware_bias_flags():
    """``has_hardware_bias`` and ``n_always_on_axons`` drive diff cause-suggestion."""
    core = _core(
        core_index=3,
        core_latency=11,
        has_hardware_bias=True,
        n_always_on_axons=2,
    )
    seg = _segment(cores=[core])
    rec = _run_record(segments={0: seg})

    csc = rec.to_hcm_subset().segments[0].cores[0]
    assert isinstance(csc, CoreSpikeCounts)
    assert csc.core_index == 3
    assert csc.core_latency == 11
    assert csc.has_hardware_bias is True
    assert csc.n_always_on_axons == 2
    # n_in_used / n_out_used translate from SANA-FE's n_axons_used / n_neurons
    assert csc.n_in_used == core.n_axons_used
    assert csc.n_out_used == core.n_neurons


def test_to_hcm_subset_preserves_compute_outputs_passthrough():
    """ComputeOp host-side outputs flow through verbatim for hybrid mappings."""
    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    rec = _run_record(compute_outputs={42: payload})

    sub = rec.to_hcm_subset()

    assert set(sub.compute_outputs.keys()) == {42}
    np.testing.assert_array_equal(sub.compute_outputs[42], payload)


def test_to_hcm_subset_preserves_segment_keys_and_order():
    """Sparse, non-contiguous stage_index keys must survive the projection."""
    rec = _run_record(segments={0: _segment(stage_index=0),
                                3: _segment(stage_index=3, stage_name="stage3")})
    sub = rec.to_hcm_subset()
    assert sorted(sub.segments.keys()) == [0, 3]
    assert sub.segments[3].stage_name == "stage3"


def test_to_hcm_subset_preserves_schedule_indices_when_set():
    """``schedule_segment_index`` / ``schedule_pass_index`` survive (HCM nullable)."""
    seg = _segment(stage_index=2)
    seg.schedule_segment_index = 5
    seg.schedule_pass_index = 1
    rec = _run_record(segments={2: seg})
    sub_seg = rec.to_hcm_subset().segments[2]
    assert sub_seg.schedule_segment_index == 5
    assert sub_seg.schedule_pass_index == 1


# ---------------------------------------------------------------------------
# Energy breakdown invariant
# ---------------------------------------------------------------------------


def test_sanafe_energy_breakdown_total_matches_components():
    """SANA-FE returns total independently; we keep an invariance check available."""
    eb = SanafeEnergyBreakdown(synapse_j=4.0, dendrite_j=1.0, soma_j=3.0,
                               network_j=2.0, total_j=10.0)
    assert eb.components_sum() == pytest.approx(10.0)
    assert eb.total_j == pytest.approx(eb.components_sum())


def test_sanafe_energy_breakdown_from_sanafe_dict():
    """Helper builds a breakdown from SANA-FE's results['energy'] dict shape."""
    d = {"total": 12.5, "synapse": 5.0, "dendrite": 1.5, "soma": 4.0, "network": 2.0}
    eb = SanafeEnergyBreakdown.from_sanafe_dict(d)
    assert eb.total_j == pytest.approx(12.5)
    assert eb.synapse_j == pytest.approx(5.0)
    assert eb.dendrite_j == pytest.approx(1.5)
    assert eb.soma_j == pytest.approx(4.0)
    assert eb.network_j == pytest.approx(2.0)


def test_sanafe_energy_breakdown_zero():
    """``zero()`` produces an all-zero breakdown — useful as an aggregation seed."""
    eb = SanafeEnergyBreakdown.zero()
    assert eb.total_j == 0.0
    assert eb.synapse_j == eb.dendrite_j == eb.soma_j == eb.network_j == 0.0


def test_sanafe_energy_breakdown_add_sums_components_and_total():
    a = SanafeEnergyBreakdown(synapse_j=1.0, dendrite_j=2.0, soma_j=3.0,
                              network_j=4.0, total_j=10.0)
    b = SanafeEnergyBreakdown(synapse_j=0.5, dendrite_j=1.5, soma_j=2.5,
                              network_j=3.5, total_j=8.0)
    c = a.add(b)
    assert c.synapse_j == pytest.approx(1.5)
    assert c.dendrite_j == pytest.approx(3.5)
    assert c.soma_j == pytest.approx(5.5)
    assert c.network_j == pytest.approx(7.5)
    assert c.total_j == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Pickle round-trip — the runner publishes the report via add_entry(..., "pickle")
# ---------------------------------------------------------------------------


def test_sanafe_run_record_pickle_round_trip():
    """The full record must survive pickle for cache persistence."""
    rec = _run_record(
        segments={0: _segment(stage_index=0),
                  1: _segment(stage_index=1, stage_name="stage1")},
        compute_outputs={5: np.arange(4, dtype=np.float32).reshape(2, 2)},
    )
    blob = pickle.dumps(rec)
    restored = pickle.loads(blob)

    assert isinstance(restored, SanafeRunRecord)
    assert restored.sample_index == rec.sample_index
    assert restored.T == rec.T
    assert sorted(restored.segments.keys()) == [0, 1]
    np.testing.assert_array_equal(
        restored.segments[0].seg_input_spike_count,
        rec.segments[0].seg_input_spike_count,
    )
    np.testing.assert_array_equal(
        restored.segments[0].per_core[0].output_spike_count,
        rec.segments[0].per_core[0].output_spike_count,
    )
    np.testing.assert_array_equal(restored.compute_outputs[5],
                                  rec.compute_outputs[5])
    assert restored.aggregate_energy.total_j == pytest.approx(rec.aggregate_energy.total_j)


def test_sanafe_run_record_pickle_round_trip_with_traces():
    """Optional per-neuron traces must also survive pickle."""
    seg = _segment(stage_index=0)
    seg.per_neuron_spike_counts = np.asarray([1, 2, 3], dtype=np.int64)
    seg.per_neuron_spike_trace = np.ones((3, 8), dtype=np.uint8)
    seg.per_neuron_potential_trace = np.zeros((3, 8), dtype=np.float32)
    seg.message_trace = [{"src": 0, "dst": 1, "hops": 2}]
    rec = _run_record(segments={0: seg})

    restored = pickle.loads(pickle.dumps(rec))

    np.testing.assert_array_equal(restored.segments[0].per_neuron_spike_counts,
                                  seg.per_neuron_spike_counts)
    np.testing.assert_array_equal(restored.segments[0].per_neuron_spike_trace,
                                  seg.per_neuron_spike_trace)
    np.testing.assert_array_equal(restored.segments[0].per_neuron_potential_trace,
                                  seg.per_neuron_potential_trace)
    assert restored.segments[0].message_trace == [{"src": 0, "dst": 1, "hops": 2}]

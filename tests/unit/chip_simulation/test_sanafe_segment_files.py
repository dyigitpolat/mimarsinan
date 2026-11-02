"""[U-SFR] A SANA-FE neural segment's record persisted at the segment's end: one file per segment holding the full
``SanafeSegmentRecord`` (its message trace streamed into the file, never built twice in memory), the file's path,
sha256 and size kept by the step, beside the totals and spike counts the step's own readers need.

Readers of the step's record, each read from the persisted form and held equal to the in-memory form:
- ``SanafeStepReport.to_snapshot_dict`` (the GUI tab, ``gui/snapshot`` builders and rebuild; the Deployment Record
  step's SANA-FE fragments: energy, per-segment timings, NoC traffic, floorplan, tiles, S and depth; the cost record
  is their projection; ``cost_extraction``'s snapshot readers);
- ``SanafeRunRecord.to_hcm_subset`` (the spike-parity gate and the spike-count certificate) and the TTFS subsets;
- ``SanafeRunRecord.total_synaptic_events`` (the event census's SANA-FE cross-check, U-S);
- the run's totals (energy, spikes, packets, the longest simulated time).

No SANA-FE here: the records are built by hand (``tests/integration/test_sanafe_flat_segments.py`` runs the step on
real SANA-FE).
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import shutil

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe.analysis import _flatten_message_trace
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeArchGeometry,
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeNocLinkLoad,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport


def _canonical(value):
    """Dataclasses by field, mappings by sorted key, arrays by dtype, shape and bytes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return ("dataclass", type(value).__name__,
                tuple((f.name, _canonical(getattr(value, f.name))) for f in dataclasses.fields(value)))
    if isinstance(value, dict):
        return ("dict", tuple(sorted(((repr(k), _canonical(v)) for k, v in value.items()), key=lambda kv: kv[0])))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tobytes())
    return ("value", type(value).__name__, repr(value))


def _eb(total):
    return SanafeEnergyBreakdown(synapse_j=0.4 * total, dendrite_j=0.1 * total, soma_j=0.3 * total,
                                 network_j=0.2 * total, total_j=total)


def _raw_message_trace(rng, cycles, per_cycle):
    """SANA-FE's per-cycle message lists, placeholders included (``_flatten_message_trace`` drops them)."""
    trace = []
    for t in range(cycles):
        events = [{"placeholder": True, "timestep": t}]
        for m in range(per_cycle):
            events.append({"timestep": t, "src_tile_id": int(rng.integers(0, 4)), "dest_tile_id": int(rng.integers(0, 4)),
                           "src_x": 0, "src_y": int(m % 2), "dest_x": 1, "dest_y": 0, "spikes": int(rng.integers(1, 5)),
                           "hops": int(rng.integers(0, 3)), "generation_delay": float(rng.uniform(0, 1e-8)),
                           "network_delay": float(rng.uniform(0, 1e-8))})
        trace.append(events)
    return trace


def _segment(stage_index, rng, *, cores=3, axons=5, neurons=4, T=8, packets=6, synaptic_events=17):
    """A segment record as the runner finalizes it, with its flattened message trace (or none at ``packets=0``)."""
    per_core = []
    for c in range(cores):
        raster = rng.integers(0, 2, size=(neurons, T)).astype(np.uint8)
        per_core.append(SanafeCoreRecord(
            core_index=c, n_neurons=neurons, n_axons_used=axons, core_latency=c % 2, has_hardware_bias=bool(c % 2),
            n_always_on_axons=1, spikes_fired=int(raster.sum()), input_neuron_spikes_fired=3,
            input_spike_count=rng.integers(0, T, size=axons).astype(np.int64),
            output_spike_count=raster.sum(axis=1).astype(np.int64), energy=_eb(0.5 + c), spike_raster=raster))
    raw = _raw_message_trace(rng, T, packets) if packets else None
    record = SanafeSegmentRecord(
        stage_index=stage_index, stage_name=f"stage{stage_index}", schedule_segment_index=stage_index,
        schedule_pass_index=0, timesteps_executed=T + 1, sim_time_s=1.25e-6 * (stage_index + 1),
        energy=_eb(2.0 + stage_index), spikes=40 + stage_index, packets_sent=packets * T, neurons_updated=99,
        neurons_fired=7, seg_input_rates=rng.uniform(0, 1, size=(1, 6)).astype(np.float32),
        seg_input_spike_count=rng.integers(0, T, size=6).astype(np.int64),
        seg_output_spike_count=rng.integers(0, T, size=cores * neurons).astype(np.int64),
        synaptic_events=synaptic_events, per_core=per_core,
        per_tile=[SanafeTileRecord(tile_index=0, cores=list(range(cores)), energy=_eb(1.0), spikes_fired=9,
                                   packets_sent=4, mesh_x=0, mesh_y=0)],
        per_neuron_spike_trace=rng.integers(0, 2, size=(cores * neurons, T)).astype(np.uint8),
        message_trace=_flatten_message_trace(raw),
        arch_geometry=SanafeArchGeometry(width=2, height=2, tiles_xy=[[0, 0], [0, 1], [1, 0], [1, 1]]),
        noc_link_load=[SanafeNocLinkLoad(from_x=0, from_y=0, to_x=1, to_y=0, packet_count=packets)],
        noc_traffic_per_cycle=[[[0, 0, 1, 0, packets]] for _ in range(T)],
        tile_packets_per_cycle=[{1: packets} for _ in range(T)],
        inter_tile_packets=packets * T, intra_tile_packets=2, input_path_packets=3, chip_spike_count=40 + stage_index,
        lif_spike_count=30)
    return record, raw


def _run_record(segments, sample_index=0, T=8):
    agg = _eb(0.0)
    for seg in segments.values():
        agg = agg.add(seg.energy)
    return SanafeRunRecord(
        arch_preset="loihi", arch_name="mimarsinan_loihi_test", sample_index=sample_index, T=T, segments=segments,
        compute_outputs={7: np.arange(4, dtype=np.float64)}, aggregate_energy=agg,
        aggregate_sim_time_s=max(s.sim_time_s for s in segments.values()),
        total_spikes=sum(s.spikes for s in segments.values()),
        total_packets=sum(s.packets_sent for s in segments.values()))


def _store(root):
    from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentStore

    return SanafeSegmentStore(root, "SANA-FE Simulation.sanafe_segments")


def _persist(store, sample_index, record, raw):
    """What the runner hands the store: the finalized record without its flattened trace, and SANA-FE's raw trace."""
    return store.persist(sample_index, dataclasses.replace(record, message_trace=None), raw)


def _both_forms(tmp_path, n_segments=3, packets=(6, 0, 11), synaptic_events=(17, 5, 23)):
    rng = np.random.default_rng(3)
    store = _store(tmp_path / "run")
    full, kept = {}, {}
    for k in range(n_segments):
        stage = 2 + 5 * k
        record, raw = _segment(stage, rng, packets=packets[k], synaptic_events=synaptic_events[k])
        full[stage] = record
        kept[stage] = _persist(store, 0, record, raw)
    return _run_record(full), _run_record(kept), store


# --------------------------------------------------------------------------------------------- the file and its record


def test_a_persisted_segment_loads_back_field_for_field_with_its_message_trace(tmp_path):
    rng = np.random.default_rng(0)
    record, raw = _segment(4, rng, packets=9)
    kept = _persist(_store(tmp_path), 0, record, raw)
    assert _canonical(kept.load()) == _canonical(record)
    assert kept.message_count == len(record.message_trace) == 9 * 8
    without = kept.load(with_message_trace=False)
    assert without.message_trace is None
    assert _canonical(dataclasses.replace(without, message_trace=record.message_trace)) == _canonical(record)


def test_a_segment_without_messages_loads_back_with_no_trace(tmp_path):
    record, raw = _segment(4, np.random.default_rng(1), packets=0)
    assert record.message_trace is None and raw is None
    kept = _persist(_store(tmp_path), 0, record, raw)
    assert kept.message_count == 0
    assert _canonical(kept.load()) == _canonical(record)
    only_placeholders = [[{"placeholder": True}], []]
    kept2 = _persist(_store(tmp_path / "b"), 0, record, only_placeholders)
    assert kept2.load().message_trace is None and kept2.message_count == 0


def test_the_kept_form_names_its_file_by_path_sha256_and_size(tmp_path):
    import hashlib

    record, raw = _segment(12, np.random.default_rng(2))
    kept = _persist(_store(tmp_path), 3, record, raw)
    path = tmp_path / kept.path
    assert path.is_file() and path.parent.name == "SANA-FE Simulation.sanafe_segments"
    assert kept.root == str(tmp_path)
    assert kept.size_bytes == path.stat().st_size
    assert kept.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    other = _persist(_store(tmp_path), 3, *_segment(17, np.random.default_rng(2)))
    assert other.path != kept.path                        # one file per sample and segment


def test_the_message_trace_is_stored_compressed(tmp_path):
    """A flattened message pickles to hundreds of bytes (its keys spelled out each time); the file stores the trace
    at under half of that even with this fixture's random delays (SANA-FE's own traces compress far further), and
    reads it back equal."""
    rng = np.random.default_rng(6)
    record, raw = _segment(9, rng, T=8, packets=1024)
    kept = _persist(_store(tmp_path), 0, record, raw)
    pickled = len(pickle.dumps(record.message_trace, protocol=pickle.HIGHEST_PROTOCOL))
    without = len(pickle.dumps(dataclasses.replace(record, message_trace=None)))
    assert kept.size_bytes - without < pickled / 2, (kept.size_bytes - without, pickled)
    assert kept.load().message_trace == record.message_trace


def test_a_changed_or_missing_file_is_refused_by_name(tmp_path):
    from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentFileRefusal
    from mimarsinan.common.refusal import ToolchainRefusal

    record, raw = _segment(2, np.random.default_rng(4))
    kept = _persist(_store(tmp_path), 0, record, raw)
    path = tmp_path / kept.path
    data = bytearray(path.read_bytes())
    data[-3] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(SanafeSegmentFileRefusal, match="sha256") as err:
        kept.load()
    assert isinstance(err.value, ToolchainRefusal)
    path.unlink()
    with pytest.raises(SanafeSegmentFileRefusal, match="missing"):
        kept.load(with_message_trace=False)


def test_the_kept_form_holds_no_trace_raster_or_per_tile_record(tmp_path):
    """What the step keeps in memory per segment: totals, the spike counts its certificates compare, and the file."""
    _full, kept_run, _store_ = _both_forms(tmp_path)
    for seg in kept_run.segments.values():
        names = {f.name for f in dataclasses.fields(seg)}
        assert not names & {"message_trace", "per_neuron_spike_trace", "per_neuron_potential_trace", "per_tile",
                            "noc_traffic_per_cycle", "cycle_energy", "connectivity"}
        for core in seg.per_core:
            assert not {f.name for f in dataclasses.fields(core)} & {"spike_raster", "energy", "final_potential"}
    small = len(pickle.dumps(kept_run))
    large = len(pickle.dumps(_full))
    assert small * 3 < large


# ----------------------------------------------------------------------------------------- readers of the step record


def test_the_spike_count_subset_reads_equal_from_the_kept_form(tmp_path):
    full, kept, _ = _both_forms(tmp_path)
    assert _canonical(kept.to_hcm_subset()) == _canonical(full.to_hcm_subset())


def test_the_run_totals_and_the_event_census_read_equal_from_the_kept_form(tmp_path):
    full, kept, _ = _both_forms(tmp_path)
    assert kept.total_synaptic_events == full.total_synaptic_events == 17 + 5 + 23
    for stage, seg in kept.segments.items():
        ref = full.segments[stage]
        assert (seg.energy, seg.sim_time_s, seg.spikes, seg.packets_sent) == (
            ref.energy, ref.sim_time_s, ref.spikes, ref.packets_sent)
    full_none, kept_none, _ = _both_forms(tmp_path / "none", synaptic_events=(17, None, 23))
    assert kept_none.total_synaptic_events is None and full_none.total_synaptic_events is None


def test_the_ttfs_subsets_read_equal_from_the_kept_form(tmp_path):
    rng = np.random.default_rng(5)
    store = _store(tmp_path)
    full, kept = {}, {}
    for stage in (1, 3):
        record, raw = _segment(stage, rng)
        for core in record.per_core:
            core.output_activation = rng.uniform(0, 1, size=core.n_neurons)
        record.contract_ttfs_cores = [{"core_index": 0, "activation": [0.25, 0.5]}]
        record.contract_ttfs_seg_output = rng.uniform(0, 1, size=4)
        full[stage] = record
        kept[stage] = _persist(store, 0, record, raw)
    a, b = _run_record(full), _run_record(kept)
    for name in ("to_ttfs_contract_subset", "to_ttfs_hardware_subset"):
        assert _canonical(getattr(b, name)(spiking_mode="ttfs")) == _canonical(getattr(a, name)(spiking_mode="ttfs"))


def test_the_report_snapshot_reads_equal_from_the_persisted_form(tmp_path):
    full, kept, _ = _both_forms(tmp_path)
    a = SanafeStepReport.from_records("loihi", [full])
    b = SanafeStepReport.from_records("loihi", [kept])
    assert a.aggregate == b.aggregate
    sa, sb = a.to_snapshot_dict(), b.to_snapshot_dict()
    assert json.dumps(sb, sort_keys=False) == json.dumps(sa, sort_keys=False)
    flags = [seg["has_message_trace"] for seg in sb["per_sample"][0]["segments"]]
    assert flags == [True, False, True]


def test_the_deployment_record_fragments_and_cost_axes_read_equal(tmp_path):
    from mimarsinan.chip_simulation import cost_extraction as ce
    from mimarsinan.deployment_record.build import from_simulators as fs

    full, kept, _ = _both_forms(tmp_path)
    sa = SanafeStepReport.from_records("loihi", [full]).to_snapshot_dict()
    sb = SanafeStepReport.from_records("loihi", [kept]).to_snapshot_dict()
    for reader in (fs.energy_record_from_sanafe, fs.segment_timings_from_sanafe, fs.noc_traffic_from_sanafe,
                   fs.tiles_from_sanafe, fs.s_global_from_sanafe, fs.depth_from_sanafe, fs.latency_steps_from_sanafe,
                   ce._neuron_steps_from_sanafe_snapshot, ce._latency_steps_from_sanafe_snapshot,
                   ce._global_s_from_sanafe_snapshot, ce._depth_from_sanafe_snapshot):
        assert reader(sb) == reader(sa), reader.__name__
    assert fs.floorplan_from_sanafe(sb, cores_per_tile=4, derivation="declared") == fs.floorplan_from_sanafe(
        sa, cores_per_tile=4, derivation="declared")


def test_a_moved_run_directory_reads_equal_after_relocation_and_the_gui_rebuilds_it(tmp_path):
    from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk

    full, kept, store = _both_forms(tmp_path)
    report = SanafeStepReport.from_records("loihi", [kept])
    run = tmp_path / "run"
    with open(run / "SANA-FE Simulation.sanafe_simulation_results.pickle", "wb") as f:
        pickle.dump(report, f)
    moved = tmp_path / "moved"
    shutil.move(str(run), str(moved))
    expected = SanafeStepReport.from_records("loihi", [full]).to_snapshot_dict()
    assert json.dumps(report.relocated(moved).to_snapshot_dict()) == json.dumps(expected)
    rebuilt = rebuild_step_snapshot_from_disk(str(moved), "SANA-FE Simulation")
    assert rebuilt is not None
    assert json.dumps(rebuilt[0]["sanafe_simulation"]) == json.dumps(expected)


def test_a_report_pickled_before_the_change_reads_as_before(tmp_path):
    """Guard: a report holding full records (the earlier persisted form) still reads, unchanged."""
    full, _kept, _ = _both_forms(tmp_path)
    report = pickle.loads(pickle.dumps(SanafeStepReport.from_records("loihi", [full])))
    snap = report.to_snapshot_dict()
    assert [s["stage_index"] for s in snap["per_sample"][0]["segments"]] == [2, 7, 12]
    assert snap["per_sample"][0]["segments"][0]["has_message_trace"] is True
    assert report.relocated(tmp_path / "elsewhere").to_snapshot_dict() == snap


def test_each_segment_record_is_read_back_in_stage_order(tmp_path):
    full, kept, _ = _both_forms(tmp_path)
    got = [(stage, _canonical(rec)) for stage, rec in kept.segment_records()]
    assert got == [(stage, _canonical(full.segments[stage])) for stage in sorted(full.segments)]
    assert [stage for stage, _rec in full.segment_records()] == sorted(full.segments)

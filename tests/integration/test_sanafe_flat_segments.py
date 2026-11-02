"""[U-SFR] In real SANA-FE: the SANA-FE Simulation step runs in flat memory across neural segments.

At each segment's end the step persists that segment's full record (one file per sample and segment, its message
trace streamed into the file) and frees it with the segment's chip; the step keeps in memory only the totals and the
spike counts its certificates compare. Before the change every segment record (its message trace about 3 KB per
packet) and the last chip stayed in memory until the step ended, so the step's peak grew with the segment count.

The fixture is a chain of identical units (``_chain``): each unit's cores read the previous unit's outputs. The step
runs on a ``MockPipeline`` with its HCM reference taken from the same program run in the runner's default mode (every
record kept in memory: the form before the change), so its parity gate and certificate run as in a session.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _have_sanafe(), reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.slow,
    pytest.mark.integration,
]

REPO = Path(__file__).resolve().parents[2]
STEP = "SANA-FE Simulation"
KEY = f"{STEP}.sanafe_simulation_results"
CORES, AXONS, NEURONS, IN, T = 16, 32, 32, 64, 16
MiB = 2 ** 20


def _chain(units: int, *, saturating: bool):
    """``units`` neural units of CORES cores (AXONS x NEURONS); unit 0 reads the IN raw inputs, unit k the outputs of
    unit k - 1. ``saturating``: every unit has the same positive weights, so from unit 1 on each segment carries the
    same traffic (every neuron fires every step); otherwise mixed-sign weights give each unit its own activity."""
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage, SegmentIOSlice
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    stages, node, size = [], -2, IN
    for u in range(units):
        rng = np.random.default_rng(0 if saturating else 11 + u)
        cores = []
        for c in range(CORES):
            core = HardCore(AXONS, NEURONS, has_bias_capability=False)
            low = 0.2 if saturating else -0.55
            core.core_matrix = rng.uniform(low, 0.9, size=(AXONS, NEURONS)).astype(np.float32)
            core.axon_sources = [SpikeSource(-2, (c * 7 + a) % size, is_input=True) for a in range(AXONS)]
            core.available_axons = core.available_neurons = 0
            core.threshold, core.latency = (4.0 if saturating else 6.0), 0
            cores.append(core)
        seg = HardCoreMapping([])
        seg.cores = cores
        seg.output_sources = np.asarray([SpikeSource(c, n) for c in range(CORES) for n in range(NEURONS)], dtype=object)
        out = CORES * NEURONS
        stages.append(HybridStage(kind="neural", name=f"unit{u}", hard_core_mapping=seg,
                                  input_map=[SegmentIOSlice(node_id=node, offset=0, size=size)],
                                  output_map=[SegmentIOSlice(node_id=u, offset=0, size=out)]))
        node, size = u, out
    return HybridHardCoreMapping(stages=stages, output_sources=np.asarray(
        [IRSource(node_id=units - 1, index=i) for i in range(size)], dtype=object))


def _sample():
    return np.random.default_rng(1).uniform(0, 1, size=(1, IN)).astype(np.float32)


def _platform():
    return {"cores": [{"max_axons": AXONS, "max_neurons": NEURONS, "count": CORES}],
            "cores_per_tile": 0, "tile_grid_rows": 0, "tile_grid_cols": 0}


def _pipeline(work: Path, mapping):
    sys.path.insert(0, str(REPO / "tests"))
    from conftest import MockPipeline, make_tiny_supermodel

    work.mkdir(parents=True, exist_ok=True)
    pipeline = MockPipeline(working_directory=str(work))
    pipeline.config.update({"spiking_mode": "lif", "firing_mode": "Default", "spike_generation_mode": "Uniform",
                            "thresholding_mode": "<", "simulation_steps": T, "sanafe_sample_count": 1,
                            "sanafe_arch_preset": "loihi", "sanafe_custom_arch_path": None})

    class _Reporter:
        def __init__(self):
            self.events = []

        def report(self, name, value):
            self.events.append((name, value))

    pipeline.reporter = _Reporter()
    pipeline.set_target_metric(0.5)
    pipeline.seed("model", make_tiny_supermodel(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", mapping, step_name="Hard Core Mapping")
    pipeline.seed("platform_constraints_resolved", _platform(), step_name="Model Configuration")
    return pipeline


def _default_record(pipeline, mapping, x):
    """The program run in the runner's default mode, as the step builds its runner: every record kept in memory."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
    from mimarsinan.pipelining.core.simulation_factory import build_deployment_contract

    runner = SanafeRunner(mapping=mapping, simulation_length=T, contract=build_deployment_contract(pipeline),
                          pass_transfer=run_pass_transfer(pipeline.config), arch_preset="loihi", custom_arch_path=None,
                          log_potential_trace=False, log_message_trace=True, cores_per_tile=0, tile_grid_rows=0,
                          tile_grid_cols=0, declared_core_capacity=CORES)
    return runner.run(x, sample_index=0)


def _run_step(monkeypatch, work: Path, mapping, *, reference=None, capture=None, runners=None):
    """The step on ``mapping`` with the default-mode run's spike counts as its HCM reference. ``capture`` collects
    each record exactly as the step hands it to the store (a deep copy, its flattened message trace attached)."""
    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as step_mod

    x = _sample()
    pipeline = _pipeline(work, mapping)
    if reference is None:
        reference = _default_record(pipeline, mapping, x).to_hcm_subset()
    monkeypatch.setattr(step_mod, "load_test_samples_by_index",
                        lambda factory, indices, num_workers=4: [torch.from_numpy(x)])
    monkeypatch.setattr(step_mod, "record_hcm_reference",
                        lambda pipeline, m, sample, sample_index=0, device=None: (None, reference))
    if capture is not None:
        from mimarsinan.chip_simulation.sanafe.analysis import _flatten_message_trace
        from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentStore

        persist = SanafeSegmentStore.persist

        def capturing(self, sample_index, record, message_trace):
            capture[(sample_index, record.stage_index)] = dataclasses.replace(
                copy.deepcopy(record), message_trace=_flatten_message_trace(message_trace))
            return persist(self, sample_index, record, message_trace)

        monkeypatch.setattr(SanafeSegmentStore, "persist", capturing)
    if runners is not None:
        base = step_mod.SanafeRunner

        class Recording(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                runners.append(self)

        monkeypatch.setattr(step_mod, "SanafeRunner", Recording)
    step = step_mod.SanafeSimulationStep(pipeline)
    step.name = STEP
    pipeline.prepare_step(step)
    step.run()
    monkeypatch.undo()
    return pipeline.cache[KEY], reference, pipeline


def _canonical(value):
    from integration.test_sanafe_placed_region import _canonical as canonical

    return canonical(value)


def _with_deltas(reference, record):
    """The base step's per-core HCM deltas (``hcm_diff``), stamped as it stamped them after the run."""
    from mimarsinan.chip_simulation.sanafe.records import SanafeCoreDiff

    for stage, seg in record.segments.items():
        ref_seg = reference.segments.get(stage)
        if ref_seg is None or not ref_seg.cores or not seg.per_core:
            continue
        by_index = {c.core_index: c for c in ref_seg.cores}
        seg.hcm_diff = [SanafeCoreDiff(core_index=int(c.core_index),
                                       input_delta_sum=int(c.input_spike_count.sum()) - int(by_index[c.core_index].input_spike_count.sum()),
                                       output_delta_sum=int(c.output_spike_count.sum()) - int(by_index[c.core_index].output_spike_count.sum()))
                        for c in seg.per_core if c.core_index in by_index]
    return record


# ------------------------------------------------------------------------------------------------------ memory


def _traced_peak(monkeypatch, work, units):
    """The step's traced peak on ``units`` identical segments, after one unmeasured run of the same step (so neither
    side carries the process's first-run allocations)."""
    mapping = _chain(units, saturating=True)
    pipeline = _pipeline(work / "ref", mapping)
    reference = _default_record(pipeline, mapping, _sample()).to_hcm_subset()
    _run_step(monkeypatch, work / "warm", mapping, reference=reference)
    gc.collect()
    tracemalloc.start()
    start = tracemalloc.get_traced_memory()[0]
    report, _ref, _p = _run_step(monkeypatch, work / "run", mapping, reference=reference)
    peak = tracemalloc.get_traced_memory()[1] - start
    tracemalloc.stop()
    return peak, report


def _count_bytes(run_record):
    """The bytes of the spike counts a run record keeps per segment (the certificates' inputs)."""
    out = []
    for seg in run_record.segments.values():
        n = seg.seg_input_rates.nbytes + seg.seg_input_spike_count.nbytes + seg.seg_output_spike_count.nbytes
        n += sum(c.input_spike_count.nbytes + c.output_spike_count.nbytes for c in seg.per_core)
        out.append(n)
    return out


def test_the_step_peak_does_not_grow_with_the_segment_count(monkeypatch, tmp_path):
    """Six identical segments peak within two segments' peak plus the four extra segments' kept spike counts and a
    fixed 1 MiB: one segment's working set at a time. Before the change each extra segment's record stayed (its
    flattened message trace about 24 MiB here), so the peak grew by about 4 x 24 MiB."""
    peak2, _ = _traced_peak(monkeypatch, tmp_path / "k2", 2)
    peak6, report6 = _traced_peak(monkeypatch, tmp_path / "k6", 6)
    counts = _count_bytes(report6.per_sample[0])
    allowance = 2 * 4 * max(counts) + 1 * MiB
    assert peak6 - peak2 <= allowance, (
        f"the step's traced peak grew by {(peak6 - peak2) / MiB:.1f} MiB from 2 to 6 segments "
        f"(allowance {allowance / MiB:.2f} MiB: one segment's working set plus a fixed overhead)")


_CHILD = r"""
import ctypes, gc, json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import integration.test_sanafe_flat_segments as F
from pathlib import Path
mode, units, work = sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])

class _MP:
    def setattr(self, obj, name, value): self._undo = getattr(self, "_undo", []) + [(obj, name, getattr(obj, name))]; setattr(obj, name, value)
    def undo(self):
        for obj, name, value in reversed(getattr(self, "_undo", [])): setattr(obj, name, value)
        self._undo = []

def hwm():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmHWM:")).split()[1]) * 1024

mapping = F._chain(units, saturating=True)
pipeline = F._pipeline(work / "ref", mapping)
record = F._default_record(pipeline, mapping, F._sample())
reference = record.to_hcm_subset()
del record

def run(tag):
    if mode == "step":
        return F._run_step(_MP(), work / tag, mapping, reference=reference)
    return F._default_record(pipeline, mapping, F._sample())

run("warm")  # one unmeasured run first: neither side carries the process's first-run allocations
gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
open("/proc/self/clear_refs", "w").write("5")
base = hwm()
kept = run("run")
print(json.dumps({"growth": hwm() - base}))
"""


def _process_growth(tmp_path, mode, units):
    out = subprocess.run([sys.executable, "-c", _CHILD, str(REPO / "src"), str(REPO / "tests"), mode, str(units),
                          str(tmp_path / f"{mode}{units}")], capture_output=True, text=True, timeout=1800,
                         env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])["growth"]


def test_the_process_peak_does_not_grow_with_the_segment_count(tmp_path):
    """The whole process (SANA-FE's chips and the allocator included): from 2 to 10 identical segments the step's peak
    resident memory grows by at most a quarter of what the same program grows by with every record kept in memory
    (the runner's default mode, the step's form before the change)."""
    kept = _process_growth(tmp_path, "kept", 10) - _process_growth(tmp_path, "kept", 2)
    step = _process_growth(tmp_path, "step", 10) - _process_growth(tmp_path, "step", 2)
    assert kept > 64 * MiB, f"the fixture's kept records grow by only {kept / MiB:.0f} MiB"
    assert step <= kept / 4, (f"the step's peak grew by {step / MiB:.0f} MiB from 2 to 10 segments, against "
                              f"{kept / MiB:.0f} MiB with every record kept")


def test_the_step_releases_each_segment_chip(monkeypatch, tmp_path):
    runners = []
    _run_step(monkeypatch, tmp_path, _chain(3, saturating=False), runners=runners)
    assert len(runners) == 1
    assert runners[0]._last_chip is None


# ----------------------------------------------------------------------------------------- the persisted record


def test_each_persisted_segment_equals_the_record_the_step_kept_in_memory_before(monkeypatch, tmp_path):
    """The file holds the record field for field, message trace included, exactly as the step finalized it (the same
    run), and the same program's default-mode run reads equal on every field SANA-FE reproduces run to run."""
    from integration.test_sanafe_placed_region import _reproducible
    from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentFile

    mapping = _chain(4, saturating=False)
    captured = {}
    report, reference, pipeline = _run_step(monkeypatch, tmp_path / "run", mapping, capture=captured)
    rec = report.per_sample[0]
    assert sorted(rec.segments) == [0, 1, 2, 3]
    assert all(isinstance(seg, SanafeSegmentFile) for seg in rec.segments.values())
    for stage, seg in rec.segments.items():
        assert Path(seg.root) == tmp_path / "run"
        assert (Path(seg.root) / seg.path).is_file()
        loaded = seg.load()
        assert loaded.message_trace and len(loaded.message_trace) == seg.message_count
        assert _canonical(loaded) == _canonical(captured[(0, stage)]), stage
    default = _with_deltas(reference, _default_record(pipeline, mapping, _sample()))
    full = dataclasses.replace(rec, segments=dict(rec.segment_records()))
    assert _reproducible(full) == _reproducible(default)


def test_every_reader_reads_the_step_record_as_the_records_kept_in_memory(monkeypatch, tmp_path, capfd):
    from mimarsinan.certification.record_certificates import certify_run_records
    from mimarsinan.chip_simulation import cost_extraction as ce
    from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentFile
    from mimarsinan.chip_simulation.sanafe.stats import SanafeStepReport
    from mimarsinan.deployment_record.build import from_simulators as fs
    from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk

    mapping = _chain(4, saturating=False)
    captured = {}
    capfd.readouterr()
    report, reference, _p = _run_step(monkeypatch, tmp_path / "run", mapping, capture=captured)
    printed = [ln for ln in capfd.readouterr().out.splitlines() if ln.startswith("[SpikeCountCertificate]")]
    rec = report.per_sample[0]
    assert all(isinstance(seg, SanafeSegmentFile) for seg in rec.segments.values())
    in_memory = dataclasses.replace(rec, segments={stage: captured[(0, stage)] for stage in rec.segments})
    before = SanafeStepReport.from_records("loihi", [in_memory])
    assert report.aggregate == before.aggregate
    # the spike-parity gate's and the certificate's input, and the printed certificate line the harvest files
    assert _canonical(rec.to_hcm_subset()) == _canonical(in_memory.to_hcm_subset())
    assert printed == [f"[SpikeCountCertificate] {certify_run_records(reference, in_memory.to_hcm_subset(), backend='sanafe').summary()}"]
    # the event census's SANA-FE cross-check (U-S)
    assert rec.total_synaptic_events == in_memory.total_synaptic_events is not None
    # the GUI tab and the Deployment Record step's SANA-FE fragments (the cost record is their projection)
    snap, snap_before = report.to_snapshot_dict(), before.to_snapshot_dict()
    assert json.dumps(snap) == json.dumps(snap_before)
    for reader in (fs.energy_record_from_sanafe, fs.segment_timings_from_sanafe, fs.noc_traffic_from_sanafe,
                   fs.tiles_from_sanafe, fs.s_global_from_sanafe, fs.depth_from_sanafe, fs.latency_steps_from_sanafe,
                   ce._neuron_steps_from_sanafe_snapshot, ce._latency_steps_from_sanafe_snapshot):
        assert reader(snap) == reader(snap_before), reader.__name__
    # the cache's pickle, read from a moved run directory (the GUI's rebuild from disk)
    run = tmp_path / "run"
    with open(run / f"{KEY}.pickle", "wb") as f:
        pickle.dump(report, f)
    shutil.move(str(run), str(tmp_path / "moved"))
    rebuilt = rebuild_step_snapshot_from_disk(str(tmp_path / "moved"), STEP)
    assert rebuilt is not None and json.dumps(rebuilt[0]["sanafe_simulation"]) == json.dumps(snap_before)

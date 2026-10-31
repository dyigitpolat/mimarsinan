"""[U-TT T-1] In real SANA-FE: the chip sized by the placed cores simulates bit-identically to the declared floorplan,
its memory does not grow with the declared capacity, and one chip is alive at a time.

The identity fixture is a two-unit program whose units hold several tiles of cores with cross-tile fan-in, so
messages walk the mesh; each declared floorplan below leaves part of the mesh unplaced. Its reference is the same
runner on a chip that defines the whole declared floorplan (the builder before U-TT).

SANA-FE 2.1.1's detailed timing model depends on memory addresses: on a congested program one chip definition gives
another simulated time and message timing from run to run (U-TT probe: 1.20592e-05 against 1.20644e-05 s on this
fixture; with address randomization off, ``setarch -R``, one process configuration reproduces itself). Over 8 runs of
each floorplan below, the fields that vary for one chip definition are the segment and aggregate simulated times, the
message trace's timings and ids, and the per-cycle NoC traffic; the region and the declared floorplan differ on no
other field (``RA/11_wave2/final_units/utt/probes/t1_sanafe_variation``). So the records are compared pickled on every
field SANA-FE reproduces run to run, the message trace as the multiset of its messages' routes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


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


def _behavior():
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig

    return NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<",
                                spike_generation_mode="Uniform")


def _unit(rng, name, in_node, in_size, out_node, n_in, n_out, nper=4):
    """One neural unit: ``n_in`` cores read the unit's input, ``n_out`` cores read neurons of far-apart first cores."""
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage, SegmentIOSlice
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    cores = []
    for c in range(n_in + n_out):
        first = c < n_in
        srcs = ([SpikeSource(-2, a, is_input=True) for a in range(in_size)] if first
                else [SpikeSource(((c - n_in) * 3 + k * 5) % n_in, k % nper) for k in range(8)])
        core = HardCore(len(srcs), nper, has_bias_capability=False)
        core.core_matrix = rng.uniform(0.2, 0.9, size=(len(srcs), nper)).astype(np.float32)
        core.axon_sources = srcs
        core.available_axons = 0
        core.available_neurons = 0
        core.threshold = 1.0
        core.latency = 0 if first else 1
        cores.append(core)
    seg = HardCoreMapping([])
    seg.cores = cores
    outs = [SpikeSource(n_in + c, n) for c in range(n_out) for n in range(nper)]
    seg.output_sources = np.asarray(outs, dtype=object)
    return HybridStage(kind="neural", name=name, hard_core_mapping=seg,
                       input_map=[SegmentIOSlice(node_id=in_node, offset=0, size=in_size)],
                       output_map=[SegmentIOSlice(node_id=out_node, offset=0, size=len(outs))])


def _program(widths):
    """Two units: (n_in, n_out) of the first reading 8 inputs, then the second reading the first's outputs."""
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping

    rng = np.random.default_rng(0)
    (a_in, a_out), (b_in, b_out) = widths
    first = _unit(rng, "u0", -2, 8, 0, a_in, a_out)
    second = _unit(rng, "u1", 0, 4 * a_out, 1, b_in, b_out)
    return HybridHardCoreMapping(
        stages=[first, second],
        output_sources=np.asarray([IRSource(node_id=1, index=i) for i in range(4 * b_out)], dtype=object))


def _full_floorplan_builder(sanafe):
    """The builder before U-TT: the whole declared floorplan, every tile and core defined."""
    from mimarsinan.chip_simulation.sanafe.arch_synth import _render_arch_yaml

    def build(spec, *, custom_arch_path=None, thresholding_mode="<=", simulation_length=1):
        assert custom_arch_path is None
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(_render_arch_yaml(spec, thresholding_mode=thresholding_mode,
                                           simulation_length=simulation_length))
        try:
            return sanafe.load_arch(handle.name)
        finally:
            os.unlink(handle.name)
    return build


# (declared capacity, cores per tile, grid rows, grid cols, unit widths): every regime of derive_arch_spec, each with
# part of the declared mesh unplaced.
FLOORPLANS = {
    "two_columns_of_a_6x4_mesh": (96, 0, 0, 0, ((18, 10), (7, 5))),
    "a_partial_column": (96, 0, 0, 0, ((8, 4), (4, 2))),
    "explicit_8x3_grid": (48, 2, 3, 8, ((18, 10), (7, 5))),
    "scheduled_replicas": (32, 0, 0, 0, ((18, 10), (7, 5))),
    "no_declared_platform": (0, 0, 0, 0, ((18, 10), (7, 5))),
}


def _run(floorplan, monkeypatch, *, full: bool):
    import mimarsinan.chip_simulation.sanafe.runner as runner_mod
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    capacity, cpt, rows, cols, widths = floorplan
    if full:
        monkeypatch.setattr(runner_mod, "build_architecture", _full_floorplan_builder(runner_mod._sanafe()))
    runner = SanafeRunner(mapping=_program(widths), simulation_length=8, arch_preset="loihi", behavior=_behavior(),
                          declared_core_capacity=capacity, cores_per_tile=cpt, tile_grid_rows=rows,
                          tile_grid_cols=cols)
    x = np.random.default_rng(1).uniform(0, 1, size=(1, 8)).astype(np.float32)
    record = runner.run(x, sample_index=0)
    chip_cores = sum(len(tile.cores) for tile in runner._arch.tiles)
    declared = sum(runner_mod.derive_arch_spec(
        runner.mapping, preset_name="loihi", cores_per_tile=cpt, tile_grid_rows=rows, tile_grid_cols=cols,
        declared_core_capacity=capacity).n_cores_per_tile)
    monkeypatch.undo()
    return pickle.dumps(record), record, chip_cores, declared


#: A message's fields SANA-FE's timing model varies run to run (its id follows the processing order).
TIMING_FIELDS = ("generation_delay", "network_delay", "receive_delay", "blocked_delay", "sent_timestamp",
                 "received_timestamp", "processed_timestamp", "mid")


def _canonical(value):
    """A comparable form: dataclasses by field, mappings by sorted key (a pickle also keeps a dict's insertion order,
    which follows SANA-FE's message order), arrays by dtype, shape and bytes."""
    import dataclasses

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


def _reproducible(record):
    """The record without what SANA-FE varies run to run: the simulated times; the message trace as the sorted
    multiset of its messages' routes (timestep, spikes, hops, source, destination); the per-cycle NoC traffic as each
    cycle's sorted bins."""
    import dataclasses

    segments = {k: dataclasses.replace(
        seg, sim_time_s=None,
        noc_traffic_per_cycle=[sorted(map(tuple, cycle)) for cycle in (seg.noc_traffic_per_cycle or [])],
        message_trace=sorted(tuple(sorted((key, value) for key, value in message.items() if key not in TIMING_FIELDS))
                             for message in (seg.message_trace or [])))
        for k, seg in record.segments.items()}
    return _canonical(dataclasses.replace(record, segments=segments, aggregate_sim_time_s=None))


@pytest.mark.parametrize("name", sorted(FLOORPLANS))
def test_the_placed_region_simulates_identically_to_the_declared_floorplan(name, monkeypatch):
    _region_bytes, record, region_cores, declared = _run(FLOORPLANS[name], monkeypatch, full=False)
    _full_bytes, full_record, full_cores, _ = _run(FLOORPLANS[name], monkeypatch, full=True)
    assert full_cores == declared                    # the reference defines the whole declared floorplan
    assert region_cores < declared, (name, region_cores, declared)
    assert record.total_packets > 0 and record.total_spikes > 0
    assert _reproducible(record) == _reproducible(full_record), name


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1]]
import numpy as np
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage, SegmentIOSlice
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
# The probe's core shape (784 axons, 512 neurons, the SD-18 twin's): the chip's memory per core follows the shape.
core = HardCore(784, 512, has_bias_capability=False)
core.core_matrix = np.zeros((784, 512), dtype=np.float32)
core.core_matrix[:4, :3] = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]]
core.axon_sources = [SpikeSource(-2, a, is_input=True) for a in range(4)] + [SpikeSource(-1, 0, is_off=True)] * 780
core.available_axons = 780; core.available_neurons = 509; core.threshold = 1.0; core.latency = 0
seg = HardCoreMapping([]); seg.cores = [core]
seg.output_sources = np.asarray([SpikeSource(0, n) for n in range(3)], dtype=object)
stage = HybridStage(kind="neural", name="tiny", hard_core_mapping=seg, input_map=[SegmentIOSlice(-2, 0, 4)],
                    output_map=[SegmentIOSlice(0, 0, 3)])
mapping = HybridHardCoreMapping(stages=[stage],
                                output_sources=np.asarray([IRSource(0, i) for i in range(3)], dtype=object))
behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<",
                                spike_generation_mode="Uniform")
def hwm():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmHWM:")).split()[1])
runner = SanafeRunner(mapping=mapping, simulation_length=8, arch_preset="loihi", behavior=behavior,
                      declared_core_capacity=int(sys.argv[2]))
before = hwm()
rec = runner.run(np.asarray([[1.0, 0.5, 0.25, 0.0]], dtype=np.float32), sample_index=0)
print(json.dumps({"hwm_kib": hwm(), "before_kib": before, "spikes": int(rec.total_spikes)}))
"""


def _child_peak(capacity: int) -> dict:
    out = subprocess.run([sys.executable, "-c", _CHILD, str(REPO / "src"), str(capacity)], capture_output=True,
                         text=True, timeout=900, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_chips_memory_does_not_grow_with_the_declared_capacity():
    """The C5 probe's defect in small: one placed 784x512 core, declared capacity 4 against 240 (60 tiles). Before
    U-TT the chip cost about 7.6 MiB per declared core at this shape (about 1.8 GiB at 240, the probe's measured
    slope); now both chips are one tile."""
    small, large = _child_peak(4), _child_peak(240)
    assert small["spikes"] == large["spikes"] > 0
    grown_mib = (large["hwm_kib"] - small["hwm_kib"]) / 1024
    assert grown_mib < 256, f"peak grew by {grown_mib:.0f} MiB from 4 to 240 declared cores"


def test_one_chip_is_alive_at_a_time_and_each_unit_gets_a_fresh_chip(monkeypatch):
    """The fresh chip per unit stays (a reused chip is not bit-identical, U-TT probe), but the previous unit's chip is
    released before the next is built: two chips alive at once doubled the step's peak."""
    import mimarsinan.chip_simulation.sanafe.runner as runner_mod
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    real = runner_mod._sanafe()
    alive_at_build: list = []

    class CountingChip:
        alive = 0

        def __init__(self, arch):
            alive_at_build.append(CountingChip.alive)
            self._chip = real.SpikingChip(arch)
            CountingChip.alive += 1

        def __getattr__(self, name):
            return getattr(self._chip, name)

        def __del__(self):
            CountingChip.alive -= 1

    proxy = type("SanafeProxy", (), {"__getattr__": lambda _self, name: getattr(real, name),
                                     "SpikingChip": CountingChip})()
    monkeypatch.setattr(runner_mod, "_sanafe", lambda: proxy)
    runner = SanafeRunner(mapping=_program(((18, 10), (7, 5))), simulation_length=8, arch_preset="loihi",
                          behavior=_behavior(), declared_core_capacity=96)
    runner.run(np.random.default_rng(1).uniform(0, 1, size=(1, 8)).astype(np.float32), sample_index=0)
    assert len(alive_at_build) == 2                 # one fresh chip per unit
    assert alive_at_build == [0, 0], alive_at_build

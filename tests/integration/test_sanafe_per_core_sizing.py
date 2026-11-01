"""[U-SF] In real SANA-FE: the chip whose cores are sized by what the program places on them simulates the program as
the chip sized at its widest core did, its memory follows the per-core sizes, and a one-axon core binds its input.

SANA-FE 2.1.1 reads a core's declared size in two places only: one ``inputs[k]`` soma unit per declared input slot
(the net builder binds input axon ``a`` to ``inputs[a]``; a slot no neuron is mapped to is never in use, and the in-use
units keep their order) and ``max_neurons_supported``, the core's mapping capacity and the length of its timestep
buffer (indexed by a neuron's offset within the core). Tiles, cores, mesh and every energy and latency attribute are the
same. The reference below is the builder before U-SF: every core at the program's widest core and
``max_neurons_supported = max(8192, 2 * (neurons + slots))``.

As in ``test_sanafe_placed_region.py``, SANA-FE's timing model varies run to run with memory addresses on a congested
program (U-TT finding F-1), so records are compared on every field SANA-FE reproduces run to run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from integration.test_sanafe_placed_region import _reproducible


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


def _unit(rng, name, in_node, in_size, out_node, reads, nper=4):
    """One neural unit. ``reads[c]`` is core ``c``'s axon count: a positive count reads that many of the unit's
    inputs, a negative one reads that many neurons of the unit's earlier cores (the unit's own fan-in, across tiles)."""
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage, SegmentIOSlice
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    n_first = sum(1 for r in reads if r > 0)
    cores = []
    for c, r in enumerate(reads):
        srcs = ([SpikeSource(-2, a % in_size, is_input=True) for a in range(r)] if r > 0
                else [SpikeSource((k * 7 + c) % n_first, k % nper) for k in range(-r)])
        core = HardCore(len(srcs), nper, has_bias_capability=False)
        core.core_matrix = rng.uniform(0.2, 0.9, size=(len(srcs), nper)).astype(np.float32)
        core.axon_sources = srcs
        core.available_axons = 0
        core.available_neurons = 0
        core.threshold = 1.0 + 0.25 * (len(srcs) // 8)
        core.latency = 0 if r > 0 else 1
        cores.append(core)
    seg = HardCoreMapping([])
    seg.cores = cores
    outs = [SpikeSource(c, n) for c in range(n_first, len(reads)) for n in range(nper)]
    seg.output_sources = np.asarray(outs, dtype=object)
    return HybridStage(kind="neural", name=name, hard_core_mapping=seg,
                       input_map=[SegmentIOSlice(node_id=in_node, offset=0, size=in_size)],
                       output_map=[SegmentIOSlice(node_id=out_node, offset=0, size=len(outs))])


def _program():
    """Two units of mixed axon counts (4 to 40): the first reads 8 inputs, the second the first's outputs."""
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping

    rng = np.random.default_rng(0)
    first_reads = [8, 24, 4, 40, 8, 16, 4, 8, 32, 8] + [-(6 + 9 * (k % 4)) for k in range(8)]
    first = _unit(rng, "u0", -2, 8, 0, first_reads)
    second = _unit(rng, "u1", 0, 32, 1, [40, 8, 4, 16, 8] + [-(4 + 12 * (k % 3)) for k in range(5)])
    return HybridHardCoreMapping(
        stages=[first, second],
        output_sources=np.asarray([IRSource(node_id=1, index=i) for i in range(20)], dtype=object))


def _widest_core_builder(sanafe):
    """The builder before U-SF: the placed region with every core at the program's widest core and R's neuron rule."""
    from mimarsinan.chip_simulation.sanafe.arch_synth import _render_arch_yaml
    from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region

    def build(spec, *, custom_arch_path=None, thresholding_mode="<=", simulation_length=1):
        assert custom_arch_path is None
        uniform = dataclasses.replace(spec, core_sizes=()) if hasattr(spec, "core_sizes") else spec
        yaml = _render_arch_yaml(placed_region(uniform), thresholding_mode=thresholding_mode,
                                 simulation_length=simulation_length)
        slots = max(1, int(spec.axons_per_core))
        yaml = re.sub(r"max_neurons_supported: \d+",
                      f"max_neurons_supported: {max(8192, 2 * (int(spec.neurons_per_core) + slots))}", yaml)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(yaml)
        try:
            return sanafe.load_arch(handle.name)
        finally:
            os.unlink(handle.name)
    return build


def _run(monkeypatch, *, widest: bool, capacity=96):
    import mimarsinan.chip_simulation.sanafe.runner as runner_mod
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    if widest:
        monkeypatch.setattr(runner_mod, "build_architecture", _widest_core_builder(runner_mod._sanafe()))
    runner = SanafeRunner(mapping=_program(), simulation_length=8, arch_preset="loihi", behavior=_behavior(),
                          declared_core_capacity=capacity)
    x = np.random.default_rng(1).uniform(0, 1, size=(1, 8)).astype(np.float32)
    record = runner.run(x, sample_index=0)
    monkeypatch.undo()
    return record, runner


def test_the_per_core_chip_simulates_the_program_as_the_widest_core_chip_did(monkeypatch):
    record, runner = _run(monkeypatch, widest=False)
    reference, ref_runner = _run(monkeypatch, widest=True)
    assert record.total_spikes > 0 and record.total_packets > 0
    axons = {int(c.axons_per_core) for s in runner.mapping.stages for c in s.hard_core_mapping.cores}
    assert len(axons) >= 5 and max(axons) == 40                     # the chips differ: mixed core classes
    assert sum(len(t.cores) for t in runner._arch.tiles) == sum(len(t.cores) for t in ref_runner._arch.tiles)
    assert _reproducible(record) == _reproducible(reference)
    assert (record.aggregate_energy, record.total_spikes, record.total_packets) == (
        reference.aggregate_energy, reference.total_spikes, reference.total_packets)


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1]]
import numpy as np
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
import mimarsinan.chip_simulation.sanafe.runner as runner_mod
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage, SegmentIOSlice
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
# 32 cores of 8 axons and one of 2,048 (4 used): sized at the widest core the chip declares 33 x 2,048 input slots.
cores = []
for c in range(33):
    axons = 2048 if c == 32 else 8
    core = HardCore(axons, 4, has_bias_capability=False)
    core.core_matrix = np.zeros((axons, 4), dtype=np.float32)
    core.core_matrix[:4, :] = 0.6
    core.axon_sources = [SpikeSource(-2, a, is_input=True) for a in range(4)] + [SpikeSource(-1, 0, is_off=True)] * (axons - 4)
    core.available_axons = axons - 4; core.available_neurons = 0; core.threshold = 1.0; core.latency = 0
    cores.append(core)
seg = HardCoreMapping([]); seg.cores = cores
seg.output_sources = np.asarray([SpikeSource(c, n) for c in range(33) for n in range(4)], dtype=object)
stage = HybridStage(kind="neural", name="mixed", hard_core_mapping=seg, input_map=[SegmentIOSlice(-2, 0, 4)],
                    output_map=[SegmentIOSlice(0, 0, 132)])
mapping = HybridHardCoreMapping(stages=[stage],
                                output_sources=np.asarray([IRSource(0, i) for i in range(132)], dtype=object))
behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<",
                                spike_generation_mode="Uniform")
def rss():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS:")).split()[1])
runner = SanafeRunner(mapping=mapping, simulation_length=8, arch_preset="loihi", behavior=behavior,
                      declared_core_capacity=36)
runner._ensure_arch()
sanafe = runner_mod._sanafe()
before = rss()
chip = sanafe.SpikingChip(runner._arch)
held = rss() - before
del chip
rec = runner.run(np.asarray([[1.0, 0.5, 0.25, 0.75]], dtype=np.float32), sample_index=0)
print(json.dumps({"chip_kib": held, "spikes": int(rec.total_spikes)}))
"""


def test_a_bare_chip_holds_the_per_core_sizes_not_the_widest_core_times_the_cores():
    """33 cores, one of 2,048 axons: sized at the widest core the bare chip held about 33 x 19.1 MiB (U-SF probe:
    about 476 KiB per core plus 9.3 KiB per input slot); per core it holds about one wide core's 19 MiB."""
    out = subprocess.run([sys.executable, "-c", _CHILD, str(REPO / "src")], capture_output=True, text=True,
                         timeout=900, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["spikes"] > 0
    assert got["chip_kib"] / 1024 < 96, f"a bare chip held {got['chip_kib'] / 1024:.0f} MiB"


def test_a_program_of_one_axon_cores_binds_its_inputs():
    """The net builder binds input axon 0 to ``inputs[0]``; a one-slot core must define that name."""
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage, SegmentIOSlice
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    cores = []
    for c in range(3):
        core = HardCore(1, 2, has_bias_capability=False)
        core.core_matrix = np.asarray([[1.0, 0.5]], dtype=np.float32)
        core.axon_sources = [SpikeSource(-2, c, is_input=True)]
        core.available_axons = core.available_neurons = 0
        core.threshold, core.latency = 1.0, 0
        cores.append(core)
    seg = HardCoreMapping([])
    seg.cores = cores
    seg.output_sources = np.asarray([SpikeSource(c, n) for c in range(3) for n in range(2)], dtype=object)
    stage = HybridStage(kind="neural", name="one_axon", hard_core_mapping=seg, input_map=[SegmentIOSlice(-2, 0, 3)],
                        output_map=[SegmentIOSlice(0, 0, 6)])
    mapping = HybridHardCoreMapping(stages=[stage],
                                    output_sources=np.asarray([IRSource(0, i) for i in range(6)], dtype=object))
    runner = SanafeRunner(mapping=mapping, simulation_length=8, arch_preset="loihi", behavior=_behavior(),
                          declared_core_capacity=4)
    record = runner.run(np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32), sample_index=0)
    assert record.total_spikes > 0

"""[W33] nevresim reads a stage input in each core's local time, as the HCM does (compiled; slow).

Before W33 the generated simulator read its T-cycle stage input at the global cycle, so a core starting at latency
L > 0 read cycle t of a host-encoded slice at global cycle t (L cycles early, its last L cycles never), identity
programs included, and a carried slice read above latency 0 and a delayed slice under COLLAPSE were refused by name.
Each test runs the program on nevresim with its refusals in force and compares the final output counts, gathered from
the stages' output counts, with the HCM run of the identity program at atol 0 on 3 seeded samples."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from integration.parity_harness import ensure_nevresim_ready
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import REPLAY_DELAYS, carried_nodes, executed_core_latencies
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V

#: semantics -> (HCM run, nevresim declaration)
SEMANTICS = {
    "streamed_lif": (R.STREAMED_LIF, dict(spike_generation_mode="Uniform", firing_mode="Default", spiking_mode="lif",
                                          pass_transfer=VERBATIM)),
    "cascaded_ttfs": (R.CASCADED_TTFS, dict(spike_generation_mode="TTFS", firing_mode="TTFS",
                                            spiking_mode="ttfs_cycle_based", pass_transfer=COLLAPSE)),
}


def _late_readers(stage, nodes=None):
    """``(core, latency)`` per core that reads a stage-input row (of ``nodes``, or any) above executed latency 0."""
    rows = {}
    for s in stage.input_map:
        rows.update({i: int(s.node_id) for i in range(int(s.offset), int(s.offset) + int(s.size))})
    hcm = stage.hard_core_mapping
    return [(h, lat) for h, (core, lat) in enumerate(zip(hcm.cores, executed_core_latencies(hcm))) if lat > 0 and any(
        getattr(src, "is_input_", False) and not getattr(src, "is_off_", False)
        and not getattr(src, "is_always_on_", False) and (nodes is None or rows.get(int(src.neuron_)) in nodes)
        for src in core.axon_sources)]


def _samples():
    return [torch.rand(1, V.IN, generator=torch.Generator().manual_seed(b)) for b in range(3)]


def _gather(mapping, node):
    return np.stack([node[int(s.node_id)][:, int(s.index)] for s in np.asarray(mapping.output_sources).ravel()], 1)


def _hcm_final(mapping, xs, semantics):
    node = {}
    flow = R.flow(mapping, **SEMANTICS[semantics][0])

    def record(stage, counts):
        c = counts.detach().cpu().numpy().astype(np.int64)
        for s in stage.output_map:
            node[int(s.node_id)] = c[:, int(s.offset):int(s.offset) + int(s.size)]
    flow.stage_count_recorder = record
    with torch.no_grad():
        flow(torch.cat(xs))
    return _gather(mapping, node)


def _nevresim_final(mapping, xs, semantics, connectivity=None):
    from mimarsinan.chip_simulation.nevresim.connectivity import default_nevresim_connectivity_mode
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

    runner = object.__new__(SimulationRunner)
    vars(runner).update(
        **SEMANTICS[semantics][1], thresholding_mode="<=",
        nevresim_connectivity_mode=connectivity or default_nevresim_connectivity_mode(),
        simulation_step_timeout_s=600.0, membrane_readout=False, membrane_half_step_charge=False, weight_type=float,
        threshold_type=float, host_compute_device="cpu", stage_timer=None, simulation_length=V.T, mapping=mapping,
        num_classes=64, test_data=[(x[0].numpy().astype(np.float64), np.zeros(1)) for x in xs],
        test_targets=[torch.tensor(0) for _ in xs], test_input=[], working_directory=tempfile.mkdtemp())
    node = {}

    def record(stage, raw):
        counts = np.rint(np.asarray(raw)).astype(np.int64)
        for s in stage.output_map:
            node[int(s.node_id)] = counts[:, int(s.offset):int(s.offset) + int(s.size)]
    runner.stage_count_recorder = record
    runner._run_hybrid(mapping)
    return _gather(mapping, node)


def _assert_equal_to_the_hcm_identity_run(program, ir, semantics, connectivity=None):
    xs = _samples()
    want = _hcm_final(build_identity_hybrid_mapping(ir_graph=ir), xs, semantics)
    assert want.std() > 0
    assert np.array_equal(_hcm_final(program, xs, semantics), want)
    got = _nevresim_final(program, xs, semantics, connectivity)
    assert np.array_equal(got.reshape(want.shape), want)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("semantics", sorted(SEMANTICS))
@pytest.mark.parametrize("kind", ["skew3", "skip", "shift_probe"])
def test_an_identity_program_with_a_raised_host_reader(kind, semantics):
    """The identity program raises a reader of a host-encoded slice above latency 0 (ALAP)."""
    ensure_nevresim_ready()
    ir = V.ir_of(V.branchy_repr(kind))
    identity = build_identity_hybrid_mapping(ir_graph=ir)
    assert any(_late_readers(s) for s in V.neural_passes(identity))
    _assert_equal_to_the_hcm_identity_run(identity, ir, semantics)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("semantics", sorted(SEMANTICS))
def test_a_carried_slice_read_above_latency_zero(semantics):
    """Chain C 3: pass 4 reads its carried slice at latency 1 (refused by name before W33)."""
    ensure_nevresim_ready()
    ir = V.ir_of(V.chain_repr(seed=0))
    u1 = V.build(ir, 3, None)
    reader = u1.stages[4]
    assert _late_readers(reader, carried_nodes(u1.stages, reader)) == [(0, 1)]
    _assert_equal_to_the_hcm_identity_run(u1, ir, semantics)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_compile_time_chip_reads_in_core_local_time():
    """The same rule on nevresim's compile-time (constant-evaluated) chip: skew3's identity program, cascaded TTFS. (A
    LIF segment's spike-count record build does not compile on the compile-time chip, before W33 as after: the
    self-tests of ``_tests/all.hpp`` evaluate ``SpikingExecution`` in a constant expression and the recorder's
    ``make_unique`` is not constant-evaluable; the default runtime chip is unaffected.)"""
    ensure_nevresim_ready()
    ir = V.ir_of(V.branchy_repr("skew3"))
    identity = build_identity_hybrid_mapping(ir_graph=ir)
    assert any(_late_readers(s) for s in V.neural_passes(identity))
    _assert_equal_to_the_hcm_identity_run(identity, ir, "cascaded_ttfs", connectivity="compile_time")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_delayed_slice_under_collapse():
    """Fuzz seed 58 C 2 under cascaded TTFS: a slice delivered 1 cycle late (refused by name before W33)."""
    ensure_nevresim_ready()
    ir = V.ir_of(V.random_rep(58))
    u1 = V.build(ir, 2, None)
    assert any(getattr(s, REPLAY_DELAYS, None) for s in V.neural_passes(u1))
    _assert_equal_to_the_hcm_identity_run(u1, ir, "cascaded_ttfs")


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_generated_simulator_holds_the_stage_input_clock_self_test():
    """nevresim's own constant-evaluated self-test (``_tests/stage_input_clock.hpp``): two cores read one front-loaded
    input, at latencies 0 and 2, and fire on every input spike of their own window: 2 and 2."""
    from mimarsinan.common.build_utils import find_cpp20_compiler

    root = ensure_nevresim_ready()
    compiler, _family = find_cpp20_compiler()
    assert compiler is not None
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "clock.cpp"
        source.write_text('#include "_tests/stage_input_clock.hpp"\n'
                          "static_assert(nevresim::tests::test_stage_input_read_in_core_local_time());\n"
                          "int main() { return nevresim::tests::test_stage_input_read_in_core_local_time() ? 0 : 1; }\n")
        built = subprocess.run([compiler, "-std=c++20", "-O1", "-I", str(Path(root) / "include"), str(source), "-o",
                                str(Path(work) / "clock")], capture_output=True, text=True, timeout=500)
        assert built.returncode == 0, built.stderr[-2000:]
        assert subprocess.run([str(Path(work) / "clock")], timeout=60).returncode == 0

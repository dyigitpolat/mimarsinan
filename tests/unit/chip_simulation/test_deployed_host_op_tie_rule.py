"""[w2-parity fix round] One tie rule on the deployed side: the host-op evaluator decides exact lattice ties itself.

The chip cores of a weight-quantized LIF program decide an exact tie on the integer membrane lattice by exact
arithmetic every cycle (``membrane_integer_lattice``). A host-run perceptron (a subsumed encoding layer, the C5 T-LIF
stem) carries an armed lattice node that snaps only inside ``measurement_plane()``. Before this round the deployed
executors evaluated such a host op under whatever plane their CALLER had opened: the nevresim runner and its
certificate twin opened it, the Soft and Hard Core Mapping metrics, the SANA-FE step (its HCM reference and its runner)
and the Lava runner did not. On C5 the deployed metric flow then agreed with itself, outside against inside the
plane, on 0.8906 (tuned) and 0.7813 (PTQ) of the gate's samples, and the gate certified one reading while the step
reported another.

The rule now lives in the one host-op evaluator every deployed executor shares
(``hybrid_run.host_compute.execute_compute_op_torch``, which the numpy evaluator calls): the module runs in the
chip-lattice plane whatever the caller's context, so every deployed reading of a program is one reading. The tests
pin the wiring, the tie itself on one armed node, and the weight-quantized encoding-skip program on inputs on the
1/T grid (where its stem meets exact ties): the deployed flow reads it alike outside and inside the plane, and the
SANA-FE, Lava and nevresim runners read it as the in-plane NF (the gate's reading) does.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    execute_compute_op_numpy,
    execute_compute_op_torch,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import in_measurement_plane, measurement_plane

from unit.spiking.encoding_skip_program import (
    D,
    D_IN,
    DEPLOYED_SEMANTICS,
    SEMANTICS,
    T,
    WQ_BITS,
    WQ_THETA_ENC,
    WQ_THETAS,
    build_program,
    count_mismatches,
    deployed_flow,
    nf_window_counts,
    per_perceptron_counts,
    scm_window_counts,
)
from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC


def _input_op(module: nn.Module, width: int) -> ComputeOp:
    """A host op reading the raw input (node -2), its module the given one."""
    flat = [IRSource(node_id=-2, index=i) for i in range(width)]
    return ComputeOp(id=0, name="enc", op_type="module", input_sources=np.array(flat, dtype=object).reshape(1, -1),
                     params={"module": module}, input_shape=(width,), output_shape=(width,))


class _PlaneRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes: list = []

    def forward(self, x):
        self.planes.append(in_measurement_plane())
        return x


# ---------------------------------------------------------------------------
# The wiring: the evaluator opens the plane itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_the_host_op_evaluator_runs_its_module_in_the_chip_lattice_plane(backend):
    recorder = _PlaneRecorder()
    op = _input_op(recorder, 3)
    assert not in_measurement_plane()
    if backend == "torch":
        execute_compute_op_torch(op, torch.rand(2, 3), {}, arithmetic=SPIKING_ARITHMETIC)
    else:
        execute_compute_op_numpy(op, np.random.rand(2, 3), {}, device="cpu", dtype=np.float32)
    assert recorder.planes == [True], "the deployed host op decides lattice ties whatever its caller's plane"
    assert not in_measurement_plane(), "the plane closes with the evaluation"


# ---------------------------------------------------------------------------
# The tie on one armed node (the t0_05 encoder tread: theta_int 328, strict '<')
# ---------------------------------------------------------------------------

THETA, THETA_INT, STEPS = 1.8761228322982788, 328.0, 4
DUST = 6e-8  # the measured CUDA-vs-CPU GEMM delta on t0_05


def _armed_encoder() -> LIFActivation:
    lif = LIFActivation(STEPS, THETA, thresholding_mode="<")
    lif.set_membrane_lattice(THETA_INT)
    return lif


def test_an_armed_host_tie_is_decided_by_the_lattice_from_any_caller():
    tread = THETA / STEPS
    x = torch.tensor([[tread, tread - DUST, tread + DUST]], dtype=torch.float32)
    lif = _armed_encoder()
    with torch.no_grad():
        bare_outside = lif(x)
        with measurement_plane():
            bare_inside = lif(x)
        deployed_outside = execute_compute_op_torch(_input_op(lif, 3), x, {}, arithmetic=SPIKING_ARITHMETIC)
        with measurement_plane():
            deployed_inside = execute_compute_op_torch(_input_op(lif, 3), x, {}, arithmetic=SPIKING_ARITHMETIC)
    assert not torch.equal(bare_outside, bare_inside), "teeth: outside the plane dust decides the tread tie"
    assert torch.equal(deployed_outside, bare_inside) and torch.equal(deployed_inside, bare_inside)
    assert torch.round(deployed_outside / THETA * STEPS).tolist() == [[0.0, 0.0, 0.0]], "the chip holds a '<' tie"


# ---------------------------------------------------------------------------
# The weight-quantized encoding-skip program on inputs on the 1/T grid
# ---------------------------------------------------------------------------

N_GRID = 256


def _grid_samples(n: int = N_GRID) -> torch.Tensor:
    """Inputs on the stem's input grid (multiples of 1/T): its membranes then sit on the lattice and meet ties."""
    g = torch.Generator().manual_seed(7)
    return (torch.randint(0, T + 1, (n, D_IN), generator=g).double() / T).float()


def _tie_program(semantics: str):
    """The program, its grid samples, and the samples whose stem the plane decides (exact ties of the stem)."""
    retimed = SEMANTICS[semantics][0]
    program = build_program(WQ_THETA_ENC, retimed=retimed, thetas=WQ_THETAS, weight_bits=WQ_BITS)
    x = _grid_samples()
    stem = program.stem().params["module"]
    with torch.no_grad():
        outside = stem(x)
        with measurement_plane():
            inside = stem(x)
    ties = torch.nonzero((outside != inside).any(dim=1)).flatten()
    return program, x, ties


@pytest.mark.parametrize("semantics", DEPLOYED_SEMANTICS)
def test_the_deployed_flow_reads_the_host_run_stem_alike_outside_and_inside_the_plane(semantics):
    program, x, ties = _tie_program(semantics)
    assert ties.numel() > 0, "teeth: the stem meets exact ties on these inputs"
    flow = deployed_flow(program, semantics, membrane_integer_lattice=True)
    with torch.no_grad():
        outside = flow(x)
        with measurement_plane():
            inside = flow(x)
    differ = int((outside != inside).sum())
    assert differ == 0, (f"{semantics}: {differ} of {outside.numel()} deployed outputs depend on the caller's plane "
                         f"({ties.numel()} of {x.shape[0]} samples carry a stem tie)")
    with torch.no_grad():
        for i in ties[:4].tolist():
            _, rec_out = flow.forward_with_recording(x[i:i + 1], sample_index=i)
            with measurement_plane():
                _, rec_in = flow.forward_with_recording(x[i:i + 1], sample_index=i)
            for oid, value in rec_out.compute_outputs.items():
                np.testing.assert_array_equal(np.asarray(value), np.asarray(rec_in.compute_outputs[oid]),
                                              err_msg=f"{semantics}: host op {oid}, sample {i}")


def _behavior():
    return NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                spike_generation_mode="Uniform")


def _reference(program, x):
    """The gate's reading: the NF and the deployed flow, both inside the plane (the streamed semantics)."""
    flow = deployed_flow(program, "streamed_lif", membrane_integer_lattice=True)
    with measurement_plane():
        nf_out, nf = nf_window_counts(program, x, SEMANTICS["streamed_lif"][2]())
        scm_out, scm, _ = scm_window_counts(program, x, flow)
    assert count_mismatches(nf, scm) == {pi: 0 for pi in scm}, "the gate's twins agree in the plane"
    torch.testing.assert_close(nf_out.double() * T / program.theta_out, scm_out.double(), atol=1e-5, rtol=0.0)
    return flow, scm_out, scm


def _tie_samples():
    program, x, ties = _tie_program("streamed_lif")
    assert ties.numel() > 0
    return program, x[ties[:4]]


def test_the_deployed_metric_flow_outside_the_plane_reads_as_the_gate():
    program, x = _tie_samples()
    flow, ref_out, ref = _reference(program, x)
    out, counts, _ = scm_window_counts(program, x, flow)  # no plane: the SCM/HCM metrics' and record_hcm_reference's read
    assert torch.equal(out, ref_out)
    assert all(np.array_equal(counts[pi], ref[pi]) for pi in ref), "window counts, positionally"


def test_the_sanafe_runner_reads_the_tie_samples_as_the_gate():
    from integration.parity_harness import have_sanafe, have_sanafe_plugins

    if not (have_sanafe() and have_sanafe_plugins()):
        pytest.skip("SANA-FE or its mimarsinan plugins are not built")
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    program, x = _tie_samples()
    _, _, ref = _reference(program, x)
    runner = SanafeRunner(mapping=program.hybrid, simulation_length=T, behavior=_behavior(),
                          host_compute_device="cpu", pass_transfer=VERBATIM)
    subsets = []
    for i in range(x.shape[0]):
        with contextlib.redirect_stdout(io.StringIO()):
            subsets.append(runner.run(x[i:i + 1].numpy().astype(np.float64), sample_index=i).to_hcm_subset())
    got = per_perceptron_counts(subsets, program.hybrid)
    assert all(np.array_equal(got[pi], ref[pi]) for pi in ref), {
        pi: int((got[pi] != ref[pi]).sum()) for pi in ref}


def test_the_lava_runner_reads_the_tie_samples_as_the_gate(monkeypatch):
    from integration.parity_harness import have_lava

    if not have_lava():
        pytest.skip("Lava is not installed")
    import mimarsinan.chip_simulation.lava_loihi.runner as lava_runner
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner

    program, x = _tie_samples()
    _, ref_out, _ = _reference(program, x)
    runner = LavaLoihiRunner(mapping=program.hybrid, simulation_length=T, behavior=_behavior(), pass_transfer=VERBATIM)
    runner.pipeline = SimpleNamespace(reporter=SimpleNamespace(report=lambda *_a, **_k: None))
    monkeypatch.setattr(runner, "_load_test_samples", lambda: (x.numpy(), np.zeros(x.shape[0], dtype=np.int64)))
    final = {}
    gather = lava_runner.gather_final_output_numpy

    def record_final(*args, **kwargs):
        final["out"] = gather(*args, **kwargs)
        return final["out"]

    monkeypatch.setattr(lava_runner, "gather_final_output_numpy", record_final)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run()
    np.testing.assert_allclose(final["out"] * T, ref_out.double().numpy(), atol=1e-5, rtol=0)


@pytest.mark.timeout(600)
def test_the_nevresim_runner_reads_the_tie_samples_as_the_gate():
    from integration.parity_harness import ensure_nevresim_ready
    from mimarsinan.chip_simulation.nevresim.connectivity import default_nevresim_connectivity_mode
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

    ensure_nevresim_ready()
    program, x = _tie_samples()
    _, _, ref = _reference(program, x)
    runner = object.__new__(SimulationRunner)
    with tempfile.TemporaryDirectory() as work:
        vars(runner).update(
            spike_generation_mode="Uniform", firing_mode="Default", spiking_mode="lif", pass_transfer=VERBATIM,
            thresholding_mode="<=", nevresim_connectivity_mode=default_nevresim_connectivity_mode(),
            simulation_step_timeout_s=600.0, membrane_readout=False, membrane_half_step_charge=False,
            weight_type=float, threshold_type=float, host_compute_device="cpu", stage_timer=None,
            simulation_length=T, mapping=program.hybrid, num_classes=D,
            test_data=[(x[i].numpy().astype(np.float64), np.zeros(1)) for i in range(x.shape[0])],
            test_targets=[torch.tensor(0) for _ in range(x.shape[0])], test_input=[], working_directory=work)
        by_node = {}

        def record(stage, raw):
            counts = np.rint(np.asarray(raw)).astype(np.float64)
            for s in stage.output_map:
                by_node[int(s.node_id)] = counts[:, int(s.offset):int(s.offset) + int(s.size)]

        runner.stage_count_recorder = record
        with contextlib.redirect_stdout(io.StringIO()):
            runner._run_hybrid(program.hybrid)  # the runner's hybrid route, without its run() plane
    got = {pi: by_node[nid] for nid, pi in program.core_perceptron_indices().items()}
    assert all(np.array_equal(got[pi], ref[pi]) for pi in ref), {
        pi: int((got[pi] != ref[pi]).sum()) for pi in ref}

"""[w2-parity, defect A] A host op that owns its scale domain reads every source at the wire gauge, on every LIF executor.

The C5 T-LIF refusal (CIFAR ResNet-18 at width 1/4: the torch-versus-deployed-sim decision gate at 0.15 and 0.40):
the first residual ``add`` reads the SUBSUMED stem. The torch twin holds every perceptron output at the wire gauge (a
rate), which is what the ``ScaleNormalizingWrapper`` the add carries expects: it decodes each source by that source's
``kappa_fold`` (theta_enc for the stem). The IR records a host-run perceptron's buffer raw (``kappa_buf = 1``,
``compute_node_buffer_scales``), and the deployed executors transcoded a raw buffer to the wire
(``boundary_normalization_scales``) only where a NEURAL stage read it, so the deployed add read the stem at
``rate * theta_enc**2``.

Every executor now reads a host op's inputs through ``segment_boundary.host_op_input_state``: a domain-owning op reads
each source through the divisor table and per-producer transcode a neural-stage input reads, then the producer shifts.
On the encoding-skip program (``encoding_skip_program``) the NF and the deployed twins hold bit-exact window counts in
both deployable LIF semantics, the deployed join reads the stem at the wire gauge under the synchronized count executor
too, and the numpy (nevresim), Lava and SANA-FE runners equal the NF on the same program. ``theta_enc == 1`` is the
inert control: raw value and wire rate coincide there, so the seam is invisible.
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
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_torch
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.pipelining.core.nf_scm_parity import compare_normalized_records
from mimarsinan.spiking.segment_boundary import boundary_normalization_scales

from unit.spiking.encoding_skip_program import (
    D,
    DEPLOYED_SEMANTICS,
    SEMANTICS,
    T,
    build_program,
    count_mismatches,
    deployed_flow,
    nf_window_counts,
    per_perceptron_counts,
    samples,
    scm_window_counts,
)
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC

THETA_ENC = 2.185
THETAS_ENC = (THETA_ENC, 1.0)


def _lif_behavior():
    return NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                spike_generation_mode="Uniform")


# ---------------------------------------------------------------------------
# The fixture carries the seam, and the seam reaches a window count
# ---------------------------------------------------------------------------


def test_the_fixture_carries_the_encoding_skip_seam():
    program = build_program(THETA_ENC)
    join, stem = program.join(), program.stem()
    divisors = boundary_normalization_scales(program.hybrid)
    # The only raw buffer is the subsumed stem's, at divisor theta_enc ...
    assert set(divisors) == {int(stem.id)}
    assert float(np.mean(divisors[int(stem.id)])) == pytest.approx(THETA_ENC)
    # ... and the join reads it, its wrapper decoding that source at theta_enc (a WIRE source by its contract).
    assert int(stem.id) in {int(s.node_id) for s in join.input_sources.flatten()}
    wrapper = join.params["module"]
    scales = [float(getattr(wrapper, f"input_scale_{i}").mean()) for i in range(wrapper._num_inputs)]
    assert any(s == pytest.approx(THETA_ENC) for s in scales), scales
    # The join feeds the second LIF core, so the seam reaches the per-perceptron comparison.
    assert sorted(program.core_perceptron_indices().values()) == [1, 2]


# ---------------------------------------------------------------------------
# NF == SCM, bit-exact, across the encoding skip (both deployable LIF semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semantics", DEPLOYED_SEMANTICS)
@pytest.mark.parametrize("theta_enc", THETAS_ENC)
def test_nf_window_counts_equal_the_deployed_twin_across_the_encoding_skip(semantics, theta_enc):
    retimed, _, policy = SEMANTICS[semantics]
    program = build_program(theta_enc, retimed=retimed)
    x = samples()
    nf_out, nf = nf_window_counts(program, x, policy())
    scm_out, scm, _ = scm_window_counts(program, x, deployed_flow(program, semantics))
    live = {pi: float(c.mean()) for pi, c in scm.items()}
    assert all(0.5 < v < T - 0.5 for v in live.values()), f"both cores must be live: {live}"
    mismatches, total, worst = compare_normalized_records(nf, scm, atol=0.0)
    assert mismatches == 0, (
        f"{semantics} theta_enc={theta_enc}: {mismatches}/{total} window counts differ "
        f"(per perceptron {count_mismatches(nf, scm)}; worst {worst})")
    torch.testing.assert_close(nf_out.double() * T / program.theta_out, scm_out.double(), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("semantics", sorted(SEMANTICS))
@pytest.mark.parametrize("theta_enc", THETAS_ENC)
def test_the_deployed_join_reads_the_encoding_layer_at_the_wire_gauge(semantics, theta_enc):
    """The join's own value on the deployed side, in every LIF semantics the hybrid flow runs: its wrapper applied to
    its sources at the wire gauge (the stem's raw buffer divided by theta_enc, the host fc2's wire buffer as stored)."""
    retimed, _, _ = SEMANTICS[semantics]
    program = build_program(theta_enc, retimed=retimed)
    join, stem = program.join(), program.stem()
    sources = sorted({int(s.node_id) for s in join.input_sources.flatten()})
    assert int(stem.id) in sources and len(sources) == 2
    x = samples(4)
    _, _, records = scm_window_counts(program, x, deployed_flow(program, semantics))
    for i, record in enumerate(records):
        outs = record.compute_outputs
        wire = {nid: torch.as_tensor(outs[nid], dtype=torch.float64) / (theta_enc if nid == int(stem.id) else 1.0)
                for nid in sources}
        with torch.no_grad():
            want = execute_compute_op_torch(join, x[i:i + 1].double(), wire,
                                            arithmetic=VALUE_ARITHMETIC.counterfactual(torch.float64))
        np.testing.assert_allclose(outs[int(join.id)], want.numpy(), atol=1e-5, rtol=0.0,
                                   err_msg=f"{semantics}: sample {i}")


# ---------------------------------------------------------------------------
# The one host-op read (the SSOT the four executors call)
# ---------------------------------------------------------------------------


def host_op_input_state(*args):
    """The SSOT under test, resolved at call time so the red run of the parity tests above reports their counts."""
    from mimarsinan.spiking.segment_boundary import host_op_input_state as read

    return read(*args)


def _wrapper_op(module: nn.Module, sources: dict) -> ComputeOp:
    """A host op over ``{node_id: width}``, its module the given one."""
    flat = [IRSource(node_id=nid, index=i) for nid, width in sources.items() for i in range(width)]
    return ComputeOp(id=9, name="join", op_type="module", input_sources=np.array(flat, dtype=object).reshape(1, -1),
                     params={"module": module}, input_shape=(len(flat),), output_shape=(len(flat),))


def _domain_owner(n_inputs=2):
    return ScaleNormalizingWrapper(nn.Identity() if n_inputs == 1 else _Sum(),
                                   input_scales=[torch.tensor(2.0)] * n_inputs, output_scale=torch.tensor(1.0))


class _Sum(nn.Module):
    def forward(self, a, b):
        return a + b


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_a_domain_owning_op_reads_a_raw_source_at_its_wire_divisor(backend):
    as_buf = (lambda a: torch.tensor(a, dtype=torch.float64)) if backend == "torch" else (
        lambda a: np.asarray(a, dtype=np.float64))
    raw, wire = as_buf([[3.0, 6.0, 9.0]]), as_buf([[0.25, 0.5, 0.75]])
    state = {4: raw, 5: wire, -2: as_buf([[1.0]])}
    op = _wrapper_op(_domain_owner(), {4: 3, 5: 3})
    view = host_op_input_state(op, state, {4: 3.0}, None)
    np.testing.assert_allclose(np.asarray(view[4]), [[1.0, 2.0, 3.0]], rtol=0, atol=0)
    assert view[5] is wire and view[-2] is state[-2], "a wire source and every other entry pass through as stored"
    np.testing.assert_array_equal(np.asarray(state[4]), [[3.0, 6.0, 9.0]])
    assert view is not state, "the caller's buffer is never mutated"


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_the_transcode_precedes_the_producer_shift_in_the_neural_stage_order(backend):
    as_buf = (lambda a: torch.tensor(a, dtype=torch.float64)) if backend == "torch" else (
        lambda a: np.asarray(a, dtype=np.float64))
    state = {4: as_buf([[3.0, 6.0]])}
    op = _wrapper_op(_domain_owner(1), {4: 2})
    view = host_op_input_state(op, state, {4: 3.0}, {4: np.asarray([0.5, 1.0])})
    np.testing.assert_allclose(np.asarray(view[4]), [[1.5, 3.0]], rtol=0, atol=0)


def test_a_per_channel_divisor_divides_each_producer_neuron():
    state = {4: torch.tensor([[2.0, 6.0, 12.0]], dtype=torch.float64)}
    op = _wrapper_op(_domain_owner(1), {4: 3})
    view = host_op_input_state(op, state, {4: np.asarray([2.0, 3.0, 4.0])}, None)
    torch.testing.assert_close(view[4], torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64), atol=0, rtol=0)


def test_an_op_that_does_not_own_its_scale_domain_reads_the_buffers_as_stored():
    """A plain op's output inherits its sources' gauge (``kappa_buf``), so it reads the raw buffer; with no divisor and
    no shift the read is the buffer itself (no copy, byte-identical for every program without the seam)."""
    state = {4: torch.tensor([[3.0, 6.0]], dtype=torch.float64)}
    plain = _wrapper_op(nn.Identity(), {4: 2})
    assert host_op_input_state(plain, state, {4: 3.0}, None) is state
    owner = _wrapper_op(_domain_owner(1), {4: 2})
    assert host_op_input_state(owner, state, {}, None) is state
    assert host_op_input_state(owner, state, {7: 3.0}, None) is state


# ---------------------------------------------------------------------------
# The same program on the deployed runners (numpy/nevresim, Lava, SANA-FE)
# ---------------------------------------------------------------------------


def _nf_streamed(theta_enc, n):
    program = build_program(theta_enc)
    x = samples(n)
    nf_out, nf = nf_window_counts(program, x, SEMANTICS["streamed_lif"][2]())
    return program, x, nf_out, nf


@pytest.mark.parametrize("theta_enc", THETAS_ENC)
def test_the_sanafe_runner_equals_the_nf_across_the_encoding_skip(theta_enc):
    from integration.parity_harness import have_sanafe, have_sanafe_plugins

    if not (have_sanafe() and have_sanafe_plugins()):
        pytest.skip("SANA-FE or its mimarsinan plugins are not built")
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    program, x, _, nf = _nf_streamed(theta_enc, 4)
    runner = SanafeRunner(mapping=program.hybrid, simulation_length=T, behavior=_lif_behavior(),
                          host_compute_device="cpu", pass_transfer=VERBATIM)
    subsets = []
    for i in range(x.shape[0]):
        with contextlib.redirect_stdout(io.StringIO()):
            subsets.append(runner.run(x[i:i + 1].numpy().astype(np.float64), sample_index=i).to_hcm_subset())
    got = per_perceptron_counts(subsets, program.hybrid)
    assert count_mismatches(nf, got) == {pi: 0 for pi in got}


@pytest.mark.parametrize("theta_enc", THETAS_ENC)
def test_the_lava_runner_equals_the_nf_across_the_encoding_skip(theta_enc, monkeypatch):
    from integration.parity_harness import have_lava

    if not have_lava():
        pytest.skip("Lava is not installed")
    import mimarsinan.chip_simulation.lava_loihi.runner as lava_runner
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner

    program, x, nf_out, _ = _nf_streamed(theta_enc, 4)
    runner = LavaLoihiRunner(mapping=program.hybrid, simulation_length=T, behavior=_lif_behavior(),
                             pass_transfer=VERBATIM)
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
    np.testing.assert_allclose(final["out"] * T, (nf_out.double() * T / program.theta_out).numpy(), atol=1e-5, rtol=0)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("theta_enc", THETAS_ENC)
def test_the_nevresim_runner_equals_the_nf_across_the_encoding_skip(theta_enc):
    from integration.parity_harness import ensure_nevresim_ready
    from mimarsinan.chip_simulation.nevresim.connectivity import default_nevresim_connectivity_mode
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

    ensure_nevresim_ready()
    program, x, _, nf = _nf_streamed(theta_enc, 4)
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
            runner._run_hybrid(program.hybrid)
    got = {pi: by_node[nid] for nid, pi in program.core_perceptron_indices().items()}
    assert count_mismatches(nf, got) == {pi: 0 for pi in got}

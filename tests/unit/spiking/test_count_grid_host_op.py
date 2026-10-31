"""[U-SC S-2, SD-10c D-10c-1] One boundary-transcoding rule for every host op that reads a raw-gauge buffer.

A host-run perceptron (the subsumed encoding layer) leaves its raw value in the deployed state buffer
(``kappa_buf = 1``), and a plain host op that reads it inherits that raw gauge: the deployed twin computes the op on
``rate * theta`` and divides by ``theta`` only at the next neural input or domain-owning op, while the NF computes the
same op on the wire rate. For an op that commutes exactly with the division (a selection or a layout move: max pool,
reshape, concatenation) the two floats are equal. For an averaging or linear op they are not: the average of four integer
window counts sits exactly on a half count, and each twin's float error decides the comb's rounding on opposite sides
(SD-10c: 6,178 of 215,040 counts differ on the ``pool_join_avg`` fixtures at theta_enc 0.731 and 2.185, 0 at 1.0).

The rule: both twins decode the buffer once to its exact window counts, evaluate the non-commuting op on that grid in
fp64, and re-encode through the one comb rule (the chip's ``llround`` tie rule); exactly commuting ops stay as they are.
These tests hold the shakedown's eight forms at T 15 equal at atol 0 on both deployable LIF semantics, with and without
weight quantization, and the Lava and SANA-FE runners equal the NF on the average-pool form. (The numpy/nevresim
runner differs from the NF and the deployed flow on these convolutional forms at every gauge, max pool and theta_enc 1
included, on RC4 as here: a host-evaluation difference outside the rule, and nevresim is the MNIST-only tier-0 check.)
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.spiking.segment_boundary import boundary_normalization_scales

from unit.spiking.host_gather_forms import (
    SEMANTICS,
    SHAKEDOWN_FORMS,
    T,
    build,
    compare,
    count_mismatches,
    nf_window_counts,
    samples,
)
from unit.spiking.encoding_skip_program import per_perceptron_counts

RAW_THETAS = (2.185, 0.731)
SEMANTICS_NAMES = tuple(SEMANTICS)


def _assert_count_exact(program, semantics, n=32, seed=0):
    mism, total, out_diff = compare(program, semantics, n, seed)
    assert total > 0
    assert mism == {pi: 0 for pi in mism}, (program.form, program.thetas, mism, total)
    assert out_diff < 1e-3, out_diff


def test_the_average_pool_reads_the_raw_encoding_buffer():
    """The fixture carries the form: the pool is a plain host op that inherits the encoding's raw gauge."""
    program = build("pool_join_avg", 0, 2.185, "streamed_lif", False)
    divisors = boundary_normalization_scales(program.hybrid)
    pools = [n for n in program.ir.nodes if isinstance(n, ComputeOp) and n.op_type == "AvgPool2d"]
    assert len(pools) == 1 and int(pools[0].id) in divisors
    assert divisors[int(pools[0].id)] == pytest.approx(2.185)


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
@pytest.mark.parametrize("theta_enc", RAW_THETAS)
def test_an_average_pool_on_the_raw_encoding_buffer_is_count_exact(theta_enc, semantics, wq):
    """SD-10c's repro: 31 of 2,240 counts differed at theta_enc 2.185 (25 with WQ), 95 and 82 at 0.731."""
    _assert_count_exact(build("pool_join_avg", 0, theta_enc, semantics, wq), semantics)


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("form", ("pool_join_avg", "pool_join_max", "encoding_skip"))
def test_theta_one_is_the_inert_control(form, wq):
    """At theta_enc 1 the raw value and the wire rate coincide: every form is exact with or without the rule."""
    _assert_count_exact(build(form, 0, 1.0, "windowed_lif", wq), "windowed_lif")


@pytest.mark.parametrize("form", SHAKEDOWN_FORMS)
def test_the_eight_forms_are_count_exact_at_t15(form):
    for wq in (False, True):
        for theta_enc in (*RAW_THETAS, 1.0):
            for semantics in SEMANTICS_NAMES:
                _assert_count_exact(build(form, 0, theta_enc, semantics, wq), semantics)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("form", SHAKEDOWN_FORMS)
def test_the_shakedown_grid_is_count_exact(form, wq):
    """SD-10c's whole grid of the eight forms: 8 seeds, theta_enc 2.185, 0.731 and 1.0, both semantics, 32 samples."""
    for seed in range(8):
        for theta_enc in (*RAW_THETAS, 1.0):
            for semantics in SEMANTICS_NAMES:
                _assert_count_exact(build(form, seed, theta_enc, semantics, wq), semantics, seed=seed)


# ---------------------------------------------------------------------------
# The same program on the deployed runners (Lava, SANA-FE)
# ---------------------------------------------------------------------------


def _lif_behavior():
    return NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                spike_generation_mode="Uniform")


def _pool_program(n=6):
    program = build("pool_join_avg", 0, 2.185, "streamed_lif", False)
    x = samples(program, n, 0)
    nf_out, nf = nf_window_counts(program, x, "streamed_lif")
    return program, x, nf_out, nf


def test_the_sanafe_runner_equals_the_nf_on_the_average_pool():
    from integration.parity_harness import have_sanafe, have_sanafe_plugins

    if not (have_sanafe() and have_sanafe_plugins()):
        pytest.skip("SANA-FE or its mimarsinan plugins are not built")
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    program, x, _, nf = _pool_program(4)
    runner = SanafeRunner(mapping=program.hybrid, simulation_length=T, behavior=_lif_behavior(),
                          host_compute_device="cpu", pass_transfer=VERBATIM)
    subsets = []
    for i in range(x.shape[0]):
        with contextlib.redirect_stdout(io.StringIO()):
            subsets.append(runner.run(x[i:i + 1].reshape(1, -1).numpy().astype(np.float64),
                                      sample_index=i).to_hcm_subset())
    got = per_perceptron_counts(subsets, program.hybrid)
    assert count_mismatches(nf, got) == {pi: 0 for pi in got}


def test_the_lava_runner_equals_the_nf_on_the_average_pool(monkeypatch):
    from integration.parity_harness import have_lava

    if not have_lava():
        pytest.skip("Lava is not installed")
    import mimarsinan.chip_simulation.lava_loihi.runner as lava_runner
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner

    program, x, nf_out, _ = _pool_program(32)
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
    theta_out = float(program.perceptrons[-1].activation_scale)
    np.testing.assert_allclose(final["out"] * T, (nf_out.double() * T / theta_out).numpy(), atol=1e-4, rtol=0)

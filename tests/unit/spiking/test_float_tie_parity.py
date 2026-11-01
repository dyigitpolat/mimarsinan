"""[U-SC2 targeted pass, review B1] NF and SCM count-exact in float mode at the float survey's located roots.

U-SC's DenseNet-BC fixture in float mode left a one-count NF/SCM difference class (final_units/usc2/probes/out/
float_ties_wide_*.json, seeds 0 to 15 at 256 samples: the base 4 of 16 programs, 26 counts; U-SC2's build 7 of 16, 69
counts, which turned U-SC's slow gate red at seed 1, sample 7, core 1). Every located root is a decision two twins took
in different arithmetic at one of the program's two discretization sites:

* (i) the LIF threshold. The deployed core integrates its float32 effective weights in float64, where the sum of the
  charges is exact, so a membrane meets the threshold exactly (seed 1, sample 7: a cumulative membrane of exactly 2.0);
  the NF integrated the same charge in float32 through the perceptron's own layers and held 2 - eps
  (probes/out/tie_margin_seed1.json, tie_at_*.json).
* (ii) the comb's half-count edge at a host op's encode. The deployed records evaluate every host kernel on one sample
  (the recorder runs at batch 1), the NF on its whole batch, and fp32 host convolutions are batch-dependent at the ulp
  level (Theorem 1's host condition, U-PS): with identical inputs the stem convolution's output differed at 338 of 512
  elements by at most 7.2e-7, and one ulp moved a rate across 13.5 counts (probes/tp/out/host_trace_base_s3_x72.json);
  with the NF on the sample alone every host op agreed bit for bit (host_trace_base_s3_x72_alone.json).

The fix: (i) inside the measurement plane a float-mode core's threshold is decided through the deployed fold (the
mapper's own unfold, the effective weight and bias, float64 sums: ``spiking/segment_policy_lif_fold.py``) wherever the
twin's own pre-activation agrees with that decomposition to within reduction dust, and the twin keeps its own arithmetic
elsewhere, so a gauge departure still shows as counts; (ii) the fixture's program declares ``host_kernel_evaluation =
per_sample`` and the NF twin follows the program's mode (``twin_as_program``), as the parity gate of record does
(``pipelining/core/nf_scm_parity.py``).
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.support.host_kernel_evaluation import program_evaluates_per_sample

from unit.spiking.host_gather_forms import SEMANTICS, build, compare, nf_window_counts, samples

SEMANTICS_NAMES = tuple(SEMANTICS)


def _densenet(seed: int, semantics: str):
    """The float survey's program at ``seed`` (U-SC's slow gate's theta rule)."""
    return build("densenet_bc", seed, 1.3 + 0.2 * (seed % 4), semantics, False)


def _assert_count_exact(program, semantics, n, seed):
    mism, total, out_diff = compare(program, semantics, n, seed)
    assert total > 0
    assert mism == {pi: 0 for pi in mism}, (program.form, seed, mism, total)
    assert out_diff < 1e-3, out_diff


@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
def test_a_float_core_decides_an_exact_threshold_tie_as_the_deployed_core(semantics):
    """Site (i): seed 1's program at U-SC's slow gate's 16 samples. Sample 7's core 1 meets its threshold exactly in the
    deployed float64 fold (a membrane of 2.0) and fired once fewer in the NF's float32 membrane."""
    _assert_count_exact(_densenet(1, semantics), semantics, 16, 1)


def test_the_fixture_program_declares_per_sample_host_evaluation():
    """Site (ii): the fixture's deployed records run every host kernel on one sample; the program says so, and the NF
    twin of every comparison follows it."""
    program = _densenet(0, "windowed_lif")
    assert program_evaluates_per_sample(program.hybrid)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_nf_reads_each_samples_host_values_alone():
    """Site (ii): a sample's NF window counts do not depend on the samples beside it. At seed 3 the batch of the
    survey's 256 samples moved sample 72's dense1_units_0_relu1 rate across a half count (the batched NF counted 13
    where the sample alone, and the deployed record, count 14)."""
    program = _densenet(3, "windowed_lif")
    x = samples(program, 256, 3)
    _, batch = nf_window_counts(program, x, "windowed_lif")
    _, alone = nf_window_counts(program, x[72:73].clone(), "windowed_lif")
    assert all((batch[pi][72] == alone[pi][0]).all() for pi in alone)


def test_the_fold_is_armed_only_inside_the_measurement_plane(monkeypatch):
    """The per-cycle deployed fold decides measurement-plane readings only: the training plane and the tuning-plane
    telemetry (the AQ D-hat, LIF Adaptation's metric) keep the NF's own arithmetic, byte for byte."""
    import mimarsinan.spiking.segment_policy_lif_fold as fold_module
    from mimarsinan.models.nn.lif_kernels import measurement_plane
    from mimarsinan.spiking.segment_forward import SegmentForwardDriver

    armed: list = []
    original = fold_module.CycleFoldSlot.__init__

    def spy(self, *args, **kwargs):
        armed.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(fold_module.CycleFoldSlot, "__init__", spy)
    program = _densenet(0, "streamed_lif")
    x = samples(program, 4, 0)
    with torch.no_grad():
        SegmentForwardDriver(program.repr_, 15, SEMANTICS["streamed_lif"][2]())(x)
    assert armed == []
    with torch.no_grad(), measurement_plane():
        SegmentForwardDriver(program.repr_, 15, SEMANTICS["streamed_lif"][2]())(x)
    assert armed


def test_the_fold_keeps_the_twins_own_arithmetic_where_the_deployed_fold_departs():
    """A hop whose own pre-activation departs from the deployed decomposition beyond reduction dust is a gauge defect,
    not a tie: the twin keeps its own arithmetic there (the reading equals its reading outside the measurement plane,
    where no fold is armed), so the departure still shows as counts. Here the output core's fold is moved by 10 percent
    after mapping: its effective weight reads the new fold, the twin's own trains the old gauges."""
    import contextlib

    from mimarsinan.models.nn.lif_kernels import measurement_plane
    from mimarsinan.spiking.segment_forward import SegmentForwardDriver

    program = build("dense_concat", 0, 1.3, "streamed_lif", False)
    core = program.perceptrons[-1]
    x = samples(program, 8, 0)

    def reading(plane: bool):
        rec: dict = {}
        with torch.no_grad(), (measurement_plane() if plane else contextlib.nullcontext()):
            SegmentForwardDriver(program.repr_, 15, SEMANTICS["streamed_lif"][2]())(x, node_value_recorder=rec)
        return rec[id(core)].clone()

    pis = getattr(core, "per_input_scales", None)
    core.per_input_scales = (torch.ones(core.layer.weight.shape[1]) if pis is None else pis.clone()) * 1.1
    try:
        assert torch.equal(reading(True), reading(False))
    finally:
        core.per_input_scales = pis


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("semantics", SEMANTICS_NAMES)
@pytest.mark.parametrize("seed", (1, 3, 4, 6, 7, 8, 10))
def test_densenet_bc_programs_are_count_exact_at_the_survey_roots(seed, semantics):
    """Every located root of the survey (review B1): the base's seed 3 samples 26 and 72, seed 4 sample 195, seeds 7
    and 8; U-SC2's build's seed 1 sample 7, seed 3 sample 105, seed 6 sample 43, seed 10 sample 252. Each program over
    the survey's 256 samples, every core's every count, at atol 0."""
    _assert_count_exact(_densenet(seed, semantics), semantics, 256, seed)

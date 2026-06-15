"""Distribution-matching calibration for the genuine TTFS single-spike cascade.

``match_activation_distributions`` grounds each block's [0,1] boundary in the
teacher ANN's activation quantile (scale-aware boundaries) and then runs a DFQ
per-neuron bias-correction loop so each perceptron's cascade channel-mean tracks
the ANN's. Together they shrink the cascade↔ANN first-moment gap and revive the
death-cascade-starved deep neurons, turning the full TTFS transform into a
smoothly recoverable ramp.
"""

import copy

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
)
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.distribution_matching import match_activation_distributions


T_STEPS = 16


def _ttfs_pipeline():
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "cascaded"
    cfg["activation_quantization"] = True
    cfg["simulation_steps"] = T_STEPS
    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5
    return pipeline


def _deployed_ttfs_model_and_teacher(seed=0):
    """A tiny model driven into the deployed single-spike-cascade TTFS state
    (as the ttfs-cycle step's caller leaves it before calibration), paired with
    a frozen continuous-ANN teacher snapshot of the SAME pre-deployment weights.
    """
    torch.manual_seed(seed)
    base = make_tiny_supermodel()
    teacher = copy.deepcopy(base).eval()

    pipeline = _ttfs_pipeline()
    model = copy.deepcopy(base)
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model, target_accuracy=0.9,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model, teacher


def _chan_mean(t):
    return t.reshape(-1, t.shape[-1]).float().mean(0)


def _ann_channel_means(teacher, cal_x):
    samples = {}
    handles = [
        p.activation.register_forward_hook(
            lambda m, i, o, k=k: samples.__setitem__(k, o.detach())
        )
        for k, p in enumerate(teacher.get_perceptrons())
        if hasattr(p, "activation")
    ]
    with torch.no_grad():
        teacher(cal_x)
    for h in handles:
        h.remove()
    return {k: _chan_mean(v) for k, v in samples.items()}


def _cascade_channel_means(model, cal_x, T):
    with torch.no_grad():
        _, nodes = TTFSSegmentForward(model.get_mapper_repr(), T).forward_with_node_values(
            cal_x
        )
    by_perc = {
        id(perceptron_of(n)): v
        for n, v in nodes.items()
        if perceptron_of(n) is not None
    }
    out = {}
    for k, p in enumerate(model.get_perceptrons()):
        c = by_perc.get(id(p))
        if c is not None:
            out[k] = c
    return out


def _mean_abs_gap(model, teacher, cal_x, T):
    ann_mu = _ann_channel_means(teacher, cal_x)
    cas = _cascade_channel_means(model, cal_x, T)
    gaps = []
    for k in ann_mu:
        if k not in cas:
            continue
        cm = _chan_mean(cas[k])
        n = min(cm.numel(), ann_mu[k].numel())
        gaps.append((cm[:n] - ann_mu[k][:n]).abs().mean().item())
    return sum(gaps) / max(1, len(gaps))


def _dead_fraction_per_perceptron(model, cal_x, T):
    cas = _cascade_channel_means(model, cal_x, T)
    return {k: (v.abs() < 1e-6).float().mean().item() for k, v in cas.items()}


@pytest.fixture
def cal_x():
    torch.manual_seed(123)
    return torch.randn(24, *default_config()["input_shape"])


class TestReturnsStats:
    def test_returns_stats_dict_with_gap_keys(self, cal_x):
        model, teacher = _deployed_ttfs_model_and_teacher()
        stats = match_activation_distributions(model, teacher, cal_x, T_STEPS)
        assert isinstance(stats, dict)
        assert "mean_gap_before" in stats
        assert "mean_gap_after" in stats
        assert stats["bias_iters"] == 15
        assert stats["quantile"] == pytest.approx(0.99)


class TestDistributionMatchingShrinksGap:
    def test_cascade_mean_closer_to_ann_after_calibration(self, cal_x):
        model, teacher = _deployed_ttfs_model_and_teacher()

        gap_before = _mean_abs_gap(model, teacher, cal_x, T_STEPS)
        stats = match_activation_distributions(model, teacher, cal_x, T_STEPS)
        gap_after = _mean_abs_gap(model, teacher, cal_x, T_STEPS)

        assert gap_after < gap_before, (
            "distribution matching must move the cascade per-perceptron channel "
            "means CLOSER to the ANN's (mean|gap| must drop)"
        )
        # The reported stats reflect the measured before/after gap.
        assert stats["mean_gap_after"] < stats["mean_gap_before"]
        assert stats["mean_gap_after"] == pytest.approx(gap_after, abs=1e-5)


def _starve_deep_layer(model, shift=-1.0):
    """Drive the deepest perceptron's cascade output to ~0 (a fully-dead deep
    layer) so the death-cascade revival is actually exercised."""
    deep = list(model.get_perceptrons())[-1]
    with torch.no_grad():
        deep.layer.bias.add_(shift)
    return len(list(model.get_perceptrons())) - 1


class TestDeathCascadeRevival:
    def test_dfq_revives_a_starved_deep_layer(self, cal_x):
        """A deep layer starved to 100% dead stays dead under boundary-only
        calibration; the DFQ bias-correction loop raises its membrane baseline
        and revives a large fraction of its neurons."""
        model, teacher = _deployed_ttfs_model_and_teacher()
        deep_layer = _starve_deep_layer(model)

        dead_starved = _dead_fraction_per_perceptron(model, cal_x, T_STEPS)[deep_layer]
        assert dead_starved == pytest.approx(1.0), "fixture must be fully starved"

        match_activation_distributions(model, teacher, cal_x, T_STEPS)
        dead_after = _dead_fraction_per_perceptron(model, cal_x, T_STEPS)[deep_layer]
        assert dead_after < dead_starved - 0.1, (
            "DFQ bias correction must revive the starved deep layer (%dead drops)"
        )

    def test_boundary_calibration_alone_does_not_revive(self, cal_x):
        """The revival is the DFQ loop's doing, not the boundary calibration:
        with ``bias_iters=0`` the starved deep layer stays fully dead."""
        model, teacher = _deployed_ttfs_model_and_teacher()
        deep_layer = _starve_deep_layer(model)

        match_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=0)
        dead_after = _dead_fraction_per_perceptron(model, cal_x, T_STEPS)[deep_layer]
        assert dead_after == pytest.approx(1.0), (
            "boundary calibration alone must not revive the starved layer "
            "(only the DFQ bias loop touches layer.bias)"
        )


class TestBiasesChanged:
    def test_biases_are_modified(self, cal_x):
        model, teacher = _deployed_ttfs_model_and_teacher()
        before = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        match_activation_distributions(model, teacher, cal_x, T_STEPS)
        after = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        assert any(
            not torch.equal(a, b) for a, b in zip(before, after)
        ), "DFQ bias correction must modify at least one perceptron bias"

    def test_zero_bias_iters_leaves_biases_unchanged(self, cal_x):
        """With ``bias_iters=0`` only the boundary calibration runs; biases are
        untouched (the DFQ loop is the only writer of ``layer.bias``)."""
        model, teacher = _deployed_ttfs_model_and_teacher()
        before = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        match_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=0)
        after = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        for a, b in zip(before, after):
            assert torch.equal(a, b)


class TestBoundaryCalibrationApplied:
    def test_activation_scale_set_to_teacher_quantile(self, cal_x):
        model, teacher = _deployed_ttfs_model_and_teacher()

        ann_samples = {}
        handles = [
            p.activation.register_forward_hook(
                lambda m, i, o, k=k: ann_samples.__setitem__(k, o.detach())
            )
            for k, p in enumerate(teacher.get_perceptrons())
            if hasattr(p, "activation")
        ]
        with torch.no_grad():
            teacher(cal_x)
        for h in handles:
            h.remove()
        expected_theta = [
            max(float(torch.quantile(ann_samples[k].abs().float().flatten(), 0.99)), 1e-2)
            for k in range(len(ann_samples))
        ]

        match_activation_distributions(model, teacher, cal_x, T_STEPS)
        for p, theta in zip(model.get_perceptrons(), expected_theta):
            # The encoding block is pinned to the data scale (1.0), not its quantile
            # (its scale is fixed by the input spike-encoding contract — retuning it
            # breaks NF↔SCM deployment parity).
            expected = 1.0 if getattr(p, "is_encoding_layer", False) else theta
            assert float(p.activation_scale) == pytest.approx(expected, abs=1e-5)

    def test_downstream_input_scale_equals_upstream_theta_out(self, cal_x):
        model, teacher = _deployed_ttfs_model_and_teacher()
        match_activation_distributions(model, teacher, cal_x, T_STEPS)
        perceptrons = list(model.get_perceptrons())
        assert float(perceptrons[1].input_activation_scale) == pytest.approx(
            float(perceptrons[0].activation_scale), abs=1e-6
        )


class TestDeterminism:
    def test_deterministic_given_fixed_seed(self, cal_x):
        model_a, teacher_a = _deployed_ttfs_model_and_teacher(seed=7)
        model_b, teacher_b = _deployed_ttfs_model_and_teacher(seed=7)

        stats_a = match_activation_distributions(model_a, teacher_a, cal_x, T_STEPS)
        stats_b = match_activation_distributions(model_b, teacher_b, cal_x, T_STEPS)

        assert stats_a["mean_gap_after"] == pytest.approx(stats_b["mean_gap_after"])
        for pa, pb in zip(model_a.get_perceptrons(), model_b.get_perceptrons()):
            if getattr(pa.layer, "bias", None) is None:
                continue
            torch.testing.assert_close(
                pa.layer.bias, pb.layer.bias, rtol=0, atol=0
            )


class TestQuantileParameter:
    def test_lower_quantile_yields_smaller_theta_out(self, cal_x):
        """A lower quantile clips more of the tail, so theta_out (activation_scale)
        is no larger; the function honours the ``quantile`` knob."""
        model_hi, teacher_hi = _deployed_ttfs_model_and_teacher(seed=3)
        model_lo, teacher_lo = _deployed_ttfs_model_and_teacher(seed=3)

        match_activation_distributions(
            model_hi, teacher_hi, cal_x, T_STEPS, quantile=0.99, bias_iters=0
        )
        match_activation_distributions(
            model_lo, teacher_lo, cal_x, T_STEPS, quantile=0.5, bias_iters=0
        )
        for p_hi, p_lo in zip(
            model_hi.get_perceptrons(), model_lo.get_perceptrons()
        ):
            assert float(p_lo.activation_scale) <= float(p_hi.activation_scale) + 1e-9

"""DFQ per-neuron bias correction for the deployed LIF cascade.

``match_lif_activation_distributions`` runs the shared DFQ loop over the deployed
LIF cycle-accurate cascade (decoded values read via the segment forward's
``node_value_recorder`` side-channel), matching each perceptron's cascade
channel-mean to the teacher ANN's. Unlike the TTFS path it does NOT recalibrate
the interior decode scale (LIF's rate code is linear; the q-quantile scale from
Activation Analysis is already the right boundary) — it is a pure first-moment
bias correction of the conversion gap.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.dfq_bias_correction import (
    perceptron_channel_mean,
    teacher_channel_means,
)
from mimarsinan.spiking.lif_distribution_matching import (
    match_lif_activation_distributions,
)


T_STEPS = 8


def _deployed_lif_model_and_teacher(seed=0):
    """A tiny model in the deployed cycle-accurate LIF state, paired with a frozen
    continuous-ANN teacher snapshot of the SAME pre-deployment weights."""
    torch.manual_seed(seed)
    base = make_tiny_supermodel()
    teacher = copy.deepcopy(base).eval()

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = T_STEPS
    cfg["cycle_accurate_lif_forward"] = True
    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5

    model = copy.deepcopy(base)
    am = AdaptationManager()
    tuner = LIFAdaptationTuner(
        pipeline, model, target_accuracy=0.5, lr=cfg["lr"], adaptation_manager=am,
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model, teacher


def _lif_cascade_means(model, cal_x, T):
    rec = {}
    with torch.no_grad():
        chip_aligned_segment_forward(model, cal_x, T, node_value_recorder=rec)
    out = {}
    for k, p in enumerate(model.get_perceptrons()):
        v = rec.get(id(p))
        if v is not None:
            out[k] = v
    return out


def _mean_abs_gap(model, teacher, cal_x, T):
    ann_mu = teacher_channel_means(teacher, cal_x)
    cas = _lif_cascade_means(model, cal_x, T)
    perceptrons = list(model.get_perceptrons())
    gaps = []
    for k in ann_mu:
        if k not in cas:
            continue
        cm = perceptron_channel_mean(perceptrons[k], cas[k])
        n = min(cm.numel(), ann_mu[k].numel())
        gaps.append((cm[:n] - ann_mu[k][:n]).abs().mean().item())
    return sum(gaps) / max(1, len(gaps))


@pytest.fixture
def cal_x():
    torch.manual_seed(321)
    return torch.randn(16, *default_config()["input_shape"])


class TestReturnsStats:
    def test_returns_stats_dict(self, cal_x):
        model, teacher = _deployed_lif_model_and_teacher()
        stats = match_lif_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=10)
        assert isinstance(stats, dict)
        assert "mean_gap_before" in stats and "mean_gap_after" in stats
        assert stats["num_perceptrons"] == len(list(model.get_perceptrons()))


class TestShrinksGap:
    def test_cascade_mean_closer_to_ann_after_matching(self, cal_x):
        model, teacher = _deployed_lif_model_and_teacher()
        gap_before = _mean_abs_gap(model, teacher, cal_x, T_STEPS)
        stats = match_lif_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=10)
        gap_after = _mean_abs_gap(model, teacher, cal_x, T_STEPS)
        assert gap_after <= gap_before + 1e-6
        assert stats["mean_gap_after"] <= stats["mean_gap_before"] + 1e-6
        # the matching must make real progress on a randomly-initialised cascade
        assert stats["mean_gap_after"] < stats["mean_gap_before"]


class TestBiasesChanged:
    def test_biases_modified(self, cal_x):
        model, teacher = _deployed_lif_model_and_teacher()
        before = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        match_lif_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=10)
        after = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))

    def test_zero_iters_leaves_biases_unchanged(self, cal_x):
        model, teacher = _deployed_lif_model_and_teacher()
        before = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        match_lif_activation_distributions(
            model, teacher, cal_x, T_STEPS, bias_iters=0,
        )
        after = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        for a, b in zip(before, after):
            assert torch.equal(a, b)


class TestDeterminism:
    def test_deterministic_given_fixed_seed(self, cal_x):
        model_a, teacher_a = _deployed_lif_model_and_teacher(seed=5)
        model_b, teacher_b = _deployed_lif_model_and_teacher(seed=5)
        stats_a = match_lif_activation_distributions(model_a, teacher_a, cal_x, T_STEPS, bias_iters=10)
        stats_b = match_lif_activation_distributions(model_b, teacher_b, cal_x, T_STEPS, bias_iters=10)
        assert stats_a["mean_gap_after"] == pytest.approx(stats_b["mean_gap_after"])
        for pa, pb in zip(model_a.get_perceptrons(), model_b.get_perceptrons()):
            if getattr(pa.layer, "bias", None) is None:
                continue
            torch.testing.assert_close(pa.layer.bias, pb.layer.bias, rtol=0, atol=0)


class TestDoesNotChangeScale:
    def test_activation_scale_untouched(self, cal_x):
        """LIF distmatch is a pure bias correction — it must NOT retune the decode
        scale (that is the TTFS scale-aware-boundary stage, a no-op for LIF)."""
        model, teacher = _deployed_lif_model_and_teacher()
        before = [float(p.activation_scale) for p in model.get_perceptrons()]
        match_lif_activation_distributions(model, teacher, cal_x, T_STEPS, bias_iters=10)
        after = [float(p.activation_scale) for p in model.get_perceptrons()]
        assert before == after

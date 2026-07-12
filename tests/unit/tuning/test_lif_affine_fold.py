"""Per-channel affine fold absorbing the LIF dead-zone bias (C4).

Estimator: layer-sequential per-channel least squares mapping the DEPLOYED
cycle-accurate rate to the float-envelope rate ``clamp(z/theta, 0, 1)``; the
FULL affine (gain + bias) is the only sound estimator at coarse grids
(bias-only folds are refuted at -4.2pp, lif_deployment_exactness.md §4 C4).
Fold currency: consumer effective-weight columns and bias (the negative-shift
fold family); terminal perceptrons absorb their affine as an effective row
scale + bias (exact under the membrane readout).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)
from mimarsinan.tuning.lif_affine_fold import (
    GAIN_MAX,
    GAIN_MIN,
    apply_lif_affine_fold,
    fit_channel_affine,
    fold_affine_into_consumer,
    fold_affine_into_readout,
)

T_STEPS = 8


class TestFitChannelAffine:
    def test_recovers_exact_affine(self):
        torch.manual_seed(0)
        dep = torch.rand(256, 5, dtype=torch.float64)
        a_true = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
        c_true = torch.tensor([-0.1, 0.0, 0.05, 0.2, -0.3], dtype=torch.float64)
        target = a_true * dep + c_true
        a, c = fit_channel_affine(dep, target)
        assert torch.allclose(a, a_true, atol=1e-9)
        assert torch.allclose(c, c_true, atol=1e-9)

    def test_gain_is_clamped(self):
        torch.manual_seed(1)
        dep = torch.rand(128, 2, dtype=torch.float64)
        target = torch.stack([dep[:, 0] * 100.0, dep[:, 1] * 0.001], dim=1)
        a, _ = fit_channel_affine(dep, target)
        assert float(a[0]) == GAIN_MAX
        assert float(a[1]) == GAIN_MIN

    def test_dead_channel_gets_identity_gain_and_mean_bias(self):
        dep = torch.zeros(64, 1, dtype=torch.float64)
        target = torch.full((64, 1), 0.25, dtype=torch.float64)
        a, c = fit_channel_affine(dep, target)
        assert float(a[0]) == 1.0
        assert float(c[0]) == 0.25


def _linear_perceptron(out_f: int, in_f: int, seed: int) -> Perceptron:
    torch.manual_seed(seed)
    p = Perceptron(out_f, in_f, normalization=nn.Identity())
    p.layer.weight.data = torch.randn(out_f, in_f)
    p.layer.bias.data = torch.randn(out_f)
    p.activation_scale.data = torch.tensor(0.8)
    return p


class TestConsumerFoldIdentity:
    def test_folded_consumer_equals_affine_corrected_input(self):
        """W'r + b' == W(a*r + c) + b in the effective (wire) domain — the fold
        is the exact realization of the input-rate affine."""
        q = _linear_perceptron(4, 3, seed=2)
        tf = PerceptronTransformer()
        w_before, b_before = tf.get_effective_parameters(q)
        w_before = w_before.clone().double()
        b_before = b_before.clone().double()

        a = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float64)
        c = torch.tensor([0.1, -0.05, 0.3], dtype=torch.float64)
        fold_affine_into_consumer(q, a, c)

        w_after, b_after = tf.get_effective_parameters(q)
        r = torch.rand(16, 3, dtype=torch.float64)
        got = r @ w_after.double().T + b_after.double()
        want = (a * r + c) @ w_before.T + b_before
        assert torch.allclose(got, want, atol=1e-6)


class TestReadoutSelfFold:
    def test_folded_readout_equals_affine_on_wire_preactivation(self):
        """z' == a*z + c in the effective (wire) domain — exact once the readout
        decodes the unquantized charge (membrane readout)."""
        p = _linear_perceptron(3, 4, seed=3)
        tf = PerceptronTransformer()
        w_before, b_before = tf.get_effective_parameters(p)
        w_before = w_before.clone().double()
        b_before = b_before.clone().double()

        a = torch.tensor([0.75, 1.5, 1.0], dtype=torch.float64)
        c = torch.tensor([0.2, -0.1, 0.0], dtype=torch.float64)
        fold_affine_into_readout(p, a, c)

        w_after, b_after = tf.get_effective_parameters(p)
        r = torch.rand(16, 4, dtype=torch.float64)
        got = r @ w_after.double().T + b_after.double()
        want = a * (r @ w_before.T + b_before) + c
        assert torch.allclose(got, want, atol=1e-6)


def _lif_config():
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = T_STEPS
    cfg["cycle_accurate_lif_forward"] = True
    return cfg


def _install_deployed_lif(model, cfg, adaptation_manager):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5
    tuner = LIFAdaptationTuner(
        pipeline, model, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=adaptation_manager,
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model


def _deployed_lif_model(seed=0, hidden_layers=2):
    """A tiny model in the deployed cycle-accurate LIF state (mirrors the LIF
    adaptation step's exit state). Two hidden layers exercise BOTH fold kinds:
    a consumer fold (hidden -> readout) and the readout self-fold."""
    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

    torch.manual_seed(seed)
    model = copy.deepcopy(make_tiny_supermodel(hidden_layers=hidden_layers))
    return _install_deployed_lif(model, _lif_config(), AdaptationManager())


def _deployed_converted_lif_flow(torch_model, seed=0):
    """A REAL converted mapper graph (pass-through nodes interleaved, host
    ComputeOp tail) in the deployed cycle-accurate LIF state — the graph shape
    the pipeline actually folds, which toy direct-edge graphs do not have."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

    cfg = _lif_config()
    torch.manual_seed(seed)
    flow = convert_torch_model(
        torch_model, input_shape=cfg["input_shape"], num_classes=cfg["num_classes"],
    ).eval()
    mark_encoding_layers(flow.get_mapper_repr())

    am = AdaptationManager()
    for p in flow.get_perceptrons():
        am.update_activation(cfg, p)
    return _install_deployed_lif(flow, cfg, am)


def _cal_x(n=16):
    torch.manual_seed(123)
    return torch.randn(n, *default_config()["input_shape"])


def _frozen_float_targets(model, x):
    from mimarsinan.tuning.lif_affine_fold import float_envelope_rates_by_perceptron

    return {
        k: v.clone() for k, v in float_envelope_rates_by_perceptron(model, x).items()
    }


def _deployed_gap(model, x, targets):
    """Mean per-channel |deployed - float| against FIXED pre-fold targets (the
    estimator's objective — the post-fold envelope is a different function)."""
    from mimarsinan.tuning.lif_affine_fold import deployed_rates_by_perceptron

    dep = deployed_rates_by_perceptron(model, x, T_STEPS)
    gaps = [
        float((dep[k] - targets[k]).mean(0).abs().mean())
        for k in dep
        if k in targets
    ]
    return sum(gaps) / max(len(gaps), 1)


class TestApplyLifAffineFold:
    def test_folds_both_kinds_and_reports(self):
        model = _deployed_lif_model()
        report = apply_lif_affine_fold(model, _cal_x(), T_STEPS)
        assert report["folded"] >= 2, f"expected both fold kinds: {report}"
        assert report["consumer_folds"] >= 1, f"no consumer fold: {report}"
        assert report["readout_folds"] == 1, f"no readout fold: {report}"

    def test_idempotent_second_pass_is_noop(self):
        model = _deployed_lif_model()
        apply_lif_affine_fold(model, _cal_x(), T_STEPS)
        params_after_first = [
            p.layer.weight.data.clone() for p in model.get_perceptrons()
        ]
        report2 = apply_lif_affine_fold(model, _cal_x(), T_STEPS)
        assert report2["folded"] == 0
        for p, saved in zip(model.get_perceptrons(), params_after_first):
            assert torch.equal(p.layer.weight.data, saved)

    def test_encoders_are_never_folded(self):
        model = _deployed_lif_model()
        apply_lif_affine_fold(model, _cal_x(), T_STEPS)
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                assert not getattr(p, "_lif_affine_folded", False)

    def test_deployed_moves_toward_float_envelope(self):
        """The pass shrinks the deployed-vs-float per-channel mean gap."""
        model = _deployed_lif_model()
        x = _cal_x()
        targets = _frozen_float_targets(model, x)

        before = _deployed_gap(model, x, targets)
        apply_lif_affine_fold(model, x, T_STEPS)
        after = _deployed_gap(model, x, targets)
        assert after <= before + 1e-9, (
            f"affine fold increased the calibrated gap: {before} -> {after}"
        )


def _converted_deep_mlp(depth=4, width=8, seed=0):
    from mimarsinan.models.deep_mlp import DeepMLP

    cfg = _lif_config()
    torch.manual_seed(seed)
    mlp = DeepMLP(
        input_shape=cfg["input_shape"], num_classes=cfg["num_classes"],
        depth=depth, width=width,
    ).eval()
    return _deployed_converted_lif_flow(mlp, seed=seed)


def _converted_mixer(seed=0):
    from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore

    cfg = _lif_config()
    torch.manual_seed(seed)
    mixer = TorchMLPMixerCore(
        input_shape=cfg["input_shape"], num_classes=cfg["num_classes"],
        patch_n_1=2, patch_m_1=2, patch_c_1=4, fc_w_1=8, fc_w_2=8,
        base_activation="ReLU", num_blocks=2,
    ).eval()
    return _deployed_converted_lif_flow(mixer, seed=seed)


class TestRealConvertedGraphDiscovery:
    """Regression for the pipeline-wide silent no-op: real converted graphs
    interleave pass-through mappers between perceptrons and end in a host
    ComputeOp classifier, so direct-edge adjacency finds NOTHING. Discovery
    must walk the mapper DAG (the M4 channel-axis walk SSOT)."""

    def test_fc_chain_finds_consumer_folds(self):
        flow = _converted_deep_mlp(depth=4)
        report = apply_lif_affine_fold(flow, _cal_x(), T_STEPS)
        assert report["consumer_folds"] == 2, f"FC chain must fold: {report}"
        assert report["folded"] == 2, f"FC chain must fold: {report}"
        folded_names = {
            p.name for p in flow.get_perceptrons()
            if getattr(p, "_lif_affine_folded", False)
        }
        assert folded_names == {"hidden_2", "hidden_4"}, report

    def test_fc_chain_host_classifier_seam_skipped_honestly(self):
        flow = _converted_deep_mlp(depth=4)
        report = apply_lif_affine_fold(flow, _cal_x(), T_STEPS)
        assert report["skipped"] == {"hidden_6": "host_compute_consumer"}, report
        assert report["readout_folds"] == 0, report

    def test_fc_chain_fold_shrinks_deployed_gap(self):
        flow = _converted_deep_mlp(depth=4)
        x = _cal_x()
        targets = _frozen_float_targets(flow, x)

        before = _deployed_gap(flow, x, targets)
        report = apply_lif_affine_fold(flow, x, T_STEPS)
        after = _deployed_gap(flow, x, targets)
        assert report["consumer_folds"] > 0, report
        assert after <= before + 1e-9, (
            f"affine fold increased the calibrated gap: {before} -> {after}"
        )

    def test_mixer_intra_block_folds_found(self):
        flow = _converted_mixer()
        report = apply_lif_affine_fold(flow, _cal_x(), T_STEPS)
        assert report["consumer_folds"] == 4, f"fc1->fc2 folds missing: {report}"
        folded_names = {
            p.name for p in flow.get_perceptrons()
            if getattr(p, "_lif_affine_folded", False)
        }
        assert folded_names == {
            "mixer_blocks_0_fc1", "mixer_blocks_1_fc1",
            "mixer_blocks_2_fc1", "mixer_blocks_3_fc1",
        }, report

    def test_mixer_fold_shrinks_deployed_gap(self):
        """The pooled-patch per-channel fits genuinely move the deployed mixer
        toward its float envelope — discovery is not just structural."""
        flow = _converted_mixer()
        x = _cal_x()
        targets = _frozen_float_targets(flow, x)

        before = _deployed_gap(flow, x, targets)
        report = apply_lif_affine_fold(flow, x, T_STEPS)
        after = _deployed_gap(flow, x, targets)
        assert report["consumer_folds"] == 4, report
        assert after <= before + 1e-9, (
            f"affine fold increased the calibrated gap: {before} -> {after}"
        )

    def test_mixer_cross_compute_op_seams_skipped_honestly(self):
        """fc2 channel axes are consumed across permute/weight-shared seams
        (blocks 0-2) or by the host classifier ComputeOp (block 3) — the walk
        voids them with the honest reason instead of folding."""
        flow = _converted_mixer()
        report = apply_lif_affine_fold(flow, _cal_x(), T_STEPS)
        assert report["skipped"] == {
            "mixer_blocks_0_fc2": "channel_axis_not_preserved",
            "mixer_blocks_1_fc2": "channel_axis_not_preserved",
            "mixer_blocks_2_fc2": "channel_axis_not_preserved",
            "mixer_blocks_3_fc2": "host_compute_consumer",
        }, report
        assert report["readout_folds"] == 0, report

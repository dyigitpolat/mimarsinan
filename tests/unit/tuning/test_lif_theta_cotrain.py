"""Per-channel trainable theta (firing-gain) co-training for the LIF deployment.

The single-scalar firing threshold (0.99-quantile ``activation_scale``) cannot serve
both wide and narrow channels of a perceptron; ``lif_theta_cotrain`` rebinds each
non-encoding perceptron's ``activation_scale`` to a per-output-channel trainable
Parameter so the gradual blend ramp co-optimises theta WITH the weights through the
deployed LIF dynamics. This is the LIF analogue of ``ttfs_theta_cotrain`` and reuses
the SAME shared promotion mechanism (``_promote_per_channel_theta`` on the base
``KDBlendAdaptationTuner``). Default OFF ⇒ the scalar theta path is byte-identical.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


def _make_tuner(
    tmp_path, *, theta=True, fast=True, steps_per_rate=3, rates=None,
):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 4
    cfg["cycle_accurate_lif_forward"] = True
    cfg["lif_blend_fast"] = fast
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_theta_cotrain"] = theta
    if rates is not None:
        cfg["lif_blend_fast_rates"] = rates
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=am,
    )
    return tuner, model, am


def _non_encoding(model):
    return [
        p for p in model.get_perceptrons()
        if not getattr(p, "is_encoding_layer", False)
    ]


class TestThetaPromotionContract:
    def test_default_off_leaves_scalar_theta_untrained(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, theta=False)
        assert tuner._lif_theta_cotrain is False
        assert tuner._theta_cotrain_params is None
        assert tuner._theta_cotrain_stats is None
        for p in _non_encoding(model):
            assert p.activation_scale.requires_grad is False

    def test_on_promotes_non_encoding_to_per_channel_trainable(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, theta=True)
        assert tuner._lif_theta_cotrain is True
        promoted = _non_encoding(model)
        assert len(promoted) >= 1
        for p in promoted:
            s = p.activation_scale
            assert s.requires_grad, "co-trained theta must require grad"
            assert s.dim() == 1 and s.numel() == p.layer.weight.shape[0], (
                "theta must be per-output-channel"
            )

    def test_lif_node_references_same_promoted_param(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, theta=True)
        for p in _non_encoding(model):
            lif_nodes = [m for m in p.modules() if isinstance(m, LIFActivation)]
            assert lif_nodes, "blend target LIFActivation must exist post-install"
            for node in lif_nodes:
                assert node.activation_scale is p.activation_scale, (
                    "LIF node theta must be the SAME object the optimiser trains"
                )

    def test_encoding_layer_theta_left_fixed(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, theta=True)
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                assert p.activation_scale.requires_grad is False

    def test_stats_reported(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, theta=True)
        assert tuner._theta_cotrain_stats == {"n_theta": len(_non_encoding(model))}


class TestThetaTrainedByFastLadder:
    def test_promoted_theta_in_fast_optimizer_and_changes(self, tmp_path, accepting_gate):
        torch.manual_seed(0)
        tuner, model, _ = _make_tuner(
            tmp_path, theta=True, steps_per_rate=3, rates=[0.5, 1.0],
        )
        before = [p.detach().clone() for p in tuner._theta_cotrain_params]
        tuner.run()
        opt_param_ids = {
            id(p) for g in tuner._fast_optimizer.param_groups for p in g["params"]
        }
        for theta in tuner._theta_cotrain_params:
            assert id(theta) in opt_param_ids, "theta must be in the fast optimizer"
        moved = sum(
            float((a.detach() - b).abs().sum())
            for a, b in zip(tuner._theta_cotrain_params, before)
        )
        assert moved > 0.0, "the optimiser must actually train the promoted theta"


class TestThetaComposesWithDistmatch:
    """theta_cotrain (per-channel scale) and distmatch (per-neuron bias) are
    orthogonal levers — the per-channel scale must not break the scalar-free
    distmatch path."""

    def test_theta_and_distmatch_both_active(self, tmp_path):
        torch.manual_seed(0)
        cfg = default_config()
        cfg["spiking_mode"] = "lif"
        cfg["firing_mode"] = "Default"
        cfg["thresholding_mode"] = "<"
        cfg["simulation_steps"] = 4
        cfg["cycle_accurate_lif_forward"] = True
        cfg["lif_blend_fast"] = True
        cfg["lif_blend_fast_steps_per_rate"] = 2
        cfg["lif_blend_fast_rates"] = [0.5, 1.0]
        cfg["lif_theta_cotrain"] = True
        cfg["lif_distmatch"] = True
        cfg["lif_distmatch_bias_iters"] = 4
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = LIFAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
            adaptation_manager=am,
        )
        assert tuner._theta_cotrain_params is not None
        tuner.run()
        assert tuner._lif_distmatch_stats is not None
        for p in _non_encoding(model):
            assert p.activation_scale.dim() == 1

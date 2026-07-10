"""LIFAffineFoldStep: pre-WQ slot, config gating, Novena S-gate, fold commit."""

from __future__ import annotations

import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.adaptation.lif_affine_fold_step import (
    LIFAffineFoldStep,
)
from mimarsinan.tuning.lif_affine_fold import AFFINE_FOLD_FLAG


def _config(**overrides):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 8
    cfg["cycle_accurate_lif_forward"] = True
    cfg["activation_quantization"] = True
    cfg["lif_affine_fold"] = True
    cfg["lif_half_step_bias"] = False
    cfg.update(overrides)
    return cfg


def _deployed_model(cfg, hidden_layers=2):
    import copy

    from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    torch.manual_seed(0)
    model = copy.deepcopy(make_tiny_supermodel(hidden_layers=hidden_layers))
    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5
    tuner = LIFAdaptationTuner(
        pipeline, model, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=AdaptationManager(),
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model


def _run_step(cfg, model):
    pipeline = MockPipeline(config=cfg)
    pipeline.seed("model", model, step_name="LIF Adaptation")
    step = LIFAffineFoldStep(pipeline)
    step.name = "LIF Affine Fold"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


class TestAppliesTo:
    def test_off_by_default(self):
        cfg = _config()
        cfg.pop("lif_affine_fold")
        assert not LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))

    def test_on_when_knob_set_for_lif(self):
        assert LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(_config()))

    def test_never_for_ttfs_modes(self):
        cfg = _config(spiking_mode="ttfs_quantized", firing_mode="TTFS",
                      thresholding_mode="<=")
        assert not LIFAffineFoldStep.applies_to(DeploymentPlan.resolve(cfg))


class TestStepOrdering:
    def test_selected_before_weight_quantization(self):
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
        )

        cfg = _config(weight_quantization=True, weight_bits=8)
        names = [name for name, _cls in get_pipeline_step_specs(cfg)]
        assert "LIF Affine Fold" in names
        assert names.index("LIF Affine Fold") < names.index("Weight Quantization")
        assert names.index("LIF Affine Fold") > names.index("LIF Adaptation")

    def test_not_selected_when_flag_off(self):
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_step_specs,
        )

        cfg = _config(lif_affine_fold=False)
        names = [name for name, _cls in get_pipeline_step_specs(cfg)]
        assert "LIF Affine Fold" not in names


class TestProcess:
    def test_folds_and_updates_model(self):
        cfg = _config()
        model = _deployed_model(cfg)
        pipeline = _run_step(cfg, model)
        assert pipeline.cache["LIF Affine Fold.model"] is model
        assert any(
            getattr(p, AFFINE_FOLD_FLAG, False) for p in model.get_perceptrons()
        ), "the step must apply at least one fold on a plain LIF chain"

    def test_novena_below_s8_is_gated(self):
        cfg = _config(firing_mode="Novena", simulation_steps=4)
        model = _deployed_model(cfg)
        weights = [p.layer.weight.data.clone() for p in model.get_perceptrons()]
        _run_step(cfg, model)
        for p, saved in zip(model.get_perceptrons(), weights):
            assert torch.equal(p.layer.weight.data, saved), (
                "Novena at S<8 must skip the fold (grid overfit, memo §4 C4)"
            )

    def test_half_step_prefold_when_enabled(self):
        from mimarsinan.mapping.support.bias_compensation import LIF_HALF_STEP_FLAG

        cfg = _config(lif_half_step_bias=True)
        model = _deployed_model(cfg)
        _run_step(cfg, model)
        non_encoders = [
            p for p in model.get_perceptrons()
            if not getattr(p, "is_encoding_layer", False)
        ]
        assert all(getattr(p, LIF_HALF_STEP_FLAG, False) for p in non_encoders), (
            "the affine estimator is fitted on the nearest (half-step) chain"
        )

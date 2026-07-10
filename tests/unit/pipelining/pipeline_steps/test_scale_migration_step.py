"""ScaleMigrationStep in isolation: exactness postcondition, stats reuse, report."""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.pipelining.pipeline_steps.adaptation import (
    scale_migration_step as sms,
)
from mimarsinan.torch_mapping.converter import convert_torch_model

INPUT_SHAPE = (1, 8, 8)
NUM_CLASSES = 4


def _tiny_mixer_flow(seed=0):
    torch.manual_seed(seed)
    model = TorchMLPMixerCore(
        input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
        patch_n_1=2, patch_m_1=2, patch_c_1=4, fc_w_1=8, fc_w_2=8,
        base_activation="ReLU", num_blocks=2,
    ).eval()
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.mul_(3.0)
                if mod.bias is not None:
                    mod.bias.copy_(torch.randn_like(mod.bias) * 0.5)
        for blk in model.mixer_blocks:
            spread = torch.linspace(0.5, 3.0, blk.fc1.weight.shape[0]).view(-1, 1)
            blk.fc1.weight.mul_(spread)
            blk.fc1.bias.mul_(spread.view(-1))
    return convert_torch_model(
        model, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
    ).eval()


def _make_step(pipeline, model):
    pipeline.config["scale_migration"] = True
    pipeline.seed("model", model)
    step = sms.ScaleMigrationStep(pipeline)
    step.name = "ScaleMigration"
    pipeline.prepare_step(step)
    return step


class TestScaleMigrationStep:
    def test_migrates_and_preserves_the_float_function(self):
        pipeline = MockPipeline()
        model = _tiny_mixer_flow()
        torch.manual_seed(3)
        x = torch.rand(8, *INPUT_SHAPE)
        with torch.no_grad():
            reference = model(x)

        step = _make_step(pipeline, model)
        step.run()

        assert pipeline.cache["ScaleMigration.model"] is model
        with torch.no_grad():
            migrated = model(x)
        assert float((reference - migrated).abs().max()) <= 1e-5

        report = pipeline.cache["ScaleMigration.scale_migration_report"]
        assert report["clip_ratio"] == pytest.approx(4.0)
        assert len(report["migrated"]) >= 4
        for hop in report["migrated"]:
            assert 1.0 / 4.0 - 1e-9 <= hop["s_min"] <= hop["s_max"] <= 4.0 + 1e-9

    def test_reuses_the_channel_stats_machinery(self, monkeypatch):
        pipeline = MockPipeline()
        model = _tiny_mixer_flow()
        calls = {}
        real = sms.collect_channel_stats

        def spy(model_arg, batches, device, **kwargs):
            calls["count"] = calls.get("count", 0) + 1
            calls["kwargs"] = kwargs
            return real(model_arg, batches, device, **kwargs)

        monkeypatch.setattr(sms, "collect_channel_stats", spy)
        step = _make_step(pipeline, model)
        step.run()
        assert calls["count"] == 1
        assert "accumulator_factory" in calls["kwargs"]

    def test_fails_loud_when_function_not_preserved(self, monkeypatch):
        pipeline = MockPipeline()
        model = _tiny_mixer_flow()
        real = sms.equalize_channel_scales

        def corrupting(model_arg, stats, *, clip_ratio):
            report = real(model_arg, stats, clip_ratio=clip_ratio)
            with torch.no_grad():
                model_arg.get_perceptrons()[1].layer.weight.add_(1.0)
            return report

        monkeypatch.setattr(sms, "equalize_channel_scales", corrupting)
        step = _make_step(pipeline, model)
        with pytest.raises(RuntimeError, match="preserve"):
            step.run()

    def test_clip_ratio_flows_from_config(self):
        pipeline = MockPipeline()
        pipeline.config["scale_migration_clip_ratio"] = 2.0
        model = _tiny_mixer_flow()
        step = _make_step(pipeline, model)
        step.run()

        report = pipeline.cache["ScaleMigration.scale_migration_report"]
        assert report["clip_ratio"] == pytest.approx(2.0)
        for hop in report["migrated"]:
            assert 0.5 - 1e-9 <= hop["s_min"] <= hop["s_max"] <= 2.0 + 1e-9

    def test_validate_returns_float_and_cleanup_closes_trainer(self):
        pipeline = MockPipeline()
        model = _tiny_mixer_flow()
        step = _make_step(pipeline, model)
        step.run()
        assert isinstance(step.validate(), float)
        assert step.trainer is not None
        step.cleanup()

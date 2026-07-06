"""Tests for ActivationAnalysisStep in isolation."""

import pytest
import torch

from conftest import (
    MockDataProviderFactory,
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
    ActivationAnalysisStep,
    scale_from_activations,
)


class TestActivationAnalysisStep:
    def _make_step(self, mock_pipeline, model=None):
        if model is None:
            model = make_tiny_supermodel()
        mock_pipeline.seed("model", model)
        step = ActivationAnalysisStep(mock_pipeline)
        step.name = "ActivationAnalysis"
        mock_pipeline.prepare_step(step)
        return step

    def test_promises_activation_scales(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        key = "ActivationAnalysis.activation_scales"
        assert key in mock_pipeline.cache
        scales = mock_pipeline.cache[key]
        assert isinstance(scales, list)
        assert len(scales) > 0
        assert all(isinstance(s, float) for s in scales)

    def test_scales_are_positive(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        scales = mock_pipeline.cache["ActivationAnalysis.activation_scales"]
        assert all(s >= 0 for s in scales)

    def test_scales_count_matches_perceptrons(self, mock_pipeline):
        model = make_tiny_supermodel()
        step = self._make_step(mock_pipeline, model)
        step.run()

        num_perceptrons = len(model.get_perceptrons())
        scales = mock_pipeline.cache["ActivationAnalysis.activation_scales"]
        assert len(scales) == num_perceptrons

    def test_writes_activation_scale_stats(self, tmp_path):
        cfg = default_config()
        pipeline = MockPipeline(
            config=cfg,
            working_directory=str(tmp_path / "pipeline_cache"),
            data_provider_factory=MockDataProviderFactory(size=64),
        )
        step = self._make_step(pipeline)
        step.run()

        scales = pipeline.cache["ActivationAnalysis.activation_scales"]
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]

        assert isinstance(stats, dict)
        assert stats["num_batches"] >= 2
        assert stats["quantile"] == pytest.approx(0.99)
        assert len(stats["layers"]) == len(scales)
        assert stats["summary"]["max_scale"] >= stats["summary"]["min_scale"]
        assert all(layer["sample_count"] >= 0 for layer in stats["layers"])

    def test_validate_returns_float(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()
        metric = step.validate()
        assert isinstance(metric, float)


    def test_cleanup_closes_trainer(self, mock_pipeline):
        """cleanup() releases the step's trainer (DataLoader workers)."""
        step = self._make_step(mock_pipeline)
        step.run()
        step.validate()
        assert step.trainer is not None
        assert step.trainer.train_loader is not None

        step.cleanup()

        assert step.trainer.train_loader is None
        assert step.trainer.validation_loader is None
        assert step.trainer.test_loader is None

    def test_single_perceptron_model(self, mock_pipeline):
        """Edge case: model with only one perceptron."""
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
        from mimarsinan.models.nn.layers import LeakyGradReLU
        from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
        from mimarsinan.mapping.mapping_utils import (
            InputMapper, PerceptronMapper, Ensure2DMapper,
            EinopsRearrangeMapper, ModuleMapper, ModelRepresentation,
        )
        import torch.nn as nn
        from conftest import default_config

        class SinglePerceptronFlow(PerceptronFlow):
            def __init__(self):
                super().__init__("cpu")
                self.input_activation = nn.Identity()
                self.p = Perceptron(4, 64)
                inp = InputMapper((1, 8, 8))
                m = ModuleMapper(inp, self.input_activation)
                out = EinopsRearrangeMapper(m, "... c h w -> ... (c h w)")
                out = Ensure2DMapper(out)
                out = PerceptronMapper(out, self.p)
                self._mapper_repr = ModelRepresentation(out)

            def get_perceptrons(self):
                return self._mapper_repr.get_perceptrons()

            def get_perceptron_groups(self):
                return self._mapper_repr.get_perceptron_groups()

            def get_mapper_repr(self):
                return self._mapper_repr

            def get_input_activation(self):
                return self.input_activation

            def set_input_activation(self, a):
                self.input_activation = a

            def forward(self, x):
                return self._mapper_repr(x)

        model = SinglePerceptronFlow()
        model.p.is_encoding_layer = True
        cfg = default_config()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            p.base_activation = LeakyGradReLU()
            p.activation = LeakyGradReLU()
            am.update_activation(cfg, p)
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 1, 8, 8))

        step = self._make_step(mock_pipeline, model)
        step.run()
        scales = mock_pipeline.cache["ActivationAnalysis.activation_scales"]
        assert len(scales) == 1

    def test_scale_from_activations_all_zeros_returns_fallback(self):
        """When all activations are pruned (zero), scale is fallback 1.0."""
        flat = torch.zeros(1000)
        assert scale_from_activations(flat) == 1.0

    def test_scale_from_activations_only_non_pruned_used(self):
        """Scale is computed from non-pruned activations only, not skewed by zeros."""
        # Many zeros (pruned) + a few large values: scale should reflect the large values.
        flat = torch.cat([torch.zeros(900), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])])
        scale = scale_from_activations(flat)
        expected = torch.quantile(
            flat[flat > 0], 0.99, interpolation="higher"
        ).item()
        assert expected == 5.0
        assert scale == expected

    def test_scale_from_activations_matches_true_quantile_by_count(self):
        """The scale statistic should match a count-based quantile, not a mass-weighted one."""
        flat = torch.cat(
            [
                torch.ones(980),
                torch.full((10,), 2.0),
                torch.full((5,), 3.0),
                torch.full((5,), 100.0),
            ]
        )
        expected = torch.quantile(
            flat[flat > 0], 0.99, interpolation="higher"
        ).item()
        assert expected == 3.0
        scale = scale_from_activations(flat)
        assert scale == expected

    def test_scale_from_activations_mixed_pruned_uses_active_only(self):
        """With many zeros (pruned), scale is computed from non-zero activations only."""
        torch.manual_seed(42)
        flat = torch.cat([torch.zeros(950), torch.rand(50) * 0.5 + 0.5])
        scale = scale_from_activations(flat)
        # All 50 active values are in [0.5, 1.0], so scale should be in that range
        assert scale >= 0.5
        assert scale <= 1.0


@pytest.mark.slow
class TestActivationAnalysisTorchConcatRegression:
    def test_squeezenet_flow_writes_named_scale_stats(self, tmp_path):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.torch_mapping.converter import convert_torch_model

        cfg = default_config()
        builder = BUILDERS_REGISTRY["torch_squeezenet11"](
            device="cpu",
            input_shape=(3, 32, 32),
            num_classes=4,
            pipeline_config=cfg,
        )
        raw_model = builder.build({})
        raw_model.eval()
        with torch.no_grad():
            raw_model(torch.randn(1, 3, 32, 32))

        flow = convert_torch_model(raw_model, input_shape=(3, 32, 32), num_classes=4)
        pipeline = MockPipeline(
            config={**cfg, "input_shape": (3, 32, 32), "num_classes": 4},
            working_directory=str(tmp_path / "pipeline_cache"),
            data_provider_factory=MockDataProviderFactory(
                input_shape=(3, 32, 32), num_classes=4, size=64
            ),
        )
        pipeline.seed("model", flow)
        step = ActivationAnalysisStep(pipeline)
        step.name = "ActivationAnalysis"
        pipeline.prepare_step(step)
        step.run()

        scales = pipeline.cache["ActivationAnalysis.activation_scales"]
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]

        assert len(scales) == len(flow.get_perceptrons()) > 0
        assert len(stats["layers"]) == len(scales)
        assert stats["num_batches"] >= 2
        assert all(layer["name"] for layer in stats["layers"])


class TestA6InstallResolutionGauge:
    """[MBH-A6] the pre-flight value gauge is computed where theta is decided."""

    def _run_step(self, tmp_path, cfg):
        pipeline = MockPipeline(
            config=cfg,
            working_directory=str(tmp_path / "pipeline_cache"),
            data_provider_factory=MockDataProviderFactory(size=64),
        )
        model = make_tiny_supermodel()
        pipeline.seed("model", model)
        step = ActivationAnalysisStep(pipeline)
        step.name = "ActivationAnalysis"
        pipeline.prepare_step(step)
        step.run()
        return pipeline

    def test_lif_emits_value_gauge_at_the_simulation_grid(self, tmp_path, capsys):
        cfg = default_config()  # lif, simulation_steps=4
        pipeline = self._run_step(tmp_path, cfg)
        out = capsys.readouterr().out
        assert "[MBH-A6] kind=value" in out
        assert "levels=4" in out
        gauge = pipeline.cache["ActivationAnalysis.install_resolution_gauge"]
        assert gauge["levels"] == 4
        assert isinstance(gauge["fails"], bool)
        assert isinstance(gauge["starved_hops"], list)
        assert len(gauge["median_effective_levels"]) == len(
            pipeline.cache["ActivationAnalysis.activation_scales"]
        )

    def test_aq_mode_uses_the_target_tq_grid(self, tmp_path, capsys):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_quantized"
        cfg["target_tq"] = 8
        pipeline = self._run_step(tmp_path, cfg)
        assert "levels=8" in capsys.readouterr().out
        assert pipeline.cache["ActivationAnalysis.install_resolution_gauge"]["levels"] == 8

    def test_analytic_ttfs_has_no_value_grid(self, tmp_path, capsys):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs"
        pipeline = self._run_step(tmp_path, cfg)
        assert "[MBH-A6]" not in capsys.readouterr().out
        gauge = pipeline.cache["ActivationAnalysis.install_resolution_gauge"]
        assert gauge["levels"] is None
        assert gauge["fails"] is False


class TestStarvationAwareQuantile:
    """[5v B1(i)] the sync full-quantile special-case becomes starvation-aware:
    a hop whose full-quantile theta leaves under 2 usable grid levels per live
    channel deflates to the mode-generic 0.99 quantile (t0_21 measured: q99
    alone lifts the AQ entry 0.10 -> 0.50)."""

    def _pipeline(self, tmp_path, cfg):
        return MockPipeline(
            config=cfg,
            working_directory=str(tmp_path / "pipeline_cache"),
            data_provider_factory=MockDataProviderFactory(size=64),
        )

    def _sync_cfg(self):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "synchronized"
        cfg["activation_quantization"] = True
        cfg["activation_scale_quantile"] = 1.0
        cfg["starvation_aware_scale_quantile"] = True
        return cfg

    def _run(self, tmp_path, cfg, model=None):
        pipeline = self._pipeline(tmp_path, cfg)
        model = model or make_tiny_supermodel()
        pipeline.seed("model", model)
        step = ActivationAnalysisStep(pipeline)
        step.name = "ActivationAnalysis"
        pipeline.prepare_step(step)
        step.run()
        return pipeline, model

    def test_starved_hop_deflates_to_the_generic_quantile(
        self, tmp_path, monkeypatch, capsys,
    ):
        # Force the gauge to read every hop as starved at the full-quantile
        # theta: every scale must then equal the 0.99-quantile scale.
        import mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step as step_mod

        monkeypatch.setattr(
            step_mod, "needs_quantile_deflation", lambda *a, **k: True,
        )
        cfg = self._sync_cfg()
        pipeline, model = self._run(tmp_path, cfg)
        out = capsys.readouterr().out
        assert "[MBH-A6] quantile-deflate" in out
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]
        assert len(stats["deflated_layers"]) == len(
            pipeline.cache["ActivationAnalysis.activation_scales"]
        )

    def test_healthy_hops_keep_the_full_quantile(self, tmp_path, monkeypatch):
        import mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step as step_mod

        monkeypatch.setattr(
            step_mod, "needs_quantile_deflation", lambda *a, **k: False,
        )
        cfg = self._sync_cfg()
        pipeline, _ = self._run(tmp_path, cfg)
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]
        assert stats["deflated_layers"] == []
        assert stats["quantile"] == pytest.approx(1.0)

    def test_knob_off_is_bit_identical(self, tmp_path):
        cfg = self._sync_cfg()
        cfg["starvation_aware_scale_quantile"] = False
        torch.manual_seed(7)
        pipeline_off, _ = self._run(tmp_path, cfg)
        scales_off = pipeline_off.cache["ActivationAnalysis.activation_scales"]

        cfg2 = self._sync_cfg()
        cfg2["starvation_aware_scale_quantile"] = False
        torch.manual_seed(7)
        pipeline_ref, _ = self._run(tmp_path, cfg2)
        assert scales_off == pipeline_ref.cache["ActivationAnalysis.activation_scales"]
        stats = pipeline_off.cache["ActivationAnalysis.activation_scale_stats"]
        assert "deflated_layers" not in stats or stats["deflated_layers"] == []

    def test_non_sync_modes_never_deflate(self, tmp_path, monkeypatch):
        # The knob rides only the sync recipe; a lif config with the knob
        # accidentally on must not deflate (mode-gated, not knob-gated alone).
        import mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step as step_mod

        monkeypatch.setattr(
            step_mod, "needs_quantile_deflation", lambda *a, **k: True,
        )
        cfg = default_config()  # lif
        cfg["starvation_aware_scale_quantile"] = True
        pipeline, _ = self._run(tmp_path, cfg)
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]
        assert stats.get("deflated_layers", []) == []

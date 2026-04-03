"""Tests for ActivationAnalysisStep in isolation."""

import pytest
import torch

from conftest import (
    MockDataProviderFactory,
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import (
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
        from mimarsinan.models.layers import LeakyGradReLU
        from mimarsinan.tuning.adaptation_manager import AdaptationManager
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

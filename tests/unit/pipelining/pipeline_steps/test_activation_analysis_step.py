"""Tests for ActivationAnalysisStep in isolation."""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

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
        from mimarsinan.models.supermodel import Supermodel
        from mimarsinan.models.preprocessing.input_cq import InputCQ
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

        flow = SinglePerceptronFlow()
        model = Supermodel("cpu", (1, 8, 8), 4, InputCQ(4), flow, 4)
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
        # 99th percentile of cumsum over [1,2,3,4,5]: cumsum = [1,3,6,10,15], norm = [1/15, ...],
        # searchsorted(0.99) gives index 4, so scale = 5.0
        assert scale == 5.0

    def test_scale_from_activations_mixed_pruned_uses_active_only(self):
        """With many zeros (pruned), scale is computed from non-zero activations only."""
        torch.manual_seed(42)
        flat = torch.cat([torch.zeros(950), torch.rand(50) * 0.5 + 0.5])
        scale = scale_from_activations(flat)
        # All 50 active values are in [0.5, 1.0], so scale should be in that range
        assert scale >= 0.5
        assert scale <= 1.0

"""Tests for ActivationAnalysisStep in isolation."""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import ActivationAnalysisStep


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

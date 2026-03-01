"""Tests for ModelBuildingStep in isolation."""

import pytest
import torch

from conftest import MockPipeline, default_config

from mimarsinan.pipelining.pipeline_steps.model_building_step import ModelBuildingStep
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder


class TestModelBuildingStep:
    def _make_step(self, mock_pipeline):
        cfg = mock_pipeline.config
        cfg.setdefault("input_shape", (1, 8, 8))
        cfg.setdefault("num_classes", 4)
        cfg.setdefault("max_axons", 256)
        cfg.setdefault("max_neurons", 256)

        builder = SimpleMLPBuilder(
            cfg["device"], cfg["input_shape"], cfg["num_classes"],
            cfg["max_axons"], cfg["max_neurons"], cfg,
        )
        model_config = {
            "mlp_width_1": 16,
            "mlp_width_2": 16,
            "base_activation": "ReLU",
        }
        mock_pipeline.seed("model_config", model_config)
        mock_pipeline.seed("model_builder", builder)

        step = ModelBuildingStep(mock_pipeline)
        step.name = "ModelBuilding"
        mock_pipeline.prepare_step(step)
        return step

    def test_promises_model_and_adaptation_manager(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        assert "ModelBuilding.model" in mock_pipeline.cache
        assert "ModelBuilding.adaptation_manager" in mock_pipeline.cache

    def test_model_is_supermodel(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        model = mock_pipeline.cache["ModelBuilding.model"]
        assert hasattr(model, "get_perceptrons")
        assert hasattr(model, "get_mapper_repr")

    def test_model_has_perceptrons(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        model = mock_pipeline.cache["ModelBuilding.model"]
        perceptrons = model.get_perceptrons()
        assert len(perceptrons) > 0

    def test_model_forward_produces_correct_shape(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        model = mock_pipeline.cache["ModelBuilding.model"]
        model.eval()
        x = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_adaptation_manager_initial_rates(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        am = mock_pipeline.cache["ModelBuilding.adaptation_manager"]
        assert am.clamp_rate == 0.0
        assert am.quantization_rate == 0.0
        assert am.shift_rate == 0.0

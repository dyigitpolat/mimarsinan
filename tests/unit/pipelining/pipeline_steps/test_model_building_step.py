"""Tests for ModelBuildingStep in isolation."""

import pytest
import torch

from conftest import MockPipeline, default_config

from mimarsinan.pipelining.pipeline_steps.config.model_building_step import ModelBuildingStep
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder


class TestModelBuildingStep:
    def _make_step(self, mock_pipeline):
        cfg = mock_pipeline.config
        cfg.setdefault("input_shape", (1, 8, 8))
        cfg.setdefault("num_classes", 4)

        builder = SimpleMLPBuilder(
            cfg["device"], cfg["input_shape"], cfg["num_classes"], cfg,
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

    def test_model_is_perceptron_flow(self, mock_pipeline):
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

class TestStaticOnchipMajorityGate:
    """W2 Q3: the t0_09-class majority-floor abort moves from post-pretrain
    SCM (~minutes) to Model Building (seconds) via the closed-form static
    twin. deep_mlp d4 subsume: fraction(w) = 3(w^2+w) / (3(w^2+w) + 795w + 10),
    so w=64 sits 0.29 pp below the 20% floor (w >= 67 required) and the
    committed w=128 respec clears it at 32.74%."""

    def _make_deepmlp_step(self, mock_pipeline, width):
        from mimarsinan.models.builders.deep_mlp_builder import DeepMLPBuilder

        cfg = mock_pipeline.config
        cfg["input_shape"] = (1, 28, 28)
        cfg["num_classes"] = 10
        builder = DeepMLPBuilder(
            cfg["device"], cfg["input_shape"], cfg["num_classes"], cfg,
        )
        mock_pipeline.seed("model_config", {"depth": 4, "width": width})
        mock_pipeline.seed("model_builder", builder)
        step = ModelBuildingStep(mock_pipeline)
        step.name = "ModelBuilding"
        mock_pipeline.prepare_step(step)
        return step

    def test_t0_09_w64_spec_fails_fast_at_build(self, mock_pipeline):
        from mimarsinan.mapping.verification.onchip_majority import (
            OnchipMajorityError,
        )

        step = self._make_deepmlp_step(mock_pipeline, width=64)
        with pytest.raises(OnchipMajorityError):
            step.run()

    def test_t0_09_committed_w128_respec_passes(self, mock_pipeline):
        step = self._make_deepmlp_step(mock_pipeline, width=128)
        step.run()
        assert "ModelBuilding.model" in mock_pipeline.cache

    def test_gate_opt_out_admits_host_majority_spec(self, mock_pipeline):
        mock_pipeline.config["onchip_majority_gate"] = False
        step = self._make_deepmlp_step(mock_pipeline, width=64)
        step.run()
        assert "ModelBuilding.model" in mock_pipeline.cache

    def test_floor_honors_config_min_fraction(self, mock_pipeline):
        from mimarsinan.mapping.verification.onchip_majority import (
            OnchipMajorityError,
        )

        mock_pipeline.config["onchip_majority_min_fraction"] = 0.5
        step = self._make_deepmlp_step(mock_pipeline, width=128)
        with pytest.raises(OnchipMajorityError):
            step.run()

    def test_supermodel_counts_on_its_own_mapper_repr(self, mock_pipeline):
        """SimpleMLP is already a perceptron flow (encoder on chip, readout the
        only host op): it is on-chip-majority and must never FX-trace."""
        step = TestModelBuildingStep()._make_step(mock_pipeline)
        step.run()
        assert "ModelBuilding.model" in mock_pipeline.cache

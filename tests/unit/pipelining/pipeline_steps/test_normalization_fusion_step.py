"""Tests for NormalizationFusionStep in isolation."""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.pipelining.pipeline_steps.normalization_fusion_step import NormalizationFusionStep


class TestNormalizationFusionStep:
    def _make_step(self, mock_pipeline, model=None):
        if model is None:
            model = make_tiny_supermodel()
        mock_pipeline.seed("model", model)
        step = NormalizationFusionStep(mock_pipeline)
        step.name = "NormFusion"
        mock_pipeline.prepare_step(step)
        return step

    def test_updates_model_entry(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()

        assert "NormFusion.model" in mock_pipeline.cache

    def test_batchnorm_fused_to_identity(self, mock_pipeline):
        model = make_tiny_supermodel()
        has_bn_before = any(
            not isinstance(p.normalization, nn.Identity)
            for p in model.get_perceptrons()
        )
        assert has_bn_before, "Fixture should have at least one BN layer"

        step = self._make_step(mock_pipeline, model)
        step.run()

        fused_model = mock_pipeline.cache["NormFusion.model"]
        for p in fused_model.get_perceptrons():
            assert isinstance(p.normalization, nn.Identity), \
                f"Perceptron {p.name} still has non-Identity normalization after fusion"

    def test_output_preserved_after_fusion(self, mock_pipeline):
        """Fusion should be mathematically equivalent."""
        model = make_tiny_supermodel()
        model.eval()
        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            out_before = model(x).clone()

        step = self._make_step(mock_pipeline, model)
        step.run()

        fused_model = mock_pipeline.cache["NormFusion.model"]
        fused_model.eval()
        with torch.no_grad():
            out_after = fused_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), \
            f"Max diff: {(out_before - out_after).abs().max()}"

    def test_model_without_batchnorm_is_noop(self, mock_pipeline):
        """Model with all Identity normalizations should pass through unchanged."""
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.normalization = nn.Identity()

        model.eval()
        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            out_before = model(x).clone()

        step = self._make_step(mock_pipeline, model)
        step.run()

        fused_model = mock_pipeline.cache["NormFusion.model"]
        fused_model.eval()
        with torch.no_grad():
            out_after = fused_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_validate_returns_metric(self, mock_pipeline):
        step = self._make_step(mock_pipeline)
        step.run()
        metric = step.validate()
        assert isinstance(metric, float)

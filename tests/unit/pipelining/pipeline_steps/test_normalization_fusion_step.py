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

    def test_pruning_buffers_preserved_after_fusion(self, mock_pipeline):
        """Fusion must preserve prune_row_mask and prune_col_mask so IR compaction can use model masks (H1)."""
        model = make_tiny_supermodel()
        # Find a perceptron with BatchNorm so fusion runs
        perceptrons = model.get_perceptrons()
        fused_idx = None
        for i, p in enumerate(perceptrons):
            if not isinstance(p.normalization, nn.Identity):
                out_f = p.layer.weight.shape[0]
                in_f = p.layer.weight.shape[1]
                p.layer.register_buffer(
                    "prune_row_mask",
                    torch.zeros(out_f, dtype=torch.bool),
                )
                p.layer.register_buffer(
                    "prune_col_mask",
                    torch.zeros(in_f, dtype=torch.bool),
                )
                fused_idx = i
                break
        assert fused_idx is not None, "Fixture should have at least one BN layer"

        step = self._make_step(mock_pipeline, model)
        step.run()

        fused_model = mock_pipeline.cache["NormFusion.model"]
        p_fused = fused_model.get_perceptrons()[fused_idx]
        assert hasattr(p_fused.layer, "prune_row_mask"), "prune_row_mask must be preserved"
        assert hasattr(p_fused.layer, "prune_col_mask"), "prune_col_mask must be preserved"
        assert p_fused.layer.prune_row_mask.dim() == 1
        assert p_fused.layer.prune_col_mask.dim() == 1
        assert p_fused.layer.prune_row_mask.shape[0] == p_fused.layer.weight.shape[0]
        assert p_fused.layer.prune_col_mask.shape[0] == p_fused.layer.weight.shape[1]

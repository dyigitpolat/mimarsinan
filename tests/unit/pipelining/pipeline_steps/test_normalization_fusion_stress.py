"""
Stress tests for NormalizationFusionStep.

Tests with known weights where we can compute the expected output by hand,
and adversarial batch norm parameters.
"""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.pipelining.pipeline_steps.normalization_fusion_step import NormalizationFusionStep


class TestNormFusionMathematicalEquivalence:
    """
    The fusion should be mathematically exact: the fused model must produce
    bitwise-close output to the unfused model for ANY input, not just the
    one we tested during development.
    """

    def _make_step(self, mock_pipeline, model):
        mock_pipeline.seed("model", model)
        step = NormalizationFusionStep(mock_pipeline)
        step.name = "NormFusion"
        mock_pipeline.prepare_step(step)
        return step

    def test_fusion_with_random_batchnorm_params(self, mock_pipeline):
        """
        Use fully random BN parameters (gamma, beta, running_mean, running_var)
        and verify output equivalence across multiple random inputs.
        """
        torch.manual_seed(77)
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            if not isinstance(p.normalization, nn.Identity):
                bn = p.normalization
                bn.weight.data = torch.randn_like(bn.weight.data)
                bn.bias.data = torch.randn_like(bn.bias.data)
                bn.running_mean.data = torch.randn_like(bn.running_mean.data)
                bn.running_var.data = torch.abs(torch.randn_like(bn.running_var.data))
        model.eval()

        inputs = [torch.randn(4, 1, 8, 8) for _ in range(5)]
        with torch.no_grad():
            outs_before = [model(x).clone() for x in inputs]

        step = self._make_step(mock_pipeline, model)
        step.run()
        fused = mock_pipeline.cache["NormFusion.model"]
        fused.eval()

        for i, x in enumerate(inputs):
            with torch.no_grad():
                out_after = fused(x)
            assert torch.allclose(outs_before[i], out_after, atol=1e-4), \
                f"Input {i}: max diff {(outs_before[i] - out_after).abs().max()}"

    def test_fusion_with_near_zero_variance(self, mock_pipeline):
        """
        Very small running_var values stress the division in BN fusion.
        BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
        With var ≈ 0: gamma / sqrt(eps) is large — test numerical stability.
        """
        torch.manual_seed(88)
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            if not isinstance(p.normalization, nn.Identity):
                bn = p.normalization
                bn.running_var.data.fill_(1e-10)
        model.eval()

        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            out_before = model(x).clone()

        step = self._make_step(mock_pipeline, model)
        step.run()
        fused = mock_pipeline.cache["NormFusion.model"]
        fused.eval()

        with torch.no_grad():
            out_after = fused(x)

        assert not torch.isnan(out_after).any(), "NaN in fused output"
        assert torch.allclose(out_before, out_after, atol=1e-3), \
            f"Max diff with near-zero variance: {(out_before - out_after).abs().max()}"

    def test_fusion_with_large_running_mean(self, mock_pipeline):
        """Large running mean values to stress the bias computation."""
        torch.manual_seed(99)
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            if not isinstance(p.normalization, nn.Identity):
                bn = p.normalization
                bn.running_mean.data.fill_(1000.0)
        model.eval()

        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            out_before = model(x).clone()

        step = self._make_step(mock_pipeline, model)
        step.run()
        fused = mock_pipeline.cache["NormFusion.model"]
        fused.eval()

        with torch.no_grad():
            out_after = fused(x)

        assert torch.allclose(out_before, out_after, atol=1e-2), \
            f"Max diff with large running mean: {(out_before - out_after).abs().max()}"

    def test_fusion_with_negative_gamma(self, mock_pipeline):
        """
        Negative BN gamma flips the sign of the normalized output.
        Fusion must handle this correctly.
        """
        torch.manual_seed(55)
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            if not isinstance(p.normalization, nn.Identity):
                bn = p.normalization
                bn.weight.data.fill_(-1.0)
        model.eval()

        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            out_before = model(x).clone()

        step = self._make_step(mock_pipeline, model)
        step.run()
        fused = mock_pipeline.cache["NormFusion.model"]
        fused.eval()

        with torch.no_grad():
            out_after = fused(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), \
            f"Max diff with negative gamma: {(out_before - out_after).abs().max()}"

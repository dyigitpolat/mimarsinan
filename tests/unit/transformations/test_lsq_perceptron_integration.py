"""Integration tests for Phase C1: LSQ weight quantizer wired into a
``Perceptron`` so end-to-end QAT works via the standard trainer/optimizer.

These tests drive the perceptron-level contract:
1. A perceptron can host a ``weight_quantizer`` (``LSQQuantizer``) as a
   proper child module so the quantizer's ``log_scale`` is picked up by
   ``nn.Module.parameters()`` and therefore by any standard optimizer.
2. ``NormalizationAwarePerceptronQuantization`` now produces quantized
   weights via the perceptron's LSQQuantizer (STE + learnable scale)
   instead of the old stochastic mixing of FP and Q tensors.
3. After the transform runs, ``perceptron.layer.weight.data`` contains
   the hard-quantized values (integer multiples of the LSQ step) so
   downstream non-differentiable code (soft-core / hard-core mapping,
   simulation) continues to see the same baked-in snapshot it did under
   the legacy transform.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestPerceptronHostsLSQQuantizer:
    def test_default_is_none(self):
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        p = Perceptron(4, 8)
        assert getattr(p, "weight_quantizer", None) is None

    def test_setter_registers_as_child_module(self):
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        p = Perceptron(4, 8)
        q = LSQQuantizer(bits=8)
        p.set_weight_quantizer(q)

        assert p.weight_quantizer is q
        # The quantizer's log_scale must appear in the perceptron's own
        # parameters() sweep so BasicTrainer's optimizer picks it up.
        param_ids = {id(param) for param in p.parameters()}
        assert id(q.log_scale) in param_ids


class TestNormalizationAwarePerceptronQuantizationUsesLSQ:
    @pytest.fixture
    def perceptron_with_data(self):
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        torch.manual_seed(0)
        p = Perceptron(4, 8, normalization=nn.Identity())
        with torch.no_grad():
            p.layer.weight.data.mul_(0.3)
            p.layer.bias.data.mul_(0.3)
        p.set_activation_scale(1.0)
        return p

    def test_transform_installs_weight_quantizer(self, perceptron_with_data):
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )
        from mimarsinan.transformations.lsq_quantization import LSQQuantizer

        p = perceptron_with_data
        assert getattr(p, "weight_quantizer", None) is None
        NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0).transform(p)
        assert isinstance(p.weight_quantizer, LSQQuantizer)

    def test_transform_rate_one_produces_integer_grid(self, perceptron_with_data):
        """After a full transform at rate=1.0 the perceptron's effective
        weight must lie on the LSQ integer grid (step * integer)."""
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )
        from mimarsinan.transformations.perceptron_transformer import (
            PerceptronTransformer,
        )

        p = perceptron_with_data
        NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0).transform(p)

        step = torch.exp(p.weight_quantizer.log_scale.detach()).item()
        eff_w = PerceptronTransformer().get_effective_weight(p)
        eff_b = PerceptronTransformer().get_effective_bias(p)
        ratio_w = eff_w / step
        ratio_b = eff_b / step
        assert torch.allclose(ratio_w, torch.round(ratio_w), atol=1e-3)
        assert torch.allclose(ratio_b, torch.round(ratio_b), atol=1e-3)

    def test_transform_is_idempotent_at_rate_one(self, perceptron_with_data):
        """Applying rate=1.0 twice must produce the same weights -- the
        transform must not drift on repeated application."""
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        p = perceptron_with_data
        tfm = NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0)
        tfm.transform(p)
        w1 = p.layer.weight.data.clone()
        b1 = p.layer.bias.data.clone()
        tfm.transform(p)
        w2 = p.layer.weight.data
        b2 = p.layer.bias.data
        assert torch.allclose(w1, w2, atol=1e-4)
        assert torch.allclose(b1, b2, atol=1e-4)

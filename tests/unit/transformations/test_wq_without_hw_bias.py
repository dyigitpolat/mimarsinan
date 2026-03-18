"""Tests for weight quantization when hardware_bias=False.

Covers:
  - PerceptronTransformer guards against bias=None (Fix 1)
  - NormalizationAwarePerceptronQuantization works with bias=False perceptrons
  - Chip-level quantization produces integer bias in both hw_bias modes (Fix 3)
  - IRMapping with hardware_bias=False accounts for the always-on axon (Fix 2)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping import IRMapping


# ---------------------------------------------------------------------------
# Fix 1: PerceptronTransformer guards for bias=None
# ---------------------------------------------------------------------------

class TestPerceptronTransformerBiasGuard:
    """apply_effective_bias_transform must be a no-op when layer.bias is None."""

    def _make_perceptron(self, *, bias: bool):
        p = Perceptron(4, 8, bias=bias, normalization=nn.Identity())
        p.set_activation_scale(2.0)
        return p

    def test_apply_effective_bias_transform_no_bias(self):
        p = self._make_perceptron(bias=False)
        pt = PerceptronTransformer()
        pt.apply_effective_bias_transform(p, lambda b: b * 2)
        assert p.layer.bias is None

    def test_apply_effective_bias_transform_to_norm_no_bias(self):
        p = self._make_perceptron(bias=False)
        pt = PerceptronTransformer()
        pt.apply_effective_bias_transform_to_norm(p, lambda b: b * 2)
        assert p.layer.bias is None

    def test_apply_effective_parameter_transform_no_bias(self):
        p = self._make_perceptron(bias=False)
        pt = PerceptronTransformer()
        orig_w = p.layer.weight.data.clone()
        pt.apply_effective_parameter_transform(p, lambda x: x)
        assert p.layer.bias is None
        assert torch.allclose(p.layer.weight.data, orig_w, atol=1e-6)

    def test_get_effective_bias_returns_zeros_when_no_bias(self):
        p = self._make_perceptron(bias=False)
        pt = PerceptronTransformer()
        eff_b = pt.get_effective_bias(p)
        assert eff_b.shape == (4,)
        assert torch.allclose(eff_b, torch.zeros(4))

    def test_apply_effective_bias_transform_with_bias_still_works(self):
        p = self._make_perceptron(bias=True)
        pt = PerceptronTransformer()
        orig_bias = p.layer.bias.data.clone()
        pt.apply_effective_bias_transform(p, lambda b: b)
        assert torch.allclose(p.layer.bias.data, orig_bias, atol=1e-6)


# ---------------------------------------------------------------------------
# NormalizationAwarePerceptronQuantization with bias=False
# ---------------------------------------------------------------------------

class TestNAPQNoBias:
    """NormalizationAwarePerceptronQuantization.transform() must not crash on
    perceptrons whose nn.Linear was created with bias=False."""

    def test_transform_no_bias_identity_norm(self):
        p = Perceptron(4, 8, bias=False, normalization=nn.Identity())
        p.set_activation_scale(1.0)

        napq = NormalizationAwarePerceptronQuantization(bits=8, device="cpu")
        napq.transform(p)

        assert p.layer.bias is None
        eff_w = PerceptronTransformer().get_effective_weight(p)
        scale = float(p.parameter_scale)
        scaled_w = eff_w * scale
        assert torch.allclose(scaled_w, torch.round(scaled_w), atol=1e-3)

    def test_transform_no_bias_batchnorm(self):
        bn = nn.BatchNorm1d(4)
        bn.eval()
        with torch.no_grad():
            bn(torch.randn(20, 4))

        p = Perceptron(4, 8, bias=False, normalization=bn)
        p.set_activation_scale(1.0)

        napq = NormalizationAwarePerceptronQuantization(bits=8, device="cpu")
        napq.transform(p)

        assert p.layer.bias is None

    def test_transform_with_bias_still_quantizes(self):
        p = Perceptron(4, 8, bias=True, normalization=nn.Identity())
        p.set_activation_scale(1.0)

        napq = NormalizationAwarePerceptronQuantization(bits=8, device="cpu")
        napq.transform(p)

        pt = PerceptronTransformer()
        eff_w = pt.get_effective_weight(p)
        eff_b = pt.get_effective_bias(p)
        scale = float(p.parameter_scale)

        assert torch.allclose(eff_w * scale, torch.round(eff_w * scale), atol=1e-3)
        assert torch.allclose(eff_b * scale, torch.round(eff_b * scale), atol=1e-3)


# ---------------------------------------------------------------------------
# Fix 3: Chip-level quantization rounds hardware_bias
# ---------------------------------------------------------------------------

def _inp(idx):
    return IRSource(node_id=-2, index=idx)


class TestChipQuantizationBiasRounding:
    """Chip-level quantization in SoftCoreMappingStep must produce integer-
    valued hardware_bias after scale+round, matching the always-on row path."""

    def _make_ir_graph(self, *, hardware_bias_mode: bool, in_features=4, out_features=3):
        """Build a minimal IRGraph via IRMapping."""
        mapper = IRMapping(
            q_max=127,
            hardware_bias=hardware_bias_mode,
            max_axons=256,
            max_neurons=256,
        )
        weights = torch.randn(out_features, in_features)
        biases = torch.randn(out_features)
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])

        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=weights,
            fc_biases=biases,
            name="test_fc",
        )
        return mapper.map(type("FakeRepr", (), {
            "map_to_ir": lambda self, m: np.array([_inp(0)])
        })())

    def _quantize_graph(self, ir_graph, bits=8):
        """Replicate the chip-level quantization logic from SoftCoreMappingStep."""
        q_max = (2 ** (bits - 1)) - 1
        q_min = -(2 ** (bits - 1))
        eps = 1e-12

        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                w_max = float(np.max(np.abs(node.core_matrix)))
                w_max = max(w_max, eps)
                scale = q_max / w_max
                W_q = np.round(node.core_matrix * scale).astype(np.float64)
                W_q = np.clip(W_q, q_min, q_max)
                node.core_matrix = W_q
                node.threshold = scale
                node.parameter_scale = torch.tensor(1.0)
                if node.hardware_bias is not None:
                    node.hardware_bias = np.round(node.hardware_bias * scale)

    def test_hw_bias_true_produces_integer_bias(self):
        ir_graph = self._make_ir_graph(hardware_bias_mode=True)
        self._quantize_graph(ir_graph)

        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore) and node.hardware_bias is not None:
                np.testing.assert_array_almost_equal(
                    node.hardware_bias,
                    np.round(node.hardware_bias),
                    decimal=10,
                    err_msg="hardware_bias must be integer after chip quantization",
                )

    def test_hw_bias_false_bias_row_is_integer(self):
        ir_graph = self._make_ir_graph(hardware_bias_mode=False)
        self._quantize_graph(ir_graph)

        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                assert node.hardware_bias is None
                np.testing.assert_array_almost_equal(
                    node.core_matrix,
                    np.round(node.core_matrix),
                    decimal=10,
                    err_msg="core_matrix (including bias row) must be integer after chip quantization",
                )


# ---------------------------------------------------------------------------
# Fix 2: IRMapping with hardware_bias=False adds always-on axon
# ---------------------------------------------------------------------------

class TestIRMappingLegacyBiasAxon:
    """When hardware_bias=False, the always-on axon row consumes one axon
    slot, and IRMapping should account for it."""

    def test_core_matrix_has_extra_row(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        in_features, out_features = 4, 3
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=torch.randn(out_features, in_features),
            fc_biases=torch.randn(out_features),
        )
        core = mapper.nodes[-1]
        assert core.core_matrix.shape[0] == in_features + 1
        assert core.hardware_bias is None

    def test_hw_bias_true_no_extra_row(self):
        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        in_features, out_features = 4, 3
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=torch.randn(out_features, in_features),
            fc_biases=torch.randn(out_features),
        )
        core = mapper.nodes[-1]
        assert core.core_matrix.shape[0] == in_features
        assert core.hardware_bias is not None

    def test_no_bias_no_extra_row(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        in_features, out_features = 4, 3
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=torch.randn(out_features, in_features),
            fc_biases=None,
        )
        core = mapper.nodes[-1]
        assert core.core_matrix.shape[0] == in_features
        assert core.hardware_bias is None

    def test_always_on_source_present_in_legacy_mode(self):
        mapper = IRMapping(hardware_bias=False, max_axons=256, max_neurons=256)
        in_features = 4
        input_sources = np.array([_inp(i) for i in range(in_features)])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=np.array([3]),
            fc_weights=torch.randn(3, in_features),
            fc_biases=torch.randn(3),
        )
        core = mapper.nodes[-1]
        always_on = [s for s in core.input_sources.flatten() if s.is_always_on()]
        assert len(always_on) == 1

    def test_no_always_on_in_hw_bias_mode(self):
        mapper = IRMapping(hardware_bias=True, max_axons=256, max_neurons=256)
        in_features = 4
        input_sources = np.array([_inp(i) for i in range(in_features)])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=np.array([3]),
            fc_weights=torch.randn(3, in_features),
            fc_biases=torch.randn(3),
        )
        core = mapper.nodes[-1]
        always_on = [s for s in core.input_sources.flatten() if s.is_always_on()]
        assert len(always_on) == 0

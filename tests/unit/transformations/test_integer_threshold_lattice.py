"""[nevresim parity] The chip threshold register is INTEGER (threshold_t=int):
the trained NAPQ grid scale, the IR threshold, and the emit must all live on
the integer lattice — a fractional theta deploys a different function than
the SSOT simulates (root cause of the 2026-08-09 nevresim count divergence).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import (
    quantize_ir_graph,
    verify_ir_graph_quantized,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

BITS = 8


def _perceptron(seed=0, bias=True):
    torch.manual_seed(seed)
    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    with torch.no_grad():
        p.layer.weight.data.mul_(0.317)
        if p.layer.bias is not None:
            if bias:
                p.layer.bias.data.mul_(0.213)
            else:
                p.layer.bias.data.zero_()
    return p


def _integral(x, tol=1e-6):
    v = float(x.item() if hasattr(x, "item") else x)
    return abs(v - round(v)) <= tol


class TestNapqIntegerScale:
    def test_single_scale_parameter_scale_is_integral(self):
        p = _perceptron(seed=1)
        NormalizationAwarePerceptronQuantization(
            BITS, "cpu", rate=1.0, two_scale=False,
        ).transform(p)
        assert _integral(p.parameter_scale), float(p.parameter_scale)
        assert float(p.parameter_scale) >= 1.0

    def test_two_scale_weight_scale_is_integral_and_ratio_snapped(self):
        p = _perceptron(seed=2)
        NormalizationAwarePerceptronQuantization(
            BITS, "cpu", rate=1.0, two_scale=True,
        ).transform(p)
        ws = float(p.parameter_scale)
        assert _integral(ws), ws
        bs = getattr(p, "bias_scale", None)
        if bs is not None and float(bs) > 0:
            ratio = ws / float(bs)
            assert abs(ratio - round(ratio)) < 1e-3 * round(ratio)


def _core_ir(matrix, parameter_scale):
    core = NeuralCore(
        id=0,
        name="c0",
        input_sources=np.asarray(
            [IRSource(-2, i) for i in range(matrix.shape[1])], dtype=object,
        ),
        core_matrix=matrix.copy(),
        threshold=1.0,
        parameter_scale=torch.tensor(float(parameter_scale)),
    )
    ir = IRGraph(
        nodes=[core],
        output_sources=np.asarray(
            [IRSource(0, i) for i in range(matrix.shape[0])], dtype=object,
        ),
    )
    return ir, core


class TestQuantizeIrGraphIntegerTheta:
    def test_fallback_scale_rounds_theta_to_integer(self):
        rng = np.random.default_rng(3)
        ir, core = _core_ir(rng.uniform(-0.7, 0.7, (4, 6)), parameter_scale=0.0)
        quantize_ir_graph(ir, BITS, weight_quantization=True)
        assert _integral(core.threshold), core.threshold
        verify_ir_graph_quantized(ir, BITS)

    def test_integral_trained_scale_passes_through(self):
        rng = np.random.default_rng(4)
        ir, core = _core_ir(rng.uniform(-1, 1, (4, 6)), parameter_scale=50.0)
        quantize_ir_graph(ir, BITS, weight_quantization=True)
        assert float(core.threshold) == 50.0

    def test_fractional_trained_scale_fails_loud(self):
        rng = np.random.default_rng(5)
        ir, _ = _core_ir(rng.uniform(-1, 1, (4, 6)), parameter_scale=49.6521)
        with pytest.raises(ValueError, match="integer lattice"):
            quantize_ir_graph(ir, BITS, weight_quantization=True)

    def test_verifier_flags_fractional_threshold(self):
        rng = np.random.default_rng(6)
        ir, core = _core_ir(rng.uniform(-0.7, 0.7, (4, 6)), parameter_scale=0.0)
        quantize_ir_graph(ir, BITS, weight_quantization=True)
        core.threshold = 4.5
        with pytest.raises(AssertionError, match="threshold"):
            verify_ir_graph_quantized(ir, BITS)


class TestEmitGuard:
    def test_int_threshold_type_rejects_fractional_theta(self):
        from mimarsinan.code_generation.cpp_chip_model import ChipModel

        with pytest.raises(ValueError, match="integer"):
            ChipModel.serialize_threshold(int, 4.5)

    def test_int_threshold_type_passes_integral_theta(self):
        from mimarsinan.code_generation.cpp_chip_model import ChipModel

        assert ChipModel.serialize_threshold(int, 5.0) == "5"
        assert ChipModel.serialize_threshold(float, 4.5) == "4.5"

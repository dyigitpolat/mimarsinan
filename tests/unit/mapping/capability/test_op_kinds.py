"""Op kinds of IR nodes: the payload and its operand structure decide, an unknown payload refuses by name."""

from __future__ import annotations

import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds, op_kind, operand_spans, stationary_operand
from mimarsinan.mapping.capability.requirements import REQUIREMENTS, UnknownOpKindError
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import BatchNormAffine

from .ir_fixtures import INPUT, Contraction, contraction_graph, core, lane_graph, op, outputs


def _unary(module, width=4):
    return op(1, module, outputs(0, width), input_shape=(width,), output_shape=(width,))


def _binary(module, width=4):
    return op(2, module, outputs(0, width) + outputs(1, width), input_shapes=((width,), (width,)),
              output_shape=(width,))


@pytest.mark.parametrize("module, kind", [
    (nn.ReLU(), "clamp"), (nn.ReLU6(), "clamp"), (nn.Hardtanh(), "clamp"), (nn.Hardsigmoid(), "clamp"),
    (nn.GELU(), "smooth"), (nn.SiLU(), "smooth"), (nn.Sigmoid(), "smooth"), (nn.Tanh(), "smooth"),
    (nn.Hardswish(), "hswish"), (nn.LayerNorm(4), "norm"), (nn.Softmax(dim=-1), "softmax"),
    (nn.AdaptiveAvgPool2d(1), "pool"), (nn.AvgPool2d(2), "pool"), (nn.MaxPool2d(2), "maxp"),
    (nn.BatchNorm1d(4), "affine"), (nn.ConvTranspose2d(4, 4, 2), "mms"),
    (nn.Conv2d(4, 4, 3, groups=4), "mmg"), (nn.Conv2d(4, 4, 3, groups=2), "mmg"),
    (nn.Conv2d(4, 4, 3), "mms"), (nn.Linear(4, 4), "mms"),
    (nn.Upsample(scale_factor=2, mode="nearest"), "wire+"), (nn.PixelShuffle(2), "wire+"),
    (nn.Upsample(scale_factor=2, mode="bilinear"), "mms"),
    (nn.MultiheadAttention(4, 1), "leaf"), (nn.LSTM(4, 4), "leaf"), (nn.GRU(4, 4), "leaf"),
    (nn.TransformerEncoderLayer(4, 1), "leaf"),
])
def test_module_payloads(module, kind):
    assert op_kind(_unary(module)) == kind


@pytest.mark.parametrize("fn, kind", [
    (F.relu, "clamp"), (torch.relu, "clamp"), (torch.clamp, "clamp"), (F.relu6, "clamp"),
    (F.gelu, "smooth"), (torch.sigmoid, "smooth"), (torch.tanh, "smooth"), (torch.exp, "smooth"),
    (F.hardswish, "hswish"), (F.softmax, "softmax"), (torch.softmax, "softmax"), (F.layer_norm, "norm"),
    (torch.mean, "pool"), (F.adaptive_avg_pool2d, "pool"), (F.max_pool2d, "maxp"),
    (operator.getitem, "wire+"), (torch.cat, "wire+"), (torch.flatten, "wire+"),
    (F.scaled_dot_product_attention, "leaf"),
])
def test_function_payloads(fn, kind):
    assert op_kind(_unary(ComputeAdapter(fn))) == kind


@pytest.mark.parametrize("fn, two_activations, one_activation", [
    (operator.add, "add", "affine"), (torch.add, "add", "affine"), (operator.sub, "add", "affine"),
    (operator.mul, "prod", "affine"), (torch.mul, "prod", "affine"),
    (operator.truediv, "data_division", "affine"),
    (torch.matmul, "dmm", "mms"), (operator.matmul, "dmm", "mms"), (torch.bmm, "dmm", "mms"),
])
def test_operand_structure_decides_binary_kinds(fn, two_activations, one_activation):
    assert op_kind(_binary(ComputeAdapter(fn))) == two_activations
    bound = ComputeAdapter(fn, bound_tensors=(torch.ones(4),))
    assert op_kind(_unary(bound)) == one_activation


def test_the_lowered_batchnorm_affine_is_an_affine():
    affine = BatchNormAffine(torch.ones(4), torch.zeros(4))
    assert op_kind(_unary(affine)) == "affine"


def test_neural_cores_are_static_matrix_products():
    assert op_kind(core(0, outputs(INPUT, 4), 4)) == "mm"


def test_a_contraction_of_two_activations_is_a_dynamic_product():
    assert op_kind(contraction_graph().nodes[2]) == "dmm"
    with_parameter = contraction_graph(roles=("activation", "parameter")).nodes[2]
    assert op_kind(with_parameter) == "mms"


def test_every_kind_the_classifier_emits_has_a_table_row():
    kinds = set(ir_op_kinds(lane_graph()).values()) | set(ir_op_kinds(contraction_graph()).values())
    assert kinds <= set(REQUIREMENTS)


def test_an_unknown_payload_refuses_by_name_naming_the_operator_and_node():
    class Frobnicate(nn.Module):
        def forward(self, x):
            return x

    node = _unary(Frobnicate())
    with pytest.raises(UnknownOpKindError) as caught:
        op_kind(node)
    assert isinstance(caught.value, ToolchainRefusal)
    assert "Frobnicate" in str(caught.value) and node.name in str(caught.value)


def test_operand_spans_follow_the_executed_input_layout():
    product = contraction_graph().nodes[2]
    assert operand_spans(product) == ((0, 8, (2, 4)), (8, 12, (4, 3)))
    assert stationary_operand(product) == (8, 12, (4, 3))
    assert operand_spans(_unary(nn.ReLU())) == ((0, 4, (4,)),)

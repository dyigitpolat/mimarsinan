"""HCM/SCM equivalence on the quantized-weight path.

The existing ``test_hardcore_ttfs_equivalence.py`` suite forces ``q_max=1``,
``threshold=1.0``, ``parameter_scale=1.0`` so weights stay in continuous
float and thresholds stay at unity.  Real deployment uses ``q_max=127``,
int8 weights, and per-core thresholds set to the quantization scale
(``q_max / w_max``).  The latter exposes precision differences in how SCM
and HCM materialise their per-core threshold and weight tensors.

This file deliberately exercises that path with realistic per-core scales.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir_latency import IRLatency
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.mappers.structural import InputMapper, EinopsRearrangeMapper
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow


CORES_CONFIG = [{"count": 512, "max_axons": 256, "max_neurons": 256}]
SIM_LENGTH = 32
Q_MAX = 127  # int8


def _quantize_ir_inplace(ir_graph) -> None:
    """Mirror ``SoftCoreMappingStep``'s weight-quantization branch on the IR.

    Sets ``threshold = scale`` and stores int8 ``core_matrix`` for every
    NeuralCore (and bank).  This reproduces the exact precision contract
    that real deployment uses for ttfs_quantized.
    """
    q_min = -(Q_MAX + 1)
    eps = 1e-12
    bank_scales: dict[int, float] = {}
    for bank_id, bank in ir_graph.weight_banks.items():
        w_max = max(float(np.max(np.abs(bank.core_matrix))), eps)
        scale = Q_MAX / w_max
        W_q = np.clip(np.round(bank.core_matrix * scale), q_min, Q_MAX).astype(np.int8)
        bank.core_matrix = W_q
        bank.parameter_scale = torch.tensor(1.0)
        bank_scales[bank_id] = scale

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.has_weight_bank():
            scale = bank_scales[node.weight_bank_id]
            node.threshold = scale
            node.parameter_scale = torch.tensor(1.0)
        else:
            w_max = max(float(np.max(np.abs(node.core_matrix))), eps)
            scale = Q_MAX / w_max
            W_q = np.clip(np.round(node.core_matrix * scale), q_min, Q_MAX).astype(np.int8)
            node.core_matrix = W_q
            node.threshold = scale
            node.parameter_scale = torch.tensor(1.0)


def _build_quantized_flows(mapper_repr, input_shape):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(
        q_max=Q_MAX, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    )
    ir_graph = ir_mapping.map(mapper_repr)
    _quantize_ir_inplace(ir_graph)
    IRLatency(ir_graph).calculate()

    soft_flow = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    ).eval()

    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph, cores_config=CORES_CONFIG,
    )
    hard_flow = SpikingHybridCoreFlow(
        input_shape, hybrid_mapping, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    ).eval()

    return soft_flow, hard_flow, ir_graph


class TestQuantizedHCMSCMEquivalence:
    """SCM and HCM must agree bit-for-bit on the analytical quantized path."""

    def test_two_layer_mlp_quantized(self):
        torch.manual_seed(42)
        input_shape = (1, 4, 4)

        p1 = Perceptron(8, 16, normalization=nn.Identity(), base_activation_name="ReLU")
        p2 = Perceptron(4, 8, normalization=nn.Identity(), base_activation_name="ReLU")

        flat = EinopsRearrangeMapper(InputMapper(input_shape), "... c h w -> ... (c h w)")
        m1 = PerceptronMapper(flat, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        soft_flow, hard_flow, ir_graph = _build_quantized_flows(repr_, input_shape)

        # Confirm we ARE on the quantized non-unity-threshold path.
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                assert node.core_matrix is None or node.core_matrix.dtype == np.int8
                assert float(node.threshold) > 1.5, (
                    f"threshold {node.threshold} suggests we're still on the unity path"
                )

        x = torch.rand(8, *input_shape)
        with torch.no_grad():
            soft_out = soft_flow(x).to(torch.float64)
            hard_out = (hard_flow(x) / SIM_LENGTH).to(torch.float64)

        max_diff = (soft_out - hard_out).abs().max().item()
        assert max_diff < 1e-9, (
            f"Quantized SCM vs HCM max diff {max_diff:.3e}.\n"
            f"soft sample: {soft_out[0]}\nhard sample: {hard_out[0]}"
        )

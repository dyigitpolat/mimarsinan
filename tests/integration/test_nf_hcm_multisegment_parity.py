"""Segment-aware torch NF forward matches HCM across a non-linear ComputeOp.

The original requirement: torch-side (NF) spike behavior must match HCM at segment
boundaries — decode the segment output to a rate, run the host ComputeOp on the
rate, re-encode for the next segment. `chip_aligned_segment_forward` is genuinely
segment-aware (it runs each mid-graph ComputeOp once on the decoded rate, not
per-cycle on spikes) and matches HCM even with a non-linear ComputeOp (LayerNorm).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation


class _TwoSegLayerNorm(nn.Module):
    """input -> [Linear+ReLU] -> LayerNorm (non-linear ComputeOp) -> [Linear+ReLU]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.act1 = nn.ReLU()
        self.ln = nn.LayerNorm(6)
        self.fc2 = nn.Linear(6, 4)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.ln(x)
        return self.act2(self.fc2(x))


def _build(T):
    torch.manual_seed(0)
    m = _TwoSegLayerNorm().eval()
    flow = convert_torch_model(m, input_shape=(8,), num_classes=4)
    repr_ = flow.get_mapper_repr()
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64).map(repr_)
    # Sanity: the LayerNorm survives as a mid-graph ComputeOp.
    assert any(isinstance(n, ComputeOp) and "LayerNorm" in n.op_type for n in ir.nodes)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 50}],
        allow_neuron_splitting=True,
    )
    hcm = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return flow, hcm


def test_segment_aware_nf_matches_hcm_across_layernorm():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    T = 8
    flow, hcm = _build(T)
    x = torch.rand(4, 8)
    with torch.no_grad():
        nf = chip_aligned_segment_forward(flow, x, T)
        hc = hcm(x) / float(T)
    torch.testing.assert_close(nf, hc, atol=1e-6, rtol=0.0)

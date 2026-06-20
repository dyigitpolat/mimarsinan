"""Cascaded ttfs_cycle_based NF == HCM across a non-linear ComputeOp boundary.

The post-LayerNorm core is on-chip (LayerNorm is transparent for encoding
marking), so HCM TTFS-encodes the decoded boundary value into the segment.
The NF driver must mirror that encode at non-encoding segment entries —
including the negative-value shift, which both sides apply before the clamp.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.support.neg_shift_bias import (
    apply_negative_value_shifts,
    calibration_forward_for_mode,
    transfer_negative_shifts_to_ir,
    propagate_negative_shifts_to_hybrid,
)


class _TwoSegLayerNorm(nn.Module):
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


def _build_cascade(T, *, shift: bool, calib_x=None):
    torch.manual_seed(0)
    flow = convert_torch_model(_TwoSegLayerNorm().eval(), (8,), 4, device="cpu")
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=T,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    flow = flow.double()
    if shift:
        apply_negative_value_shifts(
            flow, calib_x, T,
            forward_fn=calibration_forward_for_mode("ttfs_cycle_based"),
        )
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="TTFS", max_axons=64, max_neurons=64).map(repr_)
    if shift:
        transfer_negative_shifts_to_ir(flow, ir)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 50}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=True),
    )
    if shift:
        table = propagate_negative_shifts_to_hybrid(ir, hybrid)
        assert table, "shift must reach hybrid node_output_shifts"
    hcm = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
        spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
    )
    return flow, hcm


def test_cascade_nf_matches_hcm_across_layernorm():
    T = 8
    fwd = calibration_forward_for_mode("ttfs_cycle_based")
    flow, hcm = _build_cascade(T, shift=False)
    x = torch.rand(4, 8, dtype=torch.float64)
    with torch.no_grad():
        nf = fwd(flow, x, T)
        hc = hcm(x).double() / T
    torch.testing.assert_close(nf, hc, atol=1e-6, rtol=0.0)


def test_cascade_negative_shift_nf_hcm_consistent_and_recovers():
    T = 8
    torch.manual_seed(7)
    x = torch.rand(4, 8, dtype=torch.float64)
    calib = torch.cat([x, torch.rand(12, 8, dtype=torch.float64)], dim=0)
    fwd = calibration_forward_for_mode("ttfs_cycle_based")

    flow_s, hcm_s = _build_cascade(T, shift=True, calib_x=calib)
    flow_n, hcm_n = _build_cascade(T, shift=False)
    with torch.no_grad():
        nf_s, hc_s = fwd(flow_s, x, T), hcm_s(x).double() / T
        nf_n, hc_n = fwd(flow_n, x, T), hcm_n(x).double() / T

    torch.testing.assert_close(nf_s, hc_s, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(nf_n, hc_n, atol=1e-6, rtol=0.0)
    # The shift recovers the LayerNorm negatives the encode clamp would drop.
    assert not torch.allclose(nf_s, nf_n, atol=1e-6)

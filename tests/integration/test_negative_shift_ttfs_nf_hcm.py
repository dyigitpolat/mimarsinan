"""Negative-value shift for analytical TTFS: NF == HCM, shift is value-preserving.

The analytical TTFS contract path consumes segment inputs linearly, so the
shift + baked bias is an exact identity there (``W(x+s) + (B − W·s) = Wx + B``).
Its purpose is moving the boundary into the encodable ``[0, 1]`` domain for the
spike-train backends (SANA-FE / nevresim TTFS encode clamps ``clip(rate, 0, 1)``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
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


def _build_with_ttfs(mode, T, *, shift: bool, calib_x=None):
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
        if calib_x is None:
            calib_x = torch.rand(16, 8, dtype=torch.float64)
        apply_negative_value_shifts(
            flow, calib_x, T,
            forward_fn=calibration_forward_for_mode(mode),
        )
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="TTFS", max_axons=64, max_neurons=64).map(repr_)
    assert any(isinstance(n, ComputeOp) and "LayerNorm" in n.op_type for n in ir.nodes)
    if shift:
        transfer_negative_shifts_to_ir(flow, ir)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 50}],
        allow_neuron_splitting=True,
    )
    if shift:
        table = propagate_negative_shifts_to_hybrid(ir, hybrid)
        assert table, "shift must propagate to the hybrid mapping"
    hcm = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
        spiking_mode=mode,
    )
    return flow, hcm


def test_analytical_ttfs_nf_hcm_parity_with_and_without_shift():
    """ttfs_quantized: the supported analytical deployment mode (continuous
    ``ttfs`` NF uses the staircase surrogate and is a known non-goal for exact
    SCM parity — see the SoftCoreMappingStep warning)."""
    mode = "ttfs_quantized"
    T = 8
    x = torch.rand(4, 8, dtype=torch.float64)
    fwd = calibration_forward_for_mode(mode)

    flow_s, hcm_s = _build_with_ttfs(mode, T, shift=True)
    flow_n, hcm_n = _build_with_ttfs(mode, T, shift=False)
    with torch.no_grad():
        nf_s, hc_s = fwd(flow_s, x, T), hcm_s(x).double() / T
        nf_n, hc_n = fwd(flow_n, x, T), hcm_n(x).double() / T

    torch.testing.assert_close(nf_s, hc_s, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(nf_n, hc_n, atol=1e-6, rtol=0.0)
    # Linear boundary consumption: the shift + baked bias is an exact identity.
    torch.testing.assert_close(nf_s, nf_n, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized"])
def test_shift_moves_boundary_into_encodable_domain(mode):
    """The shifted boundary value (raw op min + shift) is non-negative — the
    property protecting the spike-train backends' ``clip(rate, 0, 1)``. The
    recorder captures the *raw* op output (pre-shift) by design."""
    T = 8
    torch.manual_seed(7)
    x = torch.rand(16, 8, dtype=torch.float64)
    fwd = calibration_forward_for_mode(mode)

    flow, _ = _build_with_ttfs(mode, T, shift=True, calib_x=x)
    recorder: dict = {}
    with torch.no_grad():
        fwd(flow, x, T, compute_min_recorder=recorder)
    shifted = {
        n: v for n, v in recorder.items()
        if getattr(n, "_negative_shift", None) is not None
    }
    assert shifted, "the LayerNorm boundary must have been shifted"
    saw_negative_raw = False
    for node, raw_min in shifted.items():
        saw_negative_raw |= bool(raw_min.min() < 0)
        boundary_min = raw_min + torch.as_tensor(node._negative_shift, dtype=raw_min.dtype)
        assert float(boundary_min.min()) >= -1e-9, (
            f"{node} boundary still negative after shift"
        )
    assert saw_negative_raw, (
        "test is vacuous: the unshifted boundary never goes negative"
    )

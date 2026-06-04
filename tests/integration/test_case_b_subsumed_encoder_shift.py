"""Case B: subsumed encoding perceptron fed by an unbounded negative ComputeOp.

The subsumed encoder runs cycle-accurately in HCM on uniform-encoded input
trains, so its input rate is clamped to [0, 1] (``_gather_op_input_train``).
A bare signed Linear feeding it loses its negatives there — silently, unless
the negative-value shift lifts the boundary (producer shift + encoder bias bake).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.mapping.support.neg_shift_bias import (
    apply_negative_value_shifts,
    transfer_negative_shifts_to_ir,
    propagate_negative_shifts_to_hybrid,
)
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


class _BareLinearThenEncoder(nn.Module):
    """input (2,4) -> bare Linear (signed ComputeOp) -> flatten -> [encoder] -> [perceptron].

    The flatten between the bare Linear and the encoder blocks Linear-Linear
    fusion at conversion (the mlp-mixer token-mixer shape), so the signed
    Linear survives as a host ComputeOp feeding the subsumed encoder.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.fc1 = nn.Linear(8, 4)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)
        self.a2 = nn.ReLU()

    def forward(self, x):
        x = self.lin(x)
        x = x.flatten(1)
        return self.a2(self.fc2(self.a1(self.fc1(x))))


def _build(T, *, shift: bool, calib_x=None):
    torch.manual_seed(0)
    flow = convert_torch_model(_BareLinearThenEncoder().eval(), (2, 4), 3)
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    encoder = flow.get_perceptrons()[0]
    assert getattr(encoder, "is_encoding_layer", False), (
        "the perceptron after the bare Linear must be an encoding layer"
    )
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    if shift:
        if calib_x is None:
            calib_x = torch.rand(16, 2, 4)
        shifts = apply_negative_value_shifts(flow, calib_x, T)
        assert shifts, "the bare Linear boundary must derive a shift"
        assert getattr(encoder, "_neg_shift_baked", False), (
            "the subsumed encoder's bias must be baked"
        )
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64).map(repr_)
    assert any(isinstance(n, ComputeOp) and "Linear" in n.op_type for n in ir.nodes)
    if shift:
        transfer_negative_shifts_to_ir(flow, ir)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 50}],
        allow_neuron_splitting=True,
    )
    if shift:
        table = propagate_negative_shifts_to_hybrid(ir, hybrid)
        assert table, "shift must reach hybrid node_output_shifts"
    hcm = SpikingHybridCoreFlow(
        (2, 4), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    return flow, hcm


def test_case_b_shift_restores_nf_hcm_parity_and_recovers_negatives():
    T = 8
    torch.manual_seed(3)
    x = torch.rand(4, 2, 4)
    calib = torch.cat([x, torch.rand(12, 2, 4)], dim=0)

    flow_s, hcm_s = _build(T, shift=True, calib_x=calib)
    flow_n, hcm_n = _build(T, shift=False)
    with torch.no_grad():
        nf_s, hc_s = chip_aligned_segment_forward(flow_s, x, T), hcm_s(x) / T
        nf_n, hc_n = chip_aligned_segment_forward(flow_n, x, T), hcm_n(x) / T

    # With the shift, the subsumed encoder's input boundary is lossless: NF == HCM.
    torch.testing.assert_close(nf_s, hc_s, atol=1e-6, rtol=0.0)
    # The clamp would have silenced negatives: shifted HCM differs from unshifted.
    assert not torch.allclose(hc_s, hc_n, atol=1e-6)
    # Non-vacuous: the bare Linear genuinely produced negatives on this input.
    recorder: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(flow_n, x, T, compute_min_recorder=recorder)
    assert min(float(v.min()) for v in recorder.values()) < 0.0


def test_case_b_unshifted_nf_mirrors_hcm_clamp():
    """Without the shift, NF mirrors HCM's encoder-input clamp (the loss is
    consistent across both — and announced by the boundary warning), so
    NF == HCM holds either way; only the *information* differs (see the
    recovery assertion in the shifted test)."""
    T = 8
    torch.manual_seed(3)
    x = torch.rand(4, 2, 4)
    flow_n, hcm_n = _build(T, shift=False)
    with torch.no_grad():
        nf_n, hc_n = chip_aligned_segment_forward(flow_n, x, T), hcm_n(x) / T
    torch.testing.assert_close(nf_n, hc_n, atol=1e-6, rtol=0.0)

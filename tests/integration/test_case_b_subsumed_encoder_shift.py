"""Case B: subsumed encoding perceptron fed by an unbounded negative ComputeOp.

Under the wire contract the encoder's boundary train is emitted from its host
*value* (uniform train of ``clamp(value / theta)``), and the host value path
gathers a shifted producer LIFTED (``compute_input_state_with_shifts``) so the
encoder's baked bias ``B' = B − W·s`` stays value-preserving. The host→host
handoff is therefore lossless with or without the negative-value shift: the
shift must be a no-op on results, and NF == HCM must hold in both flows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.mapping.support.bias_compensation import (
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
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=True),
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


def test_case_b_shift_is_value_preserving_and_nf_hcm_parity_holds():
    T = 8
    torch.manual_seed(3)
    x = torch.rand(4, 2, 4)
    calib = torch.cat([x, torch.rand(12, 2, 4)], dim=0)

    flow_s, hcm_s = _build(T, shift=True, calib_x=calib)
    flow_n, hcm_n = _build(T, shift=False)
    with torch.no_grad():
        nf_s, hc_s = chip_aligned_segment_forward(flow_s, x, T), hcm_s(x) / T
        nf_n, hc_n = chip_aligned_segment_forward(flow_n, x, T), hcm_n(x) / T

    # The subsumed encoder's host boundary is lossless: NF == HCM, both flows.
    torch.testing.assert_close(nf_s, hc_s, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(nf_n, hc_n, atol=1e-6, rtol=0.0)
    # The shift is value-preserving (B' = B − W·s against lifted inputs), and
    # the host value path preserves negatives by construction, so the shifted
    # flow must equal the unshifted one.
    torch.testing.assert_close(hc_s, hc_n, atol=1e-6, rtol=0.0)
    # Non-vacuous: the bare Linear genuinely produced negatives on this input.
    recorder: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(flow_n, x, T, compute_min_recorder=recorder)
    assert min(float(v.min()) for v in recorder.values()) < 0.0

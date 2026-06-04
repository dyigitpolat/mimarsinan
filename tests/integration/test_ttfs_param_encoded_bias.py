"""Torch-side parameter-encoded (always-on axon) bias must match HCM for TTFS.

Mode B (``hardware_bias=False``) delivers the bias as the weight on an always-on
axon that, for single-spike TTFS, fires once at the core's **local window start**
(global cycle ``== core.latency``). Its ramped contribution to the membrane is
``bias·(t_local+1)`` — identical to the on-chip register's constant-per-cycle add.
So the differentiable torch forward needs a single bias implementation, and mode A
(``on_chip``) and mode B (``param_encoded``) must produce identical spike timing.

Two structural cases must both hold, for subsume and offload encoding placement:
  * subsume — the encoding layer is a host ComputeOp (bias folded analytically,
    bias-mode-agnostic); the torch entry uses the analytical charge in both modes.
  * offload — the encoding layer is a real cascade core (possibly at latency ≥ 1),
    exercising the always-on-at-local-window delivery for a deeper core.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _build(T, *, hw_bias, placement="subsume", d_in=8, d_h=8, d_out=6):
    torch.manual_seed(0)
    p1 = Perceptron(d_h, d_in, normalization=nn.Identity(), base_activation_name="ReLU")
    p2 = Perceptron(d_out, d_h, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(InputMapper((d_in,)), p1), p2))
    mark_encoding_layers(repr_, placement=placement)
    ir = IRMapping(q_max=127.0, firing_mode="TTFS", max_axons=64, max_neurons=64,
                   hardware_bias=hw_bias).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 8}])
    flow = SpikingHybridCoreFlow((d_in,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
        spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded").double()
    bias_mode = "on_chip" if hw_bias else "param_encoded"
    for p in (p1, p2):
        p.double()
        p.set_activation(TTFSActivation(T=T, activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale, bias=p.layer.bias, thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False), bias_mode=bias_mode))
    drv = SegmentForwardDriver(repr_, T, TtfsSegmentPolicy())
    return drv, flow


@pytest.mark.parametrize("placement", ["subsume", "offload"])
@pytest.mark.parametrize("hw_bias", [True, False])
def test_torch_ttfs_nf_matches_hcm_both_bias_modes(hw_bias, placement):
    """torch TTFS-NF == HCM for BOTH on-chip (mode A) and parameter-encoded (mode B),
    across subsume and offload encoding placement."""
    T = 8
    drv, flow = _build(T, hw_bias=hw_bias, placement=placement)
    x = torch.rand(4, 8, dtype=torch.float64)
    with torch.no_grad():
        nf = drv(x.clone())
        hc = flow(x.clone()).double() / T
    torch.testing.assert_close(nf, hc, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("placement", ["subsume", "offload"])
def test_param_encoded_coincides_with_on_chip(placement):
    """Both deliveries give cumulative membrane ``bias·(t_local+1)`` once the always-on
    bias fires at the core's local window start, so mode A ≡ mode B in the torch forward."""
    T = 8
    x = torch.rand(4, 8, dtype=torch.float64)
    drv_b, _ = _build(T, hw_bias=False, placement=placement)
    drv_a, _ = _build(T, hw_bias=True, placement=placement)
    with torch.no_grad():
        nf_b = drv_b(x.clone()); nf_a = drv_a(x.clone())
    torch.testing.assert_close(nf_a, nf_b, atol=1e-9, rtol=0.0)

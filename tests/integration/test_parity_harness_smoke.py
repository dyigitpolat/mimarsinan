"""Smoke test for parity harness (inline to avoid import path issues)."""

import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
import torch


def test_mini_hybrid_parity_hcm():
    inp = InputMapper((16,))
    p = Perceptron(8, 16, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(inp, p))
    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=64,
        max_neurons=64,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 8}],
    )
    flow = SpikingHybridCoreFlow(
        (16,),
        hybrid,
        simulation_length=4,
        preprocessor=torch.nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<",
        spiking_mode="lif",
    )
    with torch.no_grad():
        _ = flow(torch.randn(1, 16))

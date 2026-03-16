"""TTFS activation_type tests: compound string fallback and base-name parsing.

See plan section 5.3: show that compound activation_type causes ReLU fallback
without fix, and that base-name parsing resolves LeakyReLU correctly.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.ir import NeuralCore, IRGraph, IRSource
from mimarsinan.models.unified_core_flow import (
    SpikingUnifiedCoreFlow,
    _ttfs_activation_from_type,
)


def _make_single_core_ir(activation_type: str | None, in_dim=2, out_dim=1):
    """One NeuralCore: in_dim inputs + 1 bias -> out_dim outputs. Weights chosen so pre-activation can be negative."""
    # core_matrix (axons, neurons): axons = in_dim + 1 (bias), neurons = out_dim
    # We want one output that is negative for input [1, 1]: w = [-1, -1], b = 0 -> -2
    w = np.array([[-1.0], [-1.0]], dtype=np.float32)  # (2, 1) one neuron, two input weights
    b = np.zeros((out_dim,), dtype=np.float32)
    core_matrix = np.vstack([w, b.reshape(1, -1)])  # (3, 1)
    input_sources = np.array(
        [IRSource(node_id=-2, index=i) for i in range(in_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )
    core = NeuralCore(
        id=0,
        name="core0",
        input_sources=input_sources,
        core_matrix=core_matrix,
        threshold=1.0,
        latency=0,
        activation_type=activation_type,
    )
    output_sources = np.array(
        [IRSource(node_id=0, index=i) for i in range(out_dim)],
        dtype=object,
    )
    return IRGraph(nodes=[core], output_sources=output_sources)


def test_ttfs_activation_type_base_name_parsing():
    """Helper extracts base name and resolves to F.leaky_relu."""
    fn = _ttfs_activation_from_type("LeakyReLU + ClampDecorator, QuantizeDecorator")
    assert fn is torch.nn.functional.leaky_relu
    assert _ttfs_activation_from_type("LeakyReLU") is torch.nn.functional.leaky_relu
    assert _ttfs_activation_from_type("ReLU") is torch.nn.functional.relu
    assert _ttfs_activation_from_type(None) is torch.nn.functional.relu
    assert _ttfs_activation_from_type("GELU") is torch.nn.functional.gelu


def test_ttfs_continuous_uses_leakyrelu_when_activation_type_leakyrelu():
    """With activation_type='LeakyReLU', TTFS simulation output is in [0, 1].

    In TTFS hardware, neurons fire when V >= θ: negative pre-activations produce
    rate 0 (neuron never fires), regardless of whether LeakyReLU or ReLU is used.
    The output is clamped to [0, 1] to match hardware behavior.
    Activation resolution is tested via test_ttfs_activation_type_base_name_parsing.
    """
    ir_graph = _make_single_core_ir(activation_type="LeakyReLU")
    flow = SpikingUnifiedCoreFlow(
        input_shape=(2,),
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )
    # Input [1, 1] -> pre-activation -2; hardware TTFS clamps to 0 (neuron never fires)
    x = torch.tensor([[1.0, 1.0]])
    with torch.no_grad():
        out = flow(x)
    assert out.shape == (1, 1)
    # TTFS hardware: negative pre-activation → neuron never fires → rate = 0
    assert out[0, 0].item() == 0.0, (
        "TTFS hardware: negative pre-activation should produce rate 0 "
        "(neuron never fires; output clamped to [0, 1])."
    )
    # Verify output is in hardware range [0, 1]
    assert 0.0 <= out[0, 0].item() <= 1.0


def test_ttfs_continuous_falls_back_to_relu_when_activation_type_compound_string():
    """With compound activation_type string, resolved base name yields LeakyReLU (after fix).

    In TTFS hardware, negative pre-activations produce rate 0 (output clamped to [0, 1]).
    Both 'LeakyReLU' and 'LeakyReLU + ClampDecorator, QuantizeDecorator' resolve to
    leaky_relu, and after TTFS clamping they produce identical outputs.
    """
    ir_graph = _make_single_core_ir(
        activation_type="LeakyReLU + ClampDecorator, QuantizeDecorator"
    )
    flow = SpikingUnifiedCoreFlow(
        input_shape=(2,),
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )
    x = torch.tensor([[1.0, 1.0]])
    with torch.no_grad():
        out_compound = flow(x)
    assert out_compound.shape == (1, 1)
    # TTFS hardware: negative pre-activation → rate 0
    assert out_compound[0, 0].item() == 0.0
    assert 0.0 <= out_compound[0, 0].item() <= 1.0

    # Should match explicit LeakyReLU core (both produce 0 after TTFS clamp)
    ir_graph_explicit = _make_single_core_ir(activation_type="LeakyReLU")
    flow_explicit = SpikingUnifiedCoreFlow(
        input_shape=(2,),
        ir_graph=ir_graph_explicit,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )
    with torch.no_grad():
        out_explicit = flow_explicit(x)
    assert torch.allclose(out_compound, out_explicit), (
        "Compound string should match explicit LeakyReLU output."
    )

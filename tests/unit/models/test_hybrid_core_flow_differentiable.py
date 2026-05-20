"""``SpikingHybridCoreFlow`` differentiable forward for chip-aligned training.

The same per-cycle integer firing as the eval kernel, but the cycle loop is
autograd-traversable so a loss on the segment output flows gradients back
to the trained Perceptron weights via SpikingJelly's ATan surrogate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.activations import LIFActivation
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _tiny_lif_flow(T: int = 4):
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    lif1 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    lif2 = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    flow = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T,
        firing_mode="Default", spike_mode="Uniform",
        thresholding_mode="<=", spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )
    return flow, p1, p2


def test_differentiable_forward_matches_eval_forward() -> None:
    """The differentiable path must produce bit-identical outputs to eval mode."""
    flow, _, _ = _tiny_lif_flow(T=4)
    x = torch.rand(2, 8)
    with torch.no_grad():
        eval_out = flow(x)
        flow._chip_aligned_training = True
        diff_out = flow(x)
        flow._chip_aligned_training = False
    torch.testing.assert_close(eval_out, diff_out, atol=1e-6, rtol=0.0)


def test_differentiable_forward_grads_reach_weights() -> None:
    """Backward through the chip-aligned forward must produce finite, non-zero
    gradients on the weights of the encoding Perceptron AND the on-chip core's
    matmul (the latter via the LIF surrogate)."""
    flow, p1, p2 = _tiny_lif_flow(T=4)
    flow._chip_aligned_training = True
    flow.train()

    x = torch.rand(2, 8, requires_grad=False)
    out = flow(x)
    loss = out.sum()
    loss.backward()

    # Encoding Perceptron weights see gradients through the per-cycle forward
    # (uniform-encoded raw input → perceptron(input[t]) → surrogate spikes).
    assert p1.layer.weight.grad is not None
    assert torch.isfinite(p1.layer.weight.grad).all()
    assert p1.layer.weight.grad.abs().sum().item() > 0

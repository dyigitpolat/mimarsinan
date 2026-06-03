"""TTFS spike-based node forward must match the deployed cascade.

``TTFSActivation`` + ``run_ttfs_cycle_accurate`` are the genuine spike-train KD
forward for cascaded ``ttfs_cycle_based`` (the analog of LIF's ``IFNode`` +
``run_cycle_accurate``). For the fine-tuned model to deploy without an
encode/decode mismatch, this spike forward must reproduce, bit-for-bit, what
``_run_neural_segment_rate`` computes on the same single core — multi-input
(greedy fire on partial sums) and bias included. It is also differentiable
(surrogate gradient through the fire-once dynamics).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.chip_simulation.recording import spike_modes
from mimarsinan.models.nn.activations.ttfs_spiking import (
    TTFSActivation,
    run_ttfs_cycle_accurate,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


def _hcm_value(W, theta, bias, a, S):
    out_dim, in_dim = W.shape
    core = SimpleNamespace(
        latency=None, axons_per_core=in_dim, available_axons=0,
        neurons_per_core=out_dim, available_neurons=0,
        axon_sources=[SpikeSource(-2, i, True) for i in range(in_dim)],
        core_matrix=W.T.astype(np.float32).copy(), threshold=float(theta),
        hardware_bias=(np.asarray(bias, dtype=np.float32) if bias is not None else None),
    )
    mapping = SimpleNamespace(
        cores=[core],
        output_sources=np.array([SpikeSource(0, j, False, False) for j in range(out_dim)],
                                dtype=object),
        weight_banks={}, soft_core_placements_per_hard_core=[[]],
    )
    stage = SimpleNamespace(
        hard_core_mapping=mapping, kind="neural", name="t",
        schedule_segment_index=0, schedule_pass_index=0, input_map=[], output_map=[],
    )
    flow = SpikingHybridCoreFlow(
        input_shape=(in_dim,),
        hybrid_mapping=SimpleNamespace(stages=[stage], output_sources=np.array([], dtype=object)),
        simulation_length=S, preprocessor=None, firing_mode="Default",
        spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs_cycle_based",
        ttfs_cycle_schedule="cascaded",
    )
    rate = torch.tensor(a, dtype=torch.float64)
    train = torch.stack(
        [spike_modes.to_spikes(rate, c, simulation_length=S, spike_mode="TTFS")
         for c in range(S)], dim=0,
    )
    return (flow._run_neural_segment_rate(stage, input_spike_train=train,
                                          recorder_seg=None).reshape(-1) / S).numpy()


class _OnePerceptron(nn.Module):
    def __init__(self, W, b, S):
        super().__init__()
        self.lin = nn.Linear(W.shape[1], W.shape[0])
        self.lin.weight.data = torch.tensor(W, dtype=torch.float64)
        self.lin.bias.data = torch.tensor(b, dtype=torch.float64)
        self.act = TTFSActivation(T=S, activation_scale=1.0, input_scale=1.0,
                                  bias=self.lin.bias, thresholding_mode="<=")

    def forward(self, x):
        return self.act(self.lin(x))


def test_ttfs_node_cycle_accurate_matches_hcm_cascade():
    S = 8
    rng = np.random.default_rng(1)
    for _ in range(8):
        in_dim, out_dim = 4, 3
        W = rng.uniform(-0.5, 1.0, size=(out_dim, in_dim))
        b = rng.uniform(-0.3, 0.3, size=(out_dim,))
        a = (rng.integers(0, S + 1, size=(1, in_dim)) / S).astype(np.float64)
        hcm = _hcm_value(W, 1.0, b, a, S)
        model = _OnePerceptron(W, b, S).double()
        node = run_ttfs_cycle_accurate(
            model, torch.tensor(a, dtype=torch.float64), S,
        ).detach().reshape(-1).numpy()
        np.testing.assert_allclose(node, hcm, atol=1e-9)


def test_ttfs_node_forward_is_differentiable():
    S = 8
    W = np.array([[0.4, 0.3, 0.5]]); b = np.array([0.0])
    a = torch.tensor([[0.5, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    model = _OnePerceptron(W, b, S).double()
    out = run_ttfs_cycle_accurate(model, a, S)
    out.sum().backward()
    assert model.lin.weight.grad is not None
    assert torch.isfinite(model.lin.weight.grad).all()
    assert model.lin.weight.grad.abs().sum() > 0


def test_ttfs_encoding_mode_matches_framework_spike_train():
    """Encoding mode: an ideal value V in -> a single TTFS spike at the same
    cycle the framework's ttfs_single_spike_train places it (the value->spike
    entry of a segment, analogous to LIF's value-as-current encoding)."""
    from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_single_spike_train

    S, theta = 8, 1.0
    node = TTFSActivation(T=S, activation_scale=theta, input_scale=1.0, bias=None,
                          thresholding_mode="<=", encoding=True)
    node.set_cycle_accurate(True)
    for V in [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0, 1.3]:
        node.reset_state()
        x = torch.tensor([[V]], dtype=torch.float64)
        spikes = [int(node(x).item()) for _ in range(S)]
        assert sum(spikes) <= 1, "encoding must emit at most one spike"
        fire = next((t for t, s in enumerate(spikes) if s == 1), None)
        r = min(max(V / theta, 0.0), 1.0)
        ref = ttfs_single_spike_train(np.array([[r]]), S)[0, 0]
        ref_fire = next((t for t, s in enumerate(ref) if s > 0.5), None)
        assert fire == ref_fire, f"V={V}: fire {fire} != ref {ref_fire}"


def test_ttfs_encoding_mode_is_differentiable():
    S = 8
    node = TTFSActivation(T=S, activation_scale=1.0, input_scale=1.0, bias=None,
                          thresholding_mode="<=", encoding=True)
    node.set_cycle_accurate(True)
    V = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    out = sum(node(V) for _ in range(S))
    out.sum().backward()
    assert V.grad is not None and torch.isfinite(V.grad).all()
    assert V.grad.abs().sum() > 0

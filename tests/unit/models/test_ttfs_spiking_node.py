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


class TestPerChannelThetaConvBroadcast:
    """theta_cotrain promotes ``activation_scale`` (and the upstream layer's
    ``input_scale``) to a per-OUTPUT-CHANNEL vector ``[C]``. For a Conv2D output
    ``[B, C, H, W]`` the channel axis is dim 1, not the last dim, so a raw ``[C]``
    scale broadcasts against ``W`` and crashes when ``C != W`` (the
    Conv2DPerceptronMapper ``features_3`` crash that blocks every lever-ON run).

    ``_scale_values`` must reshape a multi-element scale to broadcast on the
    channel axis for ``ndim > 2`` inputs, leaving scalar and the 2-D linear
    ``[B, C]`` case (where ``[C]`` already broadcasts correctly) untouched.
    """

    @staticmethod
    def _per_channel_scale(C):
        return torch.arange(1, C + 1, dtype=torch.float64) * 0.5 + 0.25

    def test_non_cycle_accurate_4d_conv_per_channel_scale(self):
        # eval staircase path; C != W so a [C] scale on the last dim would crash.
        C = 6
        scale = self._per_channel_scale(C)
        act = TTFSActivation(T=4, activation_scale=scale, input_scale=1.0,
                             bias=None, thresholding_mode="<=", encoding=False)
        x = torch.randn(2, C, 3, 4, dtype=torch.float64)  # C=6 != W=4
        out = act.forward(x)
        assert out.shape == x.shape

        # Manual reference: scale broadcasts on the channel axis [1, C, 1, 1].
        from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
        sv = scale.view(1, C, 1, 1).clamp(min=1e-12)
        r = (torch.relu(x) / sv).clamp(0.0, 1.0)
        ref = TTFSStaircaseFunction.apply(r, 4) * sv
        torch.testing.assert_close(out, ref)

    def test_cascade_4d_conv_per_channel_scale(self):
        # cascade branch (cycle-accurate, encoding=False) with per-channel theta
        # AND per-channel input_scale on a 4-D conv tensor, C != W.
        C = 6
        scale = self._per_channel_scale(C)
        in_scale = self._per_channel_scale(C) + 0.1
        bias = torch.arange(C, dtype=torch.float64) * 0.05
        act = TTFSActivation(T=4, activation_scale=scale, input_scale=in_scale,
                             bias=bias, thresholding_mode="<=", encoding=False)
        act.set_cycle_accurate(True)
        x = torch.randn(2, C, 3, 4, dtype=torch.float64)
        out = act.forward(x)
        assert out.shape == x.shape
        assert ((out == 0) | (out == 1)).all(), "cascade output must be 0/1 spikes"

    def test_encoding_4d_conv_per_channel_scale(self):
        # encoding branch (cycle-accurate, encoding=True) per-channel theta, 4-D.
        C = 6
        scale = self._per_channel_scale(C)
        act = TTFSActivation(T=4, activation_scale=scale, input_scale=1.0,
                             bias=None, thresholding_mode="<=", encoding=True)
        act.set_cycle_accurate(True)
        x = torch.rand(2, C, 3, 4, dtype=torch.float64)
        accum = torch.zeros_like(x)
        for _ in range(4):
            spike = act.forward(x)
            assert spike.shape == x.shape
            accum = accum + spike
        assert (accum <= 1).all(), "encoding must emit at most one spike per neuron"

    def test_cascade_4d_per_channel_matches_manual_channel_reshape(self):
        # The per-channel cascade forward must equal a forward where the scales
        # are manually pre-reshaped to [1, C, 1, 1] — i.e. the fix only moves the
        # broadcast onto the channel axis, it does not change the math.
        C = 5
        scale = self._per_channel_scale(C)
        in_scale = self._per_channel_scale(C) + 0.2
        bias = torch.arange(C, dtype=torch.float64) * 0.05
        x = torch.randn(2, C, 3, 7, dtype=torch.float64)  # C=5 != W=7

        act = TTFSActivation(T=4, activation_scale=scale, input_scale=in_scale,
                             bias=bias, thresholding_mode="<=", encoding=False)
        act.set_cycle_accurate(True)
        out = act.forward(x)

        ref_act = TTFSActivation(
            T=4, activation_scale=scale.view(1, C, 1, 1),
            input_scale=in_scale.view(1, C, 1, 1),
            bias=bias, thresholding_mode="<=", encoding=False,
        )
        ref_act.set_cycle_accurate(True)
        ref = ref_act.forward(x)
        torch.testing.assert_close(out, ref)


class TestScaleValuesByteIdentical:
    """The scalar and 2-D-linear scale paths must be untouched by the conv fix."""

    def test_scalar_scale_unchanged(self):
        # Scalar activation_scale: _scale_values returns the same 0-dim tensor.
        act = TTFSActivation(T=4, activation_scale=2.0, input_scale=3.0,
                             bias=None, thresholding_mode="<=", encoding=False)
        x = torch.randn(2, 6, 3, 4, dtype=torch.float64)
        sv, iv = act._scale_values(x)
        assert sv.dim() == 0 and float(sv) == 2.0
        # input_scale 3.0 is a python float -> stays a float.
        assert iv == 3.0
        out = act.forward(x)
        from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
        r = (torch.relu(x) / 2.0).clamp(0.0, 1.0)
        ref = TTFSStaircaseFunction.apply(r, 4) * 2.0
        torch.testing.assert_close(out, ref)

    def test_2d_linear_per_channel_scale_keeps_flat_shape(self):
        # 2-D linear [B, C]: a per-channel [C] scale already broadcasts on the
        # last (channel) dim, so _scale_values must NOT reshape it -> byte-identical.
        C = 6
        scale = torch.arange(1, C + 1, dtype=torch.float64)
        in_scale = torch.arange(1, C + 1, dtype=torch.float64) + 0.5
        act = TTFSActivation(T=4, activation_scale=scale, input_scale=in_scale,
                             bias=None, thresholding_mode="<=", encoding=False)
        x = torch.randn(3, C, dtype=torch.float64)
        sv, iv = act._scale_values(x)
        assert sv.shape == (C,), "2-D linear scale must keep its flat [C] shape"
        assert iv.shape == (C,)
        out = act.forward(x)
        from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
        r = (torch.relu(x) / scale).clamp(0.0, 1.0)
        ref = TTFSStaircaseFunction.apply(r, 4) * scale
        torch.testing.assert_close(out, ref)

    def test_3d_mixer_channel_last_per_channel_forward_unchanged(self):
        # 3-D mixer [B, tokens, F] with a per-channel [F] scale already broadcasts
        # on the last (channel) axis. ndim>2 routes it through the channel-axis
        # reshape, but the forward OUTPUT must stay byte-identical to the raw-[F]
        # broadcast (same axis -> exact equality, not just close).
        B, Tk, F = 2, 5, 7
        scale = torch.arange(1, F + 1, dtype=torch.float64)
        act = TTFSActivation(T=4, activation_scale=scale, input_scale=1.0,
                             bias=None, thresholding_mode="<=", encoding=False)
        x = torch.randn(B, Tk, F, dtype=torch.float64)
        out = act.forward(x)
        from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
        r = (torch.relu(x) / scale).clamp(0.0, 1.0)
        ref = TTFSStaircaseFunction.apply(r, 4) * scale
        assert torch.equal(out, ref), "3-D channel-last forward must be byte-identical"

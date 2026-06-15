"""Surrogate-alpha is a backward-only knob on the TTFS spike node.

``alpha`` shapes the ATan surrogate gradient and MUST NOT touch the exact
``pre > 0`` Heaviside forward: at every alpha the cycle-accurate forward output is
bit-identical; only the autograd ``.grad`` magnitude scales. ``set_surrogate_alpha``
(node-level and the ``perceptron_rate`` model-wide SSOT) threads that knob.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import (
    TTFSActivation,
    run_ttfs_cycle_accurate,
)
from mimarsinan.tuning.perceptron_rate import set_surrogate_alpha


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


def test_default_surrogate_alpha_is_two():
    node = TTFSActivation(T=8, activation_scale=1.0)
    assert node.surrogate_alpha == 2.0


def test_set_surrogate_alpha_sets_node_attr():
    node = TTFSActivation(T=8, activation_scale=1.0)
    node.set_surrogate_alpha(0.5)
    assert node.surrogate_alpha == 0.5


def test_cascade_forward_is_alpha_invariant():
    """Cascade (non-encoding) forward is bit-identical across alpha."""
    S = 8
    rng = np.random.default_rng(3)
    W = rng.uniform(-0.5, 1.0, size=(3, 4))
    b = rng.uniform(-0.3, 0.3, size=(3,))
    a = (rng.integers(0, S + 1, size=(1, 4)) / S).astype(np.float64)

    outs = {}
    for alpha in (0.5, 2.0, 8.0):
        model = _OnePerceptron(W, b, S).double()
        set_surrogate_alpha(model, alpha)
        outs[alpha] = run_ttfs_cycle_accurate(
            model, torch.tensor(a, dtype=torch.float64), S,
        ).detach().reshape(-1).numpy()

    base = outs[2.0]
    for alpha, out in outs.items():
        np.testing.assert_array_equal(out, base)


def test_encoding_forward_is_alpha_invariant():
    """Encoding-mode (value->spike) forward is bit-identical across alpha."""
    S, theta = 8, 1.0
    for V in (0.0, 0.125, 0.5, 0.875, 1.3):
        spikes = {}
        for alpha in (0.5, 2.0, 8.0):
            node = TTFSActivation(T=S, activation_scale=theta, input_scale=1.0,
                                  bias=None, thresholding_mode="<=", encoding=True)
            node.set_surrogate_alpha(alpha)
            node.set_cycle_accurate(True)
            node.reset_state()
            x = torch.tensor([[V]], dtype=torch.float64)
            spikes[alpha] = [int(node(x).item()) for _ in range(S)]
        assert spikes[0.5] == spikes[2.0] == spikes[8.0]


def test_surrogate_alpha_scales_backward_grad():
    """Only the surrogate gradient depends on alpha; larger alpha => larger grad."""
    S = 8
    W = np.array([[0.4, 0.3, 0.5]]); b = np.array([0.0])

    def _grad_for(alpha):
        a = torch.tensor([[0.5, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
        model = _OnePerceptron(W, b, S).double()
        set_surrogate_alpha(model, alpha)
        run_ttfs_cycle_accurate(model, a, S).sum().backward()
        return float(model.lin.weight.grad.abs().sum())

    g_small = _grad_for(0.5)
    g_large = _grad_for(8.0)
    assert g_small != g_large
    assert g_large > g_small


def _model_with_ttfs_nodes(n=3, S=8):
    return nn.ModuleList(
        TTFSActivation(T=S, activation_scale=1.0) for _ in range(n)
    )


def test_set_surrogate_alpha_sets_all_nodes():
    model = _model_with_ttfs_nodes(n=4)
    set_surrogate_alpha(model, 0.7)
    nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
    assert len(nodes) == 4
    assert all(node.surrogate_alpha == 0.7 for node in nodes)

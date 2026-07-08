"""End-to-end ``run_cycle_accurate`` test against a converted PyTorch model.

After the refactor, cycle-accurate forward is driven by the generic
:func:`run_cycle_accurate` utility — no model-side ``forward_cycle_accurate``
method exists. We verify two invariants here:

1. ``run_cycle_accurate(model, x, T)`` returns ``(B, num_classes)``
   logits for a converted model containing LIFActivations, without
   needing any per-layer override on Conv2DPerceptronMapper, einops,
   reshape, mean, etc.
2. When every input cycle is identical (rate exactly 0 or 1 → uniform
   encoder yields a constant train), the mean over T equals the rate-mode
   forward exactly. Other rates produce a non-constant spike pattern
   that is *not* equivalent to a broadcast rate — by design — so we
   don't pin those.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation, run_cycle_accurate
from mimarsinan.torch_mapping.converter import convert_torch_model


class _TinyLinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 3)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _swap_relu_with_lif(flow, T: int, scale: float = 1.0) -> None:
    for p in flow.get_perceptrons():
        p.activation_scale.data = torch.tensor(float(scale))
        p.activation = LIFActivation(
            T=T,
            activation_scale=p.activation_scale,
            thresholding_mode="<=",
        )


def test_run_cycle_accurate_runs_end_to_end() -> None:
    """The generic driver returns the right output shape for a converted
    model that wasn't specifically engineered for cycle-accurate."""
    model = _TinyLinearReLU().eval()
    flow = convert_torch_model(model, input_shape=(8,), num_classes=3).eval()
    _swap_relu_with_lif(flow, T=4)

    x = torch.rand(5, 8).clamp(0.0, 1.0)
    out = run_cycle_accurate(flow, x, T=4)
    assert out.shape == (5, 3)


def test_run_cycle_accurate_matches_rate_at_saturation() -> None:
    """At rate=1 the uniform encoder produces a constant 1-per-cycle
    spike train. Each cycle feeds the same input to the model, so the
    LIF integrators receive a constant-broadcast — equivalent to
    rate-mode. Mean over T must therefore equal rate-mode forward."""
    T = 4
    model = _TinyLinearReLU().eval()
    flow = convert_torch_model(model, input_shape=(8,), num_classes=3).eval()
    _swap_relu_with_lif(flow, T=T, scale=1.0)

    x = torch.ones(2, 8)
    out_ca = run_cycle_accurate(flow, x, T=T)
    out_rate = flow(x)

    torch.testing.assert_close(out_ca, out_rate, atol=1e-4, rtol=1e-4)


def test_run_cycle_accurate_restores_lif_modes() -> None:
    """After ``run_cycle_accurate`` returns, every LIFActivation is back
    in rate-mode. Subsequent ``flow(x)`` uses the rate-mode forward, so
    downstream pipeline steps see the un-modified model."""
    T = 4
    model = _TinyLinearReLU().eval()
    flow = convert_torch_model(model, input_shape=(8,), num_classes=3).eval()
    _swap_relu_with_lif(flow, T=T)

    _ = run_cycle_accurate(flow, torch.rand(3, 8), T=T)

    for m in flow.modules():
        if isinstance(m, LIFActivation):
            assert m._cycle_accurate_mode is False
            assert m.if_node.step_mode == "m"


def test_run_cycle_accurate_backward_flows_to_weights(deterministic_rng) -> None:
    """Gradients must flow through the full T-loop to every trainable
    parameter — Linear weights, LIF surrogate, the lot.

    Seeded: the assertion is only meaningful for an init that actually fires.
    Unseeded, the random init inherited whatever RNG state the worker's
    preceding tests left (``--dist worksteal``), and roughly 1 seed in 24
    (e.g. 19) produces a net where no neuron fires within T=4 — every gradient
    is then legitimately zero and the test failed for the wrong reason.

    This is the load-bearing integration test: it proves the
    cycle-accurate driver works inside a trainable setup (the loss
    has a non-zero gradient w.r.t. every Linear weight in the chain).
    Without it a silent autograd break in the T-loop would only
    surface during a full benchmark run.
    """
    T = 4
    model = _TinyLinearReLU()
    flow = convert_torch_model(model, input_shape=(8,), num_classes=3)
    _swap_relu_with_lif(flow, T=T)
    flow.train()

    x = torch.rand(4, 8, requires_grad=False)
    y = torch.tensor([0, 1, 2, 0])
    out = run_cycle_accurate(flow, x, T=T)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    # Every Linear weight in the converted graph must have received a
    # non-zero gradient. (The Perceptrons may have nested wrappers; we
    # check the raw nn.Linear children.)
    linear_modules = [m for m in flow.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_modules) >= 3  # fc1, fc2, fc3
    for lin in linear_modules:
        assert lin.weight.grad is not None
        assert float(lin.weight.grad.abs().sum()) > 0.0, (
            f"No gradient flowed to {lin}: cycle-accurate backward is broken"
        )

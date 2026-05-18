"""LIFActivation correctness tests.

These tests pin the contract the rest of the LIF deployment pipeline
relies on:

1. Forward output matches the T-level saturated staircase that a
   ``SpikingHybridCoreFlow`` LIF simulation (firing_mode='Default',
   subtractive reset) produces for a single neuron with constant
   pre-activation input. Two staircases are pinned, one per
   ``thresholding_mode``: inclusive ``<=`` keeps the classical
   ``floor(T·x/scale)`` reference, while strict ``<`` (the chip /
   nevresim / SCM default) drops the top step at each boundary
   ``x = k·scale/T``.
2. Backward flows gradients through the ATan surrogate — non-zero gradient
   at in-range inputs, bounded gradient at saturated inputs.
3. State does not leak across forward passes (``functional.reset_net``
   inside ``forward``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.activations import LIFActivation


def _expected_lif_staircase(
    x: torch.Tensor,
    scale: float,
    T: int,
    thresholding_mode: str = "<=",
) -> torch.Tensor:
    """Reference implementation of the T-level saturated LIF staircase.

    With ``thresholding_mode='<='`` (inclusive) this is the classical
    ``floor(T·x/scale)/T · scale`` shape: the staircase reaches the full
    ``scale`` at ``x = scale`` and beyond.

    With ``thresholding_mode='<'`` (strict, matching ``memb > threshold``
    on the chip) each boundary ``x = k·scale/T`` for ``k in 1..T`` drops
    by exactly one fire, so the staircase top is ``(T-1)·scale/T`` at
    ``x = scale`` and only saturates at full ``scale`` once
    ``T·x/scale`` strictly exceeds ``T``.
    """
    x_pos = x.clamp(min=0)
    raw = T * x_pos / scale
    if thresholding_mode == "<=":
        k = torch.floor(raw).clamp(max=T)
    elif thresholding_mode == "<":
        # ceil(raw) - 1 for raw > 0, 0 for raw == 0. Equivalent to
        # ``floor(raw - eps)`` for any 0 < eps < 1/(T·scale_precision).
        # Cap at T (saturation) just like the inclusive variant.
        k = torch.where(
            raw > 0,
            torch.ceil(raw) - 1,
            torch.zeros_like(raw),
        ).clamp(max=T)
    else:
        raise ValueError(f"Unsupported thresholding_mode {thresholding_mode!r}")
    return k / T * scale


@pytest.mark.parametrize("T", [1, 2, 4, 8])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 3.7])
@pytest.mark.parametrize("thresholding_mode", ["<", "<="])
def test_forward_matches_reference_staircase(
    T: int, scale: float, thresholding_mode: str
) -> None:
    act = LIFActivation(
        T=T,
        activation_scale=torch.tensor(float(scale)),
        thresholding_mode=thresholding_mode,
    )

    # Sweep a range that covers below zero, inside [0, scale], and above.
    # Use step values that include the boundary ``x = k·scale/T`` exactly,
    # so the test pins the difference between strict and inclusive modes
    # at the boundary (otherwise the float-grid sweep can skip the only
    # values where the two modes disagree).
    boundary_points = torch.tensor(
        [k / T * scale for k in range(T + 2)], dtype=torch.float32
    )
    sweep = torch.linspace(-1.0, 2.0 * scale, steps=41)
    x = torch.cat([sweep, boundary_points]).unsqueeze(0)
    y = act(x)
    expected = _expected_lif_staircase(x, float(scale), T, thresholding_mode)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=0.0)


def test_strict_mode_drops_boundary_fire() -> None:
    """The bug this guards: strict ``<`` must miss the boundary fire that
    inclusive ``<=`` would still emit at ``x = k·scale/T``, mirroring the
    chip's strict ``memb > threshold`` comparator. Failure here means
    training optimistically reports an extra fire per boundary neuron
    that nevresim / Python sim / SANA-FE / Lava all refuse to produce.
    """
    T, scale = 4, 1.0
    strict = LIFActivation(
        T=T, activation_scale=torch.tensor(scale), thresholding_mode="<",
    )
    inclusive = LIFActivation(
        T=T, activation_scale=torch.tensor(scale), thresholding_mode="<=",
    )

    # x = 0.5 = 2/T·scale: inclusive fires twice (rate 0.5), strict only
    # fires once (rate 0.25) — exact half-staircase boundary.
    x = torch.tensor([[0.5]])
    assert inclusive(x).item() == pytest.approx(0.5, abs=1e-5)
    assert strict(x).item() == pytest.approx(0.25, abs=1e-5)


def test_backward_gradient_flows_in_range() -> None:
    T, scale = 4, 2.0
    act = LIFActivation(T=T, activation_scale=torch.tensor(scale))

    # Pick a point strictly inside [0, scale] that straddles the staircase
    # boundary so the ATan surrogate produces non-trivial gradient.
    x = torch.tensor([[0.75]], requires_grad=True)
    y = act(x)
    y.sum().backward()

    assert x.grad is not None
    # Gradient should be nonzero at in-range positive inputs.
    assert float(x.grad.abs().sum()) > 0.0


def test_backward_gradient_bounded_at_saturation() -> None:
    T, scale = 4, 2.0
    act = LIFActivation(T=T, activation_scale=torch.tensor(scale))

    # Well above scale → saturated at scale.
    x_sat = torch.tensor([[5.0]], requires_grad=True)
    y_sat = act(x_sat)
    y_sat.sum().backward()
    grad_sat = float(x_sat.grad.abs().item())

    # Just inside the range → full slope from the ATan surrogate.
    x_mid = torch.tensor([[1.0]], requires_grad=True)
    y_mid = act(x_mid)
    y_mid.sum().backward()
    grad_mid = float(x_mid.grad.abs().item())

    # Saturated gradient should be strictly smaller than in-range gradient
    # (ATan surrogate decays monotonically with distance from threshold).
    assert grad_sat < grad_mid
    # And still finite / non-exploding.
    assert np.isfinite(grad_sat)


def test_no_state_leak_between_batches() -> None:
    T, scale = 4, 2.0
    act = LIFActivation(T=T, activation_scale=torch.tensor(scale))

    x = torch.tensor([[1.5]])
    y1 = act(x).detach().clone()
    y2 = act(x).detach().clone()
    # Same input, same output — no membrane state carried across calls.
    torch.testing.assert_close(y1, y2)


def test_activation_scale_tracks_perceptron_updates() -> None:
    """If the owning Perceptron updates ``activation_scale.data``, the
    LIFActivation reference must reflect the new value on next forward.
    """
    scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    act = LIFActivation(T=4, activation_scale=scale)

    x = torch.tensor([[0.5]])
    y_before = act(x).item()

    # Simulate Perceptron.set_activation_scale(new_scale).
    scale.data = torch.tensor(2.0)
    y_after = act(x).item()

    # Same pre-activation 0.5 but scale doubled → staircase bin changes:
    # before: floor(4 * 0.5 / 1.0) / 4 * 1.0 = 2/4 = 0.5
    # after:  floor(4 * 0.5 / 2.0) / 4 * 2.0 = floor(1)/4 * 2 = 0.5
    # The effective output at x=0.5 may coincide; pick a value that doesn't.
    x2 = torch.tensor([[0.75]])
    # before scale: floor(4 * 0.75 / 1.0)/4 * 1.0 = 3/4 = 0.75
    # after  scale: floor(4 * 0.75 / 2.0)/4 * 2.0 = floor(1.5)/4 * 2 = 1/4*2 = 0.5
    y_x2 = act(x2).item()
    assert y_x2 == pytest.approx(0.5, abs=1e-5)
    # And the initial y_before differs from the post-update forward at x=0.5 only
    # if the staircase bin changed. Validate that the underlying reference is live:
    assert act.activation_scale is scale

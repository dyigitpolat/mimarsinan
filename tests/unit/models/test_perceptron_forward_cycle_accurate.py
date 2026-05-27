"""LIFBlendActivation cycle-accurate behaviour.

After the refactor, cycle-accurate forward is driven entirely by
:func:`run_cycle_accurate`, which toggles each ``LIFActivation`` into
single-step mode. ``Perceptron`` and ``LIFBlendActivation`` have no
cycle-accurate-specific code paths — they just call their child
activation, which branches on its own mode flag.

These tests pin the only piece that's not exercised by
``test_lif_activation.py``: the blend module's forward must dispatch
correctly when its inner LIFActivation is in single-step mode, and the
old/lif blend must collapse to the right limit at rate=0 and rate=1.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation


def _new_blend(rate: float, T: int = 4, scale: float = 1.0) -> LIFBlendActivation:
    relu = nn.ReLU()
    lif = LIFActivation(T=T, activation_scale=torch.tensor(scale))
    return LIFBlendActivation(relu, lif, rate=rate)


def test_blend_cycle_accurate_at_rate_zero_is_relu() -> None:
    """At rate=0 the blend returns ``old_activation(x)`` regardless of the
    LIF mode flag. The single-step branch of the inner LIFActivation is
    never reached because the blend short-circuits to ReLU."""
    blend = _new_blend(rate=0.0)
    blend.lif_activation.set_cycle_accurate(True)

    x = torch.randn(2, 5)
    out = blend(x)
    torch.testing.assert_close(out, torch.relu(x), atol=0.0, rtol=0.0)


def test_blend_cycle_accurate_at_rate_one_is_single_step_lif() -> None:
    """At rate=1 the blend short-circuits to ``lif_activation(x)``. In
    cycle-accurate mode that is the single-step output ``spike * scale``,
    not the multi-step rate. Verifying this means the blend correctly
    delegates the mode decision to the LIF child.
    """
    blend = _new_blend(rate=1.0)
    blend.lif_activation.set_cycle_accurate(True)

    x = torch.tensor([[0.75]])
    # Cycle 0: integrate x_norm=0.75 → v=0.75 (no fire) → return 0.
    assert blend(x).item() == 0.0
    # Cycle 1: integrate → v=1.5 → fires → return 1.
    assert blend(x).item() == 1.0


def test_blend_cycle_accurate_mid_blend_interpolates() -> None:
    """At 0 < rate < 1 the blend's output equals the linear interp of
    ``old(x)`` and ``lif(x)``. The lif side is a single-step spike for
    this cycle; the relu side is the static ReLU of this cycle's
    pre-activation.
    """
    blend = _new_blend(rate=0.4)
    blend.lif_activation.set_cycle_accurate(True)

    x = torch.tensor([[0.6]])
    out = blend(x)
    # blend = 0.6*relu(0.6) + 0.4*lif_step(x). On cycle 0 the IFNode
    # integrates 0.6 → v=0.6 (no fire) → 0. So we should see
    # ``out = 0.6 * 0.6 + 0.4 * 0 = 0.36``.
    assert out.item() == pytest.approx(0.36, abs=1e-6)


def test_blend_in_rate_mode_unchanged() -> None:
    """Sanity check: with LIF still in rate-mode, the blend's forward is
    bitwise identical to its previous behaviour (rate input → rate output)."""
    blend = _new_blend(rate=0.5, T=4, scale=2.0)
    # LIF in default (rate) mode.
    assert blend.lif_activation._cycle_accurate_mode is False

    x = torch.tensor([[1.0]])
    out = blend(x).item()
    # Pre-refactor expected: 0.5 * relu(1) + 0.5 * lif_rate(1)
    # lif_rate(1) = floor(4 * 1 / 2) / 4 * 2 = floor(2)/4 * 2 = 1.0 (inclusive)
    # So out = 0.5 * 1.0 + 0.5 * 1.0 = 1.0.
    assert out == 1.0

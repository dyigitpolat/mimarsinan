"""Differentiable variant of ``lif_fire_and_reset`` for chip-aligned training.

The non-differentiable kernel (``lif_fire_and_reset``) is the truth at
inference / parity time. The differentiable variant must match it forward,
and on backward must provide a finite surrogate gradient consistent with the
training-side ``LIFActivation`` (SpikingJelly's ATan / StrictATan).
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.lif_kernels import (
    lif_fire_and_reset,
    lif_fire_and_reset_differentiable,
)


@pytest.mark.parametrize("thresholding_mode", ["<", "<="])
@pytest.mark.parametrize("firing_mode", ["Default", "Novena"])
def test_differentiable_forward_matches_baseline(thresholding_mode, firing_mode) -> None:
    torch.manual_seed(0)
    threshold = torch.tensor(1.0, dtype=torch.float32)
    base_memb = torch.randn(3, 5, dtype=torch.float32) * 0.7

    memb_a = base_memb.clone()
    fired_a = lif_fire_and_reset(
        memb_a, threshold,
        thresholding_mode=thresholding_mode, firing_mode=firing_mode,
        output_dtype=torch.float32,
    )

    memb_b = base_memb.clone()
    fired_b, memb_b_new = lif_fire_and_reset_differentiable(
        memb_b, threshold,
        thresholding_mode=thresholding_mode, firing_mode=firing_mode,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(fired_a, fired_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(memb_a, memb_b_new, atol=0.0, rtol=0.0)


def test_differentiable_backward_flows_surrogate_gradient() -> None:
    """Spike output must carry a non-zero surrogate gradient back to the membrane."""
    threshold = torch.tensor(1.0, dtype=torch.float32)
    memb = torch.tensor(
        [[0.5, 1.0, 1.5, -0.2]], dtype=torch.float32, requires_grad=True
    )
    fired, _ = lif_fire_and_reset_differentiable(
        memb, threshold,
        thresholding_mode="<=", firing_mode="Default",
        output_dtype=torch.float32,
    )
    # Some elements fire, some don't; backward through the spike output.
    fired.sum().backward()
    assert memb.grad is not None
    # ATan surrogate is smooth and positive across the threshold neighbourhood.
    assert torch.all(memb.grad >= 0)
    # Near-threshold elements have larger gradient than far-from-threshold ones.
    assert memb.grad[0, 1].item() > memb.grad[0, 3].item()


def test_differentiable_backward_through_membrane() -> None:
    """``memb_new``'s gradient must flow back via the subtractive-reset residual."""
    threshold = torch.tensor(1.0, dtype=torch.float32)
    memb = torch.tensor(
        [[1.5]], dtype=torch.float32, requires_grad=True
    )
    fired, memb_new = lif_fire_and_reset_differentiable(
        memb, threshold,
        thresholding_mode="<=", firing_mode="Default",
        output_dtype=torch.float32,
    )
    memb_new.sum().backward()
    assert memb.grad is not None
    # For subtractive default, memb_new = memb - threshold * fired. d(memb_new)/d(memb)
    # equals 1 - threshold * d(fired)/d(memb), which is well-defined (surrogate).
    assert torch.isfinite(memb.grad).all()


def test_differentiable_no_inplace_mutation() -> None:
    """Unlike the eval-mode kernel, the differentiable path must leave ``memb`` intact."""
    threshold = torch.tensor(1.0, dtype=torch.float32)
    memb = torch.tensor([[1.5, 0.2]], dtype=torch.float32, requires_grad=True)
    original = memb.detach().clone()
    fired, memb_new = lif_fire_and_reset_differentiable(
        memb, threshold,
        thresholding_mode="<=", firing_mode="Default",
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(memb.detach(), original, atol=0.0, rtol=0.0)

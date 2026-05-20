"""Shared LIF integrate-and-fire step (unified and hybrid core flows)."""

from __future__ import annotations

import torch

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


def lif_fire_and_reset(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    *,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Threshold ``memb``, return spike tensor, apply Novena/Default reset in-place.

    Eval / parity kernel — non-differentiable (in-place reset, integer firing).
    Use :func:`lif_fire_and_reset_differentiable` for training through SCM.
    """
    fired = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    if firing_mode == "Novena":
        memb[fired] = 0.0
    elif firing_mode == "Default":
        memb[fired] -= threshold
    if output_dtype is not None:
        return fired.to(output_dtype)
    return fired.float()


def lif_fire_and_reset_differentiable(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    *,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
):
    """Differentiable variant: forward matches :func:`lif_fire_and_reset` exactly,
    backward routes through SpikingJelly's surrogate (ATan for ``<=``, StrictATan
    for ``<``) — the *same* surrogate ``LIFActivation`` uses during NF training,
    so the chip-aligned finetune does not introduce a second gradient definition.

    Returns ``(fired, memb_new)`` as separate tensors. ``memb`` is **not**
    mutated in place (autograd-safe).
    """
    from mimarsinan.models.activations import (
        StrictATanSurrogate,
        _StrictHeavisideFunction,
    )

    threshold_b = threshold.to(memb.dtype) if isinstance(threshold, torch.Tensor) else \
        torch.tensor(float(threshold), dtype=memb.dtype, device=memb.device)

    diff = memb - threshold_b
    if thresholding_mode == "<":
        # Strict (memb > threshold) firing — StrictATan surrogate.
        fired_f32 = _StrictHeavisideFunction.apply(diff.to(torch.float32), 2.0)
        fired = fired_f32.to(memb.dtype)
    elif thresholding_mode == "<=":
        # Inclusive (memb >= threshold) firing — SpikingJelly ATan surrogate.
        # SpikingJelly's surrogate.atan is a smooth approximation; forward is
        # Heaviside via ``surrogate.atan_forward`` returning hard step.
        from spikingjelly.activation_based import surrogate as _sj_surrogate

        surrogate_fn = _sj_surrogate.ATan()
        fired_f32 = surrogate_fn(diff.to(torch.float32))
        fired = fired_f32.to(memb.dtype)
    else:
        raise ValueError(f"thresholding_mode must be '<' or '<='; got {thresholding_mode!r}")

    if firing_mode == "Default":
        memb_new = memb - threshold_b * fired
    elif firing_mode == "Novena":
        memb_new = memb * (1.0 - fired)
    else:
        raise ValueError(f"firing_mode must be 'Default' or 'Novena'; got {firing_mode!r}")

    if output_dtype is not None:
        fired = fired.to(output_dtype)
    return fired, memb_new

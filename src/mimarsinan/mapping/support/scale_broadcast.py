"""Shared scale-vector broadcast helpers for mapper and IR paths."""

from __future__ import annotations

import torch


def spread_scalar(scalar: torch.Tensor, n: int) -> torch.Tensor:
    """0-dim ``scalar`` broadcast to an OWNED ``(n,)`` vector on its own device/dtype.

    The SSOT for mean-folding a scale: broadcasting the reduction tensor keeps
    device and dtype following the input by construction, where the ``torch.full(
    ..., t.mean().item())`` idiom it replaces silently relocates the result to CPU
    in the default dtype -- and forces a host sync to do it.
    """
    return scalar.reshape(()).expand(n).clone()


def broadcast_scale_to_dim(scale: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Expand a 1-D scale vector to *target_dim* using repeat or mean-fill."""
    if scale.shape[0] == target_dim:
        return scale
    if target_dim % scale.shape[0] == 0:
        return scale.repeat_interleave(target_dim // scale.shape[0])
    return spread_scalar(scale.mean(), target_dim)


def align_scale_devices(scales: list[torch.Tensor]) -> list[torch.Tensor]:
    """Put a set of source scales on ONE device: the anchored one, if any exists.

    Scale vectors enter the propagation walk from two kinds of node.
    PARAMETER-BEARING nodes (a perceptron's theta) know the model's device and
    carry it. PARAMETERLESS structural roots -- ``InputMapper``'s unit wire scale
    -- have nothing to anchor on and can only default to CPU. A ComputeOp that
    joins an input wire to a perceptron output is where the two meet, and mixing
    them raises inside the ``torch.stack`` of ``combine_source_scales``.

    The anchored device is the model's, so it is the one that wins. An all-CPU
    walk (the default suite surface) is returned unchanged.
    """
    device = next(
        (s.device for s in scales
         if isinstance(s, torch.Tensor) and s.device.type != "cpu"),
        None,
    )
    if device is None:
        return list(scales)
    return [s.to(device) if isinstance(s, torch.Tensor) else s for s in scales]


def concat_source_scales(parts: list[torch.Tensor]) -> torch.Tensor:
    """Lane-concatenate a set of source scales, anchored onto ONE device first.

    The join used by both concat mappers. ``torch.cat`` RAISES on mixed devices,
    and the mixed-device precondition is the documented one: a parameterless
    ``InputMapper`` root contributes a CPU unit scale that meets a perceptron's
    device theta at the first structural join below it. W0.8 closed that only for
    the ComputeOp fan-in (``normalize_fan_in_scales``); the two concat joins are
    the rest of the enumeration, and they share this so there is one anchoring
    rule rather than three copies of it.
    """
    return torch.cat(align_scale_devices(parts))


def broadcast_scale_pair(s_a: torch.Tensor, s_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the shorter scale vector to match the longer one."""
    n_a, n_b = len(s_a), len(s_b)
    if n_a == n_b:
        return s_a, s_b
    if n_a < n_b:
        short, long_, flipped = s_a, s_b, True
    else:
        short, long_, flipped = s_b, s_a, False

    n_s, n_l = len(short), len(long_)
    if n_l % n_s == 0:
        expanded = short.repeat_interleave(n_l // n_s)
    else:
        # Was a bare ``torch.full((n_l,), short.mean().item())``: CPU, default
        # dtype, regardless of where the pair it is being matched against lives.
        expanded = spread_scalar(short.mean(), n_l)

    if flipped:
        return expanded, long_
    return long_, expanded

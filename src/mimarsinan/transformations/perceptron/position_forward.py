"""[U-PFOLD] The network function of a perceptron whose fold is held per position (``position_fold``).

Before any projection a per-position perceptron's function is its raw weight's: the fold only changes the deployed
weights' units, so the forward is the layer's own. Once a projection held each weight row's projected raw weight
(``position_weight``), each position's core computes with its own row, so the network function (the NF the NF and SCM
parity reads, both walks, training) reads each position at its row: forward the projected row, backward the latent
raw weight (the transform trainer's straight-through surrogate). A bias offset per position (``position_bias``) adds
its charge at that position after the normalization, in the activation's units.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from mimarsinan.transformations.pruning.committed_masks import commit_layer_pruning
from mimarsinan.transformations.perceptron.position_fold import (
    PER_INPUT,
    PER_POSITION,
    POSITION_WEIGHT,
    ROW_AXIS,
    along_outputs,
    bias_offsets,
    holds_positions,
    outputs_first,
    position_rows,
    reads_channels_first,
)

__all__ = ["add_position_offsets", "conv_positions", "position_bias_totals", "positional_preactivation",
           "reads_positions"]


def reads_positions(perceptron: Any) -> bool:
    """Whether the forward reads each position at its own row (a projection held rows, or a bias offset is held)."""
    if not holds_positions(perceptron):
        return False
    return getattr(perceptron, POSITION_WEIGHT, None) is not None or bias_offsets(perceptron) is not None


def _straight_through(latent: torch.Tensor, held: torch.Tensor) -> torch.Tensor:
    """The projected value forward, the latent's gradient backward (the transform trainer's surrogate)."""
    return latent + (held - latent).detach()


def _weights_by_row(perceptron: Any) -> tuple[torch.Tensor, torch.Tensor]:
    rows, index = position_rows(perceptron)
    held = getattr(perceptron, POSITION_WEIGHT, None)
    weight = perceptron.layer.weight
    latent = weight.reshape(1, *weight.shape).expand(int(rows.shape[0]), *weight.shape)
    return (latent if held is None else _straight_through(latent, held.to(weight.dtype))), index


def _offsets_as_charge(perceptron: Any, like: torch.Tensor) -> Optional[torch.Tensor]:
    offsets = bias_offsets(perceptron)
    if offsets is None:
        return None
    act = along_outputs(perceptron.activation_scale, PER_POSITION.bias, 2)
    return (offsets * act).to(device=like.device, dtype=like.dtype)


def positional_preactivation(perceptron: Any, x: torch.Tensor) -> torch.Tensor:
    """A linear's pre-activation with each position at its own raw weight and bias offset: ``normalization(W_p x_p +
    b) + offset_p act`` over the input's leading positions (the last axis is the fan-in)."""
    commit_layer_pruning(perceptron.layer)
    weights, index = _weights_by_row(perceptron)
    positions = int(index.numel())
    lead = tuple(x.shape[:-1])
    flat = x.reshape(-1, positions, x.shape[-1])
    out = flat.new_zeros(flat.shape[0], positions, int(weights.shape[PER_POSITION.weight]))
    for r in range(int(weights.shape[ROW_AXIS])):
        at = (index == r).nonzero().reshape(-1)
        row = outputs_first(weights.select(ROW_AXIS, r), PER_INPUT.weight)
        out = out.index_copy(1, at, torch.nn.functional.linear(flat[:, at], row.to(flat.dtype)))
    if perceptron.layer.bias is not None:
        out = out + perceptron.layer.bias.to(out.dtype)
    out = out.reshape(*lead, out.shape[-1])
    if not isinstance(perceptron.normalization, nn.Identity):
        out = perceptron.normalization(out)
    charge = _offsets_as_charge(perceptron, out)
    if charge is not None:
        out = (out.reshape(-1, positions, out.shape[-1]) + charge.movedim(PER_POSITION.bias, -1)).reshape(out.shape)
    return out


def conv_positions(perceptron: Any, x: torch.Tensor, kernel, stride, padding, dilation) -> torch.Tensor:
    """A convolution's output (``(B, out, *out_grid)``, the bias added, before its normalization) with each output
    position at its patch fold's raw weight, over the unfold of its input (the slot order of the weight's flattening,
    the positions row-major, as the mapping's patch gather)."""
    commit_layer_pruning(perceptron.layer)
    one = len(kernel) == 1
    grid = x.unsqueeze(2) if one else x
    pad = lambda v, fill: (fill, int(v[0])) if one else tuple(int(t) for t in v)  # noqa: E731
    patches = torch.nn.functional.unfold(grid, pad(kernel, 1), dilation=pad(dilation, 1), padding=pad(padding, 0),
                                         stride=pad(stride, 1))
    weights, index = _weights_by_row(perceptron)
    out = patches.new_zeros(patches.shape[0], int(weights.shape[PER_POSITION.weight]), patches.shape[2])
    for r in range(int(weights.shape[ROW_AXIS])):
        at = (index == r).nonzero().reshape(-1)
        row = outputs_first(weights.select(ROW_AXIS, r), PER_INPUT.weight)
        out = out.index_copy(2, at, torch.einsum("ok,bkl->bol", row.to(patches.dtype), patches[:, :, at]))
    if perceptron.layer.bias is not None:
        out = out + perceptron.layer.bias.to(out.dtype).reshape(1, -1, 1)
    sizes = [(int(i) + 2 * int(p) - int(d) * (int(k) - 1) - 1) // int(s) + 1
             for i, k, s, p, d in zip(x.shape[2:], kernel, stride, padding, dilation)]
    return out.reshape(out.shape[0], out.shape[1], *sizes)


def add_position_offsets(perceptron: Any, y: torch.Tensor) -> torch.Tensor:
    """A convolution's output (``(B, out, *grid)``) with each output position's bias offset added as charge (unchanged
    without offsets)."""
    charge = _offsets_as_charge(perceptron, y)
    if charge is None:
        return y
    return y + outputs_first(charge, PER_POSITION.bias).reshape(1, *y.shape[1:])


def position_bias_totals(perceptron: Any, bias: Any) -> Any:
    """A pre-activation bias (raw units, per output) with each position's offset charge where positions hold their own
    (the timing activation's resting bias), laid out as the perceptron's output sample: ``(positions, out)`` channels
    last, ``(out, positions)`` a convolution's channels first; else ``bias`` itself."""
    if bias is None:
        return bias
    charge = _offsets_as_charge(perceptron, bias)
    if charge is None:
        return bias
    totals = along_outputs(bias, PER_POSITION.bias, 2) + charge
    return outputs_first(totals, PER_POSITION.bias) if reads_channels_first(perceptron) else totals.movedim(
        PER_POSITION.bias, -1)

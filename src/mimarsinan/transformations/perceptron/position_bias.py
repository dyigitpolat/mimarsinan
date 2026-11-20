"""[U-PFOLD] The writers of a per-position perceptron's bias and weights (``position_fold`` holds its reader).

A bias written from the effective weight (a received shift's correction, its compensation at a projection, an affine
fold into a consumer: ``mapping.support.received_shift``) is corrected per position from that position's effective
weight: the shared bias takes row 0's correction and ``position_bias`` holds each position's offset over it; where every
position's correction is equal no offset is held and the bias keeps today's bytes. A bias transform runs on every
position's total. A per-input-column gain or an output channel's removal commutes with every fold, so it is the raw
weight's (and each projected row's).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from mimarsinan.transformations.perceptron.position_fold import (
    PER_INPUT,
    PER_POSITION,
    POSITION_BIAS,
    POSITION_WEIGHT,
    ROW_AXIS,
    along_outputs,
    bias_offsets,
    effective_bias_rows,
    holds_positions,
)

__all__ = ["add_bias_corrections", "apply_bias_transform", "held_per_position", "holds_positions",
           "scale_input_columns", "weight_charges", "zero_output_channel"]


def _set_offsets(perceptron: Any, offsets: Optional[torch.Tensor]) -> None:
    if offsets is None or not bool((offsets != 0).any()):
        if POSITION_BIAS in perceptron._buffers:
            del perceptron._buffers[POSITION_BIAS]
        return
    perceptron.register_buffer(POSITION_BIAS, offsets.detach().clone(), persistent=False)


def apply_bias_transform(perceptron: Any, transform: Callable[[torch.Tensor], torch.Tensor],
                         write_shared: Callable[[Callable[[torch.Tensor], torch.Tensor]], None],
                         shared: torch.Tensor) -> None:
    """A bias transform on every position's total (``transform`` on ``(positions, out)``, broadcasting over the
    positions as it does over one row): row 0's result is written as the shared bias (``write_shared``, the
    per-input writer) and every other position's as its offset over it (none where all are equal)."""
    totals = transform(effective_bias_rows(perceptron, shared))
    first = totals.select(ROW_AXIS, 0)
    write_shared(lambda _b, v=first: v.to(dtype=_b.dtype, device=_b.device))
    _set_offsets(perceptron, totals - along_outputs(first, PER_POSITION.bias, totals.dim()))


def held_per_position(perceptron: Any, stamp: torch.Tensor) -> torch.Tensor:
    """[U-RSHIFT] The stamp a perceptron keeps: where its positions hold their own bias, one row per position (a stamp
    per input repeated on each), so every reader reads it per position; else ``stamp`` as written."""
    offsets = bias_offsets(perceptron)
    if offsets is None or stamp.dim() != 1:
        return stamp
    return stamp.reshape(1, -1).expand(int(offsets.shape[ROW_AXIS]), -1).clone()


def weight_charges(weights: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """[U-PFOLD r2] Each position's charge per output neuron from its weights ``(positions, out, in)`` and the slots it
    reads (``shift_rows``): the fan-in (the last axis) summed, the output neuron on the bias view's axis."""
    return (weights * slots).sum(dim=-1).movedim(PER_POSITION.weight, PER_POSITION.bias)


def add_bias_corrections(perceptron: Any, corrections: torch.Tensor,
                         write_shared: Callable[[Callable[[torch.Tensor], torch.Tensor]], None]) -> None:
    """``B_p <- B_p - c_p`` per position (``corrections`` ``(positions, out)``, or ``(out,)`` for every position): the
    shared bias takes row 0's correction and each position's offset the rest; where every position's correction is
    equal the offsets stay as they were (today's bytes for a perceptron without any)."""
    rows = corrections if corrections.dim() > 1 else along_outputs(corrections, PER_POSITION.bias, 2)
    first = rows.select(ROW_AXIS, 0)
    write_shared(lambda b, c=first: b - c.to(dtype=b.dtype, device=b.device))
    rest = rows - along_outputs(first, PER_POSITION.bias, rows.dim())
    if rows.shape[ROW_AXIS] == 1 or not bool((rest != 0).any()):
        return
    held = bias_offsets(perceptron)
    _set_offsets(perceptron, (-rest) if held is None else held.to(rest.dtype) - rest)


def scale_input_columns(perceptron: Any, gains: torch.Tensor) -> None:
    """``W~[..., j] *= a_j`` on every position: a per-input-column gain commutes with every fold, so it is the raw
    weight's (and each projected row's) column gain."""
    with torch.no_grad():
        perceptron.layer.weight.data.mul_(gains.reshape(1, -1).to(perceptron.layer.weight.dtype))
        held = getattr(perceptron, POSITION_WEIGHT, None)
        if held is not None:
            held.mul_(gains.reshape(1, 1, -1).to(held.dtype))


def zero_output_channel(perceptron: Any, channel: int) -> None:
    """Zero output channel ``channel``'s weights on every position (an output channel is no input fold's)."""
    with torch.no_grad():
        perceptron.layer.weight.data.select(PER_INPUT.weight, channel).zero_()
        held = getattr(perceptron, POSITION_WEIGHT, None)
        if held is not None:
            held.select(PER_POSITION.weight, channel).zero_()

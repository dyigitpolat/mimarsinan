"""[U-PFOLD] The fold per core position: the one reader of a perceptron's input fold in its two forms.

A deployed core folds each input at the gauge of the producer of the element it reads (``per_input_scales``): per input
(today's form and bytes, ``(in_features,)``) where one fold per input holds every element so, else one row per core
position, ``(positions, in_features)`` (a linear's leading positions, each its own core; a convolution's output
positions, each a core reading one patch; ``fold_slot_source`` names the element each slot reads, ``-1`` a padding
slot). A projection on a grid rounds each position's weight on its own, so it holds each weight row's projected raw
weight (``position_weight``, ``(rows, out, in)``: a linear one per position, a convolution one per distinct patch fold)
beside the raw weight, which stays the latent; a bias written from the effective weight holds each position's offset
over the shared bias (``position_bias``, effective units) where the corrections differ. ``PerceptronTransformer``'s
per-input readers refuse a per-position perceptron by name (``PositionFoldError``): no reader can read it silently.
[U-PFOLD r2] The reader declares, for every view it hands out, the axis of its output neuron (``OutputAxes``: the
forms ``PER_INPUT`` and ``PER_POSITION``, a perceptron's views by ``output_axes``); every reader that works per output
neuron takes that axis from here (``along_outputs``, ``output_rows``, ``outputs_first``), never from a literal index.
[U-PFOLD r3] A reader outside this module takes the rows of a fold per position, or their count, by ``position_folds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.layers import norm_affine_params

__all__ = [
    "PER_INPUT", "PER_POSITION", "POSITION_BIAS", "POSITION_WEIGHT", "ROW_AXIS", "SLOT_SOURCE", "OutputAxes",
    "PositionFoldError", "along_outputs", "apply_weight_projection", "assert_one_fold_per_core", "bias_offsets",
    "copy_position_state", "effective_bias_rows", "effective_parameters", "effective_weight_rows", "fuse_output_scale",
    "holds_positions", "input_fold_view", "output_axes", "output_rows", "outputs_first", "position_folds",
    "position_rows", "reads_channels_first", "refuse_per_input_read", "shift_rows",
]

POSITION_WEIGHT = "position_weight"
"""The raw weight of each weight row a projection wrote (``(rows, out, in)``), a non-persistent buffer."""

POSITION_BIAS = "position_bias"
"""Each position's effective-bias offset over the shared bias (``(positions, out)``), a non-persistent buffer."""

SLOT_SOURCE = "fold_slot_source"
"""Per position and slot, the flat index of the element it reads in the per-sample input (``-1``: padding)."""

_DEGENERATE_U_EPS = 1e-6

ROW_AXIS = 0
"""The axis of a per-position view that holds its weight rows or its positions."""


@dataclass(frozen=True)
class OutputAxes:
    """The output neuron's axis in a reader's weight view (its effective weight, and what its weight transform
    receives) and in its bias view (its effective bias, and what its bias transform receives)."""

    weight: int
    bias: int


PER_INPUT = OutputAxes(weight=0, bias=0)
"""The per-input form: a weight ``(out, in, ...)``, a bias ``(out,)``; a weight row taken on ``ROW_AXIS`` is in it."""

PER_POSITION = OutputAxes(weight=1, bias=1)
"""The per-position form: weight rows ``(rows, out, in)`` (or one per position), bias rows ``(positions, out)``."""


def output_axes(perceptron: Any, positions: bool) -> OutputAxes:
    """The axes of the views a reader (``positions``: the reader in both forms) hands out for ``perceptron``: the weight
    view per position where its fold is held per position and the reader reads positions; the bias view there and
    wherever positions hold their own bias (a bias transform then runs on every position's); else per input."""
    rows = bool(positions) and holds_positions(perceptron)
    return OutputAxes(weight=(PER_POSITION if rows else PER_INPUT).weight,
                      bias=(PER_POSITION if rows or bias_offsets(perceptron) is not None else PER_INPUT).bias)


def along_outputs(values: Any, axis: int, rank: int) -> Any:
    """A per-output ``values`` viewed to broadcast against a view of ``rank`` axes whose output neuron is ``axis``."""
    if not torch.is_tensor(values) or values.dim() == 0:
        return values
    return values.reshape(*([1] * axis), -1, *([1] * (rank - axis - 1)))


def output_rows(view: torch.Tensor, axis: int) -> torch.Tensor:
    """``view`` as one row per output neuron (``axis``), ``(out, rest)``: a reduction per output neuron reads it."""
    return view.movedim(axis, 0).reshape(int(view.shape[axis]), -1)


def outputs_first(view: torch.Tensor, axis: int) -> torch.Tensor:
    """``view`` with its output neuron (``axis``) first, the layout torch's linear and the IR's weight banks read."""
    return view.movedim(axis, 0)


class PositionFoldError(ToolchainRefusal, ValueError):
    """A per-position fold read through a per-input reader, or positions of different folds sharing one core."""


def holds_positions(perceptron: Any) -> bool:
    """Whether the stamp is the per-position form (one row per core position)."""
    stamp = getattr(perceptron, "per_input_scales", None)
    return stamp is not None and stamp.dim() == 2


def reads_channels_first(perceptron: Any) -> bool:
    """A convolution's perceptron (a linear over its patches, channels-first output)."""
    return int(getattr(perceptron, "output_channel_axis", -1)) == 1


def refuse_per_input_read(perceptron: Any, what: str) -> None:
    """The per-input readers' guard: a per-position perceptron is read through this module only."""
    if holds_positions(perceptron) or bias_offsets(perceptron) is not None:
        raise PositionFoldError(
            f"{getattr(perceptron, 'name', None)!r}: its fold is held per core position, so no one {what} holds it; "
            f"its readers take it per position (transformations.perceptron.position_fold)")


def position_rows(perceptron: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, index)``: the fold rows a weight is held for and each position's row, on the weight's device. A linear
    holds one per position (``index`` the identity); a convolution one per distinct patch fold (its positions share one
    where their patches' folds are equal)."""
    stamp = perceptron.per_input_scales.to(perceptron.layer.weight.device)
    if not reads_channels_first(perceptron):
        return stamp, torch.arange(stamp.shape[0], device=stamp.device)
    rows, index = torch.unique(stamp, dim=0, return_inverse=True)
    return rows, index.reshape(-1)


def position_folds(perceptron: Any) -> torch.Tensor:
    """[U-PFOLD r3] Each core position's fold row, ``(positions, in_features)``, on the weight's device (``rows[index]``
    of ``position_rows``): the one read of a fold per position's rows and of their count outside this module."""
    rows, index = position_rows(perceptron)
    return rows[index]


def _u_and_act(perceptron: Any, like: torch.Tensor) -> tuple[Optional[torch.Tensor], Any]:
    """The normalization's fold factor per output (None for no normalization) and the activation scale, both
    broadcastable against the weight rows ``like`` along their output axis."""
    act = perceptron.activation_scale
    out = int(perceptron.layer.weight.shape[PER_INPUT.weight])
    if torch.is_tensor(act) and act.dim() >= 1 and act.numel() == out:
        act = along_outputs(act, PER_POSITION.weight, like.dim()).to(like.device)
    if isinstance(perceptron.normalization, nn.Identity):
        return None, act
    u, _beta, _mean = norm_affine_params(perceptron.normalization)
    return along_outputs(u.detach(), PER_POSITION.weight, like.dim()).to(like.device), act


def _raw_rows(perceptron: Any, n_rows: int) -> torch.Tensor:
    held = getattr(perceptron, POSITION_WEIGHT, None)
    if held is not None and int(held.shape[0]) == n_rows:
        return held
    weight = perceptron.layer.weight.data
    return weight.reshape(1, *weight.shape).expand(n_rows, *weight.shape)


def effective_weight_rows(perceptron: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights (rows, out, in), index)``: each weight row's effective weight, ``fold * W * u / act`` with ``W`` the
    row's raw weight (the projected one where a projection wrote it), and each position's row."""
    rows, index = position_rows(perceptron)
    raw = _raw_rows(perceptron, int(rows.shape[0]))
    u, act = _u_and_act(perceptron, raw)
    folded = rows.unsqueeze(PER_POSITION.weight)  # each row's fold, alike for every output neuron
    weights = folded * raw / act if u is None else folded * (raw * u) / act
    return weights, index


def effective_parameters(perceptron: Any, transformer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``(effective weight, effective bias)``: the one pair (``transformer``'s), or over a fold per position every
    position's, ``(positions, out, in)`` and ``(positions, out)``: the views ``output_axes(perceptron, True)`` names."""
    if not holds_positions(perceptron):
        return transformer.get_effective_weight(perceptron), transformer.get_effective_bias(perceptron)
    rows, index = effective_weight_rows(perceptron)
    return rows[index], effective_bias_rows(perceptron, transformer.shared_effective_bias(perceptron))


def apply_weight_projection(perceptron: Any, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """A projection of every weight row's effective weight (``transform`` on ``(rows, out, in)``) written as each row's
    raw weight (``position_weight``), inverted per row as ``PerceptronTransformer.apply_effective_weight_transform``
    inverts the one weight, from the weights installed (the rows a projection holds, else the raw weight, which stays
    the trained latent); a dead normalization channel keeps its raw weight, a pruned entry stays zero."""
    rows, _index = position_rows(perceptron)
    raw = _raw_rows(perceptron, int(rows.shape[0]))
    u, act = _u_and_act(perceptron, raw)
    folded = rows.unsqueeze(PER_POSITION.weight)
    target = transform(folded * raw / act if u is None else folded * (raw * u) / act)
    if u is None:
        new = (target * act) / folded
    else:
        dead = u.abs() < _DEGENERATE_U_EPS
        new = torch.where(dead, raw, ((target * act) / folded) / torch.where(dead, torch.ones_like(u), u))
    new = new.detach().clone()
    mask = getattr(perceptron.layer, "prune_mask", None)
    if mask is not None:
        new.masked_fill_(mask.unsqueeze(ROW_AXIS), 0.0)  # the raw weight's mask on every row
    if not bool(torch.isfinite(new).all()):
        raise RuntimeError(f"a per-position projection wrote non-finite weights into {perceptron.name!r}")
    perceptron.register_buffer(POSITION_WEIGHT, new, persistent=False)


def fuse_output_scale(perceptron: Any, factor: torch.Tensor) -> None:
    """Each projected row's raw weight scaled per output channel as the raw weight is (normalization fusion)."""
    held = getattr(perceptron, POSITION_WEIGHT, None)
    if held is not None:
        with torch.no_grad():
            held.mul_(along_outputs(factor, PER_POSITION.weight, held.dim()).to(held.dtype))


def bias_offsets(perceptron: Any) -> Optional[torch.Tensor]:
    """Each position's effective-bias offset over the shared bias, or None where every position reads the shared one."""
    return getattr(perceptron, POSITION_BIAS, None)


def effective_bias_rows(perceptron: Any, shared: torch.Tensor) -> torch.Tensor:
    """Each position's effective bias, ``(positions, out)``: the shared effective bias plus the position's offset."""
    offsets = bias_offsets(perceptron)
    row = along_outputs(shared, PER_POSITION.bias, 2)  # the shared bias as one position's
    if offsets is None:
        n = int(perceptron.per_input_scales.shape[ROW_AXIS]) if holds_positions(perceptron) else 1
        return row.expand(*[n if axis == ROW_AXIS else -1 for axis in range(row.dim())])
    return row + offsets.to(device=shared.device, dtype=shared.dtype)


def copy_position_state(source: Any, target: Any) -> None:
    """The per-position state a transform wrote on ``source`` (its projected rows and bias offsets) onto ``target``."""
    for name in (POSITION_WEIGHT, POSITION_BIAS):
        held = getattr(source, name, None)
        if held is None:
            if name in target._buffers:
                del target._buffers[name]
            continue
        current = getattr(target, name, None)
        if current is not None and current.shape == held.shape:
            current.data.copy_(held.data)
        else:
            target.register_buffer(name, held.detach().clone().to(target.layer.weight.device), persistent=False)


def _slot_source(perceptron: Any) -> tuple[np.ndarray, int]:
    """``(table, elements)``: per position and slot the flat index of the element it reads in the per-sample input
    (``-1`` a padding slot), and that input's element count. A linear's rows are its input's leading positions in C
    order (its last axis the fan-in); a convolution's table is its stamp writer's (its patch gather)."""
    held = perceptron.__dict__.get(SLOT_SOURCE)
    if held is not None:
        return np.asarray(held[0]), int(held[1])
    if reads_channels_first(perceptron):
        raise PositionFoldError(
            f"{getattr(perceptron, 'name', None)!r}: a convolution's fold per position names no element of its input "
            f"(no {SLOT_SOURCE}): its stamp is written by the stamping walk only")
    rows, width = (int(d) for d in perceptron.per_input_scales.shape)
    return np.arange(rows * width).reshape(rows, width), rows * width


def input_fold_view(perceptron: Any, like: Any) -> torch.Tensor:
    """The per-position fold as it multiplies the perceptron's batched input ``like`` (``(B, *sample)``, or a train of
    such): each element at the fold of the core position that reads it (an element no slot reads at one)."""
    table, elements = _slot_source(perceptron)
    rows = perceptron.per_input_scales.detach()
    shape = list(like.shape)
    sample: list = []
    size = 1
    while shape and size < elements:
        sample.insert(0, int(shape.pop()))
        size *= sample[0]
    if size != elements:
        raise PositionFoldError(
            f"{getattr(perceptron, 'name', None)!r}: an input of shape {tuple(like.shape)} holds no sample of the "
            f"{elements} elements its fold per position reads")
    grid = torch.ones(size, dtype=rows.dtype, device=rows.device)
    used = torch.as_tensor(table >= 0, device=rows.device)
    grid[torch.as_tensor(table, device=rows.device)[used]] = rows[used]
    return grid.reshape(*sample).to(device=like.device, dtype=like.dtype)


def shift_rows(perceptron: Any, shift: torch.Tensor) -> torch.Tensor:
    """An input shift (one value, one per input, or one per element of the per-sample input) as each position's slots
    read it, ``(positions, 1, in_features)`` against every position's weight: a padding slot reads none of it."""
    table, elements = _slot_source(perceptron)
    index = torch.as_tensor(table, device=shift.device)
    flat = shift.reshape(-1)
    if flat.numel() == elements and elements != int(table.shape[1]):
        by_element = torch.cat([flat, flat.new_zeros(1)])
        return by_element[index].unsqueeze(PER_POSITION.weight)
    if flat.numel() not in (1, int(table.shape[1])):
        raise PositionFoldError(f"{getattr(perceptron, 'name', None)!r}: a shift of {flat.numel()} values over "
                                f"{int(table.shape[1])} inputs and {elements} elements")
    rows = flat.reshape(1, -1).expand(int(table.shape[0]), int(table.shape[1]))
    return (rows * (index >= 0).to(flat.dtype)).unsqueeze(PER_POSITION.weight)


def assert_one_fold_per_core(name: Any, folds: torch.Tensor) -> None:
    """The refusal kept where positions share one core: every position a core (or a bank's cores) serves must hold one
    fold. The mappers give each distinct row its own bank, so no program they build reaches it."""
    if folds.shape[0] > 1 and not bool((folds == folds[:1]).all()):
        raise PositionFoldError(
            f"{name!r}: positions of {int(torch.unique(folds, dim=0).shape[0])} different folds would share one core, "
            f"which folds every position it serves alike")

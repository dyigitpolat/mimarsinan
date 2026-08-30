"""The convolution mappers' unfold: which input cells fill each position's slot table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mimarsinan.mapping.platform.slot_unfold import GatheredSlotUnfold


@dataclass(frozen=True)
class ConvPatchGather:
    """One convolution's unfold, computed ONCE and consumed twice.

    ``patch_index`` indexes the PADDED source grid and is what the IR mapping
    gathers its ``IRSource`` patches with. ``slot_source_index`` is the SAME
    table over the unpadded grid, flattened to ``(positions, slots)`` with
    ``-1`` marking a padding (OFF) slot — what the NF event-serial twin gathers
    upstream event multiplicities with. One computation, so the twin cannot
    fold a different order than the deployment.
    """

    in_channels: int
    source_grid: tuple[int, ...]
    out_grid: tuple[int, ...]
    kernel: tuple[int, ...]
    padding: tuple[int, ...]
    patch_index: tuple[np.ndarray, ...]
    slot_source_index: np.ndarray

    @property
    def n_positions(self) -> int:
        return int(np.prod(self.out_grid))

    @property
    def patch_size(self) -> int:
        return int(self.in_channels * np.prod(self.kernel))

    @property
    def source_size(self) -> int:
        return int(self.in_channels * np.prod(self.source_grid))

    @property
    def pad_width(self) -> tuple[tuple[int, int], ...]:
        """``np.pad`` widths for the (channel, *spatial) source grid."""
        return ((0, 0),) + tuple((p, p) for p in self.padding)

    @property
    def pads_anything(self) -> bool:
        return any(p > 0 for p in self.padding)

    def gather(self, padded_sources) -> np.ndarray:
        """The ``(positions, slots)`` patch table over an ALREADY padded grid."""
        return padded_sources[self.patch_index].reshape(
            self.n_positions, self.patch_size
        )


def _spatial(value, rank: int) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),) * rank


def conv_patch_gather(
    *,
    in_channels: int,
    source_grid: Sequence[int],
    kernel,
    stride,
    padding,
    dilation,
) -> ConvPatchGather:
    """Resolve a convolution's unfold over an ``(in_channels, *source_grid)`` input.

    Slot order within a patch is the perceptron's own input-feature order:
    channel-major, then the kernel taps in row-major order — the order
    ``nn.Conv2d``'s weight view flattens to, which is what makes the shared
    weight bank's columns and the core's axon slots the same thing.
    """
    grid = tuple(int(s) for s in source_grid)
    rank = len(grid)
    kernel_t = _spatial(kernel, rank)
    stride_t = _spatial(stride, rank)
    padding_t = _spatial(padding, rank)
    dilation_t = _spatial(dilation, rank)
    if not (len(kernel_t) == len(stride_t) == len(padding_t) == len(dilation_t) == rank):
        raise ValueError(
            f"conv_patch_gather: kernel/stride/padding/dilation must all have "
            f"rank {rank}; got {kernel_t}, {stride_t}, {padding_t}, {dilation_t}"
        )
    channels = int(in_channels)
    out_grid = tuple(
        (grid[i] + 2 * padding_t[i] - dilation_t[i] * (kernel_t[i] - 1) - 1)
        // stride_t[i] + 1
        for i in range(rank)
    )
    if any(o <= 0 for o in out_grid):
        raise ValueError(
            f"conv_patch_gather: kernel {kernel_t} / stride {stride_t} / padding "
            f"{padding_t} / dilation {dilation_t} leave no output positions on a "
            f"{grid} grid"
        )

    full_rank = 2 * rank + 1
    channel_shape = [1] * full_rank
    channel_shape[rank] = channels
    padded_axes = []
    for axis in range(rank):
        position_shape = [1] * full_rank
        position_shape[axis] = out_grid[axis]
        tap_shape = [1] * full_rank
        tap_shape[rank + 1 + axis] = kernel_t[axis]
        padded_axes.append(
            (np.arange(out_grid[axis]) * stride_t[axis]).reshape(position_shape)
            + (np.arange(kernel_t[axis]) * dilation_t[axis]).reshape(tap_shape)
        )
    broadcast = np.broadcast_arrays(
        np.arange(channels).reshape(channel_shape), *padded_axes
    )
    patch_index = tuple(broadcast)

    n_positions = int(np.prod(out_grid))
    patch_size = int(channels * np.prod(kernel_t))
    channel_index = patch_index[0]
    flat = channel_index.astype(np.int64)
    inside = np.ones(channel_index.shape, dtype=bool)
    for axis in range(rank):
        unpadded = patch_index[1 + axis] - padding_t[axis]
        inside &= (unpadded >= 0) & (unpadded < grid[axis])
        flat = flat * grid[axis] + np.clip(unpadded, 0, grid[axis] - 1)
    slot_source_index = np.where(inside, flat, -1).reshape(n_positions, patch_size)

    return ConvPatchGather(
        in_channels=channels,
        source_grid=grid,
        out_grid=out_grid,
        kernel=kernel_t,
        padding=padding_t,
        patch_index=patch_index,
        slot_source_index=slot_source_index,
    )


def conv_slot_unfold(gather: ConvPatchGather) -> GatheredSlotUnfold:
    """The NF event-serial twin's view of ``gather`` — the mapper's own table."""
    return GatheredSlotUnfold(
        slot_source_index=gather.slot_source_index,
        source_size=gather.source_size,
        output_grid=gather.out_grid,
    )


__all__ = ["ConvPatchGather", "conv_patch_gather", "conv_slot_unfold"]

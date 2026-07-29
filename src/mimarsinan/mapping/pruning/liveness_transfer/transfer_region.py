"""REGION_REDUCE derivation: avg/max pool receptive fields from module params.

Forward exactness: the avg/max of an ALL-eliminated (constant-zero) region is
exactly 0 (AvgPool padding contributes zeros; MaxPool pads with -inf, which
never wins against 0), so an output whose entire receptive field is dead is
itself dead. Backward is pure use-analysis. Any underivable geometry —
missing shapes, channel mismatch, empty windows, exotic params — falls to
OPAQUE, never an error.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Tuple

import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    OPAQUE_TRANSFER,
    TRANSFER_REGION_REDUCE,
    LivenessTransfer,
    _flat_size,
    _is_zero_preserving_or_opaque,
    _module_of,
    _relation_transfer,
)

__all__ = ["REGION_REDUCE_HOST_OP_TYPES", "derive_region_reduce"]

REGION_REDUCE_HOST_OP_TYPES: FrozenSet[type] = frozenset({
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.MaxPool1d,
    nn.MaxPool2d,
})


def _int_tuple(value: Any, rank: int) -> Tuple[int, ...] | None:
    if isinstance(value, (tuple, list)):
        vals = tuple(int(v) for v in value)
        return vals if len(vals) == rank else None
    try:
        return (int(value),) * rank
    except (TypeError, ValueError):
        return None


def _axis_windows(
    size: int, out_size: int, kernel: int, stride: int, pad: int, dilation: int
) -> List[List[int]] | None:
    """Receptive positions per output along one spatial axis; None if any
    window is empty (geometry inconsistent with the recorded shapes)."""
    if stride <= 0 or kernel <= 0 or dilation <= 0:
        return None
    windows: List[List[int]] = []
    for o in range(out_size):
        start = o * stride - pad
        valid = [
            p for p in (start + d * dilation for d in range(kernel))
            if 0 <= p < size
        ]
        if not valid:
            return None
        windows.append(valid)
    return windows


def derive_region_reduce(op: ComputeOp, n_in: int) -> LivenessTransfer:
    module = _module_of(op)
    input_shape = (getattr(op, "params", None) or {}).get(
        "input_shape", getattr(op, "input_shape", None)
    )
    output_shape = getattr(op, "output_shape", None)
    if module is None or input_shape is None or output_shape is None:
        return OPAQUE_TRANSFER
    if len(input_shape) != len(output_shape):
        return OPAQUE_TRANSFER
    rank = len(input_shape) - 1  # spatial rank; leading dim is channels
    if rank not in (1, 2):
        return OPAQUE_TRANSFER
    channels = int(input_shape[0])
    if int(output_shape[0]) != channels:
        return OPAQUE_TRANSFER
    if _flat_size(input_shape) != n_in:
        return OPAQUE_TRANSFER
    if not _is_zero_preserving_or_opaque(op):
        return OPAQUE_TRANSFER

    kernel = _int_tuple(getattr(module, "kernel_size", None), rank)
    stride_attr = getattr(module, "stride", None)
    stride = _int_tuple(
        stride_attr if stride_attr is not None else module.kernel_size, rank
    )
    padding = _int_tuple(getattr(module, "padding", 0), rank)
    dilation = _int_tuple(getattr(module, "dilation", 1), rank)
    if kernel is None or stride is None or padding is None or dilation is None:
        return OPAQUE_TRANSFER

    axis_windows: List[List[List[int]]] = []
    for axis in range(rank):
        windows = _axis_windows(
            size=int(input_shape[1 + axis]),
            out_size=int(output_shape[1 + axis]),
            kernel=kernel[axis],
            stride=stride[axis],
            pad=padding[axis],
            dilation=dilation[axis],
        )
        if windows is None:
            return OPAQUE_TRANSFER
        axis_windows.append(windows)

    in_spatial = [int(d) for d in input_shape[1:]]
    out_spatial = [int(d) for d in output_shape[1:]]
    in_plane = 1
    for d in in_spatial:
        in_plane *= d
    out_plane = 1
    for d in out_spatial:
        out_plane *= d

    out_to_ins: Dict[int, FrozenSet[int]] = {}
    for c in range(channels):
        if rank == 1:
            for o, window in enumerate(axis_windows[0]):
                out_to_ins[c * out_plane + o] = frozenset(
                    c * in_plane + p for p in window
                )
        else:
            width = in_spatial[1]
            out_width = out_spatial[1]
            for oy, wy in enumerate(axis_windows[0]):
                for ox, wx in enumerate(axis_windows[1]):
                    out_to_ins[c * out_plane + oy * out_width + ox] = frozenset(
                        c * in_plane + y * width + x for y in wy for x in wx
                    )
    return _relation_transfer(TRANSFER_REGION_REDUCE, out_to_ins, n_in)

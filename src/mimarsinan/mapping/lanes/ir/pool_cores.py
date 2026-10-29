"""Reduction cores of a pool lane (WI-5e): 0/1 weights over each window, ``theta`` the window size.

The core's accumulator is ``Σ_p q_p`` over its window's codes, so its law ``rnd_s(M0 Σ q + B)`` with
``M0 = rnd(2^s Δin / (N Δout))`` requantizes the average onto the pool lane's grid with one rounding. Windows are
grouped into cores within the declared core geometry; zero padding is an absent row (it adds 0 and counts in N).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import ComputeOp, IRSource, NeuralCore
from mimarsinan.mapping.lanes.forms import PoolForm, pool_form


#: A pool lane's reduction core carries its op's FX node name (the layer-to-type rule reads it, WI-5g).
LANE_SOURCE_NAME = "lane_source_name"


class PoolLaneError(ToolchainRefusal, ValueError):
    """A planned pool lane whose IR op has no fixed window (an adaptive size that does not divide its input)."""


def pool_windows(form: PoolForm, shape: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], List[List[int]]]:
    """``(output shape, per output (flat, channel-major) the flat input indices of its window)``."""
    channels, height, width = (int(d) for d in shape)
    (kh, kw), (sh, sw), (ph, pw) = form.kernel, form.stride, form.padding
    out_h, out_w = (height + 2 * ph - kh) // sh + 1, (width + 2 * pw - kw) // sw + 1
    windows = []
    for c in range(channels):
        for i in range(out_h):
            for j in range(out_w):
                window = [c * height * width + y * width + x
                          for y in range(i * sh - ph, i * sh - ph + kh) if 0 <= y < height
                          for x in range(j * sw - pw, j * sw - pw + kw) if 0 <= x < width]
                windows.append(window)
    return (channels, out_h, out_w), windows


def window_groups(windows: List[List[int]], max_axons: Optional[int], max_neurons: Optional[int]) -> List[List[int]]:
    """Outputs per core in order: a core grows while its union of rows and its columns fit the geometry."""
    groups: List[List[int]] = []
    current: List[int] = []
    rows: set = set()
    for index, window in enumerate(windows):
        grown = rows | set(window)
        full = (max_axons is not None and current and len(grown) > max_axons) or (
            max_neurons is not None and len(current) >= max_neurons)
        if full:
            groups.append(current)
            current, grown = [], set(window)
        current.append(index)
        rows = grown
    if current:
        groups.append(current)
    return groups


def window_rows(windows: List[List[int]], max_axons: Optional[int],
                max_neurons: Optional[int]) -> List[Tuple[List[int], List[int]]]:
    """[U-L2] Per core its outputs and its rows (its windows' inputs, sorted), as both planes state them."""
    return [(members, sorted({k for m in members for k in windows[m]}))
            for members in window_groups(windows, max_axons, max_neurons)]


#: The attribute a reduction core carries: per column, the flat (channel-major) index of the pool output it writes.
LANE_OUTPUTS = "lane_outputs"


def pool_lane_cores(op: ComputeOp, sources: np.ndarray, *, owner: int, first_id: int, max_axons: Optional[int],
                    max_neurons: Optional[int], dtype) -> Tuple[List[NeuralCore], np.ndarray, PoolForm]:
    """The reduction cores of ``op`` reading ``sources`` (its input elements), the source of each output, and the
    window."""
    shape = tuple(op.input_shape or ())
    form = pool_form(op.params.get("module"), shape)  # pyright: ignore[reportArgumentType] -- None refused below
    if form is None or len(shape) != 3:
        raise PoolLaneError(f"pool lane {op.name!r} over input {shape}: no fixed window with a constant divisor")
    out_shape, windows = pool_windows(form, shape)  # pyright: ignore[reportArgumentType] -- checked rank 3
    # An off input (a pruned producer column) adds 0, as zero padding does: no row reads it (N still counts it).
    windows = [[k for k in window if not sources[k].is_off()] for window in windows]
    if op.output_shape is not None and tuple(int(d) for d in op.output_shape) != out_shape:
        raise PoolLaneError(f"pool lane {op.name!r}: windows give {out_shape}, the op writes {op.output_shape}")
    cores: List[NeuralCore] = []
    outputs = np.empty(len(windows), dtype=object)
    for g, (members, rows) in enumerate(window_rows(windows, max_axons, max_neurons)):
        position = {k: r for r, k in enumerate(rows)}
        matrix = np.zeros((len(rows), len(members)), dtype=dtype)
        for column, m in enumerate(members):
            for k in windows[m]:
                matrix[position[k], column] = 1
            outputs[m] = IRSource(node_id=first_id + g, index=column)
        core = NeuralCore(
            id=first_id + g, name=f"{op.name}_lane{g}", input_sources=np.array([sources[k] for k in rows],
                                                                              dtype=object),
            core_matrix=matrix, threshold=float(form.divisor), hardware_bias=None,
        )
        setattr(core, LANE_OUTPUTS, np.asarray(members, dtype=np.int64))
        setattr(core, LANE_SOURCE_NAME, str(op.name))
        cores.append(core)
    return cores, outputs, form

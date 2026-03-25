"""Shared structural decision helpers for mapping backends.

Both ``LayoutIRMapping`` (shape-only) and ``IRMapping`` (full IR) call these
pure functions so that tiling mode, bias-axon counting, and psum parameters
are computed identically.  No mapping state is accessed or mutated here.
"""

from __future__ import annotations

import math
from typing import List, Literal, NamedTuple, Tuple


# ---------------------------------------------------------------------------
# Bias axon counting
# ---------------------------------------------------------------------------

def compute_core_input_count(
    n_sources: int,
    has_bias: bool,
    hardware_bias: bool,
) -> int:
    """Return the effective axon count for a core.

    When ``hardware_bias`` is True the bias lives in a dedicated register and
    does **not** consume an axon slot.  In legacy mode (``hardware_bias=False``)
    the bias occupies one extra always-on axon row.
    """
    if has_bias and not hardware_bias:
        return n_sources + 1
    return n_sources


# ---------------------------------------------------------------------------
# FC tiling mode
# ---------------------------------------------------------------------------

TilingMode = Literal["single", "coalescing", "psum", "output_tiled"]


def compute_fc_tiling_mode(
    in_features: int,
    out_features: int,
    max_axons: int | None,
    max_neurons: int | None,
    has_bias: bool,
    hardware_bias: bool,
    allow_coalescing: bool,
) -> TilingMode:
    """Decide which FC mapping path to use.

    The wide-layer test uses the effective axon count (including legacy bias
    axon when ``hardware_bias`` is False) so that both backends agree on when
    a layer exceeds ``max_axons``.
    """
    effective_in = compute_core_input_count(in_features, has_bias, hardware_bias)
    is_wide = max_axons is not None and effective_in > max_axons
    if is_wide:
        return "coalescing" if allow_coalescing else "psum"
    if max_neurons is not None and out_features > max_neurons:
        return "output_tiled"
    return "single"


# ---------------------------------------------------------------------------
# Psum parameters
# ---------------------------------------------------------------------------

class PsumParams(NamedTuple):
    """Pre-computed structural parameters for a psum-decomposed FC layer."""
    tile_count: int
    tile_slices: List[Tuple[int, int]]
    out_block_size: int
    accum_bias_axons: int


def compute_psum_params(
    in_features: int,
    out_features: int,
    max_axons: int,
    max_neurons: int | None,
    has_bias: bool,
    hardware_bias: bool,
) -> PsumParams:
    """Compute the tiling parameters for a psum-decomposed FC layer.

    Both ``LayoutIRMapping._map_fc_psum_layout`` and
    ``IRMapping._map_fc_with_psum`` must use identical values for
    ``tile_count``, ``tile_slices``, ``out_block_size``, and
    ``accum_bias_axons`` so the resulting core counts match.
    """
    eff_max_neurons = int(max_neurons) if max_neurons is not None else out_features

    tile_slices: List[Tuple[int, int]] = []
    start = 0
    while start < in_features:
        end = min(in_features, start + max_axons)
        tile_slices.append((start, end))
        start = end
    tile_count = len(tile_slices)

    accum_bias_axons = 1 if (has_bias and not hardware_bias) else 0
    max_out_by_accum = (max_axons - accum_bias_axons) // (2 * tile_count)
    if max_out_by_accum <= 0:
        raise ValueError(
            f"Cannot build psum accumulator: tile_count={tile_count} "
            f"requires at least {2 * tile_count + accum_bias_axons} axons, "
            f"but max_axons={max_axons}."
        )
    out_block_size = min(eff_max_neurons, max_out_by_accum)

    return PsumParams(
        tile_count=tile_count,
        tile_slices=tile_slices,
        out_block_size=out_block_size,
        accum_bias_axons=accum_bias_axons,
    )

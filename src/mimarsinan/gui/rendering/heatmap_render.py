"""Pure-numpy heatmap rendering: mask-aware decimation, square cells, zero margins.

A heatmap render is a pure function of ``(matrix, masks, scale, target)`` --
no pyplot, no figure state, thread-safe by construction. The output pixel grid
is the matrix grid divided by ONE integer pooling factor for both axes (square
cells; exact integer multiple whenever the dims divide), and carries zero
margins: PNG pixel ``[0, 0]`` is matrix cell ``[0, 0]``, which is what lets the
frontend position percentage overlays directly against the image box.
"""

from __future__ import annotations

import base64
from typing import Sequence

import numpy as np

from mimarsinan.gui.rendering.colormap import (
    BACKGROUND_RGB,
    DIVERGING_LUT,
    GRID_ACCENT_RGB,
    MASK_RGB,
    lut_index,
    render_colorbar_rgb,
)
from mimarsinan.gui.rendering.png_codec import encode_rgb_png

# ~2x the 200px display cell (MAX_CORE_DISPLAY_PX), so the served artifact is
# UI-resolution: decimation never lands below what the browser shows.
DEFAULT_TARGET_LONG_SIDE = 400
# The on-demand "full" variant: near-native inspection zoom, still bounded.
FULL_TARGET_LONG_SIDE = 1600
EMPTY_TILE_SIZE = 64


def symmetric_scale(matrix: np.ndarray) -> float:
    """Per-matrix symmetric color scale: p98 of |values| with the legacy floors."""
    finite = np.abs(np.asarray(matrix, dtype=np.float64).ravel())
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-12
    p98_idx = max(0, int(round(0.98 * finite.size)) - 1)
    p98 = float(np.partition(finite, p98_idx)[p98_idx])
    return float(max(p98, float(finite.max()) * 0.05, 1e-12))


def _pool_signed_absmax(matrix: np.ndarray, k: int) -> np.ndarray:
    """Block-pool by the SIGNED value of largest magnitude (extremes survive zoom)."""
    h, w = matrix.shape
    out_h, out_w = -(-h // k), -(-w // k)
    padded = np.zeros((out_h * k, out_w * k), dtype=np.float64)
    padded[:h, :w] = np.nan_to_num(matrix, nan=0.0)
    blocks = (
        padded.reshape(out_h, k, out_w, k).transpose(0, 2, 1, 3).reshape(out_h, out_w, k * k)
    )
    winner = np.abs(blocks).argmax(axis=2)
    return np.take_along_axis(blocks, winner[..., None], axis=2)[..., 0]


def _pool_mask_any(mask: np.ndarray, k: int) -> np.ndarray:
    """ANY-pool a pruned-line mask, then dilate isolated lines to >= 2px.

    The dilation only exists under decimation (k > 1): a decimated view is
    already lossy, and a 1px line would die under any further browser scaling.
    """
    n = mask.shape[0]
    out_n = -(-n // k)
    padded = np.zeros(out_n * k, dtype=bool)
    padded[:n] = mask
    pooled = padded.reshape(out_n, k).any(axis=1)
    if pooled.size >= 2:
        thick = pooled.copy()
        thick[1:] |= pooled[:-1]
        if pooled[-1]:
            thick[-2] = True
        pooled = thick
    return pooled


def _as_line_mask(mask: Sequence[bool] | None, length: int, axis_name: str) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray([bool(x) for x in mask], dtype=bool)
    if arr.shape[0] != length:
        raise ValueError(
            f"pruned_{axis_name}_mask has {arr.shape[0]} entries for {length} {axis_name}s"
        )
    return arr


def _empty_tile_rgb() -> np.ndarray:
    """A visually distinct 'no data' tile: background, border, diagonal cross."""
    size = EMPTY_TILE_SIZE
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:, :] = BACKGROUND_RGB
    rgb[0, :] = rgb[-1, :] = rgb[:, 0] = rgb[:, -1] = GRID_ACCENT_RGB
    diag = np.arange(size)
    rgb[diag, diag] = GRID_ACCENT_RGB
    rgb[diag, size - 1 - diag] = GRID_ACCENT_RGB
    return rgb


def render_heatmap_png_bytes(
    matrix: np.ndarray,
    *,
    pruned_row_mask: Sequence[bool] | None = None,
    pruned_col_mask: Sequence[bool] | None = None,
    scale: float | None = None,
    target_long_side: int = DEFAULT_TARGET_LONG_SIDE,
) -> bytes:
    """Render a 2D weight matrix as PNG bytes.

    ``scale`` is the shared symmetric color limit (``None`` falls back to this
    matrix's own :func:`symmetric_scale`); ``target_long_side`` bounds the
    output: matrices at or below it render at native 1px-per-cell size, larger
    ones decimate by ``k = ceil(long_side / target)`` on both axes.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return encode_rgb_png(_empty_tile_rgb())
    h, w = matrix.shape
    row_mask = _as_line_mask(pruned_row_mask, h, "row")
    col_mask = _as_line_mask(pruned_col_mask, w, "col")
    if scale is None:
        scale = symmetric_scale(matrix)

    k = max(1, -(-max(h, w) // int(target_long_side)))
    if k == 1:
        values = matrix
    else:
        values = _pool_signed_absmax(matrix, k)
        row_mask = _pool_mask_any(row_mask, k) if row_mask is not None else None
        col_mask = _pool_mask_any(col_mask, k) if col_mask is not None else None

    rgb = DIVERGING_LUT[lut_index(values, float(scale))].copy()
    if k == 1:
        rgb[~np.isfinite(matrix)] = BACKGROUND_RGB
    if row_mask is not None:
        rgb[row_mask, :] = MASK_RGB
    if col_mask is not None:
        rgb[:, col_mask] = MASK_RGB
    return encode_rgb_png(rgb)


def render_heatmap_png_data_uri(
    matrix: np.ndarray,
    *,
    pruned_row_mask: Sequence[bool] | None = None,
    pruned_col_mask: Sequence[bool] | None = None,
    scale: float | None = None,
    target_long_side: int = DEFAULT_TARGET_LONG_SIDE,
) -> str:
    """Render a 2D weight matrix as a base64 PNG data URI."""
    png = render_heatmap_png_bytes(
        matrix,
        pruned_row_mask=pruned_row_mask,
        pruned_col_mask=pruned_col_mask,
        scale=scale,
        target_long_side=target_long_side,
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def render_colorbar_png_bytes(*, width: int = 256, height: int = 16) -> bytes:
    """The family colorbar asset: the diverging LUT as a horizontal PNG strip."""
    return encode_rgb_png(render_colorbar_rgb(width=width, height=height))


__all__ = [
    "DEFAULT_TARGET_LONG_SIDE",
    "EMPTY_TILE_SIZE",
    "FULL_TARGET_LONG_SIDE",
    "render_colorbar_png_bytes",
    "render_heatmap_png_bytes",
    "render_heatmap_png_data_uri",
    "symmetric_scale",
]

"""Diverging colormap as a precomputed 256-entry LUT, plus reserved colors.

The LUT is built ONCE at import from the ColorBrewer BrBG-11 anchors (the same
ramp the old matplotlib renderer used), so every heatmap and the colorbar share
one immutable table. Index 0 is the brown (negative) endpoint, 255 the teal
(positive) endpoint, and the two central entries sit on the neutral midpoint.
"""

from __future__ import annotations

import numpy as np

# ColorBrewer BrBG-11: brown (negative) -> neutral -> teal (positive).
_BRBG_ANCHORS = (
    (0x54, 0x30, 0x05),
    (0x8C, 0x51, 0x0A),
    (0xBF, 0x81, 0x2D),
    (0xDF, 0xC2, 0x7D),
    (0xF6, 0xE8, 0xC3),
    (0xF5, 0xF5, 0xF5),
    (0xC7, 0xEA, 0xE5),
    (0x80, 0xCD, 0xC1),
    (0x35, 0x97, 0x8F),
    (0x01, 0x66, 0x5E),
    (0x00, 0x3C, 0x30),
)

# Reserved colors: pruned rows/columns, and the tile/NaN background.
MASK_RGB = (229, 57, 53)        # #e53935
BACKGROUND_RGB = (43, 48, 60)   # #2b303c
GRID_ACCENT_RGB = (107, 114, 128)  # #6b7280 — empty-tile pattern


def _build_lut() -> np.ndarray:
    anchors = np.array(_BRBG_ANCHORS, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, 256) * (len(anchors) - 1)
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, len(anchors) - 1)
    frac = (positions - low)[:, None]
    lut = anchors[low] * (1.0 - frac) + anchors[high] * frac
    lut = np.round(lut).astype(np.uint8)
    lut.setflags(write=False)
    return lut


DIVERGING_LUT: np.ndarray = _build_lut()


def lut_index(values: np.ndarray, scale: float) -> np.ndarray:
    """Map values in ``[-scale, scale]`` onto LUT indices 0..255.

    Monotone, exact at the extremes, and symmetric within one bin
    (an even-sized LUT has two central entries, so +v/-v mirror to
    ``255 - i`` up to one index).
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"lut_index needs a positive finite scale, got {scale!r}")
    ratio = np.asarray(values, dtype=np.float64) / scale
    # Non-finite values land on the neutral centre; the heatmap render
    # repaints native NaN cells with the background color afterwards.
    ratio = np.clip(np.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    return np.clip(np.floor(128.0 + 128.0 * ratio), 0, 255).astype(np.uint8)


def render_colorbar_rgb(*, width: int = 256, height: int = 16) -> np.ndarray:
    """The LUT as a horizontal gradient strip: negative left, positive right."""
    columns = np.floor(np.linspace(0.0, 255.0, width) + 0.5).astype(int)
    return np.tile(DIVERGING_LUT[columns][None, :, :], (height, 1, 1))


__all__ = [
    "BACKGROUND_RGB",
    "DIVERGING_LUT",
    "GRID_ACCENT_RGB",
    "MASK_RGB",
    "lut_index",
    "render_colorbar_rgb",
]

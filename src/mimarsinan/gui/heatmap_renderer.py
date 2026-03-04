"""Backend heatmap image generation for GUI snapshots.

Renders weight matrices as PNG images (with optional pruned row/col lines)
so the frontend receives only image data URIs, not raw matrices.
"""

from __future__ import annotations

import base64
import io
from typing import Sequence

import numpy as np


def render_heatmap_png_data_uri(
    matrix: np.ndarray,
    *,
    pruned_row_mask: Sequence[bool] | None = None,
    pruned_col_mask: Sequence[bool] | None = None,
    max_size: int = 1024,
    dpi: int = 150,
) -> str:
    """Render a 2D weight matrix as a PNG heatmap and return a data URI.

    The heatmap is created from the full matrix; the output image dimensions
    are capped at max_size per axis. Red is reserved for pruned row/col lines
    only; the weight colormap avoids red (blue–cyan–green–yellow).

    Returns:
        String of the form "data:image/png;base64,..." for use as img src.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.size == 0:
        return _empty_image_data_uri()

    h, w = matrix.shape
    out_h = min(h, max_size)
    out_w = min(w, max_size)
    fig_w = out_w / dpi
    fig_h = out_h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#2b303c")
    ax.set_facecolor("#2b303c")

    flat = np.abs(matrix.ravel())
    flat = np.sort(flat)
    p98_idx = max(0, int(round(0.98 * len(flat))) - 1)
    scale = max(flat[p98_idx] if len(flat) > 0 else 0, np.abs(matrix).max() * 0.05, 1e-12)

    # Diverging colormap without red (red is reserved for pruned lines): blue (neg) -> white -> brown (pos)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="BrBG",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")

    # Pruned row/col lines (red)
    if pruned_row_mask is not None:
        for i, masked in enumerate(pruned_row_mask):
            if masked:
                ax.axhline(i, color="#e53935", linewidth=1.5, alpha=0.9)
    if pruned_col_mask is not None:
        for j, masked in enumerate(pruned_col_mask):
            if masked:
                ax.axvline(j, color="#e53935", linewidth=1.5, alpha=0.9)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _empty_image_data_uri() -> str:
    """Return a minimal 1x1 dark PNG as data URI."""
    # Minimal valid 1x1 gray PNG (IHDR + IDAT + IEND)
    png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"

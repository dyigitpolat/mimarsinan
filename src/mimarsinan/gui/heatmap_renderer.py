"""Backend heatmap image generation for GUI snapshots."""

from __future__ import annotations

import base64
import io
from typing import Sequence

import numpy as np


def render_heatmap_png_bytes(
    matrix: np.ndarray,
    *,
    pruned_row_mask: Sequence[bool] | None = None,
    pruned_col_mask: Sequence[bool] | None = None,
    max_size: int = 1024,
    dpi: int = 150,
) -> bytes:
    """Render a 2D weight matrix as a PNG heatmap and return raw PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.size == 0:
        return _empty_image_bytes()

    h, w = matrix.shape
    max_side = max(h, w, 1)
    scale = max_size / max_side
    out_w = max(1, round(w * scale))
    out_h = max(1, round(h * scale))
    fig_w = out_w / dpi
    fig_h = out_h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#2b303c")
    ax.set_facecolor("#2b303c")

    flat = np.abs(matrix.ravel())
    flat = np.sort(flat)
    p98_idx = max(0, int(round(0.98 * len(flat))) - 1)
    scale = max(flat[p98_idx] if len(flat) > 0 else 0, np.abs(matrix).max() * 0.05, 1e-12)

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

    row_lw_pt = 72.0 * out_h / (h * dpi) if h else 1.0
    col_lw_pt = 72.0 * out_w / (w * dpi) if w else 1.0
    if pruned_row_mask is not None:
        for i, masked in enumerate(pruned_row_mask):
            if masked:
                ax.axhline(i, color="#e53935", linewidth=row_lw_pt, alpha=0.9)
    if pruned_col_mask is not None:
        for j, masked in enumerate(pruned_col_mask):
            if masked:
                ax.axvline(j, color="#e53935", linewidth=col_lw_pt, alpha=0.9)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    # Avoid bbox_inches='tight' so output stays exactly (out_w, out_h) to match the frontend mini-view aspect.
    plt.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_heatmap_png_data_uri(
    matrix: np.ndarray,
    *,
    pruned_row_mask: Sequence[bool] | None = None,
    pruned_col_mask: Sequence[bool] | None = None,
    max_size: int = 1024,
    dpi: int = 150,
) -> str:
    """Render a 2D weight matrix as a PNG heatmap and return a base64 data URI."""
    png = render_heatmap_png_bytes(
        matrix,
        pruned_row_mask=pruned_row_mask,
        pruned_col_mask=pruned_col_mask,
        max_size=max_size,
        dpi=dpi,
    )
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


_EMPTY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _empty_image_bytes() -> bytes:
    """Return a minimal 1x1 dark PNG as raw bytes."""
    return _EMPTY_PNG_BYTES


def _empty_image_data_uri() -> str:
    """Return a minimal 1x1 dark PNG as data URI (legacy helper)."""
    return f"data:image/png;base64,{base64.b64encode(_EMPTY_PNG_BYTES).decode('ascii')}"

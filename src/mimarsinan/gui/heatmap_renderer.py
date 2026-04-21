"""Backend heatmap image generation for GUI snapshots.

Renders weight matrices as PNG images (with optional pruned row/col lines)
so the frontend receives only image data URIs, not raw matrices.
"""

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
    """Render a 2D weight matrix as a PNG heatmap and return raw PNG bytes.

    This is the transport-friendly variant used by the lazy
    :class:`~mimarsinan.gui.resources.ResourceStore`: the GUI server streams
    the bytes directly with ``Content-Type: image/png`` instead of embedding
    a multi-megabyte base64 data URI inside a JSON snapshot.

    See :func:`render_heatmap_png_data_uri` for the colormap / pruning-line
    contract — both functions share the same implementation and differ only
    in the returned representation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.size == 0:
        return _empty_image_bytes()

    h, w = matrix.shape
    # Preserve aspect ratio when capping: scale so the longer side is max_size
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

    # Pruned row/col lines (red): thickness = one row height or one column width in the output image
    # Row line thickness in points: (out_h / h) pixels -> 72 * (out_h / h) / dpi pt. Column likewise.
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
    # Do not use bbox_inches='tight' so output dimensions stay exactly (out_w, out_h)
    # and match the frontend mini-view frame aspect (neurons/axons).
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
    """Render a 2D weight matrix as a PNG heatmap and return a base64 data URI.

    Kept for callers (and legacy tests) that still want an inline
    ``data:image/png;base64,...`` string. New code should prefer
    :func:`render_heatmap_png_bytes` + the
    :class:`~mimarsinan.gui.resources.ResourceStore`.
    """
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

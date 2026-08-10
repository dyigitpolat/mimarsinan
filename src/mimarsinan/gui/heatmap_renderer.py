"""Backend heatmap image generation for GUI snapshots.

Stable entry-point module: the implementation lives in
:mod:`mimarsinan.gui.rendering` (pure numpy + stdlib PNG encoding -- no
matplotlib, no pyplot state, thread-safe by construction). ``HeatmapSource``
calls through this module's attributes, so tests can monkeypatch
``heatmap_renderer.render_heatmap_png_bytes`` and intercept every render.
"""

from __future__ import annotations

from mimarsinan.gui.rendering import (
    DEFAULT_TARGET_LONG_SIDE,
    FULL_TARGET_LONG_SIDE,
    render_colorbar_png_bytes,
    render_heatmap_png_bytes,
    render_heatmap_png_data_uri,
    symmetric_scale,
)

__all__ = [
    "DEFAULT_TARGET_LONG_SIDE",
    "FULL_TARGET_LONG_SIDE",
    "render_colorbar_png_bytes",
    "render_heatmap_png_bytes",
    "render_heatmap_png_data_uri",
    "symmetric_scale",
]

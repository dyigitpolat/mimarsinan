"""Contract tests for the pure-numpy heatmap renderer.

The renderer is a pure function of arrays: no matplotlib, no pyplot state,
thread-safe by construction. Pins:

* PNG validity (PIL decode) and EXACT output dimensions: native size below the
  target (browser upscales), integer-factor decimation above it — ONE pooling
  factor for both axes so cells stay square, and zero margins so PNG pixel
  [0, 0] IS matrix cell [0, 0].
* Mask-aware decimation: a single pruned line survives any zoom (ANY-pooling
  plus a one-pixel dilation when decimating).
* A shared explicit ``scale`` makes pixels comparable across matrices.
* The 256-entry diverging LUT: exact endpoints, near-neutral centre, index
  symmetry within one bin.
* The empty matrix renders a distinct placeholder tile, never a stretched 1x1.
* Concurrent renders produce identical bytes.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from mimarsinan.gui.heatmap_renderer import (
    DEFAULT_TARGET_LONG_SIDE,
    render_colorbar_png_bytes,
    render_heatmap_png_bytes,
    render_heatmap_png_data_uri,
    symmetric_scale,
)
from mimarsinan.gui.rendering.colormap import (
    BACKGROUND_RGB,
    DIVERGING_LUT,
    MASK_RGB,
    lut_index,
)
from mimarsinan.gui.rendering.heatmap_render import EMPTY_TILE_SIZE


def _decode(png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png))
    return np.asarray(img.convert("RGB"))


class TestPngValidityAndDims:
    def test_small_matrix_renders_native_size(self):
        """Below the target the PNG is 1 pixel per cell: no upscaling, no margins."""
        m = np.linspace(-1.0, 1.0, 70).reshape(10, 7)
        px = _decode(render_heatmap_png_bytes(m))
        assert px.shape == (10, 7, 3)

    def test_large_matrix_decimated_with_one_factor_both_axes(self):
        """k = ceil(long/target) applies to BOTH axes so cells stay square."""
        m = np.ones((1000, 100))
        px = _decode(render_heatmap_png_bytes(m))  # k = ceil(1000/400) = 3
        assert px.shape == (334, 34, 3)

    def test_exact_multiple_pools_to_integer_grid(self):
        m = np.ones((800, 400))
        px = _decode(render_heatmap_png_bytes(m))  # k = 2 exactly
        assert px.shape == (400, 200, 3)

    def test_target_long_side_param_respected(self):
        m = np.ones((1000, 1000))
        px = _decode(render_heatmap_png_bytes(m, target_long_side=200))  # k = 5
        assert px.shape == (200, 200, 3)

    def test_top_left_pixel_is_cell_zero_zero(self):
        """Zero margins: the corner pixel carries cell [0,0]'s color exactly."""
        m = np.zeros((5, 5))
        m[0, 0] = 1.0
        px = _decode(render_heatmap_png_bytes(m, scale=1.0))
        assert tuple(px[0, 0]) == tuple(DIVERGING_LUT[255])
        assert tuple(px[4, 4]) == tuple(DIVERGING_LUT[lut_index(np.zeros(1), 1.0)[0]])


class TestEmptyMatrix:
    def test_empty_matrix_renders_placeholder_tile(self):
        px = _decode(render_heatmap_png_bytes(np.zeros((0, 0))))
        assert px.shape == (EMPTY_TILE_SIZE, EMPTY_TILE_SIZE, 3)
        colors = {tuple(c) for c in px.reshape(-1, 3)}
        assert len(colors) >= 2, "placeholder must be visually patterned, not flat"
        assert tuple(BACKGROUND_RGB) in colors

    def test_zero_row_matrix_is_also_placeholder(self):
        px = _decode(render_heatmap_png_bytes(np.zeros((0, 5))))
        assert px.shape == (EMPTY_TILE_SIZE, EMPTY_TILE_SIZE, 3)

    def test_non_2d_input_fails_loud(self):
        with pytest.raises(ValueError):
            render_heatmap_png_bytes(np.zeros(6))
        with pytest.raises(ValueError):
            render_heatmap_png_bytes(np.zeros((2, 3, 4)))


class TestMaskSurvival:
    def test_native_masked_row_is_exactly_that_row(self):
        m = np.ones((10, 10))
        mask = [False] * 10
        mask[3] = True
        px = _decode(render_heatmap_png_bytes(m, pruned_row_mask=mask))
        assert (px[3] == MASK_RGB).all()
        assert not (px[2] == MASK_RGB).all()
        assert not (px[4] == MASK_RGB).all()

    def test_single_pruned_row_survives_aggressive_decimation(self):
        """1000x1000 with ONE pruned row must stay visible at a 200px tile."""
        rng = np.random.default_rng(0)
        m = rng.standard_normal((1000, 1000))
        mask = [False] * 1000
        mask[500] = True
        px = _decode(
            render_heatmap_png_bytes(m, pruned_row_mask=mask, target_long_side=200)
        )
        assert px.shape == (200, 200, 3)
        red_rows = np.flatnonzero((px == MASK_RGB).all(axis=2).all(axis=1))
        assert red_rows.size >= 2, "a decimated pruned line must be >= 2px thick"

    def test_single_pruned_col_survives_aggressive_decimation(self):
        rng = np.random.default_rng(1)
        m = rng.standard_normal((1000, 1000))
        mask = [False] * 1000
        mask[999] = True
        px = _decode(
            render_heatmap_png_bytes(m, pruned_col_mask=mask, target_long_side=200)
        )
        red_cols = np.flatnonzero((px == MASK_RGB).all(axis=2).all(axis=0))
        assert red_cols.size >= 2

    def test_mask_length_mismatch_fails_loud(self):
        with pytest.raises(ValueError):
            render_heatmap_png_bytes(np.ones((4, 4)), pruned_row_mask=[True] * 5)


class TestSharedScale:
    def test_same_value_same_color_across_matrices(self):
        """Two matrices rendered under ONE scale give one value one color."""
        a = np.array([[1.0]])
        b = np.array([[1.0, -3.0], [0.5, 4.0]])
        pa = _decode(render_heatmap_png_bytes(a, scale=4.0))
        pb = _decode(render_heatmap_png_bytes(b, scale=4.0))
        assert tuple(pa[0, 0]) == tuple(pb[0, 0])

    def test_default_scale_is_per_matrix_p98(self):
        rng = np.random.default_rng(2)
        m = rng.standard_normal((64, 64))
        flat = np.sort(np.abs(m.ravel()))
        p98 = flat[max(0, int(round(0.98 * flat.size)) - 1)]
        expected = max(p98, np.abs(m).max() * 0.05, 1e-12)
        assert symmetric_scale(m) == pytest.approx(expected)

    def test_scale_beyond_p98_no_longer_saturates_silently(self):
        """An outlier above the shared scale clips to the LUT endpoint."""
        m = np.array([[10.0, -10.0]])
        px = _decode(render_heatmap_png_bytes(m, scale=1.0))
        assert tuple(px[0, 0]) == tuple(DIVERGING_LUT[255])
        assert tuple(px[0, 1]) == tuple(DIVERGING_LUT[0])

    def test_render_is_deterministic(self):
        rng = np.random.default_rng(3)
        m = rng.standard_normal((300, 200))
        assert render_heatmap_png_bytes(m) == render_heatmap_png_bytes(m)


class TestDivergingLut:
    def test_shape_and_endpoints(self):
        assert DIVERGING_LUT.shape == (256, 3)
        assert DIVERGING_LUT.dtype == np.uint8
        assert tuple(DIVERGING_LUT[0]) == (84, 48, 5)     # BrBG brown end (-)
        assert tuple(DIVERGING_LUT[255]) == (0, 60, 48)   # BrBG teal end (+)

    def test_centre_is_near_neutral(self):
        for i in (127, 128):
            assert np.abs(DIVERGING_LUT[i].astype(int) - 245).max() <= 12

    def test_index_map_is_symmetric_within_one_bin(self):
        v = np.linspace(0.01, 1.0, 97)
        up = lut_index(v, 1.0).astype(int)
        down = lut_index(-v, 1.0).astype(int)
        assert np.abs((up + down) - 255).max() <= 1
        assert lut_index(np.array([0.0]), 1.0)[0] in (127, 128)

    def test_index_map_is_monotone_and_exact_at_extremes(self):
        v = np.linspace(-1.5, 1.5, 301)
        idx = lut_index(v, 1.0).astype(int)
        assert (np.diff(idx) >= 0).all()
        assert idx[0] == 0 and idx[-1] == 255


class TestThreadSafety:
    def test_concurrent_renders_produce_identical_bytes(self):
        rng = np.random.default_rng(4)
        m = rng.standard_normal((500, 300))
        mask = [i % 97 == 0 for i in range(500)]
        expected = render_heatmap_png_bytes(m, pruned_row_mask=mask)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: render_heatmap_png_bytes(m, pruned_row_mask=mask),
                    range(16),
                )
            )
        assert all(r == expected for r in results)


class TestNanHandling:
    def test_nan_renders_background_at_native_resolution(self):
        px = _decode(render_heatmap_png_bytes(np.array([[np.nan, 1.0]]), scale=1.0))
        assert tuple(px[0, 0]) == tuple(BACKGROUND_RGB)
        assert tuple(px[0, 1]) == tuple(DIVERGING_LUT[255])


class TestDataUri:
    def test_data_uri_wraps_the_png_bytes(self):
        import base64

        m = np.ones((3, 3))
        uri = render_heatmap_png_data_uri(m)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",", 1)[1])
        assert decoded == render_heatmap_png_bytes(m)


class TestColorbar:
    def test_colorbar_spans_the_lut_left_to_right(self):
        px = _decode(render_colorbar_png_bytes())
        assert px.shape[0] >= 8 and px.shape[1] == 256
        assert tuple(px[0, 0]) == tuple(DIVERGING_LUT[0])
        assert tuple(px[0, -1]) == tuple(DIVERGING_LUT[255])

    def test_default_target_constant_is_ui_scaled(self):
        """~2x the 200px display cell: decimation never lands below display res."""
        assert DEFAULT_TARGET_LONG_SIDE == 400

"""FFCV's image tail must land in the torch path's dtype AND scale.

A provider that declares no normalization previously streamed raw uint8
(crashing any model), and the obvious repair — a bare Convert(float32) —
would yield [0,255] while torchvision's ToTensor() yields [0,1]: a silent
255x divergence between the two data paths, which is worse than the crash.
"""

import numpy as np
import pytest

pytest.importorskip("ffcv", reason="optional ffcv backend not installed")

from mimarsinan.data_handling.ffcv.spec_builder import image_tail_ops  # noqa: E402


def _op_names(ops):
    return [name for name, _kwargs in ops]


def _kwargs_of(ops, name):
    return next(kwargs for op, kwargs in ops if op == name)


class TestNormalizingProvider:
    def test_tail_is_unchanged_when_normalization_is_declared(self):
        ops = image_tail_ops(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        assert _op_names(ops) == [
            "NormalizeImage", "ToTensor", "ToDevice", "ToTorchImage"
        ]

    def test_declared_normalization_scales_by_255(self):
        # torchvision composes ToTensor (÷255) then Normalize(mean, std);
        # FFCV folds both into one op over raw [0,255] pixels.
        ops = image_tail_ops(mean=(0.5,), std=(0.25,))
        kw = _kwargs_of(ops, "NormalizeImage")
        np.testing.assert_allclose(kw["mean"], np.array([0.5]) * 255.0)
        np.testing.assert_allclose(kw["std"], np.array([0.25]) * 255.0)


class TestNonNormalizingProvider:
    def test_tail_still_converts_dtype(self):
        ops = image_tail_ops(mean=None, std=None)
        assert "NormalizeImage" in _op_names(ops), (
            "a non-normalizing provider must still leave the uint8 domain"
        )

    def test_it_reproduces_totensor_scaling_exactly(self):
        # ToTensor() == (x - 0) / 255 -> the identity normalization.
        kw = _kwargs_of(image_tail_ops(mean=None, std=None), "NormalizeImage")
        np.testing.assert_allclose(kw["mean"], np.zeros_like(kw["mean"]))
        np.testing.assert_allclose(kw["std"], np.full_like(kw["std"], 255.0))

    def test_output_dtype_is_float32(self):
        kw = _kwargs_of(image_tail_ops(mean=None, std=None), "NormalizeImage")
        assert kw["type"] == np.float32

    def test_numeric_parity_with_totensor(self):
        kw = _kwargs_of(image_tail_ops(mean=None, std=None), "NormalizeImage")
        pixels = np.array([0.0, 127.0, 255.0], dtype=np.float32)
        ffcv_values = (pixels - kw["mean"][0]) / kw["std"][0]
        totensor_values = pixels / 255.0
        np.testing.assert_allclose(ffcv_values, totensor_values, rtol=0, atol=0)

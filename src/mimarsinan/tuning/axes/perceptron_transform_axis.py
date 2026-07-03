"""Adapter for the stochastic perceptron-transform family (weight quant / NAPQ)."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase


class PerceptronTransformAxis(AdaptationAxisBase):
    """Uniform seam over a tuner's closure-based rate application."""

    name = "perceptron_transform"
    interpolation_mode = "stochastic_mask"
    is_stochastic = True

    def __init__(self, apply_fn, *, name=None):
        super().__init__()
        self._apply_fn = apply_fn
        if name is not None:
            self.name = name

    def set_rate(self, alpha: float) -> None:
        self._apply_fn(float(alpha))

    def descriptor(self) -> str:
        return self.name


class NAPQAxis(PerceptronTransformAxis):
    """Normalization-aware perceptron weight quantization."""

    name = "napq"

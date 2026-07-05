"""Adapter for the stochastic perceptron-transform family (weight quant / NAPQ)."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import ClosureApplyAxisBase


class PerceptronTransformAxis(ClosureApplyAxisBase):
    """Uniform seam over a tuner's closure-based rate application."""

    name = "perceptron_transform"
    interpolation_mode = "stochastic_mask"
    is_stochastic = True

    def __init__(self, apply_fn, *, replica_apply_fn=None, name=None):
        super().__init__(apply_fn, replica_apply_fn=replica_apply_fn)
        if name is not None:
            self.name = name

    def descriptor(self) -> str:
        return self.name


class NAPQAxis(PerceptronTransformAxis):
    """Normalization-aware perceptron weight quantization."""

    name = "napq"

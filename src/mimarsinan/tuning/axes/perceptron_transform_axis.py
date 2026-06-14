"""Adapter for the stochastic perceptron-transform family (weight quant / NAPQ).

The rate application is a per-parameter Bernoulli mix of prev/new transforms
applied through ``PerceptronTransformTrainer`` — state intertwined with the
tuner (the prev/new transform builders and the trainer). The axis therefore
presents the uniform ``set_rate`` seam over a tuner-provided ``apply_fn`` rather
than re-owning the mechanism; folding the mechanism into the axis is the P4
driver refactor. ``set_decision_seed`` is a no-op until the per-decision seeding
of the stochastic mask lands (P5).
"""

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

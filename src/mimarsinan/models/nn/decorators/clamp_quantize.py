"""Clamp and quantize decorators."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations.autograd import DifferentiableClamp, StaircaseFunction


class ClampDecorator:
    """Clamps activations into ``[clamp_min, clamp_max]``.

    ``clamp_max`` is either:

    - a fixed tensor (legacy behaviour — unchanged), or
    - ``None``, with a learnable ``scale_param`` (an ``nn.Parameter``)
      supplied instead. The clamp ceiling is then the scalar value of
      ``scale_param`` on the forward pass, and gradients flow back into
      ``scale_param`` through ``DifferentiableClamp`` (which passes
      through the ceiling for saturating inputs).

    When ``scale_param`` is provided, the ``ClampTuner`` is responsible
    for (a) adding it to the optimiser's parameter group, (b) applying a
    regulariser (see :func:`clamp_tuner.clamp_scale_regulariser`), and
    (c) freezing the learned scalar back onto the perceptron via
    :func:`clamp_tuner.freeze_learnable_scale` before the step commits.
    """

    def __init__(self, clamp_min, clamp_max=None, *, scale_param=None):
        assert (clamp_max is None) ^ (scale_param is None), (
            "ClampDecorator requires exactly one of clamp_max or scale_param"
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale_param = scale_param

    def _effective_clamp_max(self, x):
        # ``scale_param`` was added after some pickles were produced. Old
        # pickled ``ClampDecorator`` instances won't have this attribute at
        # all — fall back gracefully to ``clamp_max`` (which those pickles
        # do populate) so loading a pre-refactor IR graph still works.
        scale_param = getattr(self, "scale_param", None)
        if scale_param is not None:
            return scale_param.to(x.device)
        return self.clamp_max.to(x.device)

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        clamp_max = self._effective_clamp_max(x)
        return DifferentiableClamp.apply(x, self.clamp_min, clamp_max)


class QuantizeDecorator:
    def __init__(self, levels_before_c, c):
        self.levels_before_c = levels_before_c
        self.c = c

    def input_transform(self, x):
        return x

    def output_transform(self, x):
        self.levels_before_c = self.levels_before_c.to(x.device)
        self.c = self.c.to(x.device)
        return StaircaseFunction.apply(x, self.levels_before_c / self.c)


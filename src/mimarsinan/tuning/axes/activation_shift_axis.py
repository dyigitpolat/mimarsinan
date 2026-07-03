"""Adapter for the one-shot activation-shift tuner."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase


class ActivationShiftAxis(AdaptationAxisBase):
    """Uniform seam over the one-shot shift application."""

    name = "activation_shift"
    interpolation_mode = "parameter_path"
    supports_smooth = False

    def __init__(self, apply_fn):
        super().__init__()
        self._apply_fn = apply_fn

    def set_rate(self, alpha: float) -> None:
        self._apply_fn()

    def descriptor(self) -> str:
        return self.name

"""Adapter for the one-shot activation-shift tuner.

The shift is applied in full once (not a smooth 0→1 ramp), so ``supports_smooth``
is False and ``set_rate`` applies the full shift regardless of ``alpha``. The
axis presents the uniform seam over the tuner's ``apply_fn``.
"""

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
        # One-shot: the full shift is applied; alpha is advisory.
        self._apply_fn()

    def descriptor(self) -> str:
        return self.name

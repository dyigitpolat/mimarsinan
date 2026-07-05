"""Adapter for the one-shot activation-shift tuner."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import ClosureApplyAxisBase


class ActivationShiftAxis(ClosureApplyAxisBase):
    """Uniform seam over the one-shot shift application (rate-independent)."""

    name = "activation_shift"
    interpolation_mode = "parameter_path"
    supports_smooth = False

    def __init__(self, apply_fn, *, replica_apply_fn=None):
        super().__init__(
            lambda _alpha: apply_fn(), replica_apply_fn=replica_apply_fn,
        )

    def descriptor(self) -> str:
        return self.name

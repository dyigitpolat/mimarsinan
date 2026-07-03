"""Adapter for the structured-pruning family."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase


class PruningAxis(AdaptationAxisBase):
    """Uniform seam over a tuner's mask-apply + recovery-hook callables."""

    name = "pruning"
    interpolation_mode = "parameter_path"
    monotonicity = "expected"

    def __init__(self, apply_fn, *, recovery_hooks_fn=None):
        super().__init__()
        self._apply_fn = apply_fn
        self._recovery_hooks_fn = recovery_hooks_fn

    def set_rate(self, alpha: float) -> None:
        self._apply_fn(float(alpha))

    def recovery_hooks(self, alpha: float) -> list:
        if self._recovery_hooks_fn is not None:
            return self._recovery_hooks_fn(float(alpha))
        return []

    def descriptor(self) -> str:
        return self.name

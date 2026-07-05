"""Adapter for the structured-pruning family."""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import ClosureApplyAxisBase


class PruningAxis(ClosureApplyAxisBase):
    """Uniform seam over a tuner's mask-apply + recovery-hook callables."""

    name = "pruning"
    interpolation_mode = "parameter_path"
    monotonicity = "expected"

    def __init__(self, apply_fn, *, recovery_hooks_fn=None, replica_apply_fn=None):
        super().__init__(apply_fn, replica_apply_fn=replica_apply_fn)
        self._recovery_hooks_fn = recovery_hooks_fn

    def recovery_hooks(self, alpha: float) -> list:
        if self._recovery_hooks_fn is not None:
            return self._recovery_hooks_fn(float(alpha))
        return []

    def descriptor(self) -> str:
        return self.name

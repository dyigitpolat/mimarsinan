"""Teacher->genuine output blend installed as a ramping ``model.forward`` override."""

from __future__ import annotations

from mimarsinan.tuning.forward_install import LazyExecutorForward


class BlendedGenuineForward(LazyExecutorForward):
    """Picklable ``model.forward`` blending a frozen teacher with the genuine cascade.

    Live ``rate`` drives the blend: ``rate=0`` is the teacher exactly, ``rate=1``
    the freshly built :class:`TTFSSegmentForward`. The lazy executor drops on pickle.
    """

    def __init__(self, model, teacher, T: int, rate: float = 0.0,
                 *, boundary_surrogate_temp: float | None = None):
        super().__init__(model, T)
        self.teacher = teacher
        self.rate = float(rate)
        self.boundary_surrogate_temp = boundary_surrogate_temp

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )

        return TTFSSegmentForward(
            self.model.get_mapper_repr(), self.T,
            boundary_surrogate_temp=self.boundary_surrogate_temp,
        )

    def genuine_logits(self, x):
        """The pure single-spike cascade logits (the ``rate=1`` branch). Public so
        the KD loss can add a genuine-CE term without introspecting a private member."""
        return self._ensure_executor(self._build_executor)(x)

    def _run(self, x):
        rate = float(self.rate)
        if rate == 0.0:
            return self.teacher(x)
        if rate == 1.0:
            return self.genuine_logits(x)
        return (1.0 - rate) * self.teacher(x) + rate * self.genuine_logits(x)

"""Teacher->genuine output blend: a ramping ``model.forward`` override.

Installs ``out = (1 - rate) * teacher(x) + rate * genuine(x)`` as the model's
forward, where ``genuine`` is the differentiable single-spike cascade
(:class:`TTFSSegmentForward`, built lazily from the live model) and ``teacher``
is a frozen snapshot. Ramping ``rate`` 0->1 walks the output smoothly from the
continuous teacher (``rate=0`` reads the teacher exactly) to the genuine
deployed cascade (``rate=1``). Gradients flow into the model's parameters via
the genuine branch; the teacher branch carries none.
"""

from __future__ import annotations

from mimarsinan.tuning.forward_install import LazyExecutorForward


class BlendedGenuineForward(LazyExecutorForward):
    """Picklable ``model.forward`` blending a frozen teacher with the genuine cascade.

    The live ``rate`` (a settable scalar, read each call) drives the blend so an
    axis can ramp it. ``rate=0`` is the teacher output exactly, ``rate=1`` the
    freshly built :class:`TTFSSegmentForward` exactly. The lazy genuine executor
    is dropped on pickle (the teacher snapshot stays light).
    """

    def __init__(self, model, teacher, T: int, rate: float = 0.0):
        super().__init__(model, T)
        self.teacher = teacher
        self.rate = float(rate)

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )

        return TTFSSegmentForward(self.model.get_mapper_repr(), self.T)

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

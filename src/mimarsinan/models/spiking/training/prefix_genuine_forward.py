"""Prefix-conversion k-hybrid installed as a ramping ``model.forward`` override."""

from __future__ import annotations

import math

from mimarsinan.tuning.forward_install import LazyExecutorForward


def prefix_length_for_rate(rate: float, n_segments: int) -> int:
    """Converted-prefix length for a ladder rate: ``k = ceil(rate * n)``, clamped.

    Ladder rates ``i/n`` map exactly to ``k = i``. CEILING is load-bearing: a
    gate midpoint retry ``(committed + rate)/2`` must retrain the TARGET
    frontier from the restored snapshot — a frontier cannot bisect below the
    segment being converted.
    """
    n = int(n_segments)
    k = math.ceil(float(rate) * n - 1e-9)
    return max(0, min(n, k))


class PrefixGenuineForward(LazyExecutorForward):
    """Picklable ``model.forward`` walking the converted-prefix axis (T4/P4).

    Live ``rate`` drives the frontier: ``rate=0`` is the trained proxy exactly,
    ``rate=1`` the deployed single-spike cascade; every intermediate is a
    genuinely partially-deployed network. The lazy executor drops on pickle.
    """

    def __init__(self, model, T: int, rate: float = 0.0,
                 *, boundary_surrogate_temp: float | None = None):
        super().__init__(model, T)
        self.rate = float(rate)
        self.boundary_surrogate_temp = boundary_surrogate_temp

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            PrefixTTFSSegmentForward,
        )

        return PrefixTTFSSegmentForward(
            self.model.get_mapper_repr(), self.T,
            boundary_surrogate_temp=self.boundary_surrogate_temp,
        )

    def _executor_at_rate(self):
        executor = self._ensure_executor(self._build_executor)
        executor.set_prefix(prefix_length_for_rate(self.rate, executor.n_segments))
        return executor

    @property
    def n_segments(self) -> int:
        return self._ensure_executor(self._build_executor).n_segments

    @property
    def prefix_k(self) -> int:
        return prefix_length_for_rate(self.rate, self.n_segments)

    def forward_with_node_values(self, x):
        """The k-hybrid walk recording every perceptron's decoded/proxy value."""
        return self._executor_at_rate().forward_with_node_values(x)

    def _run(self, x):
        return self._executor_at_rate()(x)

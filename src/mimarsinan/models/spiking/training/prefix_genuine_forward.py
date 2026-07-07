"""Prefix-conversion k-hybrid installed as a ramping ``model.forward`` override."""

from __future__ import annotations

from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.orchestration.frontier import frontier_position


def prefix_length_for_rate(rate: float, n_segments: int) -> int:
    """Converted-prefix length for a ladder rate: the frontier-geometry SSOT mapping."""
    return frontier_position(rate, n_segments)


class PrefixGenuineForward(LazyExecutorForward):
    """Picklable ``model.forward`` walking the converted-prefix axis (T4/P4).

    Live ``rate`` drives the frontier: ``rate=0`` is the trained proxy exactly,
    ``rate=1`` the deployed single-spike cascade; every intermediate is a
    genuinely partially-deployed network. The lazy executor drops on pickle.
    """

    def __init__(self, model, T: int, rate: float = 0.0,
                 *, boundary_surrogate_temp: float | None = None,
                 hop_frontier: bool = False):
        super().__init__(model, T)
        self.rate = float(rate)
        self.boundary_surrogate_temp = boundary_surrogate_temp
        # [5v B2] frontier units: spike segments (default) or, for a
        # single-segment deep chain, the cascade hops inside it.
        self.hop_frontier = bool(hop_frontier)

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
        if self.hop_frontier:
            executor.set_hop_prefix(
                prefix_length_for_rate(self.rate, executor.n_hop_levels)
            )
        else:
            executor.set_prefix(
                prefix_length_for_rate(self.rate, executor.n_segments)
            )
        return executor

    @property
    def n_segments(self) -> int:
        return self._ensure_executor(self._build_executor).n_segments

    @property
    def frontier_units(self) -> int:
        executor = self._ensure_executor(self._build_executor)
        return executor.n_hop_levels if self.hop_frontier else executor.n_segments

    @property
    def frontier_k(self) -> int:
        return prefix_length_for_rate(self.rate, self.frontier_units)

    @property
    def prefix_k(self) -> int:
        return prefix_length_for_rate(self.rate, self.n_segments)

    def forward_with_node_values(self, x):
        """The k-hybrid walk recording every perceptron's decoded/proxy value."""
        return self._executor_at_rate().forward_with_node_values(x)

    def _run(self, x):
        return self._executor_at_rate()(x)

"""Segment-aware TTFS spike-train forward — thin wrapper over the unified segment driver."""

from __future__ import annotations

from mimarsinan.spiking.segment_forward import (
    SegmentForwardDriver,
    TtfsSegmentPolicy,
    classify_spike_producers,
    partition_perceptron_segments,
    partition_spike_segments,
)

__all__ = [
    "PrefixTTFSSegmentForward",
    "TTFSSegmentForward",
    "classify_spike_producers",
    "partition_perceptron_segments",
    "partition_spike_segments",
]


class TTFSSegmentForward:
    """Differentiable segment-aware TTFS spike forward over a ``ModelRepresentation``.

    Install as ``model.forward`` during TTFS-cycle fine-tuning; perceptron activations
    must be ``TTFSActivation`` with ``encoding`` matching ``is_encoding_layer``.
    """

    def __init__(self, mapper_repr, T: int, *, boundary_surrogate_temp: float | None = None):
        self.repr = mapper_repr
        self.T = int(T)
        policy = TtfsSegmentPolicy()
        policy.boundary_surrogate_temp = boundary_surrogate_temp
        self._driver = SegmentForwardDriver(mapper_repr, self.T, policy)

    @property
    def _segments(self) -> dict:
        return self._driver.segments

    def _segment_depths(self, seg_nodes):
        return self._driver.policy.segment_depths(self._driver, seg_nodes)

    def __call__(self, x):
        return self._driver(x)

    def forward_with_node_values(self, x):
        """Run the walk recording each perceptron node's decoded value.

        Returns ``(output, {mapper_node: value_tensor})`` — the NF side of the
        cascaded NF↔SCM per-neuron parity comparison.
        """
        recorder: dict = {}
        self._driver.policy.node_value_recorder = recorder
        try:
            out = self._driver(x)
        finally:
            self._driver.policy.node_value_recorder = None
        return out, recorder


class PrefixTTFSSegmentForward(TTFSSegmentForward):
    """The P4 k-hybrid: converted-prefix segments run the deployed cascade, the
    suffix runs the trained proxy on decoded values. ``set_prefix(n)`` IS the
    deployed forward (``genuine_segments`` collapses to None)."""

    @property
    def n_segments(self) -> int:
        return len(self._driver.segments)

    @property
    def prefix_k(self) -> int:
        genuine = self._driver.policy.genuine_segments
        return self.n_segments if genuine is None else len(genuine)

    def set_prefix(self, k: int) -> None:
        k = int(k)
        self._driver.policy.genuine_segments = (
            None if k >= self.n_segments else frozenset(range(max(0, k)))
        )

    @property
    def n_hop_levels(self) -> int:
        """Cascade-depth levels of the single spike segment ([5v B2] frontier units)."""
        driver = self._driver
        levels = 0
        for seg_nodes in driver.segments.values():
            ordered = sorted(seg_nodes, key=lambda n: driver._index[n])
            depths = driver.policy.segment_depths(driver, ordered)
            levels = max(levels, max(depths.values(), default=0) + 1)
        return levels

    def set_hop_prefix(self, k: int) -> None:
        """[5v B2] the frontier below segments: hops with depth < k run the
        deployed cascade, deeper hops the trained proxy. ``k >= n_hop_levels``
        IS the deployed forward (the frontier collapses to None)."""
        assert self.n_segments == 1, (
            "the hop frontier is defined for single-segment vehicles; "
            "multi-segment graphs walk the segment frontier (set_prefix)."
        )
        k = int(k)
        self._driver.policy.genuine_hop_frontier = (
            None if k >= self.n_hop_levels else max(0, k)
        )

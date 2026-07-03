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

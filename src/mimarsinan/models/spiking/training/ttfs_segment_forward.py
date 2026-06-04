"""Segment-aware TTFS spike-train forward — thin wrapper over the unified segment driver.

The cascaded ``ttfs_cycle_based`` deployment runs each neural segment as a
single-spike, ramp-integrate, fire-once simulation, with value-domain compute
ops between segments and a host-side *encoding layer* (value -> TTFS spike) at
each segment entry. :class:`TTFSSegmentForward` reproduces that on the trainable
model (differentiable) by driving :class:`SegmentForwardDriver` with the
:class:`TtfsSegmentPolicy` — see ``mimarsinan/spiking/segment_forward.py`` for
the shared walk and the TTFS latency-window/latch semantics.
"""

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

    Install as ``model.forward`` during TTFS-cycle fine-tuning (the analog of
    LIF's ``run_cycle_accurate`` install). Perceptron activations must be
    ``TTFSActivation`` with ``encoding`` set to match ``is_encoding_layer``.
    """

    def __init__(self, mapper_repr, T: int):
        self.repr = mapper_repr
        self.T = int(T)
        self._driver = SegmentForwardDriver(mapper_repr, self.T, TtfsSegmentPolicy())

    @property
    def _segments(self) -> dict:
        return self._driver.segments

    def _segment_depths(self, seg_nodes):
        return self._driver.policy.segment_depths(self._driver, seg_nodes)

    def __call__(self, x):
        return self._driver(x)

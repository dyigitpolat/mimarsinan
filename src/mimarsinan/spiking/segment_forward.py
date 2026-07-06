"""Unified segment-aware NF forward: one exec-graph driver, per-mode neuron policies."""

from __future__ import annotations

import torch

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.spiking.segment_partition import (
    classify_spike_producers,
    is_encoding_perceptron,
    is_value_boundary,
    partition_perceptron_segments,
    partition_spike_segments,
    perceptron_of,
)
from mimarsinan.spiking.segment_policies import (
    AnalyticalSegmentPolicy,
    LifSegmentPolicy,
    TtfsSegmentPolicy,
)
from mimarsinan.spiking.segment_policy_ttfs import segment_series_roots

__all__ = [
    "AnalyticalSegmentPolicy",
    "LifSegmentPolicy",
    "SegmentForwardDriver",
    "TtfsSegmentPolicy",
    "classify_spike_producers",
    "is_encoding_perceptron",
    "is_value_boundary",
    "partition_perceptron_segments",
    "partition_spike_segments",
    "perceptron_of",
    "segment_series_roots",
]


class SegmentForwardDriver:
    """Segment-aware forward over a ``ModelRepresentation`` exec graph.

    Owns the mode-agnostic walk (classification, partition, host ComputeOps,
    eviction); segment-internal spike dynamics are delegated to ``policy.run_segment``.
    """

    def __init__(self, mapper_repr, T: int, policy):
        self.repr = mapper_repr
        self.T = int(T)
        self.policy = policy
        self.repr._ensure_exec_graph()
        self._exec = self.repr._exec_order
        self._deps = self.repr._deps
        self._output = self.repr.output_layer_mapper
        self._index = {n: i for i, n in enumerate(self._exec)}

        self._seg_of, self._produces = partition_spike_segments(self._exec, self._deps)
        segments: dict = {}
        for node, root in self._seg_of.items():
            segments.setdefault(root, []).append(node)
        for root in segments:
            segments[root].sort(key=lambda n: self._index[n])
        self._segments = segments

        consumers: dict = {n: [] for n in self._exec}
        for node in self._exec:
            for dep in self._deps.get(node, []):
                if dep in consumers:
                    consumers[dep].append(node)
        self._consumers = consumers

        self._assert_segment_inputs_precede_segments()

    @property
    def segments(self) -> dict:
        return self._segments

    def external_consumed(self, seg_nodes):
        """Region nodes whose decoded value is needed outside the region."""
        seg_set = set(seg_nodes)
        ext = []
        for n in seg_nodes:
            if n is self._output or any(c not in seg_set for c in self._consumers.get(n, [])):
                ext.append(n)
        return ext

    def _assert_segment_inputs_precede_segments(self):
        """A segment runs atomically at its first member, so every external input
        must already be computed there; host ops interleaved inside a neural
        segment are not supported."""
        for seg_nodes in self._segments.values():
            first = min(self._index[n] for n in seg_nodes)
            seg_set = set(seg_nodes)
            for n in seg_nodes:
                for dep in self._deps.get(n, []):
                    if dep in seg_set:
                        continue
                    if dep in self._seg_of:
                        other = self._segments[self._seg_of[dep]]
                        dep_pos = min(self._index[m] for m in other)
                    else:
                        dep_pos = self._index[dep]
                    if dep_pos > first:
                        raise NotImplementedError(
                            "segment-aware forward: a neural segment input "
                            f"({dep!r}) is produced after the segment starts; "
                            "host ops interleaved inside a neural segment are "
                            "not supported"
                        )

    def __call__(
        self, x, *,
        compute_min_recorder: dict | None = None,
        node_value_recorder: dict | None = None,
        join_value_recorder: dict | None = None,
    ):
        # The recorders are pure side-channels: they never alter the forward output.
        self._node_value_recorder = node_value_recorder
        self._join_value_recorder = join_value_recorder
        self.policy.prepare(self)
        try:
            return self._run(x, compute_min_recorder)
        finally:
            self.policy.finalize(self)
            self._node_value_recorder = None
            self._join_value_recorder = None

    def _forward_node(self, node, values, x):
        d = self._deps.get(node, [])
        if len(d) == 0:
            return node.forward(x)
        if len(d) == 1:
            return node.forward(values[d[0]])
        return node.forward(tuple(values[dep] for dep in d))

    def _run_value_node(self, node, values, x, compute_min_recorder):
        value = self._forward_node(node, values, x)
        if isinstance(node, ComputeOpMapper):
            if compute_min_recorder is not None:
                cur = value.detach().amin(dim=0)
                prev = compute_min_recorder.get(node)
                compute_min_recorder[node] = (
                    cur if prev is None else torch.minimum(prev, cur)
                )
            # Positive-domain shift on the decoded value; the consumer perceptron's baked bias compensates.
            shift = getattr(node, "_negative_shift", None)
            if shift is not None:
                value = value + torch.as_tensor(shift, dtype=value.dtype, device=value.device)
            join_recorder = getattr(self, "_join_value_recorder", None)
            if join_recorder is not None and len(self._deps.get(node, [])) >= 2:
                # Post-shift: this is the value the boundary re-encode normalizes.
                join_recorder[node] = value.detach()
        values[node] = value

    def _run(self, x, compute_min_recorder):
        values: dict = {}
        output_value = None
        done = set()
        remaining = {n: len(self._consumers.get(n, [])) for n in self._exec}

        for node in self._exec:
            if self._produces[node]:
                root = self._seg_of[node]
                if root not in done:
                    seg_out = self.policy.run_segment(self, self._segments[root], values, x)
                    if seg_out is not None:
                        output_value = seg_out
                    done.add(root)
            else:
                self._run_value_node(node, values, x, compute_min_recorder)

            for dep in self._deps.get(node, []):
                if dep is None or dep is self._output:
                    continue
                remaining[dep] -= 1
                if remaining[dep] == 0 and dep in values:
                    del values[dep]

        return output_value if output_value is not None else values[self._output]

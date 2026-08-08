"""Streamed span topology [plan §9]: per-segment streaming, end-to-end reported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.verification.onchip_fraction import (
    _build_flow,
    _exec_nodes,
    _host_unit,
    _onchip_unit,
)


class NotStreamableError(ValueError):
    """The model maps no on-chip neural span at all — nothing can stream."""


@dataclass(frozen=True)
class StreamedSpanReport:
    """Span topology of a hybrid program under the streamed discipline.

    Streaming is per-Neural-Segment: spikes flow cycle-by-cycle within each
    segment; host ops between segments operate on window counts and the next
    segment re-encodes. ``end_to_end`` is the one-segment special case."""

    segments: int
    interior_host_ops: Tuple[str, ...]

    @property
    def end_to_end(self) -> bool:
        return self.segments == 1

    def describe(self) -> str:
        if self.end_to_end:
            return (
                "streamed span topology: 1 neural segment — end-to-end "
                "(spikes cross the count boundary only at encode/readout)"
            )
        ops = ", ".join(repr(n) for n in self.interior_host_ops)
        return (
            f"streamed span topology: {self.segments} neural segments; "
            f"host op(s) {ops} run between segments on window counts "
            f"(per-segment streaming; end-to-end needs a pooling-free "
            f"architecture, e.g. strided conv)"
        )


def _report_from_kinds(kinds: Sequence[Tuple[str, str]]) -> StreamedSpanReport:
    """``kinds`` is the ordered ``(name, "neural"|"host")`` classification
    (structural nodes excluded)."""
    neural_positions = [i for i, (_, kind) in enumerate(kinds) if kind == "neural"]
    if not neural_positions:
        raise NotStreamableError(
            "streamed lif requires an on-chip neural span, but the model maps "
            "no neural cores at all (everything runs host-side)."
        )
    first, last = neural_positions[0], neural_positions[-1]
    interior = tuple(
        name for i, (name, kind) in enumerate(kinds)
        if kind == "host" and first < i < last
    )
    segments = 1 + sum(
        1 for prev, cur in zip(neural_positions, neural_positions[1:])
        if cur - prev > 1
    )
    return StreamedSpanReport(segments=segments, interior_host_ops=interior)


def streamed_span_report_ir(ir_graph) -> StreamedSpanReport:
    """Span topology of the mapped IR (authoritative post-mapping view)."""
    kinds = [
        (node.name, "neural" if isinstance(node, NeuralCore) else "host")
        for node in ir_graph.nodes
    ]
    return _report_from_kinds(kinds)


def streamed_span_report_model(
    model, input_shape, num_classes, *, encoding_placement: str = "subsume",
) -> StreamedSpanReport:
    """Static span topology of the model SPEC (before pretraining): classifies
    the mapper exec order with the SAME host/on-chip units the on-chip-majority
    estimate uses."""
    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    kinds: List[Tuple[str, str]] = []
    for node in _exec_nodes(flow) or ():
        if _host_unit(node) is not None:
            kinds.append((getattr(node, "name", type(node).__name__), "host"))
        elif _onchip_unit(node) is not None:
            kinds.append((getattr(node, "name", type(node).__name__), "neural"))
    return _report_from_kinds(kinds)

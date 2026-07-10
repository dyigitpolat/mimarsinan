"""Depth-balancing relay insertion for unequal-depth intra-segment fan-in (C5/V6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph

# Sub-1/T identity margin: the strict "<" comparator never fires on an
# exact-theta charge, so a relay's identity weight must exceed theta; the
# accumulated subtractive-reset residual n*margin stays below theta for any
# realistic window (n < 2**20 cycles).
RELAY_WEIGHT_MARGIN = 2.0 ** -20

RELAY_MARKER = "_depth_relay"


class UnbalancedFanInError(ValueError):
    """A live intra-segment edge spans a latency gap > 1 (head-drop + stale re-read)."""


class DeadRelayError(ValueError):
    """A relay's identity weight does not clear its threshold under the comparator."""


@dataclass(frozen=True)
class DepthGapViolation:
    producer_id: int
    consumer_id: int
    gap: int


def relay_weight(thresholding_mode: str) -> float:
    """Identity relay weight that fires on every input spike under the comparator."""
    if thresholding_mode == "<":
        return 1.0 + RELAY_WEIGHT_MARGIN
    return 1.0


def _live_axon_flags(node: NeuralCore, ir_graph: IRGraph) -> list[bool]:
    """Per-axon liveness (any nonzero weight in the axon's row); conservative
    all-live when the matrix rows cannot be aligned to the sources."""
    sources = list(node.input_sources.flatten())
    matrix = node.get_core_matrix(ir_graph)
    if matrix is None or matrix.shape[0] != len(sources):
        return [True] * len(sources)
    return [bool(np.any(np.abs(matrix[i, :]) > 0.0)) for i in range(len(sources))]


def _segment_depths(seg: NeuralSegment, ir_graph: IRGraph) -> dict[int, int]:
    """Longest live intra-segment path depth per node (list order is topological)."""
    ids = {node.id for node in seg.nodes}
    depth: dict[int, int] = {}
    for node in seg.nodes:
        live = _live_axon_flags(node, ir_graph)
        upstream = [
            depth[src.node_id]
            for i, src in enumerate(node.input_sources.flatten())
            if isinstance(src, IRSource)
            and src.node_id in ids
            and live[i]
            and src.node_id in depth
        ]
        depth[node.id] = (max(upstream) + 1) if upstream else 0
    return depth


def find_intra_segment_depth_gaps(ir_graph: IRGraph) -> list[DepthGapViolation]:
    """Every live intra-segment edge whose latency gap exceeds 1."""
    violations: list[DepthGapViolation] = []
    seen: set[tuple[int, int]] = set()
    for seg in partition_ir_graph(ir_graph):
        if not isinstance(seg, NeuralSegment):
            continue
        depth = _segment_depths(seg, ir_graph)
        for node in seg.nodes:
            live = _live_axon_flags(node, ir_graph)
            for i, src in enumerate(node.input_sources.flatten()):
                if not isinstance(src, IRSource) or src.node_id not in depth:
                    continue
                if not live[i]:
                    continue
                gap = depth[node.id] - depth[src.node_id]
                key = (src.node_id, node.id)
                if gap > 1 and key not in seen:
                    seen.add(key)
                    violations.append(
                        DepthGapViolation(src.node_id, node.id, gap)
                    )
    return violations


def assert_intra_segment_gap1(ir_graph: IRGraph) -> None:
    """Loud post-latency join-safety guard: every live intra-segment edge must
    have gap <= 1 (Theorem 2's window-coverage precondition)."""
    violations = find_intra_segment_depth_gaps(ir_graph)
    if violations:
        detail = ", ".join(
            f"{v.producer_id}->{v.consumer_id} (gap {v.gap})" for v in violations
        )
        raise UnbalancedFanInError(
            f"unequal-depth intra-segment fan-in: {detail}. The consumer both "
            f"drops the shallow branch's early spikes and re-reads its stale "
            f"final buffer (V6). Exact remedy: depth-balancing relay insertion "
            f"(lif_depth_balancing_relays)."
        )


def assert_relays_alive(ir_graph: IRGraph, *, thresholding_mode: str) -> None:
    """Dead-relay lattice guard: under the strict '<' comparator an exact-theta
    identity weight NEVER fires; fail loud before emitting a silent mapping."""
    for node in ir_graph.nodes:
        if not getattr(node, RELAY_MARKER, False):
            continue
        assert isinstance(node, NeuralCore), f"relay {node.name} is not a NeuralCore"
        matrix = node.get_core_matrix(ir_graph)
        assert matrix is not None, f"relay {node.name} lost its core matrix"
        diag_min = float(np.min(np.diagonal(matrix)))
        threshold = float(node.threshold)
        dead = (
            diag_min <= threshold
            if thresholding_mode == "<"
            else diag_min < threshold
        )
        if dead:
            raise DeadRelayError(
                f"depth relay {node.name!r} (id={node.id}) is dead: identity "
                f"weight {diag_min!r} vs threshold {threshold!r} under "
                f"{thresholding_mode!r} never fires (an exact-theta charge is "
                f"silent under the strict comparator — the integer-lattice "
                f"hazard, V9). Re-quantize with headroom or use '<='."
            )


def _relay_activation_scale(producer: NeuralCore, channels: list[int]):
    scale = torch.as_tensor(producer.activation_scale).detach().clone()
    if scale.dim() >= 1 and scale.numel() > 1:
        return scale.reshape(-1)[channels].clone()
    return scale


def _build_relay_chain(
    producer: NeuralCore,
    channels: list[int],
    length: int,
    *,
    weight: float,
    next_id: int,
) -> list[NeuralCore]:
    chain: list[NeuralCore] = []
    k = len(channels)
    prev_id = producer.id
    prev_indices = channels
    for d in range(1, length + 1):
        relay = NeuralCore(
            id=next_id,
            name=f"{producer.name}_depth_relay_{d}",
            input_sources=np.asarray(
                [IRSource(prev_id, idx) for idx in prev_indices], dtype=object,
            ),
            core_matrix=np.eye(k, dtype=np.float64) * weight,
            threshold=1.0,
            activation_scale=_relay_activation_scale(producer, channels),
            parameter_scale=torch.tensor(0.0),
            latency=(
                producer.latency + d if producer.latency is not None else None
            ),
        )
        setattr(relay, RELAY_MARKER, True)
        chain.append(relay)
        prev_id = relay.id
        prev_indices = list(range(k))
        next_id += 1
    return chain


def insert_depth_balancing_relays(
    ir_graph: IRGraph, *, thresholding_mode: str,
) -> int:
    """Insert identity relay chains on every gap>1 intra-segment edge.

    Consumers at gap ``g`` are rewired to relay ``g-1`` of the shared chain,
    making every live intra-segment edge gap-1 exact. No-op (zero mutation)
    on gap-free graphs; idempotent. Returns the number of relays inserted.
    """
    violations = find_intra_segment_depth_gaps(ir_graph)
    if not violations:
        return 0

    weight = relay_weight(thresholding_mode)
    next_id = max(node.id for node in ir_graph.nodes) + 1
    inserted = 0

    by_producer: dict[int, list[DepthGapViolation]] = {}
    for violation in violations:
        by_producer.setdefault(violation.producer_id, []).append(violation)

    for producer_id, viols in by_producer.items():
        producer = ir_graph.get_node_by_id(producer_id)
        assert isinstance(producer, NeuralCore)
        consumers: list[tuple[NeuralCore, int]] = []
        for v in viols:
            consumer_node = ir_graph.get_node_by_id(v.consumer_id)
            assert isinstance(consumer_node, NeuralCore)
            consumers.append((consumer_node, v.gap))
        channels = sorted({
            int(src.index)
            for consumer, _gap in consumers
            for i, src in enumerate(consumer.input_sources.flatten())
            if isinstance(src, IRSource)
            and src.node_id == producer_id
            and _live_axon_flags(consumer, ir_graph)[i]
        })
        chain = _build_relay_chain(
            producer,
            channels,
            max(gap for _c, gap in consumers) - 1,
            weight=weight,
            next_id=next_id,
        )
        next_id += len(chain)
        inserted += len(chain)

        position = ir_graph.nodes.index(producer) + 1
        ir_graph.nodes[position:position] = chain

        for consumer, gap in consumers:
            target = chain[gap - 2]
            live = _live_axon_flags(consumer, ir_graph)
            flat = consumer.input_sources.flatten()
            for i, src in enumerate(flat):
                if (
                    isinstance(src, IRSource)
                    and src.node_id == producer_id
                    and live[i]
                ):
                    flat[i] = IRSource(target.id, channels.index(int(src.index)))
            consumer.input_sources = flat.reshape(consumer.input_sources.shape)

    assert_relays_alive(ir_graph, thresholding_mode=thresholding_mode)
    assert_intra_segment_gap1(ir_graph)
    return inserted

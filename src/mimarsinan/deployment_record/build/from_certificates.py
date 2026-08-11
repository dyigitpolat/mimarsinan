"""Eager reduction of spike-count-gate node counts into boundary traffic records.

The reduction runs AT THE GATE SCOPE, right after the certificate passes:
each ``(B, n)`` count tensor collapses to scalar totals/maxima and is never
retained (bounded memory; raw tensors never persist — schema §9.2).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema import BoundaryTrafficRecord


def boundary_traffic_from_node_counts(
    node_counts: Mapping[int, Any],
    *,
    stage_index_by_node: Optional[Mapping[int, int]] = None,
) -> Tuple[BoundaryTrafficRecord, ...]:
    """Reduce ``{node_id: (B, n) count tensor}`` to per-node totals/maxima.

    ``total_count`` sums every (sample, neuron) window count; ``max_neuron_count``
    is the largest single window count observed. Records hold only scalars.
    """
    records = []
    for node_id in sorted(int(key) for key in node_counts):
        counts = node_counts[node_id]
        if len(counts.shape) != 2:
            raise ValueError(
                f"node {node_id}: expected a (B, n) count tensor, "
                f"got shape {tuple(counts.shape)}"
            )
        samples, neurons = (int(dim) for dim in counts.shape)
        empty = samples == 0 or neurons == 0
        stage_index = (
            None if stage_index_by_node is None
            else stage_index_by_node.get(node_id)
        )
        records.append(BoundaryTrafficRecord(
            node_id=node_id,
            producing_stage_index=None if stage_index is None else int(stage_index),
            neurons=neurons,
            samples=samples,
            total_count=0 if empty else int(counts.sum().item()),
            max_neuron_count=0 if empty else int(counts.max().item()),
        ))
    return tuple(records)

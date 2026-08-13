"""The sealed on-chip/host split, from the same SSOTs the validity gate reads."""

from __future__ import annotations

from typing import Any, Sequence

from mimarsinan.deployment_record.schema import ComputePartitionRecord
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import compute_onchip_fraction


def partition_record_from_breakdowns(params: Any, macs: Any) -> ComputePartitionRecord:
    """Assemble the record from the gate's own breakdown objects.

    ``params`` is an ``OnchipParamBreakdown`` (mapped-IR remainder), ``macs`` an
    ``OnchipFractionEstimate`` with ``metric == "macs"`` — duck-typed so this stays a
    pure field mapping.
    """
    if getattr(macs, "metric", "macs") != "macs":
        raise ValueError(
            f"partition wants the forward-MAC estimate, got metric {macs.metric!r}"
        )
    return ComputePartitionRecord(
        onchip_params=int(params.onchip_params),
        host_params=int(params.host_params),
        total_params=int(params.total_params),
        onchip_macs=int(macs.onchip),
        host_macs=int(macs.host),
        total_macs=int(macs.total),
    )


def compute_partition_record(
    ir_graph: Any,
    model: Any,
    input_shape: Sequence[int],
    num_classes: int,
    *,
    encoding_placement: str,
) -> ComputePartitionRecord:
    """The logical split of a mapped deployment — an ungated census, not a gate.

    Params come from the mapped IR's host remainder (``compute_onchip_fraction``),
    MACs from the model's forward census under the resolved placement
    (``estimate_onchip_fraction``) — the exact SSOT pair behind
    ``assert_onchip_validity_or_raise``, so the sealed fact and the gate can never
    disagree.
    """
    total_params = int(sum(p.numel() for p in model.parameters()))
    params = compute_onchip_fraction(ir_graph, total_params=total_params)
    macs = estimate_onchip_fraction(
        model,
        input_shape,
        num_classes,
        encoding_placement=encoding_placement,
        metric="macs",
    )
    return partition_record_from_breakdowns(params, macs)

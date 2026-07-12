"""Membrane-augmented readout: exact charge decode for final-only LIF output cores (Theorem 0)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)


def final_only_output_nodes(hybrid_mapping: HybridHardCoreMapping) -> set[int]:
    """Node ids whose decoded value feeds NOTHING but the network output.

    Only these may claim the residual membrane: a node re-consumed by any other
    stage travels the wire as spikes, where hardware exports counts only.
    """
    final_ids = {
        int(src.node_id)
        for src in hybrid_mapping.output_sources.flatten()
        if isinstance(src, IRSource) and src.node_id >= 0
    }
    consumed: set[int] = set()
    for stage in hybrid_mapping.stages:
        if stage.kind == "neural":
            consumed.update(
                int(s.node_id)
                for s in stage.input_map
                if s.node_id is not None and int(s.node_id) >= 0
            )
        elif stage.kind == "compute" and stage.compute_op is not None:
            consumed.update(
                int(src.node_id)
                for src in stage.compute_op.input_sources.flatten()
                if isinstance(src, IRSource) and src.node_id >= 0
            )
    return final_ids - consumed


def membrane_readout_slices(
    hybrid_mapping: HybridHardCoreMapping, stage: HybridStage,
) -> list[tuple[int, int, int]]:
    """``(node_id, start, end)`` segment-output slices of ``stage`` eligible
    for the membrane readout."""
    eligible = final_only_output_nodes(hybrid_mapping)
    return [
        (int(s.node_id), int(s.offset), int(s.offset) + int(s.size))
        for s in stage.output_map
        if s.node_id is not None and int(s.node_id) in eligible
    ]


def membrane_readout_ranges(
    hybrid_mapping: HybridHardCoreMapping, stage: HybridStage,
) -> list[tuple[int, int]]:
    """Segment-output index ranges of ``stage`` eligible for the membrane readout."""
    return [
        (start, end)
        for _node_id, start, end in membrane_readout_slices(hybrid_mapping, stage)
    ]


def stash_membrane_readout_correction(
    host,
    *,
    seg: dict,
    stage: HybridStage,
    output_counts: torch.Tensor,
    output_spans,
    neuron_states: list[dict],
    thresholds: list[torch.Tensor],
    single_spike: bool,
    readout_corrections: Dict[int, torch.Tensor] | None,
) -> None:
    """Gate + compute the C2 correction for a finished segment window (LIF only;
    eligible slices memoized on the segment tensor cache).

    The correction is a HOST-READ decode: it never touches ``output_counts``
    (the per-neuron spike-count currency that parity gathers and cross-sim
    comparisons read). Eligible node slices land in ``readout_corrections``
    keyed by node id, for the decode-to-logits boundary to consume; a ``None``
    side channel means the caller does not decode logits and gets pure counts.
    """
    if (
        readout_corrections is None
        or not getattr(host, "membrane_readout", False)
        or single_spike
        or not is_lif(host.spiking_mode)
    ):
        return
    eligible = seg.get("membrane_readout_slices")
    if eligible is None:
        eligible = membrane_readout_slices(host.hybrid_mapping, stage)
        seg["membrane_readout_slices"] = eligible
    if not eligible:
        return
    correction = torch.zeros_like(output_counts)
    accumulate_membrane_correction(
        correction,
        output_spans=output_spans,
        neuron_states=neuron_states,
        thresholds=thresholds,
        eligible_ranges=[(start, end) for _node_id, start, end in eligible],
        half_step_charge=(
            0.5 if getattr(host, "membrane_readout_half_step", True) else 0.0
        ),
    )
    for node_id, start, end in eligible:
        readout_corrections[node_id] = correction[:, start:end]


def accumulate_membrane_correction(
    correction: torch.Tensor,
    *,
    output_spans,
    neuron_states: list[dict],
    thresholds: list[torch.Tensor],
    eligible_ranges: list[tuple[int, int]],
    half_step_charge: float,
) -> None:
    """Add ``m_T/theta - half_step_charge`` in-place on eligible core spans.

    ``Q_T = theta*c_T + m_T`` (Theorem 0), so counts plus this correction
    report the exact unquantized terminal charge in threshold units. Spans
    without a membrane (``input``/``on``) are never claimed.
    """
    for sp in output_spans:
        if sp.kind != "core":
            continue
        d0 = int(sp.dst_start)
        d1 = int(sp.dst_end)
        for e0, e1 in eligible_ranges:
            lo = max(d0, e0)
            hi = min(d1, e1)
            if lo >= hi:
                continue
            core = int(sp.src_core)
            s0 = int(sp.src_start) + (lo - d0)
            memb = neuron_states[core]["memb"][:, s0 : s0 + (hi - lo)]
            correction[:, lo:hi] += memb / thresholds[core] - half_step_charge


def apply_membrane_corrections_to_logits(
    logits: torch.Tensor,
    readout_corrections: Dict[int, torch.Tensor],
    output_sources: np.ndarray,
) -> torch.Tensor:
    """Add the stashed membrane terms onto the final logits, in count units.

    This is the decode-to-logits boundary — the only currency that may carry
    the sign-carrying charge readout. Mutates ``logits`` in place (callers pass
    a freshly assembled tensor) and returns it; a no-op when nothing was stashed.
    """
    if not readout_corrections:
        return logits
    for idx, src in enumerate(output_sources.flatten()):
        if not isinstance(src, IRSource):
            continue
        correction = readout_corrections.get(int(src.node_id))
        if correction is None:
            continue
        logits[:, idx] += correction[:, int(src.index)].to(logits.dtype)
    return logits

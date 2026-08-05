"""Value-domain neural-segment kernel: one affine pass per core, latency-tier order."""

from __future__ import annotations

import weakref
from typing import Dict, List

import torch

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore.matrix_placement import (
    same_core_matrix_payloads,
)
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid
from mimarsinan.models.spiking.signal_spans import SpanFillPlan

_SEGMENT_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class _PreparedValueSegment:
    """Per-(mapping, dtype, device) tensors: weights, biases, gather plans, order.

    ``resident_from`` aliases the residency-chain head's uploaded weight and
    bias tensors ([wsm V3]: ``schedule_weights_resident`` passes reuse the
    programmed bank content per ordinal — verified at build by the
    placement-geometry SUBSET check — so re-upload is pure waste)."""

    def __init__(
        self, hcm, device: torch.device, dtype: torch.dtype,
        resident_from: "_PreparedValueSegment | None" = None,
        upload_memo: "dict | None" = None,
    ) -> None:
        ensure_core_latencies(hcm)
        self.order: List[int] = sorted(
            range(len(hcm.cores)), key=lambda i: int(hcm.cores[i].latency or 0)
        )
        if resident_from is not None:
            assert len(hcm.cores) <= len(resident_from.weights)
            self.weights = resident_from.weights[: len(hcm.cores)]
            self.biases = resident_from.biases[: len(hcm.cores)]
        else:
            self.weights = [
                _upload(core, dtype, device, upload_memo)
                for core in hcm.cores
            ]
            self.biases = [
                None if (bias := getattr(core, "hardware_bias", None)) is None
                else torch.as_tensor(bias, dtype=dtype, device=device).reshape(1, -1)
                for core in hcm.cores
            ]
        self.thresholds = []
        self.plans = []
        self.entry_transforms = []
        for core in hcm.cores:
            self.thresholds.append(float(core.threshold))
            plan = SpanFillPlan(core.get_axon_source_spans(), device)
            self.plans.append(plan)
            # [mvm AQ] the boundary grid snaps ENTRY cores only (no upstream
            # core sources); the plan applies it to the columns it owns.
            grid = getattr(core, "boundary_grid", None)
            armed = (
                grid if (grid is not None and grid.armed
                         and not plan.has_upstream_core_sources) else None
            )
            self.entry_transforms.append(
                None if armed is None
                else (lambda t, g=armed: quantize_to_value_grid(t, g.scale, g.bits))
            )
        self.output_plan = SpanFillPlan(
            compress_spike_sources(list(hcm.output_sources.flatten())), device
        )
        self.output_size = int(len(hcm.output_sources.flatten()))
        self.axon_counts = [int(w.shape[0]) for w in self.weights]


def ensure_core_latencies(hcm) -> None:
    """Latency tiers double as the value walk's dependency order (invariant:
    a live consumer's latency exceeds every live source's)."""
    if any(core.latency is None for core in hcm.cores):
        ChipLatency(hcm).calculate()


def _upload(core, dtype, device, memo: "dict | None"):
    """Device tensor for the core's weight grid; cores resolving to
    byte-identical grids share one tensor ([F2] — content-keyed, so cores
    whose padded grid is now a transient composite still upload once).
    The memo retains the payloads and re-checks them by identity, so a
    freed-and-reallocated array can never alias through a stale key."""
    if memo is None:
        return torch.as_tensor(core.get_core_matrix(), dtype=dtype, device=device)
    key = (core.core_matrix_key(), dtype, str(device))
    payloads = core.core_matrix_payloads()
    hit = memo.get(key)
    if hit is not None and same_core_matrix_payloads(hit[0], payloads):
        return hit[1]
    tensor = torch.as_tensor(core.get_core_matrix(), dtype=dtype, device=device)
    memo[key] = (payloads, tensor)
    return tensor


def _prepared(
    hcm, device: torch.device, dtype: torch.dtype, resident_head=None,
    upload_memo: "dict | None" = None,
) -> _PreparedValueSegment:
    by_key = _SEGMENT_CACHE.setdefault(hcm, {})
    key = (str(device), dtype)
    prepared = by_key.get(key)
    if prepared is None:
        head = (
            None if resident_head is None or resident_head is hcm
            else _prepared(resident_head, device, dtype, upload_memo=upload_memo)
        )
        prepared = _PreparedValueSegment(
            hcm, device, dtype, resident_from=head, upload_memo=upload_memo
        )
        by_key[key] = prepared
    return prepared


def prepared_segment_cache_for_testing() -> "weakref.WeakKeyDictionary":
    """The live prepared-segment cache (tests assert residency aliasing)."""
    return _SEGMENT_CACHE


def run_neural_segment_values(
    hcm, seg_input: torch.Tensor, resident_head=None,
    upload_memo: "dict | None" = None,
) -> torch.Tensor:
    """Execute one packed segment in the value domain: y_core = (x @ W + b) / theta.

    ``theta`` is the weight-quantization dequant scale stamped by
    ``quantize_ir_graph`` (1.0 on float programs); always-on axons read 1.0
    (the param-encoded bias row); fused cores execute as one wide dot product.
    Entry cores carrying a ``boundary_grid`` snap their input-sourced
    columns onto it with the same formula the model-side
    ``ValueGridQuantizer`` applies.
    """
    device, dtype = seg_input.device, seg_input.dtype
    prepared = _prepared(
        hcm, device, dtype, resident_head=resident_head, upload_memo=upload_memo
    )
    batch = seg_input.shape[0]

    buffers: Dict[int, torch.Tensor] = {}
    for idx in prepared.order:
        signal = torch.empty(
            batch, prepared.axon_counts[idx], device=device, dtype=dtype
        )
        prepared.plans[idx].apply(
            signal, input_spikes=seg_input, buffers=buffers, on_value=1.0,
            input_transform=prepared.entry_transforms[idx],
        )
        out = signal @ prepared.weights[idx]
        bias = prepared.biases[idx]
        if bias is not None:
            out = out + bias
        theta = prepared.thresholds[idx]
        if theta != 1.0:
            out = out / theta
        buffers[idx] = out

    seg_output = torch.empty(
        batch, prepared.output_size, device=device, dtype=dtype
    )
    prepared.output_plan.apply(
        seg_output, input_spikes=seg_input, buffers=buffers, on_value=1.0
    )
    return seg_output

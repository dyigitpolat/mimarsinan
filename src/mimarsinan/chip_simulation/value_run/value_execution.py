"""Value-domain neural-segment kernel: one affine pass per core, latency-tier order."""

from __future__ import annotations

import weakref
from typing import Dict, List

import torch

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.signal_spans import SpanFillPlan

_SEGMENT_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class _PreparedValueSegment:
    """Per-(mapping, dtype, device) tensors: weights, biases, gather plans, order."""

    def __init__(self, hcm, device: torch.device, dtype: torch.dtype) -> None:
        ensure_core_latencies(hcm)
        self.order: List[int] = sorted(
            range(len(hcm.cores)), key=lambda i: int(hcm.cores[i].latency or 0)
        )
        self.weights = [
            torch.as_tensor(core.core_matrix, dtype=dtype, device=device)
            for core in hcm.cores
        ]
        self.biases = []
        self.thresholds = []
        self.plans = []
        for core in hcm.cores:
            bias = getattr(core, "hardware_bias", None)
            self.biases.append(
                None if bias is None
                else torch.as_tensor(bias, dtype=dtype, device=device).reshape(1, -1)
            )
            self.thresholds.append(float(core.threshold))
            self.plans.append(SpanFillPlan(core.get_axon_source_spans(), device))
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


def _prepared(hcm, device: torch.device, dtype: torch.dtype) -> _PreparedValueSegment:
    by_key = _SEGMENT_CACHE.setdefault(hcm, {})
    key = (str(device), dtype)
    prepared = by_key.get(key)
    if prepared is None:
        prepared = _PreparedValueSegment(hcm, device, dtype)
        by_key[key] = prepared
    return prepared


def run_neural_segment_values(hcm, seg_input: torch.Tensor) -> torch.Tensor:
    """Execute one packed segment in the value domain: y_core = (x @ W + b) / theta.

    ``theta`` is the weight-quantization dequant scale stamped by
    ``quantize_ir_graph`` (1.0 on float programs); always-on axons read 1.0
    (the param-encoded bias row); fused cores execute as one wide dot product.
    """
    device, dtype = seg_input.device, seg_input.dtype
    prepared = _prepared(hcm, device, dtype)
    batch = seg_input.shape[0]

    buffers: Dict[int, torch.Tensor] = {}
    for idx in prepared.order:
        signal = torch.empty(
            batch, prepared.axon_counts[idx], device=device, dtype=dtype
        )
        prepared.plans[idx].apply(
            signal, input_spikes=seg_input, buffers=buffers, on_value=1.0
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

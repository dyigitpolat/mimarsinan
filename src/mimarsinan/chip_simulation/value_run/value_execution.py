"""Value-domain neural-segment kernel: one affine pass per core, latency-tier order."""

from __future__ import annotations

import weakref
from typing import Dict, List

import torch

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid
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
        self.entry_input_cols = []
        self.entry_scales = []
        for core in hcm.cores:
            bias = getattr(core, "hardware_bias", None)
            self.biases.append(
                None if bias is None
                else torch.as_tensor(bias, dtype=dtype, device=device).reshape(1, -1)
            )
            self.thresholds.append(float(core.threshold))
            plan = SpanFillPlan(core.get_axon_source_spans(), device)
            self.plans.append(plan)
            # [mvm AQ] boundary quantization applies only at entry cores
            # (no upstream core sources), on the input-sourced columns.
            self.entry_input_cols.append(
                None if plan.has_upstream_core_sources else plan.input_destination()
            )
            scale = getattr(core, "input_activation_scale", None)
            if scale is None:
                self.entry_scales.append(0.0)
            elif isinstance(scale, torch.Tensor):
                self.entry_scales.append(float(scale.max()))
            else:
                self.entry_scales.append(float(scale))
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


def run_neural_segment_values(
    hcm, seg_input: torch.Tensor, activation_bits: "int | None" = None
) -> torch.Tensor:
    """Execute one packed segment in the value domain: y_core = (x @ W + b) / theta.

    ``theta`` is the weight-quantization dequant scale stamped by
    ``quantize_ir_graph`` (1.0 on float programs); always-on axons read 1.0
    (the param-encoded bias row); fused cores execute as one wide dot product.
    ``activation_bits`` arms boundary quantization of entry cores' input
    columns onto their calibrated ``input_activation_scale`` grid — the same
    formula the model-side ``ValueGridQuantizer`` applies.
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
        if activation_bits:
            columns = prepared.entry_input_cols[idx]
            scale = prepared.entry_scales[idx]
            if columns is not None and scale > 0.0:
                kind, selector = columns
                if kind == "slice":
                    lo, hi = selector
                    signal[:, lo:hi] = quantize_to_value_grid(
                        signal[:, lo:hi], scale, activation_bits
                    )
                else:
                    signal[:, selector] = quantize_to_value_grid(
                        signal[:, selector], scale, activation_bits
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

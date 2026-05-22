"""Numpy TTFS neural-segment execution (shared reference for HCM and SANA-FE parity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from mimarsinan.mapping.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.spike_source_spans import SpikeSourceSpan, compress_spike_sources
from mimarsinan.models.ttfs_kernels import ttfs_quantized_activation_np


@dataclass
class SegmentTtfsArrays:
    core_params: List[np.ndarray]
    thresholds: List[np.ndarray]
    hw_biases: List[np.ndarray | None]
    latencies: List[int]
    axon_spans: List[List[SpikeSourceSpan]]
    output_spans: List[SpikeSourceSpan]
    n_output: int
    n_axons_per_core: List[int]
    n_neurons_per_core: List[int]


def segment_ttfs_arrays_from_mapping(mapping: Any) -> SegmentTtfsArrays:
    """Extract numpy segment arrays from a ``HardCoreMapping``."""
    cores = mapping.cores
    axon_spans = []
    for c in cores:
        if hasattr(c, "get_axon_source_spans"):
            axon_spans.append(c.get_axon_source_spans())
        else:
            axon_spans.append(compress_spike_sources(c.axon_sources))
    if hasattr(mapping, "get_output_source_spans"):
        output_spans = mapping.get_output_source_spans()
    else:
        output_spans = compress_spike_sources(list(mapping.output_sources.flatten()))

    weight_banks = getattr(mapping, "weight_banks", None) or {}
    placements = getattr(mapping, "soft_core_placements_per_hard_core", []) or []

    core_params: List[np.ndarray] = []
    hw_biases: List[np.ndarray | None] = []
    thresholds: List[np.ndarray] = []
    latencies: List[int] = []

    for core_idx, core in enumerate(cores):
        used_ax = used_axons(core, min_one=True)
        used_neu = used_neurons(core, min_one=True)
        placement_dicts = placements[core_idx] if core_idx < len(placements) else []
        core_weight: np.ndarray | None = None

        if len(placement_dicts) == 1:
            pd = placement_dicts[0]
            bid = pd.get("weight_bank_id")
            ao = int(pd.get("axon_offset", 0))
            ne_off = int(pd.get("neuron_offset", 0))
            a = int(pd.get("axons", 0))
            n = int(pd.get("neurons", 0))
            if (
                bid is not None
                and ao == 0
                and ne_off == 0
                and a == used_ax
                and n == used_neu
            ):
                bank_mat = weight_banks.get(int(bid))
                if bank_mat is not None:
                    ba0, ba1 = pd.get("bank_axon_range") or (0, a)
                    bn0, bn1 = pd.get("bank_neuron_range") or (0, n)
                    core_weight = np.asarray(
                        bank_mat[int(bn0):int(bn1), int(ba0):int(ba1)],
                        dtype=np.float64,
                    ).T

        if core_weight is None:
            tile = np.asarray(core.core_matrix[:used_ax, :used_neu], dtype=np.float64)
            core_weight = tile.T

        if core_weight.shape != (used_neu, used_ax):
            padded = np.zeros((used_neu, used_ax), dtype=np.float64)
            padded[: core_weight.shape[0], : core_weight.shape[1]] = core_weight
            core_weight = padded
        core_params.append(core_weight)
        bias = getattr(core, "hardware_bias", None)
        if bias is None:
            hw_biases.append(None)
        else:
            hw_biases.append(np.asarray(bias[:used_neu], dtype=np.float64))
        thresholds.append(np.array(float(core.threshold), dtype=np.float64))
        latencies.append(int(core.latency) if core.latency is not None else 0)

    n_out = len(mapping.output_sources.flatten())
    return SegmentTtfsArrays(
        core_params=core_params,
        thresholds=thresholds,
        hw_biases=hw_biases,
        latencies=latencies,
        axon_spans=axon_spans,
        output_spans=output_spans,
        n_output=n_out,
        n_axons_per_core=[used_axons(c, min_one=True) for c in cores],
        n_neurons_per_core=[used_neurons(c, min_one=True) for c in cores],
    )


def _assign_span_slice(
    out: np.ndarray,
    *,
    d0: int,
    d1: int,
    src: np.ndarray,
    s0: int,
) -> None:
    """Copy ``src[:, s0:s0+take]`` into ``out[:, d0:d1]``, clamping to buffer bounds."""
    take = int(d1) - int(d0)
    if take <= 0:
        return
    available = max(0, int(src.shape[1]) - int(s0))
    n = min(take, available)
    if n > 0:
        out[:, d0 : d0 + n] = src[:, s0 : s0 + n]


def _fill_signal_numpy(
    out: np.ndarray,
    *,
    input_activations: np.ndarray,
    buffers: Sequence[np.ndarray],
    spans: List[SpikeSourceSpan],
) -> None:
    out.fill(0.0)
    dst_axons = int(out.shape[1])
    for sp in spans:
        d0 = int(sp.dst_start)
        d1 = min(int(sp.dst_end), dst_axons)
        if d0 >= d1:
            continue
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            out[:, d0:d1] = 1.0
            continue
        if sp.kind == "input":
            _assign_span_slice(
                out, d0=d0, d1=d1, src=input_activations, s0=int(sp.src_start),
            )
            continue
        _assign_span_slice(
            out,
            d0=d0,
            d1=d1,
            src=buffers[int(sp.src_core)],
            s0=int(sp.src_start),
        )


def gather_segment_ttfs_output_from_cores(
    seg: SegmentTtfsArrays,
    seg_input: np.ndarray,
    per_core_activations: Sequence[np.ndarray | None],
) -> np.ndarray:
    """Gather segment output activations from per-core TTFS buffers (HCM parity)."""
    batch = int(seg_input.shape[0])
    buffers: List[np.ndarray] = [
        np.zeros((batch, seg.n_neurons_per_core[i]), dtype=np.float64)
        for i in range(len(seg.core_params))
    ]
    for ci, act in enumerate(per_core_activations):
        if act is None or ci >= len(buffers):
            continue
        a = np.asarray(act, dtype=np.float64).reshape(-1)
        n = min(a.size, buffers[ci].shape[1])
        if n > 0:
            buffers[ci][0, :n] = a[:n]
    return _gather_segment_output(
        input_activations=np.asarray(seg_input, dtype=np.float64),
        buffers=buffers,
        output_spans=seg.output_spans,
        n_output=seg.n_output,
    )


def _gather_segment_output(
    *,
    input_activations: np.ndarray,
    buffers: Sequence[np.ndarray],
    output_spans: List[SpikeSourceSpan],
    n_output: int,
) -> np.ndarray:
    batch = input_activations.shape[0]
    output = np.zeros((batch, n_output), dtype=np.float64)
    for sp in output_spans:
        d0 = int(sp.dst_start)
        d1 = min(int(sp.dst_end), n_output)
        if d0 >= d1:
            continue
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            output[:, d0:d1] = 1.0
            continue
        if sp.kind == "input":
            _assign_span_slice(
                output, d0=d0, d1=d1, src=input_activations, s0=int(sp.src_start),
            )
            continue
        _assign_span_slice(
            output,
            d0=d0,
            d1=d1,
            src=buffers[int(sp.src_core)],
            s0=int(sp.src_start),
        )
    return output


def ttfs_core_membrane_voltages(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
) -> List[np.ndarray]:
    """Per-core membrane charge ``V = W @ a + b`` before TTFS activation (HCM parity)."""
    batch = input_activations.shape[0]
    n_cores = len(seg.core_params)
    buffers = [
        np.zeros((batch, seg.n_neurons_per_core[i]), dtype=np.float64)
        for i in range(n_cores)
    ]
    input_signals = [
        np.zeros((batch, seg.n_axons_per_core[i]), dtype=np.float64)
        for i in range(n_cores)
    ]
    membrane: List[np.ndarray] = [
        np.zeros((batch, seg.n_neurons_per_core[i]), dtype=np.float64)
        for i in range(n_cores)
    ]
    topo = sorted(range(n_cores), key=lambda i: seg.latencies[i])
    for ci in topo:
        _fill_signal_numpy(
            input_signals[ci],
            input_activations=input_activations,
            buffers=buffers,
            spans=seg.axon_spans[ci],
        )
        v = input_signals[ci] @ seg.core_params[ci].T
        if seg.hw_biases[ci] is not None:
            v = v + seg.hw_biases[ci]
        membrane[ci] = v.astype(np.float64, copy=False)
    return membrane


def run_ttfs_continuous_segment(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Analytical TTFS: ``relu(W @ x + b) / θ`` per core in latency order."""
    membrane = ttfs_core_membrane_voltages(seg, input_activations)
    batch = input_activations.shape[0]
    n_cores = len(seg.core_params)
    buffers = [
        np.zeros((batch, seg.n_neurons_per_core[i]), dtype=np.float64)
        for i in range(n_cores)
    ]
    for ci in range(n_cores):
        safe_th = np.maximum(seg.thresholds[ci], 1e-12)
        buffers[ci] = np.clip(np.maximum(membrane[ci], 0.0) / safe_th, 0.0, 1.0)
    out = _gather_segment_output(
        input_activations=input_activations,
        buffers=buffers,
        output_spans=seg.output_spans,
        n_output=seg.n_output,
    )
    return out, list(buffers)


def run_ttfs_quantized_segment(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    simulation_length: int,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Closed-form ``ttfs_quantized`` per core (matches HCM / ``ttfs_kernels``)."""
    s = int(simulation_length)
    membrane = ttfs_core_membrane_voltages(seg, input_activations)
    batch = input_activations.shape[0]
    n_cores = len(seg.core_params)
    buffers = [
        np.zeros((batch, seg.n_neurons_per_core[i]), dtype=np.float64)
        for i in range(n_cores)
    ]
    for ci in range(n_cores):
        buffers[ci] = ttfs_quantized_activation_np(membrane[ci], seg.thresholds[ci], s)
    out = _gather_segment_output(
        input_activations=input_activations,
        buffers=buffers,
        output_spans=seg.output_spans,
        n_output=seg.n_output,
    )
    return out, list(buffers)

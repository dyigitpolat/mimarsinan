import heapq
from typing import List, Sequence

import numpy as np

from mimarsinan.chip_simulation.ttfs.segment_arrays import (
    SegmentTtfsArrays,
    segment_ttfs_arrays_from_mapping as segment_ttfs_arrays_from_mapping,
)
from mimarsinan.chip_simulation.ttfs.float_membrane import float_membrane
from mimarsinan.chip_simulation.ttfs.membrane_lattice import lattice_membrane
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation_np
from mimarsinan.models.spiking.wire_semantics import ttfs_quantized_staircase_lattice_np

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
        a = np.asarray(act, dtype=np.float64)
        if a.ndim == 1:
            n = min(a.size, buffers[ci].shape[1])
            if n > 0:
                buffers[ci][0, :n] = a[:n]
        else:
            n = min(a.shape[1], buffers[ci].shape[1])
            if n > 0:
                buffers[ci][:, :n] = a[:, :n]
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


def dependency_order(seg: SegmentTtfsArrays) -> List[int]:
    """[U1 R6] Cores in dependency order (every core after the cores it reads), ties by stored latency then index: the
    stored latency alone is no order once an ALAP alignment raised a producer above a shallower consumer."""
    n_cores = len(seg.core_params)
    reads = [{int(sp.src_core) for sp in seg.axon_spans[i] if sp.kind not in ("off", "on", "input")} - {i}
             for i in range(n_cores)]
    readers: List[List[int]] = [[] for _ in range(n_cores)]
    for i, sources in enumerate(reads):
        for p in sources:
            readers[p].append(i)
    pending = [len(sources) for sources in reads]
    ready = [(seg.latencies[i], i) for i in range(n_cores) if not pending[i]]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        _, ci = heapq.heappop(ready)
        order.append(ci)
        for reader in readers[ci]:
            pending[reader] -= 1
            if not pending[reader]:
                heapq.heappush(ready, (seg.latencies[reader], reader))
    if len(order) != n_cores:
        raise ValueError("TTFS segment: cyclic core dependency")
    return order


def _run_ttfs_segment_ordered(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    *,
    simulation_length: int = 1,
    spiking_mode: str,
    comparator_half_step: bool = False,
) -> tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Execute TTFS cores in dependency order; propagate activations between cores."""
    from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization

    batch = input_activations.shape[0]
    n_cores = len(seg.core_params)
    quantized = forces_activation_quantization(spiking_mode)
    s = max(int(simulation_length), 1)

    buffers: List[np.ndarray] = [
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
    for ci in dependency_order(seg):
        _fill_signal_numpy(
            input_signals[ci],
            input_activations=input_activations,
            buffers=buffers,
            spans=seg.axon_spans[ci],
        )
        hw_bias = seg.hw_biases[ci]
        if quantized:
            # The integer chip's membrane lattice, per neuron from its own inputs (``membrane_lattice``), and every
            # other neuron's float membrane as a fixed function of its own terms (``float_membrane``): float dust and
            # the summation order never decide a staircase tie (t0_15; the per-axon split; co-residency; R1).
            lattice = lattice_membrane(seg, ci, input_signals[ci], s)
            if lattice is not None and bool(lattice[0].all()):
                v = lattice[1] / (2 * s)
            else:
                v = float_membrane(input_signals[ci], seg.core_params[ci], hw_bias)
                if lattice is not None:
                    v = np.where(lattice[0], lattice[1] / (2 * s), v)
            membrane[ci] = v
            fired = ttfs_quantized_activation_np(
                v, seg.thresholds[ci], s,
                comparator_half_step=comparator_half_step,
            )
            if lattice is not None:
                fired = np.where(lattice[0], ttfs_quantized_staircase_lattice_np(
                    lattice[1], seg.thresholds[ci], s, comparator_half_step=comparator_half_step), fired)
            buffers[ci] = fired
        else:
            v = input_signals[ci] @ seg.core_params[ci].T
            if hw_bias is not None:
                v = v + hw_bias
            v = v.astype(np.float64, copy=False)
            membrane[ci] = v
            safe_th = np.maximum(seg.thresholds[ci], 1e-12)
            buffers[ci] = np.clip(np.maximum(v, 0.0) / safe_th, 0.0, 1.0)
    out = _gather_segment_output(
        input_activations=input_activations,
        buffers=buffers,
        output_spans=seg.output_spans,
        n_output=seg.n_output,
    )
    return out, list(buffers), membrane


def ttfs_core_membrane_voltages(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    *,
    simulation_length: int = 1,
    spiking_mode: str = "ttfs",
) -> List[np.ndarray]:
    """Per-core membrane charge ``V = W @ a + b`` before TTFS activation (HCM parity)."""
    _, _, membrane = _run_ttfs_segment_ordered(
        seg,
        input_activations,
        simulation_length=simulation_length,
        spiking_mode=spiking_mode,
    )
    return membrane


def run_ttfs_segment(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    *,
    simulation_length: int,
    spiking_mode: str,
    comparator_half_step: bool = False,
) -> tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """One execution returning ``(out, per-core buffers, per-core membrane V)``."""
    return _run_ttfs_segment_ordered(
        seg,
        input_activations,
        simulation_length=simulation_length,
        spiking_mode=spiking_mode,
        comparator_half_step=comparator_half_step,
    )


def run_ttfs_continuous_segment(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Analytical TTFS: ``relu(W @ x + b) / θ`` per core in latency order."""
    out, buffers, _ = _run_ttfs_segment_ordered(
        seg, input_activations, spiking_mode="ttfs",
    )
    return out, buffers


def run_ttfs_quantized_segment(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    simulation_length: int,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Closed-form ``ttfs_quantized`` per core (matches HCM / ``ttfs_kernels``)."""
    out, buffers, _ = _run_ttfs_segment_ordered(
        seg,
        input_activations,
        simulation_length=simulation_length,
        spiking_mode="ttfs_quantized",
    )
    return out, buffers

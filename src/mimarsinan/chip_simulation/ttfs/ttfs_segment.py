import numpy as np

from typing import List, Sequence

from mimarsinan.chip_simulation.ttfs.segment_arrays import SegmentTtfsArrays, segment_ttfs_arrays_from_mapping
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation_np

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


def _run_ttfs_segment_ordered(
    seg: SegmentTtfsArrays,
    input_activations: np.ndarray,
    *,
    simulation_length: int = 1,
    spiking_mode: str,
) -> tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Execute TTFS cores in latency order; propagate activations between cores."""
    from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization

    batch = input_activations.shape[0]
    n_cores = len(seg.core_params)
    quantized = forces_activation_quantization(spiking_mode)
    s = max(int(simulation_length), 1)

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
        v = v.astype(np.float64, copy=False)
        membrane[ci] = v
        if quantized:
            buffers[ci] = ttfs_quantized_activation_np(v, seg.thresholds[ci], s)
        else:
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

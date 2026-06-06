"""Numpy TTFS neural-segment execution (shared reference for HCM and SANA-FE parity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan, compress_spike_sources
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation_np


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
                    # ``bank.core_matrix`` is stored ``(axons, neurons)`` (see
                    # ``register_weight_bank``); slice axons with the axon range
                    # and neurons with the neuron range, then transpose to the
                    # ``(neurons, axons)`` core_params layout. (Equal for square
                    # cores, which is why the swapped form went unnoticed.)
                    core_weight = np.asarray(
                        bank_mat[int(ba0):int(ba1), int(bn0):int(bn1)],
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

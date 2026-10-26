"""Numpy TTFS neural-segment execution (shared reference for HCM and SANA-FE parity)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from mimarsinan.mapping.packing.executed_reads import bank_read
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan, compress_spike_sources

#: Detects float dust on an integral weight, bias or threshold (the integer chip's check).
INTEGRAL_DUST = 1e-6


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

    def integral_neurons(self, ci: int) -> np.ndarray:
        """``(N,)`` bool: the neurons of core ``ci`` on the integer chip's lattice, their weights and bias integral
        and the core's threshold an integer >= 1. Such a neuron's membrane over inputs on the ``1/(2S)`` grid is
        exact in integers (``membrane_lattice``), so float dust never decides a staircase tie (on t0_15 a true tie
        at V == theta was decided by f-noise: V=16.99999943 against the plugin's exact 17)."""
        cache = getattr(self, "_integral_neurons", None)
        if cache is None:
            cache = self._integral_neurons = {}
        hit = cache.get(ci)
        if hit is None:
            def _int(a) -> np.ndarray:
                arr = np.asarray(a, dtype=np.float64)
                return np.abs(arr - np.round(arr)) <= INTEGRAL_DUST

            hit = np.all(_int(self.core_params[ci]), axis=1)
            bias = self.hw_biases[ci]
            if bias is not None:
                hit = hit & _int(bias)
            theta = np.asarray(self.thresholds[ci], dtype=np.float64)
            if not bool(np.all(_int(theta) & (theta >= 1.0))):
                hit = np.zeros_like(hit)
            cache[ci] = hit
        return hit

    def integer_lattice(self) -> bool:
        """True when every neuron of every core is on the integer chip's lattice (``integral_neurons``)."""
        return all(bool(self.integral_neurons(ci).all()) for ci in range(len(self.core_params)))


# Keyed by mapping identity: a HardCoreMapping is not mutated after packing, so
# the derived arrays are reused across stages/batches instead of re-extracted per
# call (W3 wall); WeakKey keeps pickles clean and lets mappings be collected.
_ARRAYS_CACHE: "weakref.WeakKeyDictionary[Any, SegmentTtfsArrays]" = (
    weakref.WeakKeyDictionary()
)


def segment_ttfs_arrays_from_mapping(mapping: Any) -> SegmentTtfsArrays:
    """Extract numpy segment arrays from a ``HardCoreMapping`` (cached per mapping)."""
    try:
        cached = _ARRAYS_CACHE.get(mapping)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    arrays = _segment_ttfs_arrays_from_mapping_uncached(mapping)
    try:
        _ARRAYS_CACHE[mapping] = arrays
    except TypeError:
        pass
    return arrays


def _segment_ttfs_arrays_from_mapping_uncached(mapping: Any) -> SegmentTtfsArrays:
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

    core_params: List[np.ndarray] = []
    hw_biases: List[np.ndarray | None] = []
    thresholds: List[np.ndarray] = []
    latencies: List[int] = []

    for core_idx, core in enumerate(cores):
        used_ax = used_axons(core, min_one=True)
        used_neu = used_neurons(core, min_one=True)
        core_weight: np.ndarray | None = None

        # [W15] the bank fast path's one definition (``executed_reads.bank_read``), which the static check reads too.
        bank = bank_read(mapping, core_idx)
        if bank is not None:
            bank_mat = weight_banks.get(bank.bank_id)
            if bank_mat is not None:
                # bank.core_matrix is (axons, neurons); slice by axon/neuron range then transpose to the (neurons, axons) core_params layout.
                core_weight = np.asarray(
                    bank_mat[bank.axons[0]:bank.axons[1], bank.neurons[0]:bank.neurons[1]],
                    dtype=np.float64,
                ).T

        if core_weight is None:
            grid = core.get_core_matrix()
            tile = np.asarray(grid[:used_ax, :used_neu], dtype=np.float64)
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

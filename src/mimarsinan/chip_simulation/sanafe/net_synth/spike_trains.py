from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.neuron_model import (
    input_neuron_attributes,
    ttfs_continuous_model_attributes,
    ttfs_quantized_model_attributes,
)
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons


def set_input_spike_trains(
    core_input_neurons: Dict[Tuple[int, int], Any],
    hcm: Any,
    encoded: np.ndarray,
) -> None:
    """Inject spike trains into every per-core input neuron.

    Feeds ``encoded[0, k, :]`` (the ``(1, seg_in_size, T)`` binary tensor) to the
    neuron for axon logical index ``k``; per-core copies of ``k`` share the train.
    """
    for (core_idx, axon_idx), neuron in core_input_neurons.items():
        src = hcm.cores[core_idx].axon_sources[axon_idx]
        k = int(src.neuron_)
        if k >= encoded.shape[1]:
            train: list[int] = []
        else:
            train = [int(b) for b in encoded[0, k, :].tolist()]
        neuron.set_attributes(
            model_attributes=input_neuron_attributes(train),
        )


def set_always_on_spike_trains(
    core_always_on_neurons: Dict[int, Any],
    T: int,
    *,
    spiking_mode: str = "lif",
    core_latencies: Dict[int, int] | None = None,
) -> None:
    """Inject always-on (bias) input spikes as a per-timestep binary mask.

    rate/LIF fire every cycle; single-spike TTFS places one spike at ``core_latency``
    so SANA-FE's one-cycle input delay lands it at the soma's gated ``active_start``.
    """
    from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

    ttfs = requires_ttfs_firing(spiking_mode)
    for core_idx, neuron in core_always_on_neurons.items():
        if not ttfs:
            train = [1] * T
        elif core_latencies is not None:
            lat = int(core_latencies.get(core_idx, 0))
            train = [0] * lat + [1]
        else:
            train = [1]
        neuron.set_attributes(
            model_attributes=input_neuron_attributes(train),
        )


def apply_ttfs_preset_membranes(
    core_to_group: Dict[int, Any],
    hcm: Any,
    membrane_voltages: list,
    *,
    spiking_mode: str,
    simulation_length: Optional[int] = None,
    firing_mode: str = "Default",
) -> None:
    """Stamp analytical ``V = W @ a + b`` on TTFS somas so SANA-FE matches HCM."""
    for core_idx, group in core_to_group.items():
        core = hcm.cores[core_idx]
        used_neurons = _used_neurons(core)
        if used_neurons <= 0:
            continue
        V = membrane_voltages[core_idx]
        core_latency = (int(core.latency)
                        if getattr(core, "latency", None) is not None
                        else 0)
        active_start = ((core_latency + 1)
                        if simulation_length is not None else None)
        from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization

        is_quantized = forces_activation_quantization(spiking_mode)
        if is_quantized and simulation_length is not None:
            active_length = int(simulation_length)
        elif simulation_length is not None:
            active_length = 1
        else:
            active_length = None
        for n_idx in range(used_neurons):
            bias = None
            if core.hardware_bias is not None:
                bias = float(np.asarray(core.hardware_bias)[n_idx])
            preset = float(V[0, n_idx])
            if is_quantized:
                attrs = ttfs_quantized_model_attributes(
                    threshold=float(core.threshold),
                    hardware_bias=bias,
                    active_start=active_start,
                    active_length=active_length,
                    preset_membrane=preset,
                )
            else:
                attrs = ttfs_continuous_model_attributes(
                    threshold=float(core.threshold),
                    hardware_bias=bias,
                    active_start=active_start,
                    active_length=active_length,
                    preset_membrane=preset,
                )
            group[n_idx].set_attributes(model_attributes=attrs)


def set_ttfs_input_spike_trains(
    core_input_neurons: Dict[Tuple[int, int], Any],
    hcm: Any,
    seg_input_rates: np.ndarray,
    simulation_length: int,
) -> None:
    """Inject latched TTFS spike times from segment input rates ``(1, D)``.

    Used by ``ttfs``/``ttfs_quantized``; ``ttfs_cycle_based`` instead injects the
    binary single-spike train so the spike *timing* reaches the genuine soma.
    """
    from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_input_spike_times_1based

    s = int(simulation_length)
    for (core_idx, axon_idx), neuron in core_input_neurons.items():
        src = hcm.cores[core_idx].axon_sources[axon_idx]
        k = int(src.neuron_)
        if k >= seg_input_rates.shape[1]:
            times: list[int] = []
        else:
            rate = float(seg_input_rates[0, k])
            times = ttfs_input_spike_times_1based(rate, s)
        neuron.set_attributes(
            model_attributes=input_neuron_attributes(times),
        )

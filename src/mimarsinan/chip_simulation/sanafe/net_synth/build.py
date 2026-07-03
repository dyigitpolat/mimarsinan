"""Build a SANA-FE Network from a single ``HardCoreMapping`` neural segment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import mimarsinan.chip_simulation.sanafe.net_synth as _net_synth
from mimarsinan.chip_simulation.sanafe.neuron_model import input_neuron_attributes
from mimarsinan.chip_simulation.sanafe.presets import SOMA_INPUT_RANGE_NAME, SYNAPSE_NAME
from mimarsinan.mapping.support.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons


def _pack_tile_index(core_global_idx: int, cores_per_tile: int) -> Tuple[int, int]:
    if cores_per_tile <= 0:
        return 0, core_global_idx
    return core_global_idx // cores_per_tile, core_global_idx % cores_per_tile


def build_network_for_segment(
    arch: Any,
    hcm: Any,
    *,
    tile_offset: int,
    core_offset: int,
    cores_per_tile: int = 0,
    simulation_length: Optional[int] = None,
    firing_mode: str = "Default",
    spiking_mode: str = "lif",
    ttfs_cycle_schedule: str = "cascaded",
) -> Tuple[
    Any,
    Dict[int, Any],
    Dict[Tuple[int, int], Any],
    Dict[int, Any],
]:
    """Construct a ``sanafe.Network`` for one neural segment.

    Returns ``(network, core_to_group, core_input_neurons, core_always_on_neurons)``.
    ``simulation_length`` (HCM ``T``), when set, gates each LIF soma to the window
    ``[core.latency, core.latency+T)``; ``None`` keeps somas active all simulation.
    """
    sanafe = _net_synth._sanafe()
    net = sanafe.Network()
    from mimarsinan.chip_simulation.spiking_semantics import (
        forces_activation_quantization,
        is_cascaded_ttfs,
        is_synchronized_ttfs,
        is_ttfs_cycle_based,
        requires_ttfs_firing,
    )
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
    from mimarsinan.chip_simulation.ttfs.ttfs_cycle_genuine import latency_groups

    policy = policy_for_spiking_mode(spiking_mode, ttfs_cycle_schedule)
    soma_hw = policy.soma_hw_name()
    is_ttfs = requires_ttfs_firing(spiking_mode)
    is_cycle = is_synchronized_ttfs(spiking_mode, ttfs_cycle_schedule)
    is_cascade = is_cascaded_ttfs(spiking_mode, ttfs_cycle_schedule)
    is_quantized = (forces_activation_quantization(spiking_mode)
                    and not is_ttfs_cycle_based(spiking_mode))
    log_potential = policy.log_potential

    cycle_core_group = None
    if is_cycle:
        _, cycle_core_group = latency_groups(
            [getattr(c, "latency", None) for c in hcm.cores]
        )

    core_to_group: Dict[int, Any] = {}
    core_input_neurons: Dict[Tuple[int, int], Any] = {}
    core_always_on_neurons: Dict[int, Any] = {}

    def _resolve_core(hardcore_idx: int) -> Any:
        tile_local = core_offset + hardcore_idx
        tile_idx, core_in_tile = _pack_tile_index(tile_local, cores_per_tile)
        tile_idx += tile_offset
        return arch.tiles[tile_idx].cores[core_in_tile]

    for core_idx, core in enumerate(hcm.cores):
        sanafe_core = _resolve_core(core_idx)
        used_neurons = _used_neurons(core)
        used_axons = _used_axons(core)

        if used_neurons > 0:
            group = net.create_neuron_group(
                f"core{core_idx}", used_neurons, model_attributes={},
            )
            core_latency = (int(core.latency)
                            if getattr(core, "latency", None) is not None
                            else 0)
            # SANA-FE delivers an input spike to its consumer's synapse one cycle
            # after emission, so active_start is shifted +1 (else depth-0 cores
            # miss their first integration step).
            if is_cycle and simulation_length is not None:
                lat_group = int(cycle_core_group[core_idx])
                active_start = (lat_group + 1) * int(simulation_length)
                active_length = int(simulation_length)
            elif is_cascade:
                active_start = ((core_latency + 1)
                                if simulation_length is not None else None)
                active_length = (int(simulation_length)
                                 if simulation_length is not None else None)
            else:
                active_start = ((core_latency + 1)
                                if simulation_length is not None else None)
                if is_ttfs and simulation_length is not None:
                    if is_quantized:
                        active_length = int(simulation_length)
                    else:
                        active_length = 1
                else:
                    active_length = (int(simulation_length)
                                     if simulation_length is not None else None)
            for n_idx in range(used_neurons):
                neuron = group[n_idx]
                bias = None
                if core.hardware_bias is not None:
                    bias = float(np.asarray(core.hardware_bias)[n_idx])
                model_attrs = policy.soma_model_attributes(
                    threshold=float(core.threshold),
                    hardware_bias=bias,
                    active_start=active_start,
                    active_length=active_length,
                    firing_mode=firing_mode,
                )
                neuron.set_attributes(
                    soma_hw_name=soma_hw,
                    default_synapse_hw_name=SYNAPSE_NAME,
                    log_spikes=True,
                    log_potential=log_potential,
                    model_attributes=model_attrs,
                )
                neuron.map_to_core(sanafe_core)
            core_to_group[core_idx] = group

        input_axon_indices = []
        always_on_axon_indices = []
        for a in range(used_axons):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_input_", False):
                input_axon_indices.append(a)
            elif getattr(src, "is_always_on_", False):
                always_on_axon_indices.append(a)

        if input_axon_indices:
            in_group = net.create_neuron_group(
                f"core{core_idx}_in", len(input_axon_indices),
                model_attributes={},
            )
            for offset, a in enumerate(input_axon_indices):
                n = in_group[offset]
                n.set_attributes(
                    soma_hw_name=f"{SOMA_INPUT_RANGE_NAME}[{a}]",
                    default_synapse_hw_name=SYNAPSE_NAME,
                    log_spikes=True,
                    model_attributes=input_neuron_attributes(),
                )
                n.map_to_core(sanafe_core)
                core_input_neurons[(core_idx, a)] = n

        if always_on_axon_indices:
            ao_slot = always_on_axon_indices[0]
            ao_group = net.create_neuron_group(
                f"core{core_idx}_on", 1, model_attributes={},
            )
            ao_group[0].set_attributes(
                soma_hw_name=f"{SOMA_INPUT_RANGE_NAME}[{ao_slot}]",
                default_synapse_hw_name=SYNAPSE_NAME,
                log_spikes=True,
                model_attributes=input_neuron_attributes(),
            )
            ao_group[0].map_to_core(sanafe_core)
            core_always_on_neurons[core_idx] = ao_group[0]

    for core_idx, core in enumerate(hcm.cores):
        dst_group = core_to_group.get(core_idx)
        if dst_group is None:
            continue
        used_axons = _used_axons(core)
        used_neurons = _used_neurons(core)
        if used_neurons <= 0:
            continue

        accum: Dict[Tuple[int, int], float] = {}
        sources_by_key: Dict[int, Any] = {}

        for a in range(used_axons):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            src_neuron: Optional[Any] = None
            if getattr(src, "is_input_", False):
                src_neuron = core_input_neurons.get((core_idx, a))
            elif getattr(src, "is_always_on_", False):
                src_neuron = core_always_on_neurons.get(core_idx)
            else:
                src_core_idx = int(src.core_)
                src_group = core_to_group.get(src_core_idx)
                if src_group is None:
                    continue
                src_neuron = src_group[int(src.neuron_)]

            if src_neuron is None:
                continue
            src_key = id(src_neuron)
            sources_by_key[src_key] = src_neuron

            for n_idx in range(used_neurons):
                w = float(core.core_matrix[a, n_idx])
                if w == 0.0:
                    continue
                key = (src_key, n_idx)
                accum[key] = accum.get(key, 0.0) + w

        for (src_key, n_idx), w in accum.items():
            sources_by_key[src_key].connect_to_neuron(
                dst_group[n_idx],
                {"weight": w, "synapse_hw_name": SYNAPSE_NAME},
            )

    return net, core_to_group, core_input_neurons, core_always_on_neurons

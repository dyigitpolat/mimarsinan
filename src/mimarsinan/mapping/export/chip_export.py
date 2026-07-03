"""Convert HardCoreMapping to ChipModel for nevresim."""

from __future__ import annotations

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import (
    ChipModel,
    Connection,
    Core,
    Neuron,
    SpikeSource,
)


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    return tensor_or_array.detach().cpu().numpy()


def generate_core_weights(
    neurons_count,
    axons_count,
    weight_tensor,
    outs,
    thresh,
    latency,
    bias_tensor=None,
):
    neurons = []
    for idx in range(neurons_count):
        if idx < outs:
            row = weight_tensor[idx]
            pad = axons_count - row.shape[0]
            neuron_ws = row.tolist() + ([0] * pad) if pad > 0 else row.tolist()
        else:
            neuron_ws = [0] * axons_count

        bias = 0.0
        if (bias_tensor is not None) and (idx < outs):
            bias = bias_tensor[idx]

        neurons.append(Neuron(neuron_ws, thresh, bias))

    return Core(neurons, latency)


def generate_core_connection_info(axons_count, ins, core, is_input_core):
    axon_sources = [SpikeSource(core, i, is_input_core) for i in range(ins)]
    for _ in range(axons_count - ins):
        axon_sources.append(SpikeSource(-1, 0, False, True))

    return Connection(axon_sources)


def hard_cores_to_chip(
    input_size,
    hardcore_mapping,
    axons_per_core,
    neurons_per_core,
    leak,
    weight_type,
    threshold_type=None,
):
    """Convert a HardCoreMapping into a ChipModel for nevresim, padding heterogeneous cores to uniform axons/neurons.

    Bias handling, in priority order: explicit hardware_bias array, else legacy always-on bias row (folded to per-neuron bias), else no bias.
    """
    output_sources = hardcore_mapping.output_sources

    hardcores = []
    for hardcore in hardcore_mapping.cores:
        outs = int(hardcore.neurons_per_core)
        hw_bias = getattr(hardcore, "hardware_bias", None)
        has_bias_cap = getattr(hardcore, "has_bias_capability", True)

        if hw_bias is not None:
            weight_tensor = hardcore.core_matrix.transpose()
            hardcores.append(
                generate_core_weights(
                    neurons_per_core,
                    axons_per_core,
                    weight_tensor,
                    outs,
                    hardcore.threshold,
                    hardcore.latency,
                    bias_tensor=hw_bias,
                )
            )
        elif has_bias_cap:
            # Parameter-encoded bias: keep the last core_matrix row as an always-on-axon weight; do NOT fold it into on-chip bias_, which breaks spike parity.
            weight_tensor = hardcore.core_matrix.transpose()
            hardcores.append(
                generate_core_weights(
                    neurons_per_core,
                    axons_per_core,
                    weight_tensor,
                    outs,
                    hardcore.threshold,
                    hardcore.latency,
                )
            )
        else:
            hardcores.append(
                generate_core_weights(
                    neurons_per_core,
                    axons_per_core,
                    hardcore.core_matrix.transpose(),
                    outs,
                    hardcore.threshold,
                    hardcore.latency,
                )
            )

    hardcore_connections = []
    for hardcore in hardcore_mapping.cores:
        axon_sources = list(hardcore.axon_sources)
        while len(axon_sources) < axons_per_core:
            axon_sources.append(SpikeSource(-1, 0, is_input=False, is_off=True))
        hardcore_connections.append(Connection(axon_sources))

    chip = ChipModel(
        axons_per_core,
        neurons_per_core,
        len(hardcores),
        input_size,
        len(output_sources),
        leak,
        hardcore_connections,
        output_sources,
        hardcores,
        weight_type,
        threshold_type=threshold_type,
    )

    chip.load_from_json(chip.get_chip_json())

    return chip

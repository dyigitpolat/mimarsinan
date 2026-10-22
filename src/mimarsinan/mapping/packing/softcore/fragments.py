"""Neuron-split fragments of a real SoftCore: each owns a column window of the parent's matrix, bias and law."""

from __future__ import annotations

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore


def build_real_fragment(
    parent,
    *,
    window,
    matrix_slice,
    hardware_bias_slice,
    bank_neuron_slice,
    fragment_index,
    group_id,
    original_neurons,
):
    """The fragment of ``parent`` computing its columns ``window = (lo, hi)``."""
    lo, hi = window
    frag = SoftCore(
        core_matrix=matrix_slice,
        axon_sources=[
            SpikeSource(s.core_, s.neuron_, s.is_input_, s.is_off_, s.is_always_on_)
            for s in parent.axon_sources
        ],
        id=parent.id,
        activation_scale=parent.activation_scale,
        parameter_scale=parent.parameter_scale,
        input_activation_scale=parent.input_activation_scale,
        boundary_grid=getattr(parent, "boundary_grid", None),
        name=parent.name,
        psum_group_id=parent.psum_group_id,
        psum_role=parent.psum_role,
    )
    frag.threshold = parent.threshold
    frag.latency = parent.latency
    frag.neuron_offset_in_original = (
        getattr(parent, "neuron_offset_in_original", 0) + lo
    )
    frag.split_group_id = group_id
    frag.split_fragment_index = fragment_index
    frag.split_original_neurons = original_neurons
    frag.perceptron_index = getattr(parent, "perceptron_index", None)
    frag.perceptron_output_slice = getattr(parent, "perceptron_output_slice", None)
    if hardware_bias_slice is not None:
        frag.hardware_bias = hardware_bias_slice
    law = getattr(parent, "readout_law", None)
    if law is not None:
        frag.readout_law = law.take_columns(np.arange(lo, hi))
    if bank_neuron_slice is not None:
        frag.weight_bank_id = parent.weight_bank_id
        frag.bank_axon_slice = parent.bank_axon_slice
        frag.bank_neuron_slice = bank_neuron_slice
        frag.bank_includes_bias_row = parent.bank_includes_bias_row
    return frag


def make_real_fragments(*, softcore, first_neurons, remaining_neurons, group_id):
    """The two fragments of a neuron split at ``first_neurons``."""
    total = first_neurons + remaining_neurons
    hb = getattr(softcore, "hardware_bias", None)
    bns = None
    if getattr(softcore, "weight_bank_id", None) is not None:
        bns_parent = getattr(softcore, "bank_neuron_slice", None) or (0, total)
        bns = (
            (int(bns_parent[0]), int(bns_parent[0] + first_neurons)),
            (int(bns_parent[0] + first_neurons), int(bns_parent[1])),
        )
    parent_dense = softcore.get_core_matrix()
    windows = ((0, first_neurons), (first_neurons, total))
    return tuple(
        build_real_fragment(
            softcore,
            window=(lo, hi),
            matrix_slice=parent_dense[:, lo:hi].copy(),
            hardware_bias_slice=None if hb is None else hb[lo:hi].copy(),
            bank_neuron_slice=None if bns is None else bns[index],
            fragment_index=index,
            group_id=group_id,
            original_neurons=total,
        )
        for index, (lo, hi) in enumerate(windows)
    )

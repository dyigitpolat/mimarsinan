"""Neuron-split fragments of a real SoftCore: each owns a column window of the parent's matrix, bias and law."""

from __future__ import annotations

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir.lane_law import OPERAND_AXONS
from mimarsinan.mapping.packing.softcore.sign_split import is_sign_split, logical_view, split_soft_core
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS


def sibling_operand_rows(law, lo: int, hi: int) -> set:
    """The rows of ``law``'s fused operands that belong to columns outside ``[lo, hi)`` (WI-5e): a fragment keeps
    them as axons (every parent row stays) but off, so no executor decodes a sibling's operand on its own grid. Their
    weights are 0 in every column, so no accumulator changes."""
    rows = getattr(law, "operand_row", None)
    if rows is None:
        return set()
    columns = np.arange(len(rows))
    outside = (columns < lo) | (columns >= hi)
    return {int(r) for r in np.asarray(rows)[outside].tolist() if r >= 0}


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
    axon_sources=None,
):
    """The fragment of ``parent`` computing its columns ``window = (lo, hi)`` over ``axon_sources`` (the
    parent's, by default)."""
    lo, hi = window
    law = getattr(parent, "readout_law", None)
    siblings = sibling_operand_rows(law, lo, hi)
    frag = SoftCore(
        core_matrix=matrix_slice,
        axon_sources=[
            SpikeSource(-1, 0, is_input=False, is_off=True) if row in siblings
            else SpikeSource(s.core_, s.neuron_, s.is_input_, s.is_off_, s.is_always_on_)
            for row, s in enumerate(parent.axon_sources if axon_sources is None else axon_sources)
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
    # [W04] A fragment keeps its parent's residency class (its singleton values are the parent's): before, it read
    # as an ungrouped core and shared no core with its own layer, so the deployed program used cores the shape-only
    # plane, whose fragments keep the class, did not count.
    frag.residency_class_id = getattr(parent, "residency_class_id", None)
    frag.neuron_offset_in_original = (
        getattr(parent, "neuron_offset_in_original", 0) + lo
    )
    frag.split_group_id = group_id
    frag.split_fragment_index = fragment_index
    frag.split_original_neurons = original_neurons
    frag.perceptron_index = getattr(parent, "perceptron_index", None)
    frag.perceptron_output_slice = getattr(parent, "perceptron_output_slice", None)
    bits = getattr(parent, CORE_WEIGHT_BITS, None)
    if bits is not None:  # [WI-5g] a fragment binds to its parent's core type
        setattr(frag, CORE_WEIGHT_BITS, bits)
    if hardware_bias_slice is not None:
        frag.hardware_bias = hardware_bias_slice
    axons = getattr(parent, OPERAND_AXONS, None)
    if axons:  # [W08] every parent operand row stays an axon of each fragment (priced as the operand port)
        setattr(frag, OPERAND_AXONS, int(axons))
    if law is not None:
        frag.readout_law = law.take_columns(np.arange(lo, hi))
    role = getattr(parent, "core_role", None)
    if role is not None:
        frag.core_role = role  # pyright: ignore[reportAttributeAccessIssue] -- declared-only (N2)
    if bank_neuron_slice is not None:
        frag.weight_bank_id = parent.weight_bank_id
        frag.bank_axon_slice = parent.bank_axon_slice
        frag.bank_neuron_slice = bank_neuron_slice
        frag.bank_includes_bias_row = parent.bank_includes_bias_row
    return frag


def make_real_fragments(*, softcore, first_neurons, remaining_neurons, group_id):
    """The two fragments of a neuron split at ``first_neurons``. A per-axon parent's fragments are cut from its
    logical rows and each re-derives its own split, so a row mixed only outside a fragment takes one row there."""
    total = first_neurons + remaining_neurons
    hb = getattr(softcore, "hardware_bias", None)
    bns = None
    if getattr(softcore, "weight_bank_id", None) is not None:
        bns_parent = getattr(softcore, "bank_neuron_slice", None) or (0, total)
        bns = (
            (int(bns_parent[0]), int(bns_parent[0] + first_neurons)),
            (int(bns_parent[0] + first_neurons), int(bns_parent[1])),
        )
    signed = is_sign_split(softcore)
    parent_dense, sources = logical_view(softcore) if signed else (softcore.get_core_matrix(), None)
    windows = ((0, first_neurons), (first_neurons, total))
    fragments = tuple(
        build_real_fragment(
            softcore,
            window=(lo, hi),
            matrix_slice=parent_dense[:, lo:hi].copy(),
            hardware_bias_slice=None if hb is None else hb[lo:hi].copy(),
            bank_neuron_slice=None if bns is None else bns[index],
            fragment_index=index,
            group_id=group_id,
            original_neurons=total,
            axon_sources=sources,
        )
        for index, (lo, hi) in enumerate(windows)
    )
    if signed:
        for fragment in fragments:
            split_soft_core(fragment)
    return fragments

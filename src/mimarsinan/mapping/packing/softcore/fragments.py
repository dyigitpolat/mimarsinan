"""Neuron-split fragments of a real SoftCore: each windows a column range of the parent's weight payload (by
reference, through the compaction descriptor) and owns its window of every value held per neuron (its threshold
registers, bias and law: ``compaction.take_neuron_columns``)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir.lane_law import OPERAND_AXONS
from mimarsinan.mapping.packing.softcore.compaction import take_neuron_columns
from mimarsinan.mapping.packing.softcore.matrix_placement import materialize_compacted
from mimarsinan.mapping.packing.softcore.sign_split import (
    is_sign_split,
    logical_sources,
    logical_starts,
    logical_view,
    memoized_array,
    split_soft_core,
)
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS

#: A compaction descriptor: ``materialize_compacted(source, rows, cols)`` is the dense matrix it stands for.
Descriptor = Tuple[Any, np.ndarray, np.ndarray]


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
    bank_neuron_slice,
    fragment_index,
    group_id,
    original_neurons,
    axon_sources=None,
    descriptor: Optional[Descriptor] = None,
):
    """The fragment of ``parent`` computing its columns ``window = (lo, hi)`` over ``axon_sources`` (the
    parent's, by default): its weights are ``matrix_slice`` (an owned copy of the window) or, by reference,
    ``descriptor`` (a compaction descriptor over the parent's payload that resolves to that window)."""
    lo, hi = window
    law = getattr(parent, "readout_law", None)
    siblings = sibling_operand_rows(law, lo, hi)
    frag = SoftCore(
        core_matrix=matrix_slice if descriptor is None else None,
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
    if descriptor is not None:
        frag.compact_source_matrix, frag.compact_keep_rows, frag.compact_keep_cols = descriptor
    take_neuron_columns(parent, np.arange(lo, hi), into=frag)  # [U-LIVE r2] every value held per neuron, windowed
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
    axons = getattr(parent, OPERAND_AXONS, None)
    if axons:  # [W08] every parent operand row stays an axon of each fragment (priced as the operand port)
        setattr(frag, OPERAND_AXONS, int(axons))
    role = getattr(parent, "core_role", None)
    if role is not None:
        frag.core_role = role  # pyright: ignore[reportAttributeAccessIssue] -- declared-only (N2)
    if bank_neuron_slice is not None:
        frag.weight_bank_id = parent.weight_bank_id
        frag.bank_axon_slice = parent.bank_axon_slice
        frag.bank_neuron_slice = bank_neuron_slice
        frag.bank_includes_bias_row = parent.bank_includes_bias_row
    return frag


def window_descriptor(parent, lo: int, hi: int, memo: Dict[Any, Any]) -> Optional[Descriptor]:
    """[P1 R1] The descriptor over ``parent``'s payload whose ``materialize_compacted`` is the copy recipe's window
    ``[lo, hi)`` (the parent's dense matrix, a per-axon parent's logical matrix) in bytes, dtype, shape and
    C-contiguity, or ``None`` where no descriptor is. A stored float64 payload windows all its rows and the columns
    ``lo:hi``; a descriptor parent (a compaction, an earlier fragment) its own rows and the window of its columns;
    a per-axon parent the rows of its logical view (each payload row once) and the window of its columns. A stored
    payload of another dtype has none: the descriptor's recipe casts to float64, the copy keeps the payload's dtype.
    Index arrays are shared through ``memo``."""
    if is_sign_split(parent):
        rows = np.asarray(parent.compact_keep_rows)[logical_starts(parent)]
        source, cols = parent.compact_source_matrix, parent.compact_keep_cols
    elif parent.core_matrix is not None:
        source = parent.core_matrix
        if getattr(source, "dtype", None) != np.float64 or np.ndim(source) != 2:
            return None
        rows, cols = np.arange(source.shape[0]), np.arange(source.shape[1])
    elif parent.compact_keep_rows is not None:
        source, rows, cols = parent.compact_source_matrix, parent.compact_keep_rows, parent.compact_keep_cols
    else:
        return None
    return (source, memoized_array(memo, np.asarray(rows, dtype=np.intp)),
            memoized_array(memo, np.ascontiguousarray(np.asarray(cols, dtype=np.intp)[lo:hi])))


def keeps_a_signed_zero(fragment) -> bool:
    """Whether a per-axon fragment by reference resolves an all-zero row with a negative zero: the copy recipe's
    logical view writes every zero as +0, and ``materialize_signed`` keeps an all-zero row as it is, so such a
    window by reference would differ from the copy in that bit."""
    zero = np.asarray(fragment.compact_row_signs) == 0
    if not zero.any():
        return False
    block = materialize_compacted(fragment.compact_source_matrix, np.asarray(fragment.compact_keep_rows)[zero],
                                  fragment.compact_keep_cols)
    return bool(np.signbit(block).any())


def make_real_fragments(*, softcore, first_neurons, remaining_neurons, group_id, memo=None):
    """The two fragments of a neuron split at ``first_neurons``. [P1 R1] Each references its parent's weight
    payload through a compaction descriptor (``window_descriptor``) and owns no weight array, so a split costs no
    weight storage; a window no descriptor resolves to byte for byte (``window_descriptor`` is ``None``,
    ``keeps_a_signed_zero``) keeps its copy, as every fragment did before. A per-axon parent's fragments are cut from
    its logical rows and each re-derives its own split, so a row mixed only outside a fragment takes one row there.
    ``memo`` shares index arrays across the splits of one placement."""
    memo = {} if memo is None else memo
    total = first_neurons + remaining_neurons
    bns = None
    if getattr(softcore, "weight_bank_id", None) is not None:
        bns_parent = getattr(softcore, "bank_neuron_slice", None) or (0, total)
        bns = (
            (int(bns_parent[0]), int(bns_parent[0] + first_neurons)),
            (int(bns_parent[0] + first_neurons), int(bns_parent[1])),
        )
    signed = is_sign_split(softcore)
    sources = logical_sources(softcore) if signed else None
    dense: list = []

    def copied(lo: int, hi: int) -> np.ndarray:
        if not dense:  # the parent's dense matrix (a per-axon parent's logical one), resolved once
            dense.append(logical_view(softcore)[0] if signed else softcore.get_core_matrix())
        return dense[0][:, lo:hi].copy()

    def fragment(index: int, lo: int, hi: int, descriptor: Optional[Descriptor]):
        frag = build_real_fragment(
            softcore,
            window=(lo, hi),
            matrix_slice=None if descriptor is not None else copied(lo, hi),
            descriptor=descriptor,
            bank_neuron_slice=None if bns is None else bns[index],
            fragment_index=index,
            group_id=group_id,
            original_neurons=total,
            axon_sources=sources,
        )
        if signed:
            split_soft_core(frag, memo)
        return frag

    fragments = []
    for index, (lo, hi) in enumerate(((0, first_neurons), (first_neurons, total))):
        descriptor = window_descriptor(softcore, lo, hi, memo)
        frag = fragment(index, lo, hi, descriptor)
        if descriptor is not None and signed and keeps_a_signed_zero(frag):
            frag = fragment(index, lo, hi, None)
        fragments.append(frag)
    return tuple(fragments)

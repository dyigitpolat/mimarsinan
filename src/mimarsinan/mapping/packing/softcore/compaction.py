"""Post-pruning soft-core dimension compaction and the geometry it yields."""

from mimarsinan.code_generation.cpp_chip_model import SpikeSource

import numpy as np

from mimarsinan.mapping.packing.spike_source import (
    source_is_always_on,
    source_is_input,
    source_is_off,
    source_is_special,
)


def compaction_masks_apply(
    n_axons: int, n_neurons: int, pruned_row_mask, pruned_col_mask,
) -> bool:
    """Whether these masks describe THIS matrix; a length disagreement compacts nothing."""
    return (
        pruned_row_mask is not None
        and pruned_col_mask is not None
        and len(pruned_row_mask) == n_axons
        and len(pruned_col_mask) == n_neurons
    )


def compaction_keep_indices(
    *,
    n_axons: int,
    n_neurons: int,
    pruned_row_mask,
    pruned_col_mask,
    keep_last_axon: bool,
) -> "tuple[list[int], list[int]]":
    """Rows and columns surviving pruning-mask compaction.

    The always-on bias row is never dropped.
    """
    if not compaction_masks_apply(
        n_axons, n_neurons, pruned_row_mask, pruned_col_mask,
    ):
        return list(range(n_axons)), list(range(n_neurons))

    keep_rows = [r for r in range(n_axons) if not pruned_row_mask[r]]
    keep_cols = [c for c in range(n_neurons) if not pruned_col_mask[c]]
    if keep_last_axon and n_axons and (n_axons - 1) not in keep_rows:
        keep_rows.append(n_axons - 1)
        keep_rows.sort()
    return keep_rows, keep_cols


def compacted_core_extent(core) -> "tuple[int, int]":
    """The ``(axons, neurons)`` an IR ``NeuralCore`` occupies once compacted.

    The scheduler's sizing SSOT: bank-backed instances carry their elimination
    as masks until :func:`compact_soft_core_mapping` materializes it, so the
    un-compacted matrix is never the extent a pass must budget for.
    """
    sources = core.input_sources.flatten()
    n_axons = int(len(sources))
    n_neurons = int(core.get_output_count())
    row_mask = getattr(core, "pruned_row_mask", None)
    col_mask = getattr(core, "pruned_col_mask", None)
    if not compaction_masks_apply(n_axons, n_neurons, row_mask, col_mask):
        return n_axons, n_neurons

    last = sources[-1] if n_axons else None
    keep_rows, keep_cols = compaction_keep_indices(
        n_axons=n_axons,
        n_neurons=n_neurons,
        pruned_row_mask=row_mask,
        pruned_col_mask=col_mask,
        keep_last_axon=bool(last is not None and last.is_always_on()),
    )
    # BIAS_ONLY: no axon survives, but the bias-driven columns still need one row.
    return len(keep_rows) or 1, len(keep_cols)


def eliminated_core_extent(core) -> "tuple[int, int]":
    """``(axons, neurons)`` elimination removed since the core was emitted.

    The correction a PRE-elimination record (a layout softcore spec) needs to
    describe the instance the packer will actually place: zero whenever nothing
    was eliminated, so an untouched geometry keeps its recorded extent.
    """
    pre_rows = getattr(core, "pre_pruning_row_mask", None)
    pre_cols = getattr(core, "pre_pruning_col_mask", None)
    post_axons, post_neurons = compacted_core_extent(core)
    pre_axons = (
        len(pre_rows) if pre_rows is not None
        else int(len(core.input_sources.flatten()))
    )
    pre_neurons = (
        len(pre_cols) if pre_cols is not None else int(core.get_output_count())
    )
    return pre_axons - post_axons, pre_neurons - post_neurons


def compact_soft_core_mapping(cores, output_sources):
    """Compact soft cores from IR pruning masks; return core_id → neuron reindex maps."""
    reindex_maps = {}
    n_compacted = 0
    n_skipped = 0

    for core in cores:
        mat = np.asarray(core.core_matrix, dtype=np.float64)
        n_axons, n_neurons = mat.shape
        keep_rows, keep_cols = compaction_keep_indices(
            n_axons=n_axons,
            n_neurons=n_neurons,
            pruned_row_mask=getattr(core, "pruned_row_mask", None),
            pruned_col_mask=getattr(core, "pruned_col_mask", None),
            keep_last_axon=bool(
                core.axon_sources
                and source_is_always_on(core.axon_sources[-1])
            ),
        )

        if len(keep_rows) < n_axons or len(keep_cols) < n_neurons:
            n_compacted += 1
            if getattr(core, "weight_bank_id", None) is not None:
                core.weight_bank_id = None
                core.bank_axon_slice = None
                core.bank_neuron_slice = None
                core.bank_includes_bias_row = False
            from mimarsinan.mapping.pruning.pruning_apply import compact_hardware_bias_columns

            if keep_rows and keep_cols:
                core.core_matrix = mat[np.ix_(keep_rows, keep_cols)].copy()
                core.axon_sources = [core.axon_sources[r] for r in keep_rows]
                core.hardware_bias = compact_hardware_bias_columns(
                    core.hardware_bias, keep_cols
                )
            elif keep_cols:
                # BIAS_ONLY core: collapse to a single OFF-source axon but keep the live bias-driven columns; collapsing to (1,1) would silently delete live neurons still referenced downstream.
                core.core_matrix = np.zeros((1, len(keep_cols)), dtype=np.float64)
                core.axon_sources = [SpikeSource(-1, 0, False, True)]
                core.hardware_bias = compact_hardware_bias_columns(
                    core.hardware_bias, keep_cols
                )
            else:
                raise AssertionError(
                    f"compact_soft_core_mapping: SoftCore id={core.id} has "
                    f"no surviving neurons (keep_cols={keep_cols}). "
                    "Liveness analysis should have removed it via "
                    "IRGraph.remove_nodes before soft-core mapping. "
                    "This indicates a regression in prune_ir_graph or a "
                    "stale pickle that bypassed the liveness pass."
                )
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)}
            reindex_maps[core.id] = remap
        else:
            n_skipped += 1
            reindex_maps[core.id] = {j: j for j in range(n_neurons)}

        core._axon_source_spans = None

    for core in cores:
        new_sources = []
        for s in core.axon_sources:
            if source_is_off(s):
                new_sources.append(s)
            elif source_is_special(s):
                new_sources.append(s)
            else:
                cid, nidx = int(s.core_), int(s.neuron_)
                if cid in reindex_maps and nidx in reindex_maps[cid]:
                    new_sources.append(SpikeSource(cid, reindex_maps[cid][nidx], False, False))
                else:
                    new_sources.append(SpikeSource(-1, 0, False, True))
        core.axon_sources = new_sources
        core._axon_source_spans = None

    had_output_refs = any(
        not source_is_off(s) for s in output_sources
    )
    new_out = []
    for s in output_sources:
        if source_is_off(s):
            continue
        if source_is_input(s) or source_is_always_on(s):
            new_out.append(s)
        else:
            cid, nidx = int(s.core_), int(s.neuron_)
            if cid in reindex_maps and nidx in reindex_maps[cid]:
                new_out.append(SpikeSource(cid, reindex_maps[cid][nidx], False, False))
    if had_output_refs and len(new_out) == 0:
        raise ValueError(
            "compact_soft_core_mapping: all output_sources were dropped by compaction "
            "(every output neuron was pruned). At least one output must remain; check pruning masks "
            "and initial mask assignment (e.g. 1:1 vs tiled layer matching)."
        )
    output_sources.clear()
    output_sources.extend(new_out)

    return reindex_maps

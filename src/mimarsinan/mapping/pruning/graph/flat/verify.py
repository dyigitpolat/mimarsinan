"""Verification seams of the flat engine: the mask mirror and the contract lock."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.pruning.graph.flat.state import FlatState

__all__ = ["FlatEngineQuiescenceError", "_sync_masks"]


class FlatEngineQuiescenceError(RuntimeError):
    """The change-tracked lattice gather missed a re-arm: the final FULL gather
    still produced facts. Loud and unconditional, because a missed re-arm is
    otherwise silent -- it yields fewer folds and descents, never wrong ones."""


def _sync_masks(ctx, state: FlatState, row_dead: np.ndarray, col_dead: np.ndarray) -> None:
    """Mirror the context's per-node sets into the flat masks, EXACTLY.

    The ONE sync point (start of every wave, after the lattice commit): the
    lattice commit and the node commits both mutate the ctx sets, and the sets
    are REPLACED wholesale by the kernels, so the mirror is rebuilt rather
    than accumulated -- an add-only mirror went stale the first time a fold
    killed a row outside the node loop, which is precisely how the flat
    engine silently lost the bias-only collapse.
    """
    row_dead[:] = False
    col_dead[:] = False
    for k, nid in enumerate(state.node_ids):
        rows = state.rows_of(k)
        cols = state.cols_of(k)
        for i in ctx.pruned_rows.get(nid, ()):
            if 0 <= i < rows.stop - rows.start:
                row_dead[rows.start + i] = True
        for j in ctx.pruned_cols.get(nid, ()):
            if 0 <= j < cols.stop - cols.start:
                col_dead[cols.start + j] = True

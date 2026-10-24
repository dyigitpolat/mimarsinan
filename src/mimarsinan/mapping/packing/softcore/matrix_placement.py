"""Placement descriptors: cores reference shared payloads; dense grids materialize transiently."""

from __future__ import annotations

import numpy as np


def materialize_compacted(source, keep_rows, keep_cols) -> np.ndarray:
    """The eager compaction recipe, replayed on read: f64 cast then ``np.ix_``.

    Byte-identical to the retired ``np.asarray(src, float64)[np.ix_(...)].copy()``
    (fancy indexing already returns a fresh C-contiguous array).
    """
    mat = np.asarray(source, dtype=np.float64)
    return mat[np.ix_(keep_rows, keep_cols)]


def materialize_signed(source, keep_rows, keep_cols, row_signs) -> np.ndarray:
    """The compaction recipe, then each signed row keeps only its sign's entries (a per-axon W+ or W- row);
    a row of sign 0 (all zero) is kept as is."""
    mat = materialize_compacted(source, keep_rows, keep_cols)
    signs = np.asarray(row_signs)[:, None]
    return np.where((signs == 0) | (np.sign(mat) == signs), mat, 0.0)


class MatrixPlacement:
    """One softcore weight block placed on a hard core.

    ``source`` is the SHARED payload (bank matrix, memoized bank column view,
    or an owned array) — pickling stores it by reference so duplicate cores
    memoize one payload per bank. ``keep_rows``/``keep_cols`` replay pruning
    compaction on read; offsets and extents place the block on the padded grid.
    """

    __slots__ = (
        "source", "keep_rows", "keep_cols",
        "axon_offset", "neuron_offset", "axons", "neurons",
    )

    def __init__(
        self, *, source, keep_rows, keep_cols,
        axon_offset: int, neuron_offset: int, axons: int, neurons: int,
    ) -> None:
        assert (keep_rows is None) == (keep_cols is None)
        self.source = source
        self.keep_rows = keep_rows
        self.keep_cols = keep_cols
        self.axon_offset = int(axon_offset)
        self.neuron_offset = int(neuron_offset)
        self.axons = int(axons)
        self.neurons = int(neurons)

    def is_plain(self) -> bool:
        """Whether the block IS the source payload (no compaction replay)."""
        return self.keep_rows is None

    def block_dtype(self) -> np.dtype:
        if self.keep_rows is not None:
            return np.dtype(np.float64)
        dtype = getattr(self.source, "dtype", None)
        return np.dtype(dtype) if dtype is not None else np.dtype(np.float64)

    def materialize(self):
        if self.keep_rows is None:
            return self.source
        return materialize_compacted(self.source, self.keep_rows, self.keep_cols)

    def content_key(self) -> tuple:
        """Hashable fragment identity: equal keys imply byte-identical blocks
        (``id`` components stay valid while the owning mapping is alive)."""
        return (
            id(self.source),
            None if self.keep_rows is None else self.keep_rows.tobytes(),
            None if self.keep_cols is None else self.keep_cols.tobytes(),
            self.axon_offset, self.neuron_offset, self.axons, self.neurons,
        )


class SignedRowPlacement(MatrixPlacement):
    """A per-axon soft core's block: the descriptor's rows repeat each mixed row, ``row_signs`` keeps one sign
    per row. A subclass, so a per-synapse placement keeps its slots and its pickle."""

    __slots__ = ("row_signs",)

    def __init__(self, *, row_signs, **placement) -> None:
        super().__init__(**placement)
        assert self.keep_rows is not None and len(row_signs) == len(self.keep_rows)
        self.row_signs = row_signs

    def is_plain(self) -> bool:
        return False

    def materialize(self):
        return materialize_signed(self.source, self.keep_rows, self.keep_cols, self.row_signs)

    def content_key(self) -> tuple:
        return (*super().content_key(), np.asarray(self.row_signs).tobytes())


def _is_exact_fit(placements, axons_per_core: int, neurons_per_core: int) -> bool:
    if len(placements) != 1:
        return False
    p = placements[0]
    return (
        p.axon_offset == 0 and p.neuron_offset == 0
        and p.axons == axons_per_core and p.neurons == neurons_per_core
    )


def resolve_core_matrix(
    owned, placements, axons_per_core: int, neurons_per_core: int, owner: str,
):
    """Full logical weight grid for a core, always ``(axons_per_core, neurons_per_core)``.

    Owned-dense returns as-is; an exact-fit single PLAIN placement returns the
    shared payload object unchanged (stable identity — the pickle/upload dedup
    mechanism); everything else materializes a TRANSIENT composite: zeros in
    the first-placed block's dtype with diagonal block writes, byte-identical
    to the retired eager ``add_softcore`` paste.
    """
    if owned is not None:
        return owned
    if not placements:
        raise ValueError(
            f"{owner}: no owned core_matrix and no matrix placements to "
            f"resolve a weight grid from."
        )
    if _is_exact_fit(placements, axons_per_core, neurons_per_core):
        return placements[0].materialize()
    comp = np.zeros(
        (axons_per_core, neurons_per_core), dtype=placements[0].block_dtype(),
    )
    for p in placements:
        comp[
            p.axon_offset:p.axon_offset + p.axons,
            p.neuron_offset:p.neuron_offset + p.neurons,
        ] = p.materialize()
    return comp


def core_matrix_content_key(
    owned, placements, axons_per_core: int, neurons_per_core: int,
) -> tuple:
    """Stable content identity: equal keys imply byte-identical resolved grids."""
    if owned is not None:
        return ("owned", id(owned))
    if _is_exact_fit(placements, axons_per_core, neurons_per_core) and (
        placements[0].is_plain()
    ):
        return ("base", id(placements[0].source))
    return (
        "composite", axons_per_core, neurons_per_core,
        tuple(p.content_key() for p in placements),
    )


def core_matrix_dtype(owned, placements) -> "np.dtype | None":
    """Grid dtype without materialization (first-placed block rule)."""
    if owned is not None:
        dtype = getattr(owned, "dtype", None)
        return np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
    if not placements:
        return None
    return placements[0].block_dtype()


def core_matrix_payloads(owned, placements) -> tuple:
    """The shared arrays a content key is derived from, in key order.

    A memo keyed by :func:`core_matrix_content_key` must retain these and
    re-check them by identity: the key embeds payload ``id``s, which a freed
    and re-allocated array could otherwise collide with.
    """
    if owned is not None:
        return (owned,)
    return tuple(p.source for p in placements)


def same_core_matrix_payloads(left: tuple, right: tuple) -> bool:
    """Element-wise identity (``==`` on ndarrays is not a truth value)."""
    return len(left) == len(right) and all(
        a is b for a, b in zip(left, right)
    )

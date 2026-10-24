"""Per-axon weight sign: a soft core's mixed-sign rows split into an adjacent W+ then W- pair."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.softcore.matrix_placement import materialize_signed
from mimarsinan.mapping.platform.platform_constraints import PER_AXON


class SignSplitError(ToolchainRefusal, ValueError):
    """A per-axon split of weights off the integer grid: W+ + W- is exact per entry, but a blocked sum of float
    weights may reorder, so the split is refused by name rather than run approximately."""


def mixed_rows(matrix: Any) -> np.ndarray:
    """Per row, whether it holds both a positive and a negative weight."""
    w = np.asarray(matrix)
    return np.asarray((w > 0).any(axis=1) & (w < 0).any(axis=1))


def sign_split_rows(matrix: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(origin, signs)`` of the physical rows: every logical row in order, a mixed row as ``(j, +1), (j, -1)``,
    a single-sign row with its sign, an all-zero row with 0."""
    w = np.asarray(matrix)
    positive, negative = (w > 0).any(axis=1), (w < 0).any(axis=1)
    mixed = positive & negative
    repeats = 1 + mixed.astype(np.intp)
    origin = np.repeat(np.arange(w.shape[0], dtype=np.intp), repeats)
    signs = np.repeat(np.where(positive, 1, np.where(negative, -1, 0)).astype(np.int8), repeats)
    signs[(np.cumsum(repeats) - 1)[mixed]] = -1
    return origin, signs


def is_sign_split(core: Any) -> bool:
    """Whether ``core`` carries per-axon row signs (declared-only: absent on every per-synapse core)."""
    return getattr(core, "compact_row_signs", None) is not None


def _require_integer_grid(core: Any, matrix: np.ndarray) -> None:
    w = np.asarray(matrix)
    if np.issubdtype(w.dtype, np.integer):
        return
    if not (np.isfinite(w).all() and (w == np.round(w)).all()):
        raise SignSplitError(
            f"soft core {getattr(core, 'name', None) or getattr(core, 'id', '?')}: per-axon weight sign needs "
            f"integer weights (quantize the weights for this target); a weight off the integer grid was found")


def _memoized(memo: Dict[Any, Any], array: np.ndarray) -> np.ndarray:
    key = (array.dtype.str, array.tobytes())
    hit = memo.get(key)
    if hit is None:
        memo[key] = hit = array
    return hit


def _copied(source: SpikeSource) -> SpikeSource:
    return SpikeSource(source.core_, source.neuron_, source.is_input_, source.is_off_, source.is_always_on_)


def split_soft_core(core: Any, memo: Optional[Dict[Any, Any]] = None) -> None:
    """Re-describe ``core`` by its physical rows, in place: the shared payload stays the source of a descriptor
    whose rows repeat each mixed row, signed by ``compact_row_signs``; each source is duplicated as a distinct
    object (the packer remaps placed sources in place). Bank bookkeeping is dropped, as compaction drops it, so
    no reader slices the logical bank by a physical row count."""
    if is_sign_split(core):
        return
    memo = {} if memo is None else memo
    matrix = core.get_core_matrix()
    _require_integer_grid(core, matrix)
    origin, signs = sign_split_rows(matrix)
    if core.compact_keep_rows is not None:
        source, rows, cols = core.compact_source_matrix, core.compact_keep_rows[origin], core.compact_keep_cols
    else:
        source, rows = core.core_matrix, origin
        cols = np.arange(np.shape(source)[1], dtype=np.intp)
    sources = core.axon_sources
    core.axon_sources = [sources[j] if i == 0 or origin[i - 1] != j else _copied(sources[j])
                         for i, j in enumerate(origin.tolist())]
    core.compact_source_matrix = source
    core.compact_keep_rows = _memoized(memo, np.asarray(rows, dtype=np.intp))
    core.compact_keep_cols = _memoized(memo, np.asarray(cols, dtype=np.intp))
    core.compact_row_signs = _memoized(memo, signs)
    core.core_matrix = None
    core.weight_bank_id = None
    core.bank_axon_slice = None
    core.bank_neuron_slice = None
    core.bank_includes_bias_row = False
    core._axon_source_spans = None


def apply_sign_split(cores: Iterable[Any], granularity: str) -> None:
    """Split every soft core of a segment under per-axon sign (after compaction, before packing); a no-op that
    sets no attribute under per-synapse sign."""
    if granularity != PER_AXON:
        return
    memo: Dict[Any, Any] = {}
    for core in cores:
        split_soft_core(core, memo)


def logical_view(core: Any) -> Tuple[np.ndarray, List[SpikeSource]]:
    """The logical matrix and sources of a split core: the rows of one origin summed (one of each pair is zero
    per entry, so the sum is exact), each origin's first source."""
    physical = materialize_signed(core.compact_source_matrix, core.compact_keep_rows, core.compact_keep_cols,
                                  core.compact_row_signs)
    rows = np.asarray(core.compact_keep_rows)
    starts = np.concatenate(([True], rows[1:] != rows[:-1]))
    first, group = np.flatnonzero(starts), np.cumsum(starts) - 1
    logical = np.zeros((len(first), physical.shape[1]), dtype=np.float64)
    np.add.at(logical, group, physical)
    return logical, [core.axon_sources[i] for i in first.tolist()]

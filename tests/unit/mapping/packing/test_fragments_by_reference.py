"""[P1 R1] Neuron-split fragments by reference: a fragment windows its parent's weight payload through the compaction
descriptor (``compact_source_matrix`` + ``compact_keep_rows`` + ``compact_keep_cols``) instead of owning a dense copy
of its window, so a split costs no weight storage.

The oracle is the copy recipe the packer used before (``_copied_fragments``: each fragment owns
``parent_dense[:, lo:hi].copy()``, a per-axon parent's logical window split again). Every fragment, at every step of a
chain of splits and on every hard-core placement, resolves to the oracle's grid in bytes, dtype, shape and
C-contiguity, and carries the oracle's other attributes. A stored payload of another dtype keeps the copy (the
descriptor's recipe casts to float64, the copy keeps the payload's dtype), and so does a per-axon window whose
all-zero row holds a negative zero (the logical view's sum writes it as +0, the descriptor would keep its sign).
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import WeightBank
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping, SoftCore, compact_soft_core_mapping
from mimarsinan.mapping.packing.softcore.fragments import build_real_fragment, make_real_fragments
from mimarsinan.mapping.packing.softcore.sign_split import (
    apply_sign_split,
    is_sign_split,
    logical_view,
    split_soft_core,
)
from mimarsinan.mapping.packing.softcore.soft_core_mapper import SoftCoreMapping
from mimarsinan.mapping.platform.platform_constraints import PER_AXON

#: Attributes that hold a fragment's weight storage: compared through ``get_core_matrix`` and the placements.
STORAGE = {"core_matrix", "compact_source_matrix", "compact_keep_rows", "compact_keep_cols", "_axon_source_spans"}


def _copied_fragments(parent, first, rest, group_id=0):
    """The copy recipe of the head before P1, verbatim: each fragment owns a dense copy of its window."""
    total = first + rest
    hb = getattr(parent, "hardware_bias", None)
    bns = None
    if getattr(parent, "weight_bank_id", None) is not None:
        lo0, hi0 = getattr(parent, "bank_neuron_slice", None) or (0, total)
        bns = ((int(lo0), int(lo0 + first)), (int(lo0 + first), int(hi0)))
    signed = is_sign_split(parent)
    dense, sources = logical_view(parent) if signed else (parent.get_core_matrix(), None)
    fragments = tuple(
        build_real_fragment(parent, window=(lo, hi), matrix_slice=dense[:, lo:hi].copy(),
                            bank_neuron_slice=None if bns is None else bns[index], fragment_index=index,
                            group_id=group_id, original_neurons=total, axon_sources=sources)
        for index, (lo, hi) in enumerate(((0, first), (first, total))))
    # [U-LIVE r2] the fragment now takes its bias with every value held per neuron (``take_neuron_columns``); the
    # recipe keeps its own verbatim window of the bias, so the comparison below still reads the two apart
    for fragment, (lo, hi) in zip(fragments, ((0, first), (first, total))):
        fragment.hardware_bias = None if hb is None else hb[lo:hi].copy()
    if signed:
        for fragment in fragments:
            split_soft_core(fragment)
    return fragments


def _source(s):
    return (int(s.core_), int(s.neuron_), bool(s.is_input_), bool(s.is_off_), bool(s.is_always_on_))


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and a.tobytes() == b.tobytes())
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and torch.equal(a, b))
    if isinstance(a, SpikeSource) or isinstance(b, SpikeSource):
        return isinstance(a, SpikeSource) and isinstance(b, SpikeSource) and _source(a) == _source(b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return type(a) is type(b) and _same(vars(a), vars(b))
    return a == b


def _grid_identity(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.flags.c_contiguous == want.flags.c_contiguous
    assert got.tobytes() == want.tobytes()


def _placed(fragment, *, offset: bool) -> HardCore:
    """``fragment`` on an exact-fit hard core, or (``offset``) after a 1 x 1 block on a core one larger each way."""
    axons, neurons = fragment.get_input_count(), fragment.get_output_count()
    if not offset:
        core = HardCore(axons, neurons)
    else:
        core = HardCore(axons + 1, neurons + 1)
        core.add_softcore(SoftCore(np.full((1, 1), 3.0), [SpikeSource(-1, 0, is_input=False, is_off=True)], id=-9))
    core.latency = None
    core.add_softcore(fragment)
    return core


def assert_same_fragment(got, want) -> None:
    """``got`` (by reference) and ``want`` (the copy recipe) are the same fragment to every reader."""
    _grid_identity(got.get_core_matrix(), want.get_core_matrix())
    assert (got.get_input_count(), got.get_output_count()) == (want.get_input_count(), want.get_output_count())
    names = (set(vars(got)) | set(vars(want))) - STORAGE
    for name in sorted(names):
        assert _same(getattr(got, name, None), getattr(want, name, None)), name
    for offset in (False, True):
        a, b = _placed(got, offset=offset), _placed(want, offset=offset)
        _grid_identity(a.get_core_matrix(), b.get_core_matrix())
        assert a.core_matrix_dtype() == b.core_matrix_dtype()
        assert [_source(s) for s in a.axon_sources] == [_source(s) for s in b.axon_sources]


def assert_by_reference(fragment, root: np.ndarray) -> None:
    assert fragment.core_matrix is None, "a fragment owns a dense copy of its window"
    assert fragment.compact_source_matrix is not None
    assert np.shares_memory(fragment.compact_source_matrix, root)


def _sources(n: int, *, bias: bool = True):
    out = [SpikeSource(0, i, is_input=True, is_off=False) for i in range(n - (1 if bias else 0))]
    if bias:
        out.append(SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True))
    return out


def _int_matrix(rng, rows, cols):
    w = rng.integers(-5, 6, size=(rows, cols)).astype(np.float64)
    w[rng.random((rows, cols)) < 0.25] = -0.0  # negative zeros keep their sign through a window by reference
    return w


def _owned(rng):
    w = _int_matrix(rng, 7, 13)
    return SoftCore(w, _sources(7), id=3, name="owned"), w


def _bank(rng, view: bool):
    bank = WeightBank(id=0, core_matrix=_int_matrix(rng, 7, 17))
    lo, hi = (2, 15) if view else (0, 17)
    matrix = bank.column_slice(lo, hi)
    parent = SoftCore(matrix, _sources(7), id=4, name="bank", weight_bank_id=0, bank_axon_slice=(0, 7),
                      bank_neuron_slice=(lo, hi), bank_includes_bias_row=True)
    if view:
        assert not matrix.flags.c_contiguous and np.shares_memory(matrix, bank.core_matrix)
    return parent, bank.core_matrix


def _compacted(rng, banked: bool):
    if banked:
        parent, root = _bank(rng, view=True)
    else:
        parent, root = _owned(rng)
    n_rows, n_cols = parent.get_input_count(), parent.get_output_count()
    parent.pruned_row_mask = [bool(i in (1, 4)) for i in range(n_rows)]
    parent.pruned_col_mask = [bool(j in (0, 5, 6)) for j in range(n_cols)]
    compact_soft_core_mapping([parent], [])
    assert parent.core_matrix is None and parent.compact_keep_rows is not None
    return parent, root


def _signed(rng, *, compacted: bool):
    parent, root = _compacted(rng, banked=False) if compacted else _owned(rng)
    apply_sign_split([parent], PER_AXON)
    assert is_sign_split(parent)
    return parent, root


PARENTS = {
    "owned": lambda rng: _owned(rng),
    "full bank": lambda rng: _bank(rng, view=False),
    "bank column view": lambda rng: _bank(rng, view=True),
    "compaction": lambda rng: _compacted(rng, banked=False),
    "compacted bank view": lambda rng: _compacted(rng, banked=True),
    "per-axon split": lambda rng: _signed(rng, compacted=False),
    "per-axon split after compaction": lambda rng: _signed(rng, compacted=True),
}


def _payload(core) -> np.ndarray:
    return core.core_matrix if core.core_matrix is not None else core.compact_source_matrix


def _expect_copy(parent, lo: int, hi: int) -> bool:
    """Whether the window ``[lo, hi)`` of ``parent`` keeps the copy: a stored payload of another dtype than float64,
    or a per-axon window with an all-zero row that holds a negative zero."""
    if parent.core_matrix is not None:
        return np.asarray(parent.core_matrix).dtype != np.float64
    if not is_sign_split(parent):
        return False
    rows = np.asarray(parent.compact_keep_rows)
    logical = rows[np.concatenate(([True], rows[1:] != rows[:-1]))]
    block = np.asarray(parent.compact_source_matrix, dtype=np.float64)[
        np.ix_(logical, np.asarray(parent.compact_keep_cols)[lo:hi])]
    zero = ~(block != 0).any(axis=1)
    return bool(np.signbit(block[zero]).any())


def _check_split(parent, got, want, first: int) -> None:
    total = parent.get_output_count()
    for g, w, (lo, hi) in zip(got, want, ((0, first), (first, total))):
        assert_same_fragment(g, w)
        if _expect_copy(parent, lo, hi):
            assert not np.shares_memory(_payload(g), _payload(parent))
        else:
            assert_by_reference(g, _payload(parent))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", sorted(PARENTS))
def test_a_fragment_resolves_to_the_copy_recipe_by_reference(kind, seed):
    rng = np.random.default_rng(1000 + seed)
    parent, root = PARENTS[kind](rng)
    assert np.shares_memory(_payload(parent), root)
    total = parent.get_output_count()
    first = int(rng.integers(1, total))
    got = make_real_fragments(softcore=parent, first_neurons=first, remaining_neurons=total - first, group_id=0)
    _check_split(parent, got, _copied_fragments(parent, first, total - first), first)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", sorted(PARENTS))
def test_a_chain_of_splits_resolves_to_the_copy_recipe_by_reference(kind, seed):
    """Two to six successive splits of the remainder: a fragment of a fragment windows the same payload (the root's,
    unless a guard kept a copy on the way)."""
    rng = np.random.default_rng(2000 + seed)
    parent, _root = PARENTS[kind](rng)
    got_tail, want_tail = parent, parent
    for step in range(int(rng.integers(2, 7))):
        total = got_tail.get_output_count()
        if total < 2:
            break
        first = int(rng.integers(1, total))
        got = make_real_fragments(softcore=got_tail, first_neurons=first, remaining_neurons=total - first,
                                  group_id=step)
        want = _copied_fragments(want_tail, first, total - first, group_id=step)
        _check_split(got_tail, got, want, first)
        got_tail, want_tail = got[1], want[1]


def test_the_random_parents_reach_both_branches_of_the_guard():
    """The per-axon parents above draw negative zeros, so the chains exercise the copy branch as well as the
    reference branch."""
    kept = referenced = 0
    for seed in range(6):
        for kind in ("per-axon split", "per-axon split after compaction"):
            parent, _ = PARENTS[kind](np.random.default_rng(3000 + seed))
            total = parent.get_output_count()
            for first in range(1, total):
                for lo, hi in ((0, first), (first, total)):
                    if _expect_copy(parent, lo, hi):
                        kept += 1
                    else:
                        referenced += 1
    assert kept and referenced


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.int8])
def test_a_stored_payload_of_another_dtype_keeps_the_copy(dtype):
    rng = np.random.default_rng(7)
    w = rng.integers(-5, 6, size=(5, 9)).astype(dtype)
    parent = SoftCore(w, _sources(5), id=1)
    got = make_real_fragments(softcore=parent, first_neurons=4, remaining_neurons=5, group_id=0)
    want = _copied_fragments(parent, 4, 5)
    for g, wnt in zip(got, want):
        assert_same_fragment(g, wnt)
        assert g.core_matrix is not None and g.core_matrix.dtype == np.dtype(dtype)
        assert not np.shares_memory(g.core_matrix, w)


def test_a_compaction_of_another_dtype_windows_by_reference():
    """A descriptor parent resolves through the float64 recipe whatever its payload's dtype, so its window does too."""
    rng = np.random.default_rng(8)
    w = rng.integers(-5, 6, size=(6, 10)).astype(np.int16)
    parent = SoftCore(w, _sources(6), id=2)
    parent.pruned_row_mask = [False, True, False, False, False, False]
    parent.pruned_col_mask = [False] * 9 + [True]
    compact_soft_core_mapping([parent], [])
    got = make_real_fragments(softcore=parent, first_neurons=4, remaining_neurons=5, group_id=0)
    want = _copied_fragments(parent, 4, 5)
    for g, wnt in zip(got, want):
        assert_same_fragment(g, wnt)
        assert_by_reference(g, w)


class TestTheNegativeZeroGuard:
    """Under per-axon sign the copy recipe sums a row's W+ and W- (``logical_view``), which writes every zero as +0;
    an all-zero row of a fragment's window keeps its entries as they are, so a negative zero there would keep its
    sign by reference. Such a fragment keeps the copy; its sibling, with no such row, windows the payload."""

    def _parent(self):
        w = np.array([[3.0, -2.0, 0.0, 0.0],
                      [1.0, 2.0, 4.0, -1.0],
                      [-0.0, 0.0, 5.0, -0.0],
                      [0.0, 0.0, 0.0, 0.0]])
        parent = SoftCore(w, _sources(4, bias=False), id=5)
        apply_sign_split([parent], PER_AXON)
        return parent, w

    def test_the_fragment_with_a_negative_zero_in_a_zero_row_keeps_the_copy(self):
        parent, w = self._parent()
        got = make_real_fragments(softcore=parent, first_neurons=2, remaining_neurons=2, group_id=0)
        want = _copied_fragments(parent, 2, 2)
        for g, wnt in zip(got, want):
            assert_same_fragment(g, wnt)
        grid = want[0].get_core_matrix()
        assert not (np.signbit(grid) & (grid == 0)).any()     # the copy writes +0 in row 2 of the window [0, 2)
        assert _expect_copy(parent, 0, 2) and not _expect_copy(parent, 2, 4)
        assert not np.shares_memory(got[0].compact_source_matrix, w)
        assert_by_reference(got[1], w)                         # row 2 of [2, 4) is signed, rows 0 and 3 hold +0

    def test_a_negative_zero_in_a_signed_row_windows_by_reference(self):
        w = np.array([[3.0, -0.0, 2.0], [-1.0, -0.0, 4.0]])
        parent = SoftCore(w, _sources(2, bias=False), id=6)
        apply_sign_split([parent], PER_AXON)
        got = make_real_fragments(softcore=parent, first_neurons=2, remaining_neurons=1, group_id=0)
        want = _copied_fragments(parent, 2, 1)
        for g, wnt in zip(got, want):
            assert_same_fragment(g, wnt)
            assert_by_reference(g, w)


def test_split_bank_instances_retain_no_weight_copy():
    """K bank-backed instances of one 2N-neuron core packed with neuron splitting onto N-neuron cores retain under
    0.1 B per split cell (tracemalloc over the pack); the copy recipe retained 8 B per cell (one float64 each)."""
    a, n, k = 32, 8192, 16
    bank = np.random.default_rng(9).integers(-7, 8, size=(a, 2 * n)).astype(np.float64)
    soft = SoftCoreMapping()
    for i in range(k):
        core = SoftCore(bank, _sources(a, bias=False), id=i, name=f"position{i}", weight_bank_id=0,
                        bank_axon_slice=(0, a), bank_neuron_slice=(0, 2 * n))
        core.latency = 0
        soft.cores.append(core)
    soft.weight_banks = {0: bank}
    hcm = HardCoreMapping([HardCore(a, n, has_bias_capability=False) for _ in range(2 * k)])
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        hcm.map(soft, allow_neuron_splitting=True, allow_coalescing=False)
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert len(hcm.cores) == 2 * k
    assert sum(len(p) for p in hcm.soft_core_placements_per_hard_core) == 2 * k
    cells = k * a * 2 * n
    assert retained / cells < 0.1, f"{retained / cells:.3f} B retained per split cell"
    for core in hcm.cores:
        (placement,) = core.matrix_placements
        assert np.shares_memory(placement.source, bank)

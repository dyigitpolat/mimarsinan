"""CL-SG: per-axon weight sign. A soft core's mixed-sign rows split into an adjacent W+ then W- pair after
compaction and before packing; single-sign and all-zero rows keep one row; the tiler budgets half the rows."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing.softcore import (
    HardCore,
    SoftCore,
    compact_soft_core_mapping,
    compacted_core_extent,
    deployed_core_extent,
)
from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments
from mimarsinan.mapping.packing.softcore.matrix_placement import (
    MatrixPlacement,
    SignedRowPlacement,
    core_matrix_content_key,
)
from mimarsinan.mapping.packing.softcore.sign_split import (
    SignSplitError,
    apply_sign_split,
    is_sign_split,
    logical_view,
    sign_split_rows,
)
from mimarsinan.mapping.platform.platform_constraints import (
    PER_AXON,
    PER_SYNAPSE,
    SignGranularityError,
    pool_sign_granularity,
    resolve_platform_mapping_params,
    rows_per_input,
)

ODIN = {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False, "weight_sign_granularity": PER_AXON}


def _int_matrix(rng, rows, cols, low=-7, high=7, zero_rows=()):
    w = rng.integers(low, high + 1, size=(rows, cols)).astype(np.float64)
    for r in zero_rows:
        w[r] = 0.0
    return w


def _sources(n, *, bias=True):
    out = [SpikeSource(0, i, is_input=True, is_off=False) for i in range(n - (1 if bias else 0))]
    if bias:
        out.append(SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True))
    return out


def _soft(matrix, *, bias=True, core_id=0):
    return SoftCore(core_matrix=matrix, axon_sources=_sources(matrix.shape[0], bias=bias), id=core_id)


def _fold(physical, origin):
    """The logical matrix back from the physical rows: the rows of one origin summed (one is zero per entry)."""
    out = np.zeros((int(origin.max()) + 1, physical.shape[1]))
    np.add.at(out, origin, physical)
    return out


class TestTheDeclaredProperty:
    def test_an_absent_granularity_is_per_synapse_and_one_per_axon_type_makes_the_pool_per_axon(self):
        base = {"max_axons": 64, "max_neurons": 64, "count": 2}
        assert pool_sign_granularity([base]) == PER_SYNAPSE
        assert pool_sign_granularity([dict(base, weight_sign_granularity=PER_SYNAPSE)]) == PER_SYNAPSE
        assert pool_sign_granularity([base, dict(base, weight_sign_granularity=PER_AXON)]) == PER_AXON
        assert (rows_per_input(PER_SYNAPSE), rows_per_input(PER_AXON)) == (1, 2)

    def test_an_unknown_granularity_is_refused_by_name(self):
        with pytest.raises(SignGranularityError, match="per_row"):
            pool_sign_granularity([{"max_axons": 8, "max_neurons": 8, "weight_sign_granularity": "per_row"}])
        assert issubclass(SignGranularityError, ToolchainRefusal)

    def test_the_fan_in_tiler_budgets_half_the_rows(self):
        """ODIN (256 rows, no bias lane): 127 logical inputs and the bias row, at most 2 x 128 = 256 rows (the
        row capacity a full tile of mixed rows takes)."""
        odin = resolve_platform_mapping_params([ODIN])
        assert (odin.effective_max_axons, odin.rows_per_input, odin.row_capacity) == (127, 2, 256)
        lane = resolve_platform_mapping_params([dict(ODIN, has_bias=True)])
        assert (lane.effective_max_axons, lane.row_capacity) == (128, 256)

    def test_a_per_synapse_platform_resolves_as_before(self):
        for has_bias, expected in ((False, 255), (True, 256)):
            params = resolve_platform_mapping_params([{"max_axons": 256, "max_neurons": 128, "has_bias": has_bias}])
            assert (params.effective_max_axons, params.rows_per_input, params.row_capacity) == (
                expected, 1, expected)

    def test_a_mixed_pool_tiles_at_the_per_axon_budget(self):
        mixed = [dict(ODIN, count=2), {"max_axons": 128, "max_neurons": 256, "count": 2, "has_bias": False}]
        assert resolve_platform_mapping_params(mixed).effective_max_axons == 127

    def test_a_per_axon_core_without_room_for_one_input_is_refused_by_name(self):
        with pytest.raises(SignGranularityError, match="max_axons 3"):
            resolve_platform_mapping_params([dict(ODIN, max_axons=3)])


class TestPhysicalRows:
    def test_the_hand_matrix(self):
        w = np.array([[2, -1], [3, 0], [0, -4], [0, 0], [-5, 6]], dtype=np.float64)
        origin, signs = sign_split_rows(w)
        assert origin.tolist() == [0, 0, 1, 2, 3, 4, 4]
        assert signs.tolist() == [1, -1, 1, -1, 0, 1, -1]

    @pytest.mark.parametrize("seed", range(12))
    def test_every_row_carries_one_sign_and_the_fold_is_the_matrix(self, seed):
        rng = np.random.default_rng(seed)
        w = _int_matrix(rng, 9, 7, zero_rows=(2,))
        w[4] = np.abs(w[4])
        w[5] = -np.abs(w[5])
        soft = _soft(w)
        apply_sign_split([soft], PER_AXON)
        physical = soft.get_core_matrix()
        origin = soft.compact_keep_rows
        mixed = int(((w > 0).any(axis=1) & (w < 0).any(axis=1)).sum())
        assert soft.get_input_count() == physical.shape[0] == 9 + mixed
        assert np.array_equal(_fold(physical, origin), w)
        for row, sign in zip(physical, soft.compact_row_signs):
            assert (row >= 0).all() or (row <= 0).all()
            if sign:
                assert (row * sign >= 0).all()
        pairs = np.flatnonzero(np.diff(origin) == 0)
        assert all(soft.compact_row_signs[i] == 1 and soft.compact_row_signs[i + 1] == -1 for i in pairs)
        assert (np.diff(origin) >= 0).all()

    def test_the_bias_pair_is_at_the_tail_and_reads_the_always_on_source(self):
        w = np.array([[1, 2], [-1, 3], [2, -2]], dtype=np.float64)
        soft = _soft(w)
        apply_sign_split([soft], PER_AXON)
        assert soft.compact_keep_rows.tolist() == [0, 1, 1, 2, 2]
        assert [s.is_always_on_ for s in soft.axon_sources] == [False, False, False, True, True]

    def test_a_duplicated_source_is_a_distinct_object(self):
        """The packer remaps every placed source in place (``_finalize_sources``), once per object."""
        w = np.array([[1, -1], [2, 2]], dtype=np.float64)
        soft = SoftCore(core_matrix=w, axon_sources=[SpikeSource(3, 1), SpikeSource(3, 2)], id=0)
        apply_sign_split([soft], PER_AXON)
        first, second = soft.axon_sources[0], soft.axon_sources[1]
        assert first is not second and (first.core_, first.neuron_) == (second.core_, second.neuron_) == (3, 1)

    def test_per_synapse_is_a_no_op_that_sets_no_attribute(self):
        w = np.array([[1, -1], [2, 2]], dtype=np.float64)
        soft = _soft(w)
        sources = list(soft.axon_sources)
        apply_sign_split([soft], PER_SYNAPSE)
        assert soft.core_matrix is w and soft.axon_sources == sources
        assert "compact_row_signs" not in vars(soft) and not is_sign_split(soft)

    def test_a_soft_core_without_a_mixed_row_carries_its_signs_and_keeps_its_rows(self):
        w = np.array([[1, 2], [-1, -3], [0, 0]], dtype=np.float64)
        soft = _soft(w)
        apply_sign_split([soft], PER_AXON)
        assert soft.get_input_count() == 3 and soft.compact_row_signs.tolist() == [1, -1, 0]
        assert np.array_equal(soft.get_core_matrix(), w)

    def test_the_split_composes_with_pruning_compaction(self):
        w = np.array([[1, -1, 2], [3, 4, 0], [-2, 5, -6], [1, 1, -1]], dtype=np.float64)
        soft = _soft(w)
        soft.pruned_row_mask = [False, True, False, False]
        soft.pruned_col_mask = [False, False, True]
        compact_soft_core_mapping([soft], [SpikeSource(0, 0)])
        apply_sign_split([soft], PER_AXON)
        kept = w[np.ix_([0, 2, 3], [0, 1])]
        assert np.array_equal(_fold(soft.get_core_matrix(), np.repeat(np.arange(3), [2, 2, 1])), kept)
        assert soft.compact_keep_rows.tolist() == [0, 0, 2, 2, 3]

    def test_the_shared_payload_stays_shared_and_the_bank_bookkeeping_is_dropped(self):
        bank = np.array([[1, -1], [2, 2], [-3, 1]], dtype=np.float64)
        cores = []
        for core_id in range(3):
            soft = _soft(bank, core_id=core_id)
            soft.weight_bank_id, soft.bank_axon_slice, soft.bank_neuron_slice = 0, (0, 3), (0, 2)
            cores.append(soft)
        apply_sign_split(cores, PER_AXON)
        assert all(c.compact_source_matrix is bank for c in cores)
        assert cores[0].compact_keep_rows is cores[1].compact_keep_rows is cores[2].compact_keep_rows
        assert cores[0].compact_row_signs is cores[2].compact_row_signs
        assert all(c.weight_bank_id is None and c.bank_axon_slice is None for c in cores)

    @pytest.mark.parametrize("bad", [0.5, np.nan, np.inf])
    def test_a_weight_off_the_integer_grid_is_refused_by_name(self, bad):
        w = np.array([[1, -1], [2, bad]], dtype=np.float64)
        with pytest.raises(SignSplitError, match="integer"):
            apply_sign_split([_soft(w, core_id=7)], PER_AXON)
        assert issubclass(SignSplitError, ToolchainRefusal)

    def test_an_integer_typed_matrix_splits(self):
        w = np.array([[3, -2], [-8, -1]], dtype=np.int8)
        soft = _soft(w)
        apply_sign_split([soft], PER_AXON)
        assert np.array_equal(_fold(soft.get_core_matrix(), soft.compact_keep_rows), w.astype(np.float64))


class TestPlacementIdentity:
    def _hard(self, soft):
        hc = HardCore(16, 4, has_bias_capability=False)
        hc.add_softcore(soft)
        return hc

    def test_a_split_soft_core_places_a_signed_descriptor(self):
        w = np.array([[1, -1], [2, 2]], dtype=np.float64)
        plain, split = _soft(w.copy()), _soft(w)
        apply_sign_split([split], PER_AXON)
        assert type(self._hard(plain).matrix_placements[0]) is MatrixPlacement
        placement = self._hard(split).matrix_placements[0]
        assert isinstance(placement, SignedRowPlacement) and placement.source is w
        grid = self._hard(split).get_core_matrix()
        assert np.array_equal(grid[:3, :2], [[1, 0], [0, -1], [2, 2]]) and not grid[3:].any()

    def test_the_content_key_carries_the_signs(self):
        """Two placements alike in payload and rows but not in signs must never share a resolved grid."""
        source = np.array([[1, -1], [2, -2]], dtype=np.float64)
        rows, cols = np.array([0, 0, 1, 1]), np.array([0, 1])
        a = SignedRowPlacement(source=source, keep_rows=rows, keep_cols=cols, row_signs=np.array([1, -1, 1, -1]),
                               axon_offset=0, neuron_offset=0, axons=4, neurons=2)
        b = SignedRowPlacement(source=source, keep_rows=rows, keep_cols=cols, row_signs=np.array([-1, 1, 1, -1]),
                               axon_offset=0, neuron_offset=0, axons=4, neurons=2)
        assert a.content_key() != b.content_key()
        assert core_matrix_content_key(None, [a], 4, 2) != core_matrix_content_key(None, [b], 4, 2)
        assert not np.array_equal(a.materialize(), b.materialize())

    def test_a_split_hard_core_pickles_and_resolves_alike(self):
        w = np.array([[1, -1], [2, 2]], dtype=np.float64)
        soft = _soft(w)
        apply_sign_split([soft], PER_AXON)
        hc = self._hard(soft)
        again = pickle.loads(pickle.dumps(hc))
        assert np.array_equal(again.get_core_matrix(), hc.get_core_matrix())
        assert isinstance(again.matrix_placements[0], SignedRowPlacement)


class TestFragments:
    @pytest.mark.parametrize("seed", range(8))
    def test_fragments_re_derive_their_split_from_the_logical_columns(self, seed):
        rng = np.random.default_rng(100 + seed)
        w = _int_matrix(rng, 6, 9)
        w[:, :4] = np.abs(w[:, :4])
        parent = _soft(w)
        apply_sign_split([parent], PER_AXON)
        first, second = make_real_fragments(softcore=parent, first_neurons=4, remaining_neurons=5, group_id=0)
        for frag, (lo, hi) in ((first, (0, 4)), (second, (4, 9))):
            assert is_sign_split(frag)
            logical, sources = logical_view(frag)
            assert np.array_equal(logical, w[:, lo:hi])
            assert [s.is_always_on_ for s in sources] == [False] * 5 + [True]
            mixed = int(((w[:, lo:hi] > 0).any(axis=1) & (w[:, lo:hi] < 0).any(axis=1)).sum())
            assert frag.get_input_count() == 6 + mixed <= parent.get_input_count()
        assert first.get_input_count() == 6
        assert first.axon_sources[0] is not parent.axon_sources[0]

    def test_a_fragment_of_a_per_synapse_core_is_as_before(self):
        w = np.array([[1, -1, 3], [2, 2, -1]], dtype=np.float64)
        parent = _soft(w)
        first, second = make_real_fragments(softcore=parent, first_neurons=1, remaining_neurons=2, group_id=0)
        # [P1 R1] a fragment windows its parent's payload by reference: its grid is read through the accessor
        assert not is_sign_split(first) and np.array_equal(first.get_core_matrix(), w[:, :1])
        assert np.array_equal(second.get_core_matrix(), w[:, 1:])


def _ir_graph(rng, *, banked=False, pruned=False):
    """Two cores: an owned (or bank-backed) hidden layer over 6 inputs and the bias, and an owned output layer."""
    w1 = _int_matrix(rng, 7, 5)
    w1[1] = np.abs(w1[1])
    src1 = np.array([IRSource(node_id=-2, index=i) for i in range(6)] + [IRSource(node_id=-3, index=0)],
                    dtype=object)
    banks = {}
    if banked:
        banks[0] = WeightBank(id=0, core_matrix=w1)
        hidden = NeuralCore(id=0, name="hidden", input_sources=src1, core_matrix=None, weight_bank_id=0,
                            weight_row_slice=(0, 5), threshold=1.0, latency=0)
    else:
        hidden = NeuralCore(id=0, name="hidden", input_sources=src1, core_matrix=w1, threshold=1.0, latency=0)
    if pruned:
        hidden.pruned_row_mask = [False, False, True, False, False, False, False]
        hidden.pruned_col_mask = [False, False, False, False, False]
    w2 = _int_matrix(rng, 6, 3)
    src2 = np.array([IRSource(node_id=0, index=i) for i in range(5)] + [IRSource(node_id=-3, index=0)],
                    dtype=object)
    out = NeuralCore(id=1, name="out", input_sources=src2, core_matrix=w2, threshold=1.0, latency=1)
    graph = IRGraph(nodes=[hidden, out], output_sources=np.array([IRSource(node_id=1, index=i) for i in range(3)],
                                                                 dtype=object), weight_banks=banks)
    return graph


class TestDeployedExtent:
    @pytest.mark.parametrize("banked", [False, True])
    @pytest.mark.parametrize("pruned", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_the_deployed_extent_is_the_runtime_soft_cores(self, banked, pruned, seed):
        graph = _ir_graph(np.random.default_rng(seed), banked=banked, pruned=pruned)
        mapping = build_hybrid_hard_core_mapping(ir_graph=graph, cores_config=[dict(ODIN, count=4)])
        placed = {}
        for stage in mapping.stages:
            for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
                for p in placements:
                    placed[p["ir_node_id"]] = (p["axons"], p["neurons"])
        for core in graph.get_neural_cores():
            assert deployed_core_extent(core, PER_AXON, graph=graph) == placed[core.id]
            assert deployed_core_extent(core, PER_SYNAPSE, graph=graph) == compacted_core_extent(core)
        assert any(placed[c.id][0] > compacted_core_extent(c)[0] for c in graph.get_neural_cores())

"""[Defect B] Hard cores carry placement descriptors, never per-core dense f64 grids.

The defect this pins (real vehicle, 2026-08-05): ``compact_soft_core_mapping``
materialized a per-instance ``np.ix_`` float64 copy and nulled bank identity,
and ``HardCore.add_softcore`` padded dense composites — 33.94 GB of unique
padded grids in a 35.36 GB hybrid_mapping pickle while the actual weights are
57 MB. The contract now: cores store placements (shared payload reference +
keep-index arrays + block offsets + padded shape); dense float64 materializes
TRANSIENTLY via ``get_core_matrix()``, byte-identical to the retired eager
path, with the exact-fit single-placement fast path returning the shared base
object (today's aliasing).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.ir.legacy_convert import neural_core_to_soft_core
from mimarsinan.mapping.packing.softcore import (
    HardCore,
    HardCoreMapping,
    SoftCore,
    compact_soft_core_mapping,
)
from mimarsinan.mapping.packing.softcore.soft_core_mapper import SoftCoreMapping


# ---------------------------------------------------------------- frozen oracle
def _eager_softcore_dense(node: NeuralCore, graph: IRGraph) -> np.ndarray:
    """The RETIRED eager recipe, reimplemented independently (byte-verified
    against the pre-descriptor code): SRC = bank | bank column view | owned;
    masks -> f64 ``np.ix_`` copy with the always-on last axon never dropped;
    BIAS_ONLY -> ``(1, n)`` f64 zeros; untouched -> SRC itself."""
    if node.weight_bank_id is not None:
        bank = graph.weight_banks[node.weight_bank_id].core_matrix
        wrs = node.weight_row_slice
        src = bank if wrs is None else bank[:, wrs[0]:wrs[1]]
    else:
        src = node.core_matrix
    n_axons, n_neurons = src.shape
    row_mask = getattr(node, "pruned_row_mask", None)
    col_mask = getattr(node, "pruned_col_mask", None)
    if (
        row_mask is None or col_mask is None
        or len(row_mask) != n_axons or len(col_mask) != n_neurons
    ):
        return src
    sources = node.input_sources.flatten()
    keep_rows = [r for r in range(n_axons) if not row_mask[r]]
    keep_cols = [c for c in range(n_neurons) if not col_mask[c]]
    if (
        len(sources) and sources[-1].is_always_on()
        and n_axons and (n_axons - 1) not in keep_rows
    ):
        keep_rows = sorted(keep_rows + [n_axons - 1])
    if len(keep_rows) == n_axons and len(keep_cols) == n_neurons:
        return src
    if not keep_rows:
        return np.zeros((1, len(keep_cols)), dtype=np.float64)
    return np.asarray(src, dtype=np.float64)[np.ix_(keep_rows, keep_cols)].copy()


def _eager_hardcore_dense(hardcore, placements, block_by_node_id) -> np.ndarray:
    """The retired eager composite: exact-fit single block IS the grid, else
    zeros in the FIRST-placed block's dtype with diagonal block writes."""
    blocks = [block_by_node_id[p["ir_node_id"]] for p in placements]
    if (
        len(placements) == 1
        and placements[0]["axon_offset"] == 0
        and placements[0]["neuron_offset"] == 0
        and blocks[0].shape
        == (hardcore.axons_per_core, hardcore.neurons_per_core)
    ):
        return blocks[0]
    comp = np.zeros(
        (hardcore.axons_per_core, hardcore.neurons_per_core),
        dtype=blocks[0].dtype,
    )
    for p, blk in zip(placements, blocks):
        a0, n0 = p["axon_offset"], p["neuron_offset"]
        comp[a0:a0 + p["axons"], n0:n0 + p["neurons"]] = blk
    return comp


def _assert_bytes_equal(got: np.ndarray, want: np.ndarray, label: str) -> None:
    assert got.dtype == want.dtype, f"{label}: dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{label}: shape {got.shape} != {want.shape}"
    assert (
        np.ascontiguousarray(got).tobytes()
        == np.ascontiguousarray(want).tobytes()
    ), f"{label}: dense bytes diverge from the eager oracle"


# ---------------------------------------------------------------- vehicles
BANK_IN, BANK_OUT = 6, 4
ROW_MASK = [False, True, False, False, False, True]  # row 5 is the bias row
COL_MASK = [False, True, False, False]
KEEP_ROWS_A = [0, 2, 3, 4, 5]  # row 5 forced back: always-on axons never drop
KEEP_COLS_A = [0, 2, 3]


def _mk_sources(n_weight_axons: int, bias_row: bool) -> np.ndarray:
    src = [IRSource(node_id=-2, index=i) for i in range(n_weight_axons)]
    if bias_row:
        src.append(IRSource(node_id=-3, index=0))
    return np.array(src, dtype=object)


def _edge_case_graph():
    """int8 bank + per-instance masks + BIAS_ONLY + slices + owned + hw bias."""
    rng = np.random.default_rng(7)
    bank_mat = rng.integers(-128, 128, size=(BANK_IN, BANK_OUT)).astype(np.int8)
    bank = WeightBank(id=0, core_matrix=bank_mat)

    core_a = NeuralCore(
        id=0, name="a", input_sources=_mk_sources(5, True), core_matrix=None,
        weight_bank_id=0, weight_row_slice=None,
    )
    core_a.pruned_row_mask = list(ROW_MASK)
    core_a.pruned_col_mask = list(COL_MASK)

    core_b = NeuralCore(
        id=1, name="b", input_sources=_mk_sources(5, True), core_matrix=None,
        weight_bank_id=0, weight_row_slice=(1, 3),
    )
    core_b.pruned_row_mask = list(ROW_MASK)
    core_b.pruned_col_mask = COL_MASK[1:3]

    core_c = NeuralCore(
        id=2, name="c", input_sources=_mk_sources(5, True), core_matrix=None,
        weight_bank_id=0, weight_row_slice=None,
    )
    core_c.pruned_row_mask = [False] * BANK_IN
    core_c.pruned_col_mask = [False] * BANK_OUT

    own_mat = rng.integers(-128, 128, size=(3, 4)).astype(np.int8)
    hb_d = rng.integers(-128, 128, size=(4,)).astype(np.int8)
    core_d = NeuralCore(
        id=3, name="d", input_sources=_mk_sources(3, False),
        core_matrix=own_mat, hardware_bias=hb_d,
    )
    core_d.pruned_row_mask = [False, True, False]
    core_d.pruned_col_mask = [False, False, True, False]

    core_e = NeuralCore(
        id=4, name="e", input_sources=_mk_sources(2, False),
        core_matrix=rng.integers(-5, 5, size=(2, 3)).astype(np.int8),
        hardware_bias=rng.integers(-5, 5, size=(3,)).astype(np.int8),
    )
    core_e.pruned_row_mask = [True, True]
    core_e.pruned_col_mask = [False, False, False]

    graph = IRGraph(
        nodes=[core_a, core_b, core_c, core_d, core_e],
        output_sources=np.array([IRSource(0, 0)], dtype=object),
        weight_banks={0: bank},
    )
    return graph, bank_mat, own_mat, hb_d


def _compacted_softcores(graph):
    softs = [neural_core_to_soft_core(n, graph=graph) for n in graph.nodes]
    for sc in softs:
        sc.threshold = 1.0
        sc.latency = 1
    compact_soft_core_mapping(softs, [SpikeSource(0, 0)])
    return softs


def _scm_for(softs, bank_mat, out=(0, 0)):
    scm = SoftCoreMapping()
    scm.cores = list(softs)
    scm.output_sources = [SpikeSource(*out)]
    scm.weight_banks = {0: bank_mat}
    for sc in scm.cores:
        sc.residency_class_id = 42
        sc.latency = 1
    return scm


def _masked_bank_ir(n_instances: int, rows: int = 129, cols: int = 64):
    """One shared f64 bank, every instance compacted by the SAME masks —
    the real-vehicle shape whose eager pickle scaled with instance count."""
    rng = np.random.default_rng(3)
    bank = WeightBank(
        id=0, core_matrix=rng.normal(size=(rows, cols)).astype(np.float64)
    )
    row_mask = [(r % 2 == 1) for r in range(rows)]
    col_mask = [(c % 2 == 1) for c in range(cols)]
    nodes = []
    for k in range(n_instances):
        nodes.append(NeuralCore(
            id=k, name=f"inst{k}",
            input_sources=_mk_sources(rows - 1, True), core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, cols),
            threshold=1.0, latency=0,
        ))
        nodes[-1].pruned_row_mask = row_mask
        nodes[-1].pruned_col_mask = col_mask
    out = np.array(
        [IRSource(k, 0) for k in range(n_instances)], dtype=object
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def _grid_payload_bytes(rows: int = 129, cols: int = 64) -> int:
    keep_rows = sum(1 for r in range(rows) if r % 2 == 0) + 1  # + forced bias row
    keep_cols = sum(1 for c in range(cols) if c % 2 == 0)
    return keep_rows * keep_cols * 8


# ---------------------------------------------------------------- tests
class TestCompactionKeepsBankStorage:
    def test_compacted_bank_core_references_the_bank_not_a_copy(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        a = softs[0]
        assert a.core_matrix is None, "compaction must not materialize dense"
        assert a.compact_source_matrix is bank_mat, (
            "the descriptor must keep referencing the SHARED bank payload"
        )
        assert a.compact_keep_rows.tolist() == KEEP_ROWS_A
        assert a.compact_keep_cols.tolist() == KEEP_COLS_A
        # Public bank metadata still nulls exactly as before: the packer's
        # affinity/placement bookkeeping must not see compacted cores as banked.
        assert a.weight_bank_id is None and a.bank_axon_slice is None
        assert a.bank_neuron_slice is None and a.bank_includes_bias_row is False

    def test_sliced_bank_core_references_the_memoized_column_view(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        b = softs[1]
        view = graph.weight_banks[0].column_slice(1, 3)
        assert b.compact_source_matrix is view
        assert np.shares_memory(view, bank_mat)

    def test_dense_resolution_matches_the_eager_recipe(self):
        graph, _bank, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        for node, sc in zip(graph.nodes, softs):
            want = _eager_softcore_dense(node, graph)
            _assert_bytes_equal(sc.get_core_matrix(), want, f"softcore {sc.id}")

    def test_uncompacted_core_still_is_the_bank_object(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        assert softs[2].core_matrix is bank_mat
        assert softs[2].get_core_matrix() is bank_mat
        assert softs[2].weight_bank_id == 0
        assert softs[2].bank_includes_bias_row is True

    def test_bias_only_collapse_stays_eager_and_exact(self):
        graph, _bank, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        e = softs[4]
        assert e.core_matrix is not None and e.core_matrix.shape == (1, 3)
        assert e.core_matrix.dtype == np.float64
        assert not e.core_matrix.any()
        assert len(e.axon_sources) == 1 and e.axon_sources[0].is_off_
        assert e.hardware_bias.dtype == np.int8

    def test_hardware_bias_compacts_eagerly_dtype_preserving(self):
        graph, _bank, _own, hb_d = _edge_case_graph()
        softs = _compacted_softcores(graph)
        d = softs[3]
        assert d.hardware_bias.dtype == np.int8
        assert d.hardware_bias.tobytes() == hb_d[[0, 1, 3]].tobytes()

    def test_extent_accessors_read_the_descriptor(self):
        graph, _bank, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        assert softs[0].get_input_count() == len(KEEP_ROWS_A)
        assert softs[0].get_output_count() == len(KEEP_COLS_A)
        assert softs[4].get_output_count() == 3  # BIAS_ONLY keeps live columns

    def test_keep_index_arrays_are_shared_across_same_mask_instances(self):
        graph = _masked_bank_ir(n_instances=3, rows=9, cols=4)
        softs = [neural_core_to_soft_core(n, graph=graph) for n in graph.nodes]
        compact_soft_core_mapping(
            softs, [SpikeSource(0, 0), SpikeSource(1, 0), SpikeSource(2, 0)]
        )
        assert softs[0].compact_keep_rows is softs[1].compact_keep_rows
        assert softs[0].compact_keep_cols is softs[2].compact_keep_cols

    def test_resolving_without_any_matrix_fails_loud(self):
        sc = SoftCore(core_matrix=None, axon_sources=[], id=9)
        with pytest.raises(ValueError, match="core_matrix"):
            sc.get_core_matrix()


class TestFrozenOracleDifferential:
    def test_identity_mapping_matches_eager_oracle_per_core(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hcm = HardCoreMapping([])
        hcm.map_identity(_scm_for(softs, bank_mat))
        blocks = {
            n.id: _eager_softcore_dense(n, graph) for n in graph.nodes
        }
        for idx, hc in enumerate(hcm.cores):
            placements = hcm.soft_core_placements_per_hard_core[idx]
            want = _eager_hardcore_dense(hc, placements, blocks)
            _assert_bytes_equal(hc.get_core_matrix(), want, f"hardcore {idx}")

    def test_pool_padded_composite_matches_eager_oracle(self):
        graph, bank_mat, _own, hb_d = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hcm = HardCoreMapping(
            [HardCore(8, 6, has_bias_capability=True) for _ in range(4)]
        )
        hcm.map(
            _scm_for([softs[0], softs[3]], bank_mat),
            allow_neuron_splitting=False, allow_coalescing=True,
        )
        assert len(hcm.cores) == 1, "vehicle must co-pack into one padded core"
        hc = hcm.cores[0]
        placements = hcm.soft_core_placements_per_hard_core[0]
        assert placements[1]["axon_offset"] == placements[0]["axons"]
        assert placements[1]["neuron_offset"] == placements[0]["neurons"]
        blocks = {n.id: _eager_softcore_dense(n, graph) for n in graph.nodes}
        want = _eager_hardcore_dense(hc, placements, blocks)
        _assert_bytes_equal(hc.get_core_matrix(), want, "padded composite")
        # hardware_bias composite stays eager: full-width f64 zeros + block.
        d_place = next(p for p in placements if p["ir_node_id"] == 3)
        want_hb = np.zeros(6)
        n0 = d_place["neuron_offset"]
        want_hb[n0:n0 + 3] = hb_d[[0, 1, 3]]
        assert hc.hardware_bias.dtype == np.float64
        assert hc.hardware_bias.tobytes() == want_hb.tobytes()

    def test_composite_dtype_follows_the_first_placed_block(self):
        def _inputs(n):
            return [SpikeSource(-2, i, is_input=True) for i in range(n)]

        t1 = SoftCore(
            core_matrix=np.full((2, 2), -3, dtype=np.int8),
            axon_sources=_inputs(2), id=1,
        )
        t2 = SoftCore(
            core_matrix=np.full((2, 2), 7.0, dtype=np.float64),
            axon_sources=_inputs(2), id=2,
        )
        hc = HardCore(8, 6)
        hc.add_softcore(t1)
        hc.add_softcore(t2)
        got = hc.get_core_matrix()
        assert got.dtype == np.int8, "composite dtype follows the FIRST block"
        assert got[2, 2] == 7 and got[3, 3] == 7  # f64 block casts in exactly

    def test_fused_core_composite_matches(self):
        from mimarsinan.mapping.packing.softcore.hard_core_mapping import (
            RuntimeMaterializer,
        )

        hcm = HardCoreMapping([])
        fused = RuntimeMaterializer(hcm).fuse_hardcores(
            [HardCore(4, 6), HardCore(4, 6)]
        )
        assert fused.fused_component_axons == [4, 4]
        rng = np.random.default_rng(5)
        wide = SoftCore(
            core_matrix=rng.normal(size=(7, 5)),
            axon_sources=[SpikeSource(-2, i, is_input=True) for i in range(7)],
            id=11,
        )
        fused.add_softcore(wide)
        want = np.zeros((8, 6), dtype=np.float64)
        want[:7, :5] = wide.core_matrix
        _assert_bytes_equal(fused.get_core_matrix(), want, "fused composite")

    def test_composite_materializes_fresh_and_is_never_cached(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hcm = HardCoreMapping([])
        hcm.map_identity(_scm_for([softs[0]], bank_mat))
        hc = hcm.cores[0]
        first = hc.get_core_matrix()
        second = hc.get_core_matrix()
        assert first is not second, "transient dense must not be retained"
        _assert_bytes_equal(second, first, "repeated materialization")
        assert hc.core_matrix is None, "resolution must never write the core"

    def test_pickle_round_trip_preserves_dense_bytes(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hcm = HardCoreMapping([])
        hcm.map_identity(_scm_for(softs, bank_mat))
        before = [hc.get_core_matrix() for hc in hcm.cores]
        loaded = pickle.loads(pickle.dumps(hcm))
        for idx, hc in enumerate(loaded.cores):
            _assert_bytes_equal(
                hc.get_core_matrix(), before[idx], f"reloaded core {idx}"
            )


class TestExactFitAliasing:
    def _uncompacted_exact_fit(self):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hard = HardCore(
            axons_per_core=BANK_IN, neurons_per_core=BANK_OUT,
        )
        hard.add_softcore(softs[2])
        return hard, bank_mat

    def test_exact_fit_returns_the_shared_base_object(self):
        hard, bank_mat = self._uncompacted_exact_fit()
        assert hard.get_core_matrix() is bank_mat
        assert hard.get_core_matrix() is hard.get_core_matrix()

    def test_identity_mapping_shares_one_bank_payload_across_cores(self):
        graph = _masked_bank_ir(n_instances=4, rows=9, cols=4)
        for node in graph.nodes:  # no elimination: pure sharing vehicle
            node.pruned_row_mask = None
            node.pruned_col_mask = None
        bank_mat = graph.weight_banks[0].core_matrix
        softs = [neural_core_to_soft_core(n, graph=graph) for n in graph.nodes]
        scm = SoftCoreMapping()
        scm.cores = softs
        scm.output_sources = [SpikeSource(0, 0)]
        scm.weight_banks = {0: bank_mat}
        for sc in softs:
            sc.threshold = 1.0
            sc.latency = 1
        hcm = HardCoreMapping([])
        hcm.map_identity(scm)
        for hc in hcm.cores:
            assert hc.get_core_matrix() is bank_mat

    def test_owned_dense_write_stays_authoritative(self):
        hard, bank_mat = self._uncompacted_exact_fit()
        owned = np.zeros((BANK_IN, BANK_OUT), dtype=np.float64)
        hard.core_matrix = owned
        assert hard.get_core_matrix() is owned
        assert hard.has_core_matrix()

    def test_resolving_an_empty_hardcore_fails_loud(self):
        hc = HardCore(4, 4)
        assert not hc.has_core_matrix()
        with pytest.raises(ValueError, match="core_matrix"):
            hc.get_core_matrix()

    def test_content_keys_agree_exactly_when_bytes_agree(self):
        graph = _masked_bank_ir(n_instances=3, rows=9, cols=4)
        softs = [neural_core_to_soft_core(n, graph=graph) for n in graph.nodes]
        for sc in softs:
            sc.threshold = 1.0
            sc.latency = 1
        compact_soft_core_mapping(
            softs, [SpikeSource(0, 0), SpikeSource(1, 0), SpikeSource(2, 0)]
        )
        scm = SoftCoreMapping()
        scm.cores = softs
        scm.output_sources = [SpikeSource(0, 0)]
        scm.weight_banks = {0: graph.weight_banks[0].core_matrix}
        hcm = HardCoreMapping([])
        hcm.map_identity(scm)
        keys = [hc.core_matrix_key() for hc in hcm.cores]
        assert keys[0] == keys[1] == keys[2], (
            "same bank + same keeps must dedup under one content key"
        )
        different = HardCore(4, 4)
        different.core_matrix = np.ones((4, 4))
        assert different.core_matrix_key() != keys[0]


class TestConsumersResolveOncePerCore:
    """Hoisting is load-bearing, not stylistic: a padded composite
    materializes per call, so a consumer that resolves inside a per-neuron
    or per-element loop rebuilds the whole grid every iteration."""

    def _counted_mapping(self, monkeypatch):
        graph, bank_mat, _own, _hb = _edge_case_graph()
        softs = _compacted_softcores(graph)
        hcm = HardCoreMapping(
            [HardCore(8, 6, has_bias_capability=True) for _ in range(4)]
        )
        hcm.map(
            _scm_for([softs[0], softs[3]], bank_mat),
            allow_neuron_splitting=False, allow_coalescing=True,
        )
        calls = {"n": 0}
        original = HardCore.get_core_matrix

        def counting(core):
            calls["n"] += 1
            return original(core)

        monkeypatch.setattr(HardCore, "get_core_matrix", counting)
        return hcm, calls

    def test_chip_latency_resolves_each_core_once(self, monkeypatch):
        from mimarsinan.code_generation.cpp_chip_model import SpikeSource as SS
        from mimarsinan.mapping.latency.chip import ChipLatency

        hcm, calls = self._counted_mapping(monkeypatch)
        n_cores = len(hcm.cores)
        ChipLatency(hcm).calculate()
        assert calls["n"] <= n_cores, (
            f"{calls['n']} resolutions for {n_cores} cores in calculate()"
        )

        # The recursive walk reads ONE COLUMN per (core, neuron): without a
        # per-core resolution it rebuilds the padded grid per neuron.
        calls["n"] = 0
        walker = ChipLatency(hcm)
        for ci, core in enumerate(hcm.cores):
            for j in range(core.neurons_per_core):
                walker.get_delay_for(SS(ci, j, False, False))
        n_neurons = sum(c.neurons_per_core for c in hcm.cores)
        assert n_neurons > n_cores  # the vehicle can tell the two apart
        assert calls["n"] <= n_cores, (
            f"{calls['n']} resolutions for {n_cores} cores over {n_neurons} "
            f"neurons: the delay walk is re-materializing per neuron"
        )

    def test_nevresim_export_resolves_each_core_once(self, monkeypatch):
        from mimarsinan.mapping.export.chip_export import hard_cores_to_chip

        hcm, calls = self._counted_mapping(monkeypatch)
        hard_cores_to_chip(
            input_size=8, hardcore_mapping=hcm,
            axons_per_core=8, neurons_per_core=6,
            leak=0.0, weight_type=float,
        )
        assert calls["n"] == len(hcm.cores)


class TestUploadMemoIdentityGuard:
    def test_stale_content_key_never_aliases_a_reallocated_payload(self):
        """``core_matrix_key`` embeds payload ids, so the memo must retain the
        payloads and re-check them — otherwise a freed-then-reallocated array
        could serve another mapping's weights."""
        import torch

        from mimarsinan.chip_simulation.value_run.value_execution import _upload

        hc = HardCore(2, 2)
        hc.core_matrix = np.ones((2, 2), dtype=np.float64)
        memo: dict = {}
        first = _upload(hc, torch.float64, torch.device("cpu"), memo)
        assert _upload(hc, torch.float64, torch.device("cpu"), memo) is first

        stale_key = next(iter(memo))
        memo[stale_key] = ((np.zeros((2, 2)),), torch.zeros(2, 2, dtype=torch.float64))
        again = _upload(hc, torch.float64, torch.device("cpu"), memo)
        assert torch.equal(again, torch.ones(2, 2, dtype=torch.float64)), (
            "a stale memo entry was served instead of the core's own weights"
        )


class TestStorageRegression:
    def test_hybrid_pickle_grows_sublinearly_in_instance_count(self):
        """The regression test for the 33.94 GB defect itself: adding
        instances of a shared bank must never add dense grid payloads."""
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_identity_hybrid_mapping,
        )

        def cost(n: int) -> int:
            return len(pickle.dumps(
                build_identity_hybrid_mapping(ir_graph=_masked_bank_ir(n))
            ))

        payload = _grid_payload_bytes()
        few, many = cost(2), cost(12)
        assert many - few < 10 * payload // 2, (
            f"pickle grew {many - few} bytes across 10 extra instances "
            f"(dense payload is {payload} bytes) — per-core grids are back"
        )

    def test_pickled_mapping_carries_one_payload_and_no_dense_grids(self):
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_identity_hybrid_mapping,
        )

        graph = _masked_bank_ir(6)
        hybrid = build_identity_hybrid_mapping(ir_graph=graph)
        want = {
            hc_idx: hc.get_core_matrix()
            for stage in hybrid.stages if stage.kind == "neural"
            for hc_idx, hc in enumerate(stage.hard_core_mapping.cores)
        }
        loaded = pickle.loads(pickle.dumps(hybrid))
        cores = [
            hc for stage in loaded.stages if stage.kind == "neural"
            for hc in stage.hard_core_mapping.cores
        ]
        assert len(cores) == 6
        assert all(hc.core_matrix is None for hc in cores), (
            "a pickled hard core must never own a dense grid"
        )
        sources = {
            id(p.source) for hc in cores for p in hc.matrix_placements
        }
        assert len(sources) == 1, "every instance must share ONE bank payload"
        keep_rows = {
            id(p.keep_rows) for hc in cores for p in hc.matrix_placements
        }
        assert len(keep_rows) == 1, "same-mask instances share keep arrays"
        for idx, hc in enumerate(cores):
            _assert_bytes_equal(
                hc.get_core_matrix(), want[idx], f"reloaded core {idx}"
            )

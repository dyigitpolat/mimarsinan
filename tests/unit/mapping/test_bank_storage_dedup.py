"""[wsm V3] Bank storage dedup: instances share ONE ndarray so pickles memoize.

Numpy views pickle as region COPIES, so object identity is the dedup
mechanism: full-range slices return the bank array itself, partial slices
come from a per-bank memo, and exact-fit 1:1 packing aliases instead of
copying into a padded grid (measured: a scheduled ViT run materialized
4778 duplicate instance matrices into a 13 GB hybrid_mapping.pickle).
"""

import pickle

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.softcore.hard_core import HardCore


def _bank_and_nodes(n_tokens=3, rows=5, cols=4, row_slice=(0, 4)):
    bank = WeightBank(
        id=0,
        core_matrix=np.arange(rows * cols, dtype=np.float64).reshape(rows, cols),
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * (rows - 1) + i) for i in range(rows - 1)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"tok{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=row_slice,
            perceptron_index=0, perceptron_output_column=tok,
        ))
    graph = IRGraph(
        nodes=nodes,
        output_sources=np.array(
            [IRSource(n.id, j) for n in nodes for j in range(cols)], dtype=object
        ),
        weight_banks={0: bank},
    )
    return bank, nodes, graph


class TestSliceObjectIdentity:
    def test_full_range_slice_is_the_bank_array_itself(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 4))
        mats = [n.get_core_matrix(graph) for n in nodes]
        assert all(m is bank.core_matrix for m in mats)

    def test_partial_slices_are_memoized_per_range(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 2))
        first = nodes[0].get_core_matrix(graph)
        second = nodes[1].get_core_matrix(graph)
        assert first is second  # one view object per (bank, range)
        np.testing.assert_array_equal(first, bank.core_matrix[:, 0:2])

    def test_memo_invalidates_when_bank_matrix_is_replaced(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 2))
        stale = nodes[0].get_core_matrix(graph)
        bank.core_matrix = bank.core_matrix * 2.0  # quantize-style rewrite
        fresh = nodes[0].get_core_matrix(graph)
        assert fresh is not stale
        np.testing.assert_array_equal(fresh, bank.core_matrix[:, 0:2])


class TestExactFitAliasPacking:
    def _softcore(self, graph, node):
        from mimarsinan.mapping.ir.legacy_convert import neural_core_to_soft_core
        return neural_core_to_soft_core(node, graph)

    def test_exact_fit_aliases_the_shared_matrix(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 4))
        soft = self._softcore(graph, nodes[0])
        hard = HardCore(axons_per_core=5, neurons_per_core=4)
        hard.add_softcore(soft)
        assert hard.core_matrix is bank.core_matrix
        assert hard.available_axons == 0 and hard.available_neurons == 0

    def test_non_exact_fit_keeps_the_padded_copy(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 4))
        soft = self._softcore(graph, nodes[0])
        hard = HardCore(axons_per_core=8, neurons_per_core=8)
        hard.add_softcore(soft)
        assert hard.core_matrix is not bank.core_matrix
        assert hard.core_matrix.shape == (8, 8)
        np.testing.assert_array_equal(hard.core_matrix[:5, :4], bank.core_matrix)

    def test_exact_fit_never_accepts_a_second_softcore(self):
        bank, nodes, graph = _bank_and_nodes(row_slice=(0, 4))
        hard = HardCore(axons_per_core=5, neurons_per_core=4)
        hard.add_softcore(self._softcore(graph, nodes[0]))
        with pytest.raises(AssertionError):
            hard.add_softcore(self._softcore(graph, nodes[1]))


class TestScheduledStageDedup:
    def _scheduled_build(self, n_tokens=7, count=2):
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_hybrid_hard_core_mapping,
        )
        from mimarsinan.mapping.platform.mapping_structure import (
            ChipCapabilities,
            MappingStrategy,
        )
        _bank, _nodes, graph = _bank_and_nodes(
            n_tokens=n_tokens, rows=5, cols=4, row_slice=(0, 4)
        )
        strategy = MappingStrategy.resolve(ChipCapabilities(
            allow_scheduling=True, schedule_policy="bank_clustered",
        ))
        # 8x8 pool cores: the (5,4) instances pad into grids (non-exact fit).
        return build_hybrid_hard_core_mapping(
            ir_graph=graph,
            cores_config=[{"max_axons": 8, "max_neurons": 8, "count": count}],
            strategy=strategy,
        )

    def test_head_stage_duplicates_share_one_grid(self):
        hybrid = self._scheduled_build()
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        head_cores = neural[0].hard_core_mapping.cores
        assert len(head_cores) == 2  # full pool, padded grids
        assert head_cores[0].core_matrix is head_cores[1].core_matrix

    def test_resident_stages_alias_the_head_grid(self):
        hybrid = self._scheduled_build()
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        head_matrix = neural[0].hard_core_mapping.cores[0].core_matrix
        assert len(neural) > 1
        for stage in neural[1:]:
            for core in stage.hard_core_mapping.cores:
                assert core.core_matrix is head_matrix

    def test_pass_chain_pickles_one_grid_payload(self):
        hybrid = self._scheduled_build(n_tokens=24, count=2)
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        blob = len(pickle.dumps([s.hard_core_mapping for s in neural]))
        grid_bytes = neural[0].hard_core_mapping.cores[0].core_matrix.nbytes
        n_cores = sum(len(s.hard_core_mapping.cores) for s in neural)
        assert blob < 2 * grid_bytes + n_cores * 8192


class TestPickleDedup:
    def test_duplicate_cores_pickle_the_bank_once(self):
        rows, cols, n = 65, 64, 24
        bank, nodes, graph = _bank_and_nodes(
            n_tokens=n, rows=rows, cols=cols, row_slice=(0, cols)
        )
        from mimarsinan.mapping.ir.legacy_convert import neural_core_to_soft_core
        hards = []
        for node in nodes:
            hard = HardCore(axons_per_core=rows, neurons_per_core=cols)
            hard.add_softcore(neural_core_to_soft_core(node, graph))
            hards.append(hard)
        blob = len(pickle.dumps(hards))
        bank_bytes = bank.core_matrix.nbytes
        # One shared payload + per-core overhead — far below n copies.
        assert blob < 2 * bank_bytes + n * 4096

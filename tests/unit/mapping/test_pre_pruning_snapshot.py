"""Pre-pruning heatmaps must share storage, never per-core copies.

The defect this pins (real vehicle, 2026-08-05): ``_attach_bank_metadata``
materialized ``np.asarray(bank_slice, float32)`` onto every bank-backed core
— 4,925 copies of 25 shared matrices = 45.08 GB of a 45.86 GB pickle, +42 GB
prune RSS, while the actual weights are 57 MB. The contract now: ONE
dtype-preserving snapshot per bank, cores resolve their heatmap lazily
through the graph (the ``get_core_matrix`` pattern); the GUI converts at
render time, so what it displays is bit-identical.
"""

import pickle

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph


def _srcs(pairs):
    return np.array([IRSource(node_id=n, index=i) for n, i in pairs], dtype=object)


def _bank_graph(n_instances=3):
    """One 4x4 bank shared by ``n_instances`` sliced cores (2 cols each)."""
    rng = np.random.default_rng(3)
    bank = WeightBank(id=0, core_matrix=rng.normal(size=(4, 4)).astype(np.float64))
    nodes = []
    for k in range(n_instances):
        lo = (k % 2) * 2
        nodes.append(NeuralCore(
            id=k, name=f"inst{k}",
            input_sources=_srcs([(-2, 0), (-2, 1), (-2, 2), (-3, 0)]),
            core_matrix=None, threshold=1.0, latency=0,
            weight_bank_id=0, weight_row_slice=(lo, lo + 2),
        ))
    out = _srcs([(k, j) for k in range(n_instances) for j in range(2)])
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank}), bank


SEEDS = {0: ([False, True, False, False], [False, True, False, False])}


class TestSnapshotStorageIsShared:
    def test_bank_backed_cores_carry_no_per_core_heatmap(self):
        graph, bank = _bank_graph()
        pre = bank.core_matrix.copy()
        pruned = prune_ir_graph(graph, initial_pruned_per_bank=SEEDS,
                                store_heatmap=True)
        for node in pruned.nodes:
            assert node.pre_pruning_heatmap is None, (
                f"core {node.id}: a bank-backed core must not own heatmap storage"
            )
        snap = pruned.weight_banks[0].pre_pruning_snapshot
        assert snap is not None
        assert snap.dtype == pre.dtype, "snapshot preserves the bank dtype"
        assert np.array_equal(snap, pre), "snapshot is the PRE-compaction bank"

    def test_accessor_resolves_the_slice_bit_exactly(self):
        graph, bank = _bank_graph()
        pre = bank.core_matrix.copy()
        pruned = prune_ir_graph(graph, initial_pruned_per_bank=SEEDS,
                                store_heatmap=True)
        for node in pruned.nodes:
            got = node.resolve_pre_pruning_heatmap(pruned)
            lo, hi = node.weight_row_slice
            want = pre[:, lo:hi]
            assert got is not None and np.array_equal(got, want), node.id
            # what the GUI renders (float conversion at the seam) is unchanged
            assert np.array_equal(
                np.asarray(got, dtype=np.float32),
                np.asarray(want, dtype=np.float32),
            )

    def test_store_heatmap_off_stores_nothing(self):
        graph, _bank = _bank_graph()
        pruned = prune_ir_graph(graph, initial_pruned_per_bank=SEEDS,
                                store_heatmap=False)
        assert pruned.weight_banks[0].pre_pruning_snapshot is None
        for node in pruned.nodes:
            assert node.resolve_pre_pruning_heatmap(pruned) is None

    def test_pickle_stores_one_payload_per_bank(self):
        """The regression test for the 46 GB defect itself: the HEATMAP's
        pickle cost (on minus off) must not scale with the instance count —
        one payload per bank, never one per core."""
        def cost(n, heat):
            g, _ = _bank_graph(n_instances=n)
            return len(pickle.dumps(prune_ir_graph(
                g, initial_pruned_per_bank=SEEDS, store_heatmap=heat)))

        payload = 4 * 4 * 8   # one full bank snapshot, float64
        heat_few = cost(2, True) - cost(2, False)
        heat_many = cost(12, True) - cost(12, False)
        assert heat_few >= payload, "heatmap on must store one bank snapshot"
        assert heat_many - heat_few < payload, (
            f"heatmap cost grew {heat_many - heat_few} bytes across 10 extra "
            f"instances — storage is being copied per core again"
        )

    def test_owned_cores_keep_their_per_core_heatmap(self):
        core = NeuralCore(
            id=0, name="owned",
            input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=np.arange(12, dtype=np.float64).reshape(3, 4),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[core], output_sources=_srcs([(0, 0), (0, 2)]),
                        weight_banks={})
        before = core.core_matrix.copy()
        pruned = prune_ir_graph(graph, store_heatmap=True)
        node = pruned.nodes[0]
        got = node.resolve_pre_pruning_heatmap(pruned)
        assert got is not None
        assert np.array_equal(np.asarray(got, np.float32),
                              np.asarray(before, np.float32))

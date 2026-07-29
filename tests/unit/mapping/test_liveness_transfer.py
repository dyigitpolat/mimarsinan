"""W4b liveness-transfer registry: per-op derivation + cross-op propagation.

Covers the three implemented transfer classes (ELEMENTWISE_1TO1,
INDEX_BIJECTION, REGION_REDUCE), the conservative OPAQUE default, chain
composition through the graph-level index, forward AND backward relaying in
all three propagation arms, and the ``computeop_liveness_transfers``
kill-switch (``identity_only`` reproduces the pre-W4b barrier).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.liveness_transfer import (
    COMPUTEOP_LIVENESS_TRANSFERS_FULL,
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    COMPUTEOP_LIVENESS_TRANSFERS_KEY,
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    TRANSFER_ELEMENTWISE_1TO1,
    TRANSFER_INDEX_BIJECTION,
    TRANSFER_OPAQUE,
    TRANSFER_REGION_REDUCE,
    build_computeop_transfer_index,
    derive_liveness_transfer,
    require_computeop_liveness_transfers,
    resolve_computeop_liveness_transfers,
)


def _srcs(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _op(module, n_in, *, input_shape=None, output_shape=None, op_type=None,
        producer=0, params_extra=None):
    params = {"module": module}
    if input_shape is not None:
        params["input_shape"] = tuple(input_shape)
    if params_extra:
        params.update(params_extra)
    return ComputeOp(
        id=50, name="probe",
        input_sources=_srcs([(producer, j) for j in range(n_in)]),
        op_type=op_type or type(module).__name__,
        params=params,
        input_shape=tuple(input_shape) if input_shape else None,
        output_shape=tuple(output_shape) if output_shape else None,
    )


class _Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class _Unstable(nn.Module):
    """Answers a different permutation on every call: probe must refuse."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls % 2:
            return x
        return x.flip(-1)


class TestDeriveLivenessTransfer:
    @pytest.mark.parametrize("module", [
        nn.ReLU(), nn.LeakyReLU(), nn.GELU(), nn.Identity(), nn.Dropout(),
    ])
    def test_zero_preserving_activation_is_elementwise_1to1(self, module):
        transfer = derive_liveness_transfer(
            _op(module, 4, input_shape=(4,), output_shape=(4,))
        )
        assert transfer.kind == TRANSFER_ELEMENTWISE_1TO1
        assert transfer.out_to_ins[2] == frozenset({2})
        assert transfer.in_to_outs[2] == frozenset({2})

    def test_elementwise_width_mismatch_is_opaque(self):
        transfer = derive_liveness_transfer(
            _op(nn.ReLU(), 4, input_shape=(4,), output_shape=(2,))
        )
        assert transfer.kind == TRANSFER_OPAQUE

    def test_non_zero_preserving_activation_is_opaque(self):
        transfer = derive_liveness_transfer(
            _op(nn.Sigmoid(), 4, input_shape=(4,), output_shape=(4,))
        )
        assert transfer.kind == TRANSFER_OPAQUE

    def test_unknown_module_is_opaque_never_an_error(self):
        class Mystery(nn.Module):
            def forward(self, x):
                return x + 1.0

        transfer = derive_liveness_transfer(
            _op(Mystery(), 4, input_shape=(4,), output_shape=(4,))
        )
        assert transfer.kind == TRANSFER_OPAQUE

    @pytest.mark.parametrize("module", [
        nn.LayerNorm(4), nn.Softmax(dim=-1), nn.BatchNorm1d(4),
    ])
    def test_normalizing_ops_are_opaque(self, module):
        transfer = derive_liveness_transfer(
            _op(module, 4, input_shape=(4,), output_shape=(4,))
        )
        assert transfer.kind == TRANSFER_OPAQUE

    def test_multi_input_join_is_opaque(self):
        op = _op(
            nn.Identity(), 8, output_shape=(4,),
            params_extra={"input_shapes": [(4,), (4,)]},
        )
        assert derive_liveness_transfer(op).kind == TRANSFER_OPAQUE

    def test_declared_identity_without_module_is_elementwise(self):
        op = ComputeOp(
            id=50, name="relay", input_sources=_srcs([(0, 0), (0, 1)]),
            op_type="identity",
        )
        transfer = derive_liveness_transfer(op)
        assert transfer.kind == TRANSFER_ELEMENTWISE_1TO1
        assert transfer.out_to_ins[1] == frozenset({1})

    def test_flatten_derives_identity_bijection(self):
        transfer = derive_liveness_transfer(
            _op(nn.Flatten(), 4, input_shape=(2, 2), output_shape=(4,))
        )
        assert transfer.kind == TRANSFER_INDEX_BIJECTION
        assert transfer.out_to_ins == {j: frozenset({j}) for j in range(4)}

    def test_transpose_derives_true_index_map(self):
        transfer = derive_liveness_transfer(
            _op(_Transpose(), 6, input_shape=(2, 3), output_shape=(3, 2),
                op_type="transpose")
        )
        assert transfer.kind == TRANSFER_INDEX_BIJECTION
        # (2,3) -> transpose -> (3,2): out (r,c) reads in (c,r).
        want = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2, 5: 5}
        assert transfer.out_to_ins == {
            o: frozenset({i}) for o, i in want.items()
        }

    def test_unstable_probe_falls_to_opaque(self):
        transfer = derive_liveness_transfer(
            _op(_Unstable(), 4, input_shape=(4,), output_shape=(4,),
                op_type="permute")
        )
        assert transfer.kind == TRANSFER_OPAQUE

    def test_avgpool2d_regions(self):
        transfer = derive_liveness_transfer(
            _op(nn.AvgPool2d(2), 8, input_shape=(1, 2, 4),
                output_shape=(1, 1, 2))
        )
        assert transfer.kind == TRANSFER_REGION_REDUCE
        assert transfer.out_to_ins[0] == frozenset({0, 1, 4, 5})
        assert transfer.out_to_ins[1] == frozenset({2, 3, 6, 7})
        assert transfer.in_to_outs[5] == frozenset({0})

    def test_maxpool2d_regions_per_channel(self):
        transfer = derive_liveness_transfer(
            _op(nn.MaxPool2d(2), 16, input_shape=(2, 2, 4),
                output_shape=(2, 1, 2))
        )
        assert transfer.kind == TRANSFER_REGION_REDUCE
        # channel 1 plane starts at flat 8; regions never cross channels.
        assert transfer.out_to_ins[2] == frozenset({8, 9, 12, 13})

    def test_strided_pool_leaves_skipped_inputs_uncovered(self):
        transfer = derive_liveness_transfer(
            _op(nn.AvgPool1d(kernel_size=1, stride=2), 4,
                input_shape=(1, 4), output_shape=(1, 2))
        )
        assert transfer.kind == TRANSFER_REGION_REDUCE
        assert transfer.in_to_outs[1] == frozenset()
        assert transfer.in_to_outs[3] == frozenset()

    def test_pool_without_shapes_is_opaque(self):
        op = ComputeOp(
            id=50, name="pool", input_sources=_srcs([(0, j) for j in range(8)]),
            op_type="AvgPool2d", params={"module": nn.AvgPool2d(2)},
        )
        assert derive_liveness_transfer(op).kind == TRANSFER_OPAQUE

    def test_identity_only_policy_relays_identity_ops_only(self):
        pool = _op(nn.AvgPool2d(2), 8, input_shape=(1, 2, 4),
                   output_shape=(1, 1, 2))
        relu = _op(nn.ReLU(), 4, input_shape=(4,), output_shape=(4,))
        identity = ComputeOp(
            id=50, name="relay", input_sources=_srcs([(0, 0), (0, 1)]),
            op_type="identity",
        )
        policy = COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY
        assert derive_liveness_transfer(pool, policy=policy).kind == TRANSFER_OPAQUE
        assert derive_liveness_transfer(relu, policy=policy).kind == TRANSFER_OPAQUE
        assert (
            derive_liveness_transfer(identity, policy=policy).kind
            == TRANSFER_ELEMENTWISE_1TO1
        )


class TestPolicyAxis:
    def test_default_is_full(self):
        assert DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS == COMPUTEOP_LIVENESS_TRANSFERS_FULL
        assert resolve_computeop_liveness_transfers({}) == "full"

    def test_resolve_reads_config_key(self):
        config = {COMPUTEOP_LIVENESS_TRANSFERS_KEY: "identity_only"}
        assert resolve_computeop_liveness_transfers(config) == "identity_only"

    def test_unknown_policy_fails_loud(self):
        with pytest.raises(ValueError, match="computeop_liveness_transfers"):
            require_computeop_liveness_transfers("everything")


# ── graph vehicles ───────────────────────────────────────────────────────────


def _act_chain_graph(act_module, n_mid=4):
    """NC0 -> activation op (n_mid wide) -> NC1 -> 2 logits."""
    rng = np.random.default_rng(7)
    core0 = NeuralCore(
        id=0, name="c0", input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=rng.standard_normal((3, n_mid)), threshold=1.0, latency=0,
    )
    op = ComputeOp(
        id=1, name="act", input_sources=_srcs([(0, j) for j in range(n_mid)]),
        op_type=type(act_module).__name__,
        params={"module": act_module, "input_shape": (n_mid,)},
        input_shape=(n_mid,), output_shape=(n_mid,),
    )
    core1 = NeuralCore(
        id=2, name="c1",
        input_sources=_srcs([(1, j) for j in range(n_mid)] + [(-3, 0)]),
        core_matrix=rng.standard_normal((n_mid + 1, 2)), threshold=1.0,
        latency=1,
    )
    return IRGraph(
        nodes=[core0, op, core1], output_sources=_srcs([(2, 0), (2, 1)]),
    )


def _pool_chain_graph(pool_module=None):
    """NC0 (8 cols = 1x2x4 plane) -> pool 2x2 -> NC1 (2 pool outs + bias)."""
    rng = np.random.default_rng(11)
    core0 = NeuralCore(
        id=0, name="c0", input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=rng.standard_normal((3, 8)), threshold=1.0, latency=0,
    )
    module = pool_module if pool_module is not None else nn.AvgPool2d(2)
    op = ComputeOp(
        id=1, name="pool", input_sources=_srcs([(0, j) for j in range(8)]),
        op_type=type(module).__name__,
        params={"module": module, "input_shape": (1, 2, 4)},
        input_shape=(1, 2, 4), output_shape=(1, 1, 2),
    )
    core1 = NeuralCore(
        id=2, name="c1",
        input_sources=_srcs([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=rng.standard_normal((3, 2)), threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[core0, op, core1], output_sources=_srcs([(2, 0), (2, 1)]),
    )


def _transpose_chain_graph():
    """NC0 (6 cols = 2x3) -> transpose -> NC1 (6 axons + bias)."""
    rng = np.random.default_rng(13)
    core0 = NeuralCore(
        id=0, name="c0", input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=rng.standard_normal((3, 6)), threshold=1.0, latency=0,
    )
    op = ComputeOp(
        id=1, name="tr", input_sources=_srcs([(0, j) for j in range(6)]),
        op_type="transpose",
        params={"module": _Transpose(), "input_shape": (2, 3)},
        input_shape=(2, 3), output_shape=(3, 2),
    )
    core1 = NeuralCore(
        id=2, name="c1",
        input_sources=_srcs([(1, j) for j in range(6)] + [(-3, 0)]),
        core_matrix=rng.standard_normal((7, 2)), threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[core0, op, core1], output_sources=_srcs([(2, 0), (2, 1)]),
    )


def _run(graph, *, seeds_node=None, mode="cascade", policy="full"):
    exempt_rows = {0: frozenset({0, 1})}
    exempt_cols = {graph.nodes[-1].id: frozenset({0, 1})}
    return compute_global_pruned_sets(
        graph,
        zero_threshold=1e-8,
        initial_per_node=seeds_node,
        initial_per_bank=None,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=mode,
        computeop_liveness_transfers=policy,
    )


class TestElementwisePropagation:
    @pytest.mark.parametrize("act", [nn.ReLU(), nn.GELU()])
    def test_forward_through_activation(self, act):
        """Producer col dead => act out is constant act(0)=0 => consumer axon dies."""
        graph = _act_chain_graph(act)
        for mode in ("closure", "cascade"):
            res = _run(graph, seeds_node={0: (set(), {1})}, mode=mode)
            assert 1 in res.pruned_rows_per_node[2], mode
            assert 0 not in res.pruned_rows_per_node[2], mode
        masked = _run(graph, seeds_node={0: (set(), {1})}, mode="masked")
        assert 1 not in masked.pruned_rows_per_node[2]

    @pytest.mark.parametrize("act", [nn.ReLU(), nn.GELU()])
    def test_backward_through_activation(self, act):
        """All consumer rows for act out i dead => producer col i loses its consumer."""
        graph = _act_chain_graph(act)
        for mode in ("closure", "cascade"):
            res = _run(graph, seeds_node={2: ({2}, set())}, mode=mode)
            assert 2 in res.pruned_cols_per_node[0], mode
            assert 1 not in res.pruned_cols_per_node[0], mode
        masked = _run(graph, seeds_node={2: ({2}, set())}, mode="masked")
        assert 2 not in masked.pruned_cols_per_node[0]

    def test_sigmoid_stays_a_barrier_both_directions(self):
        graph = _act_chain_graph(nn.Sigmoid())
        res = _run(graph, seeds_node={0: (set(), {1})})
        assert 1 not in res.pruned_rows_per_node[2]
        res = _run(graph, seeds_node={2: ({2}, set())})
        assert 2 not in res.pruned_cols_per_node[0]


class TestRegionReducePropagation:
    @pytest.mark.parametrize("pool", [nn.AvgPool2d(2), nn.MaxPool2d(2)])
    def test_forward_requires_entire_region_dead(self, pool):
        graph = _pool_chain_graph(pool)
        partial = _run(graph, seeds_node={0: (set(), {0, 1, 4})})
        assert 0 not in partial.pruned_rows_per_node[2], (
            "pool out 0 still receives live input 5 — its axon must survive"
        )
        full = _run(graph, seeds_node={0: (set(), {0, 1, 4, 5})})
        assert 0 in full.pruned_rows_per_node[2]
        assert 1 not in full.pruned_rows_per_node[2]

    def test_backward_requires_all_covering_outputs_dead(self):
        graph = _pool_chain_graph()
        res = _run(graph, seeds_node={2: ({0}, set())})
        assert res.pruned_cols_per_node[0] >= {0, 1, 4, 5}
        assert 2 not in res.pruned_cols_per_node[0]
        assert 6 not in res.pruned_cols_per_node[0]

    def test_closure_couples_through_pool_one_hop(self):
        graph = _pool_chain_graph()
        res = _run(graph, seeds_node={0: (set(), {0, 1, 4, 5})}, mode="closure")
        assert 0 in res.pruned_rows_per_node[2]


class TestIndexBijectionPropagation:
    def test_forward_follows_the_permutation(self):
        graph = _transpose_chain_graph()
        res = _run(graph, seeds_node={0: (set(), {1})})
        # in 1 = (0,1) -> out (1,0) = flat 2: axon 2 dies, axon 1 survives.
        assert 2 in res.pruned_rows_per_node[2]
        assert 1 not in res.pruned_rows_per_node[2]

    def test_backward_follows_the_inverse_permutation(self):
        graph = _transpose_chain_graph()
        res = _run(graph, seeds_node={2: ({2}, set())})
        assert 1 in res.pruned_cols_per_node[0]
        assert 2 not in res.pruned_cols_per_node[0]


class TestChainComposition:
    def test_forward_composes_through_act_then_pool(self):
        """NC0 -> ReLU -> pool -> NC1: the transfer chain composes statically."""
        rng = np.random.default_rng(17)
        core0 = NeuralCore(
            id=0, name="c0", input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=rng.standard_normal((3, 8)), threshold=1.0, latency=0,
        )
        relu = ComputeOp(
            id=1, name="act", input_sources=_srcs([(0, j) for j in range(8)]),
            op_type="ReLU", params={"module": nn.ReLU(), "input_shape": (1, 2, 4)},
            input_shape=(1, 2, 4), output_shape=(1, 2, 4),
        )
        pool = ComputeOp(
            id=2, name="pool", input_sources=_srcs([(1, j) for j in range(8)]),
            op_type="AvgPool2d",
            params={"module": nn.AvgPool2d(2), "input_shape": (1, 2, 4)},
            input_shape=(1, 2, 4), output_shape=(1, 1, 2),
        )
        core1 = NeuralCore(
            id=3, name="c1", input_sources=_srcs([(2, 0), (2, 1), (-3, 0)]),
            core_matrix=rng.standard_normal((3, 2)), threshold=1.0, latency=1,
        )
        graph = IRGraph(
            nodes=[core0, relu, pool, core1],
            output_sources=_srcs([(3, 0), (3, 1)]),
        )
        index = build_computeop_transfer_index(graph)
        assert index.forward_producers[(2, 0)] == frozenset(
            {(0, 0), (0, 1), (0, 4), (0, 5)}
        )
        assert index.effective_consumers[(0, 5)] == frozenset({(3, 0)})
        res = _run(graph, seeds_node={0: (set(), {0, 1, 4, 5})})
        assert 0 in res.pruned_rows_per_node[3]
        # Backward: killing NC1 axon 1 orphans the other pooled plane half.
        res = _run(graph, seeds_node={3: ({1}, set())})
        assert res.pruned_cols_per_node[0] >= {2, 3, 6, 7}


class TestOpaqueGuards:
    def test_opaque_op_reference_still_starvation_guards(self):
        """A neuron consumed only by an OPAQUE op keeps a live consumer."""
        graph = _act_chain_graph(nn.Sigmoid())
        res = _run(graph, seeds_node=None)
        assert res.pruned_cols_per_node[0] == set()

    def test_port_reaching_model_output_through_op_is_protected(self):
        """NC col -> ReLU -> model output: never orphan-killed."""
        rng = np.random.default_rng(19)
        core0 = NeuralCore(
            id=0, name="c0", input_sources=_srcs([(-2, 0), (-3, 0)]),
            core_matrix=rng.standard_normal((2, 2)), threshold=1.0, latency=0,
        )
        op = ComputeOp(
            id=1, name="act", input_sources=_srcs([(0, 0), (0, 1)]),
            op_type="ReLU", params={"module": nn.ReLU(), "input_shape": (2,)},
            input_shape=(2,), output_shape=(2,),
        )
        graph = IRGraph(
            nodes=[core0, op], output_sources=_srcs([(1, 0), (1, 1)]),
        )
        index = build_computeop_transfer_index(graph)
        assert (0, 0) in index.protected_ports
        assert (0, 1) in index.protected_ports
        res = compute_global_pruned_sets(
            graph, zero_threshold=1e-8, initial_per_node=None,
            initial_per_bank=None,
            exempt_rows_per_node={0: frozenset({0})},
            exempt_cols_per_node={},
        )
        assert res.pruned_cols_per_node[0] == set()


class TestKillSwitch:
    def test_identity_only_reproduces_the_pre_w4b_barrier(self):
        graph = _pool_chain_graph()
        seeds = {0: (set(), {0, 1, 4, 5})}
        old = _run(graph, seeds_node=seeds, policy="identity_only")
        assert 0 not in old.pruned_rows_per_node[2]
        # backward likewise blocked: op-referenced neurons blanket-protected.
        old_b = _run(graph, seeds_node={2: ({0}, set())}, policy="identity_only")
        assert old_b.pruned_cols_per_node[0] == set()
        new = _run(graph, seeds_node=seeds, policy="full")
        assert 0 in new.pruned_rows_per_node[2]

    def test_identity_only_still_relays_declared_identity_ops(self):
        graph = _act_chain_graph(nn.ReLU())
        op = graph.nodes[1]
        op.op_type = "identity"
        op.params = {}
        res = _run(graph, seeds_node={0: (set(), {1})}, policy="identity_only")
        assert 1 in res.pruned_rows_per_node[2]

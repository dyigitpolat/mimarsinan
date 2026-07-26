"""[wsm V2] bank-clustered scheduling: weights program once, instances stream."""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.packing.schedule_bank_clustered import (
    try_bank_clustered_passes,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.weight_programming import weight_programming_report


def _banked_tokens(n_tokens, in_features=4, out_features=4, bank_id=0,
                   id_base=0, perceptron_index=0):
    rows = in_features + 1
    rng = np.random.default_rng(7 + bank_id)
    bank = WeightBank(
        id=bank_id,
        core_matrix=rng.normal(size=(rows, out_features)).astype(np.float32),
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * in_features + i) for i in range(in_features)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=id_base + tok, name=f"b{bank_id}_col{tok}", input_sources=srcs,
            core_matrix=None, weight_bank_id=bank_id,
            weight_row_slice=(0, out_features), latency=0,
            perceptron_index=perceptron_index, perceptron_output_column=tok,
            perceptron_output_slice=(0, out_features),
        ))
    return bank, nodes


def _token_graph(n_tokens=7):
    bank, nodes = _banked_tokens(n_tokens)
    out = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(4)], dtype=object
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def _two_bank_graph():
    bank_a, nodes_a = _banked_tokens(4, bank_id=0, id_base=0,
                                     perceptron_index=0)
    bank_b, nodes_b = _banked_tokens(4, bank_id=1, id_base=100,
                                     perceptron_index=1)
    nodes = nodes_a + nodes_b
    out = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(4)], dtype=object
    )
    return IRGraph(nodes=nodes, output_sources=out,
                   weight_banks={0: bank_a, 1: bank_b})


def _sched_strategy(policy):
    return MappingStrategy.resolve(ChipCapabilities(
        allow_scheduling=True, schedule_policy=policy,
    ))


class TestPassComposition:
    def test_chunks_stream_instances(self):
        graph = _token_graph(5)
        chunks = try_bank_clustered_passes(
            cores=list(graph.nodes),
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}],
            weight_banks=graph.weight_banks,
            max_schedule_passes=8,
        )
        assert chunks is not None
        # Load-minimizing: 5 instances under an 8-pass budget need ONE
        # resident core (loads(b)=1), streaming one instance per pass.
        assert [len(c) for c in chunks] == [1, 1, 1, 1, 1]

    def test_owned_core_disqualifies(self):
        graph = _token_graph(3)
        w = np.ones((3, 2), dtype=np.float32)
        graph.nodes.append(NeuralCore(
            id=99, name="own",
            input_sources=np.array(
                [IRSource(-2, 0), IRSource(-2, 1), IRSource(-3, 0)],
                dtype=object),
            core_matrix=w, latency=0,
        ))
        assert try_bank_clustered_passes(
            cores=list(graph.nodes),
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}],
            weight_banks=graph.weight_banks,
            max_schedule_passes=8,
        ) is None

    def test_intra_segment_dependency_disqualifies(self):
        bank, nodes = _banked_tokens(2)
        nodes[1].input_sources = np.array(
            [IRSource(nodes[0].id, i) for i in range(4)] + [IRSource(-3, 0)],
            dtype=object,
        )
        assert try_bank_clustered_passes(
            cores=nodes,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}],
            weight_banks={0: bank},
            max_schedule_passes=8,
        ) is None


class TestScheduledBuild:
    def _build(self, graph, policy, count=2):
        return build_hybrid_hard_core_mapping(
            ir_graph=graph,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": count}],
            strategy=_sched_strategy(policy),
        )

    def test_truthful_pass_indices(self):
        # A 1-core 8x8 pool cannot co-pack two (5,4) instances: the capacity
        # split produces real multi-pass structure with truthful indices.
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=_token_graph(7),
            cores_config=[{"max_axons": 8, "max_neurons": 8, "count": 1}],
            strategy=_sched_strategy("pool"),
        )
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert len(neural) > 1
        assert {s.schedule_segment_index for s in neural} == {0}
        assert [s.schedule_pass_index for s in neural] == list(range(len(neural)))
        assert not any(s.schedule_weights_resident for s in neural)

    def test_bank_clustered_marks_residency(self):
        hybrid = self._build(_token_graph(7), "bank_clustered")
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert len(neural) == 7  # one resident core streams all 7 instances
        assert neural[0].schedule_weights_resident is False
        assert all(s.schedule_weights_resident for s in neural[1:])
        assert [s.schedule_pass_index for s in neural] == list(range(7))

    def test_weight_programming_counts_resident_passes_free(self):
        graph = _token_graph(7)
        pool = weight_programming_report(self._build(graph, "pool"))
        clustered = weight_programming_report(
            self._build(_token_graph(7), "bank_clustered")
        )
        assert pool.params_programmed == 7 * 20
        assert clustered.params_programmed == 20  # one resident core, once
        assert clustered.params_unique == 20
        assert clustered.reuse_factor == pytest.approx(1.0)
        assert clustered.reuse_factor > pool.reuse_factor

    def test_value_equivalence_across_policies(self):
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        graph = _token_graph(7)
        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=_token_graph(7)),
            dtype=torch.float64,
        )
        x = torch.randn(3, 28)
        with torch.no_grad():
            want = identity(x)
            for policy in ("pool", "bank_clustered"):
                got = ValueHybridCoreFlow(
                    self._build(_token_graph(7), policy), dtype=torch.float64
                )(x)
                torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_two_banks_share_the_residency_set(self):
        graph = _two_bank_graph()
        hybrid = self._build(graph, "bank_clustered", count=4)
        report = weight_programming_report(hybrid)
        # Two banks, one resident core each: the ideal — programmed == unique.
        assert report.params_programmed == 2 * 20
        assert report.params_unique == 2 * 20
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert all(s.schedule_weights_resident for s in neural[1:])

    def test_interleaving_pathology_pool_vs_clustered(self):
        # [wsm F3] two banks' instances in one segment: the fresh-pool truth
        # reprograms EVERY instance (W_prog >> unique); bank_clustered
        # collapses it to the unique bytes.
        pool = weight_programming_report(
            self._build(_two_bank_graph(), "pool", count=1)
        )
        clustered = weight_programming_report(
            self._build(_two_bank_graph(), "bank_clustered", count=4)
        )
        assert pool.params_unique == 2 * 20
        assert pool.params_programmed == 8 * 20  # every instance reprograms
        assert clustered.params_programmed == 2 * 20
        assert clustered.reuse_factor == pytest.approx(1.0)

    def test_value_equivalence_across_policies_with_boundary_aq(self):
        # [wsm F5 x mvm AQ] the boundary grid composes with scheduling: entry
        # cores of every pass snap the same input columns as the unscheduled
        # identity program — bit-exact across policies.
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        def armed(n_tokens=7):
            graph = _token_graph(n_tokens)
            for node in graph.nodes:
                node.input_activation_scale = torch.tensor(2.5)
            return graph

        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=armed()),
            dtype=torch.float64, activation_bits=8,
        )
        x = torch.randn(3, 28)
        with torch.no_grad():
            want = identity(x)
            unarmed = ValueHybridCoreFlow(
                build_identity_hybrid_mapping(ir_graph=armed()),
                dtype=torch.float64,
            )(x)
            assert float((want - unarmed).abs().max()) > 0  # grid load-bearing
            for policy in ("pool", "bank_clustered"):
                flow = ValueHybridCoreFlow(
                    self._build(armed(), policy), dtype=torch.float64,
                    activation_bits=8,
                )
                torch.testing.assert_close(
                    flow(x), want, atol=1e-12, rtol=1e-12
                )

    def test_resident_stages_alias_the_head_weights(self):
        # [wsm V3] schedule_weights_resident IS the executor contract: a
        # resident pass reuses the chain head's uploaded tensors (no
        # re-upload), verified by object identity; pool passes never alias.
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
        from mimarsinan.chip_simulation.value_run.value_execution import (
            prepared_segment_cache_for_testing,
        )

        flow = ValueHybridCoreFlow(
            self._build(_token_graph(7), "bank_clustered"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            flow(torch.randn(2, 28))
        neural = [
            s for s in flow.hybrid_mapping.stages if s.kind == "neural"
        ]
        cache = prepared_segment_cache_for_testing()
        prepared = [
            cache[s.hard_core_mapping][("cpu", torch.float64)] for s in neural
        ]
        head = prepared[0]
        assert all(p.weights[0] is head.weights[0] for p in prepared[1:])

        pool_flow = ValueHybridCoreFlow(
            self._build(
                _token_graph(7), "pool", count=1
            ),
            dtype=torch.float64,
        )
        with torch.no_grad():
            pool_flow(torch.randn(2, 28))
        pool_neural = [
            s for s in pool_flow.hybrid_mapping.stages if s.kind == "neural"
        ]
        pool_prepared = [
            cache[s.hard_core_mapping][("cpu", torch.float64)]
            for s in pool_neural
        ]
        assert len(pool_neural) > 1
        assert not any(
            p.weights[0] is pool_prepared[0].weights[0]
            for p in pool_prepared[1:]
        )

    def test_pruned_bank_rows_stay_equivalent(self):
        # [wsm F6] masked (pruned) bank rows: zeros program like any value —
        # the report grid is structural, and the twin stays exact.
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        def pruned(n_tokens=5):
            graph = _token_graph(n_tokens)
            graph.weight_banks[0].core_matrix[1, :] = 0.0
            return graph

        report = weight_programming_report(
            self._build(pruned(), "bank_clustered")
        )
        assert report.params_programmed == 20  # structural, not nnz
        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=pruned()),
            dtype=torch.float64,
        )
        packed = ValueHybridCoreFlow(
            self._build(pruned(), "bank_clustered"), dtype=torch.float64
        )
        x = torch.randn(3, 20)
        with torch.no_grad():
            torch.testing.assert_close(
                packed(x), identity(x), atol=1e-12, rtol=1e-12
            )

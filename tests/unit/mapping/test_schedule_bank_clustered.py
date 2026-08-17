"""[wsm V2/U1] Residency streaming: weights program once, instances stream.

The composition is planned by the ONE pass planner over the builder's own
post-compaction specs; these vehicles pin the allocation law's observable
behaviour (pool-filling expansion, type-awareness, budget floor) and the
builder's materialization (truthful indices, residency marks, programming
counts, value equivalence).
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.packing.hybrid_build_pool import (
    _segment_specs,
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_segment_passes,
)
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget,
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


def _plan(graph, cores_config, max_schedule_passes=8):
    """The planner over the builder's own specs — the composition deployment runs."""
    specs, _, group_ids = _segment_specs(list(graph.nodes), ir_graph=graph)
    hw_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]
    return plan_segment_passes(
        specs,
        effective_core_budget(list(cores_config)),
        core_types=hw_types,
        max_schedule_passes=max_schedule_passes,
        coalescing_group_ids=group_ids,
    )


def _sched_strategy():
    return MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True))


class TestPassComposition:
    def test_chunks_fill_the_pool_for_parallelism(self):
        plan = _plan(
            _token_graph(5),
            [{"max_axons": 32, "max_neurons": 32, "count": 2}],
        )
        assert plan.residency_applied
        # Weight reuse must not idle the chip: the bank duplicates onto BOTH
        # pool cores (one extra programming) so every pass runs 2 instances
        # in parallel — 3 passes, not 5 one-instance passes.
        assert [len(chunk) for chunk in plan.pass_lists] == [2, 2, 1]

    def test_expansion_never_exceeds_instances(self):
        plan = _plan(
            _token_graph(3),
            [{"max_axons": 32, "max_neurons": 32, "count": 8}],
        )
        assert plan.residency_applied
        # Duplicates beyond the instance count would program weights nothing
        # ever runs: 3 instances -> 3 resident cores, one pass.
        assert [len(chunk) for chunk in plan.pass_lists] == [3]

    def test_expansion_is_core_type_aware(self):
        # Instances need 5 axons: only the 8-axon type fits (count 2); the
        # 4-axon type's 6 cores are NOT usable duplicates for this bank.
        plan = _plan(
            _token_graph(6),
            [
                {"max_axons": 8, "max_neurons": 8, "count": 2},
                {"max_axons": 4, "max_neurons": 8, "count": 6},
            ],
        )
        assert plan.residency_applied
        assert [len(chunk) for chunk in plan.pass_lists] == [2, 2, 2]

    def test_infeasible_minimal_residency_falls_back(self):
        # Even the minimal resident set exceeds the pool under the budget:
        # 9 instances / 2 passes -> 5 cores > 4-core pool. The capacity
        # split answers instead.
        plan = _plan(
            _token_graph(9),
            [{"max_axons": 32, "max_neurons": 32, "count": 4}],
            max_schedule_passes=2,
        )
        assert plan.residency_applied is False
        assert plan.feasible

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
        plan = _plan(
            graph, [{"max_axons": 32, "max_neurons": 32, "count": 2}],
        )
        assert plan.residency_applied is False

    def test_intra_segment_dependency_disqualifies(self):
        """The dependent core sits one latency level deeper (the IR mapper's
        longest-path rule), which is exactly what the planner declines on."""
        bank, nodes = _banked_tokens(2)
        nodes[1].input_sources = np.array(
            [IRSource(nodes[0].id, i) for i in range(4)] + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes[1].latency = 1
        graph = IRGraph(
            nodes=nodes,
            output_sources=np.array(
                [IRSource(nodes[1].id, j) for j in range(4)], dtype=object
            ),
            weight_banks={0: bank},
        )
        plan = _plan(
            graph, [{"max_axons": 32, "max_neurons": 32, "count": 2}],
        )
        assert plan.residency_applied is False


class TestScheduledBuild:
    def _build(self, graph, count=2, *, scheduled=True):
        strategy = (
            _sched_strategy() if scheduled
            else MappingStrategy.resolve(ChipCapabilities())
        )
        return build_hybrid_hard_core_mapping(
            ir_graph=graph,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": count}],
            strategy=strategy,
        )

    def test_truthful_pass_indices_on_a_one_core_stream(self):
        # A 1-core 8x8 pool streams seven (5,4) instances one at a time:
        # seven passes, the single resident copy programmed once.
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=_token_graph(7),
            cores_config=[{"max_axons": 8, "max_neurons": 8, "count": 1}],
            strategy=_sched_strategy(),
        )
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert len(neural) == 7
        assert {s.schedule_segment_index for s in neural} == {0}
        assert [s.schedule_pass_index for s in neural] == list(range(7))
        assert neural[0].schedule_weights_resident is False
        assert all(s.schedule_weights_resident for s in neural[1:])

    def test_two_core_stream_marks_residency(self):
        hybrid = self._build(_token_graph(7))
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        # Both pool cores hold the bank; 7 instances stream in 4 passes.
        assert len(neural) == 4
        assert neural[0].schedule_weights_resident is False
        assert all(s.schedule_weights_resident for s in neural[1:])
        assert [s.schedule_pass_index for s in neural] == list(range(4))

    def test_weight_programming_counts_resident_passes_free(self):
        """The reprogramming-minimization headline: the scheduled build
        programs the resident copy set; the unscheduled build programs one
        copy per instance."""
        unscheduled = weight_programming_report(
            self._build(_token_graph(7), scheduled=False)
        )
        streamed = weight_programming_report(self._build(_token_graph(7)))
        assert unscheduled.params_programmed == 7 * 20
        # Two resident duplicates (full pool), each programmed once.
        assert streamed.params_programmed == 2 * 20
        assert streamed.params_unique == 20
        assert streamed.reuse_factor == pytest.approx(0.5)
        assert streamed.reuse_factor > unscheduled.reuse_factor

    def test_value_equivalence_across_compositions(self):
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=_token_graph(7)),
            dtype=torch.float64,
        )
        x = torch.randn(3, 28)
        with torch.no_grad():
            want = identity(x)
            for build in (
                self._build(_token_graph(7)),
                build_hybrid_hard_core_mapping(
                    ir_graph=_token_graph(7),
                    cores_config=[
                        {"max_axons": 8, "max_neurons": 8, "count": 1}
                    ],
                    strategy=_sched_strategy(),
                ),
            ):
                got = ValueHybridCoreFlow(build, dtype=torch.float64)(x)
                torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

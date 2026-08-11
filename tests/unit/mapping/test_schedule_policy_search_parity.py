"""The searched pass structure must be the one deployment will schedule.

`compute_mapping_stats` (the layout answer every search candidate, the wizard
mini-view and the agent introspection surface read) took THREE permission bits
and never learned `schedule_policy` — while the 12 literature IMC presets all
declare `bank_clustered`. So a scheduled platform was searched against the
capacity-split pool composition and deployed against the weight-stationary one:
different pass counts, different sync barriers, different programming.

These pin the divergence closed: on a bank-clustered platform the shape-only
answer equals what `build_hybrid_hard_core_mapping` actually emits, and the pool
composition (which is a DIFFERENT number here) is no longer what search sees.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_neural_core
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)

_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 2}]


def _token_graph(n_tokens: int = 7, in_features: int = 4, out_features: int = 4):
    """One shared bank streamed over ``n_tokens`` spatial positions (the conv shape)."""
    rng = np.random.default_rng(7)
    bank = WeightBank(
        id=0,
        core_matrix=rng.normal(
            size=(in_features + 1, out_features)
        ).astype(np.float32),
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * in_features + i) for i in range(in_features)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"b0_col{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, out_features), latency=0,
            perceptron_index=0, perceptron_output_column=tok,
            perceptron_output_slice=(0, out_features),
        ))
    out = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(out_features)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def _softcores(graph):
    return [
        spec_from_neural_core(
            core, hardware_bias=False, fallback_residency_class_id=-(i + 1),
        )
        for i, core in enumerate(graph.get_neural_cores())
    ]


def _deployed_pass_count(graph, policy: str) -> int:
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph,
        cores_config=_CORES,
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True, schedule_policy=policy)
        ),
    )
    return len([s for s in hybrid.stages if s.kind == "neural"])


def _searched(policy: str, softcores, **kwargs):
    stats, err = compute_mapping_stats(
        softcores=softcores,
        core_types=[LayoutHardCoreType(**ct) for ct in _CORES],
        allow_scheduling=True,
        schedule_policy=policy,
        **kwargs,
    )
    return stats, err


class TestDivergenceIsReal:
    def test_the_two_policies_deploy_different_pass_structures(self):
        """Without this the test below could pass on a coincidence."""
        assert _deployed_pass_count(_token_graph(7), "bank_clustered") == 4
        assert _deployed_pass_count(_token_graph(7), "pool") == 1


class TestSearchMatchesDeployment:
    def test_bank_clustered_search_reports_the_deployed_pass_count(self):
        graph = _token_graph(7)
        deployed = _deployed_pass_count(graph, "bank_clustered")
        stats, err = _searched("bank_clustered", _softcores(graph))
        assert err is None
        assert stats.feasible
        assert stats.schedule_pass_count == deployed == 4

    def test_bank_clustered_search_reports_the_deployed_sync_barriers(self):
        graph = _token_graph(7)
        stats, _ = _searched("bank_clustered", _softcores(graph))
        # One segment, four passes ⇒ three barriers between them.
        assert stats.schedule_sync_count == 3

    def test_the_pool_answer_is_no_longer_what_a_clustered_platform_sees(self):
        graph = _token_graph(7)
        pool_stats, _ = _searched("pool", _softcores(graph))
        clustered_stats, _ = _searched("bank_clustered", _softcores(graph))
        assert pool_stats.schedule_pass_count != clustered_stats.schedule_pass_count

    def test_the_pass_budget_is_honored_like_deployment_honors_it(self):
        """``max_schedule_passes`` is the residency floor; dropping it disarms the policy."""
        graph = _token_graph(7)
        tight, _ = _searched("bank_clustered", _softcores(graph), max_schedule_passes=1)
        # A 1-pass budget needs 7 resident cores; the 2-core pool cannot, so
        # the policy declines and the capacity path answers (as at build time).
        assert tight.schedule_pass_count != 4


class TestUnchangedWhereThePolicyDoesNotApply:
    def test_pool_platforms_keep_their_exact_previous_answer(self):
        graph = _token_graph(7)
        softcores = _softcores(graph)
        stats, err = compute_mapping_stats(
            softcores=softcores,
            core_types=[LayoutHardCoreType(**ct) for ct in _CORES],
            allow_scheduling=True,
        )
        pool_stats, pool_err = _searched("pool", softcores)
        assert (stats, err) == (pool_stats, pool_err)

    def test_owned_weight_cores_fall_back_to_the_capacity_path(self):
        """No bank ⇒ outside the policy's proven class; the pool answer stands."""
        graph = _token_graph(7)
        softcores = [
            sc if i else type(sc)(**{**sc.__dict__, "bank_id": None})
            for i, sc in enumerate(_softcores(graph))
        ]
        clustered, _ = _searched("bank_clustered", softcores)
        pool, _ = _searched("pool", softcores)
        assert clustered == pool

    def test_an_unscheduled_platform_never_composes_passes(self):
        graph = _token_graph(7)
        stats, err = compute_mapping_stats(
            softcores=_softcores(graph),
            core_types=[LayoutHardCoreType(**ct) for ct in _CORES],
            allow_scheduling=False,
            schedule_policy="bank_clustered",
        )
        assert err is None
        assert stats.schedule_pass_count == 0

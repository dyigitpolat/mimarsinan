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

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
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

from .bank_clustered_vehicles import (
    TWO_CORES as _CORES,
    deployed_pass_count as _deployed_pass_count,
    softcores_of as _softcores,
    token_graph as _token_graph,
)


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


class TestOnARealConvolution:
    """The synthetic vehicle above is hand-built; this one goes through the mapper."""

    CORES = [{"max_axons": 32, "max_neurons": 32, "count": 3}]

    def _repr(self):
        import torch
        from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
        from mimarsinan.mapping.mappers.structural import (
            EinopsRearrangeMapper,
            InputMapper,
        )
        from mimarsinan.mapping.model_representation import ModelRepresentation
        from mimarsinan.mapping.support.per_source_scales import (
            compute_per_source_scales,
        )

        torch.manual_seed(3)
        conv = Conv2DPerceptronMapper(
            InputMapper((1, 8, 8)), in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0, bias=True,
            use_batchnorm=False, base_activation_name="Identity",
        )
        model = ModelRepresentation(
            EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
        )
        compute_per_source_scales(model)
        model.assign_perceptron_indices()
        return model

    def _deployed(self, policy: str) -> int:
        from mimarsinan.mapping.ir_mapping_class import IRMapping

        irm = IRMapping(q_max=1, firing_mode="Default", max_axons=32, max_neurons=32)
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=irm.map(self._repr()),
            cores_config=self.CORES,
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True, schedule_policy=policy)
            ),
        )
        return len([s for s in hybrid.stages if s.kind == "neural"])

    def _searched(self, policy: str):
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping

        layout = LayoutIRMapping(max_axons=32, max_neurons=32)
        softcores = layout.collect_layout_softcores(self._repr())
        stats, error = compute_mapping_stats(
            softcores=softcores,
            core_types=[LayoutHardCoreType(**ct) for ct in self.CORES],
            allow_scheduling=True,
            schedule_policy=policy,
        )
        assert error is None
        return stats

    def test_the_shape_only_answer_equals_the_deployed_pass_count(self):
        deployed = self._deployed("bank_clustered")
        assert deployed > 1  # the policy really composes here
        assert self._searched("bank_clustered").schedule_pass_count == deployed

    def test_the_pool_answer_now_agrees_too(self):
        """[C4] The policy decides WHICH schedule is composed, never whether one
        exists: a scheduled pool platform reports the passes it will run."""
        assert self._deployed("pool") != self._deployed("bank_clustered")
        assert self._searched("pool").schedule_pass_count == self._deployed("pool")


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

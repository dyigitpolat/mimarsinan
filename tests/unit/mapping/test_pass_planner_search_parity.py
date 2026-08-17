"""The searched pass structure must be the one deployment will schedule.

`compute_mapping_stats` (the layout answer every search candidate, the wizard
mini-view and the agent introspection surface read) and the hard-core builder
consume the SAME planner [U1], so agreement holds by construction — these pins
keep the ESTIMAND honest on vehicles the synthetic fixtures cannot reach: a
real convolution through the mappers, an unbanked segment (capacity
composition), a tight pass budget, and the unscheduled platform.
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
    softcores_of as _softcores,
    token_graph as _token_graph,
)


def _searched(softcores, **kwargs):
    stats, err = compute_mapping_stats(
        softcores=softcores,
        core_types=[LayoutHardCoreType(**ct) for ct in _CORES],
        allow_scheduling=True,
        **kwargs,
    )
    return stats, err


class TestSearchMatchesDeployment:
    def test_the_search_reports_the_streamed_composition(self):
        graph = _token_graph(7)
        stats, err = _searched(_softcores(graph))
        assert err is None
        assert stats.feasible
        assert stats.schedule_pass_count == 4
        # One segment, four passes => three barriers between them.
        assert stats.schedule_sync_count == 3

    def test_an_unbanked_segment_takes_the_capacity_composition(self):
        """No bank => outside the streamed class; everything fits in one pass
        here, so the capacity composition is the single pass the chip runs."""
        graph = _token_graph(7)
        softcores = [
            sc if i else type(sc)(**{**sc.__dict__, "bank_id": None})
            for i, sc in enumerate(_softcores(graph))
        ]
        stats, err = _searched(softcores)
        assert err is None
        assert stats.schedule_pass_count == 1

    def test_the_pass_budget_is_honored_like_deployment_honors_it(self):
        """A 1-pass budget needs 7 resident cores; the 2-core pool cannot, so
        the streamed composition declines and the capacity split answers —
        one fitting pass, exactly what the builder emits under this budget."""
        graph = _token_graph(7)
        tight, err = _searched(_softcores(graph), max_schedule_passes=1)
        assert err is None
        assert tight.schedule_pass_count == 1


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

    def _deployed(self) -> int:
        from mimarsinan.mapping.ir_mapping_class import IRMapping

        irm = IRMapping(q_max=1, firing_mode="Default", max_axons=32, max_neurons=32)
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=irm.map(self._repr()),
            cores_config=self.CORES,
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        return len([s for s in hybrid.stages if s.kind == "neural"])

    def _searched_conv(self):
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping

        layout = LayoutIRMapping(max_axons=32, max_neurons=32)
        softcores = layout.collect_layout_softcores(self._repr())
        stats, error = compute_mapping_stats(
            softcores=softcores,
            core_types=[LayoutHardCoreType(**ct) for ct in self.CORES],
            allow_scheduling=True,
        )
        assert error is None
        return stats

    def test_the_shape_only_answer_equals_the_deployed_pass_count(self):
        deployed = self._deployed()
        assert deployed > 1  # the composition really schedules here
        assert self._searched_conv().schedule_pass_count == deployed


class TestUnscheduledPlatformsNeverCompose:
    def test_an_unscheduled_platform_reports_the_flat_pack(self):
        graph = _token_graph(7)
        stats, err = compute_mapping_stats(
            softcores=_softcores(graph),
            core_types=[LayoutHardCoreType(**ct) for ct in _CORES],
            allow_scheduling=False,
        )
        assert err is None
        assert stats.schedule_pass_count == 0

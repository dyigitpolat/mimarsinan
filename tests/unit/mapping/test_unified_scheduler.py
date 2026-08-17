"""[U1] One scheduler: residency-first, capacity fallback, no policy knob.

The owner directive that retired the ``schedule_policy`` enum: weight
programming is the PRIMARY objective, and other scheduling remains available
where residency does not apply. The token vehicle states the whole argument in
two numbers — a fitting flat pack programs SEVEN bank copies in one pass while
the residency composition programs TWO copies over four passes — so the
chooser adopts residency whenever the law composes it, never gating on pass
count. Pass inflation is bounded by ``max_schedule_passes`` (the law's
feasibility floor) and its sync cost is priced, so search sees the trade.
"""

from __future__ import annotations

import inspect

from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.support.schedule import pass_planner
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_program_passes,
    plan_segment_passes,
)
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget,
)

from .bank_clustered_vehicles import (
    TWO_CORES,
    hard_core_types,
    multi_segment_graph,
    softcores_of,
    token_graph,
    two_layer_dependency_graph,
)

DEPENDENCY_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 3}]


def _segment_plan(graph, cores=TWO_CORES, **kwargs):
    return plan_segment_passes(
        softcores_of(graph),
        effective_core_budget(cores),
        core_types=hard_core_types(cores),
        allow_coalescing=False,
        allow_splitting=False,
        max_schedule_passes=kwargs.pop("max_schedule_passes", 8),
        **kwargs,
    )


class TestReprogrammingFirstChooser:
    def test_residency_composes_even_over_a_fitting_flat_pack(self):
        """The pin that retires pass-count dominance: the flat pack FITS
        (one pass), yet the scheduler takes the residency composition,
        because one pass here means seven programmed bank copies."""
        plan = _segment_plan(token_graph(7))
        assert plan.feasible
        assert plan.residency_applied
        assert plan.pass_count == 4

    def test_the_programmed_copy_set_is_smaller_than_the_instance_count(self):
        """Pass 0 IS the programmed set; every later pass streams over it."""
        plan = _segment_plan(token_graph(7))
        placed = sum(len(chunk) for chunk in plan.pass_lists)
        assert placed == 7
        assert len(plan.pass_lists[0]) == 2

    def test_resident_flags_credit_every_pass_after_the_first(self):
        plan = _segment_plan(token_graph(7))
        assert plan.resident_flags == (False, True, True, True)


class TestCapacityFallback:
    def test_a_dependent_segment_falls_back_to_the_capacity_split(self):
        """Two latency levels in one segment: outside the residency class."""
        plan = plan_segment_passes(
            softcores_of(two_layer_dependency_graph(3)),
            effective_core_budget(DEPENDENCY_CORES),
            core_types=hard_core_types(DEPENDENCY_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        assert plan.feasible
        assert not plan.residency_applied
        assert plan.resident_flags == tuple(False for _ in plan.pass_lists)

    def test_unbanked_softcores_fall_back(self):
        from dataclasses import replace

        specs = [
            replace(spec, bank_id=None)
            for spec in softcores_of(token_graph(7))
        ]
        plan = plan_segment_passes(
            specs,
            effective_core_budget(TWO_CORES),
            core_types=hard_core_types(TWO_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        assert plan.feasible
        assert not plan.residency_applied


class TestProgramPlan:
    def test_segments_plan_independently_and_in_order(self):
        plan = plan_program_passes(
            softcores_of(multi_segment_graph(6)),
            hard_core_types(TWO_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        assert plan.feasible
        assert plan.per_segment_passes == {sid: 1 for sid in range(6)}
        assert plan.total_pass_count == 6

    def test_sync_barriers_count_only_within_segments(self):
        """Six one-pass segments need no barrier; one four-pass segment
        needs three."""
        six = plan_program_passes(
            softcores_of(multi_segment_graph(6)),
            hard_core_types(TWO_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        streamed = plan_program_passes(
            softcores_of(token_graph(7)),
            hard_core_types(TWO_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        assert six.sync_count == 0
        assert streamed.sync_count == 3

    def test_flat_resident_flags_align_with_the_flat_pass_lists(self):
        plan = plan_program_passes(
            softcores_of(token_graph(7)),
            hard_core_types(TWO_CORES),
            allow_coalescing=False,
            allow_splitting=False,
            max_schedule_passes=8,
        )
        assert len(plan.resident_flags) == len(plan.pass_lists)
        assert plan.resident_flags == (False, True, True, True)


class TestThePlannerHasNoPolicyKnob:
    def test_no_schedule_policy_parameter_survives(self):
        for fn in (plan_segment_passes, plan_program_passes):
            assert "schedule_policy" not in inspect.signature(fn).parameters

    def test_the_enum_constant_is_gone(self):
        assert not hasattr(pass_planner, "BANK_CLUSTERED")


class TestTheBuilderRunsThePlannersProgram:
    """The twin-wrapper consolidation: the deployed pass structure IS the
    planner's — one planner, two materializers, no re-decision."""

    def _deployed_neural_stages(self, graph, cores):
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=graph,
            cores_config=[dict(ct) for ct in cores],
            strategy=MappingStrategy.resolve(
                ChipCapabilities(allow_scheduling=True)
            ),
        )
        return [stage for stage in hybrid.stages if stage.kind == "neural"]

    def test_the_bank_vehicle_deploys_the_streamed_composition(self):
        graph = token_graph(7)
        stages = self._deployed_neural_stages(graph, TWO_CORES)
        plan = plan_program_passes(
            softcores_of(graph), hard_core_types(TWO_CORES),
            allow_coalescing=False, allow_splitting=False,
            max_schedule_passes=8,
        )
        assert len(stages) == plan.total_pass_count == 4
        deployed_resident = [
            bool(getattr(stage, "schedule_weights_resident", False))
            for stage in stages
        ]
        assert deployed_resident == list(plan.resident_flags)

    def test_the_dependent_vehicle_deploys_the_capacity_composition(self):
        graph = two_layer_dependency_graph(3)
        stages = self._deployed_neural_stages(graph, DEPENDENCY_CORES)
        plan = plan_program_passes(
            softcores_of(graph), hard_core_types(DEPENDENCY_CORES),
            allow_coalescing=False, allow_splitting=False,
            max_schedule_passes=8,
        )
        assert plan.feasible
        assert len(stages) == plan.total_pass_count
        assert not any(
            getattr(stage, "schedule_weights_resident", False)
            for stage in stages
        )

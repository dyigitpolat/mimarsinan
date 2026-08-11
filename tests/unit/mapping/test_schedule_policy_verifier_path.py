"""The OTHER half of the layout answer: ``verify_hardware_config``.

``compute_mapping_stats`` is what a search candidate reads; ``verify_hardware_config``
is what the GUI ``/api/hw_config_verify`` route, the wizard snapshot
(``verify_planned_mapping_performance`` via ``build_layout_plan``) and every
capacity gate read. Both must compose the passes the hard-core builder composes,
so these pin the SAME behaviour change on the verifier path — a bank-clustered
platform yields the deployment pass structure even when the flat pack fits, and
the pool platform keeps reporting exactly what it reported before.

The memo in front of the verifier is keyed on the WHOLE declaration for the same
reason: a key that forgot the policy would serve one platform's program as
another's on the second question of a run.
"""

from __future__ import annotations

from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.verifier import verify_hardware_config
from mimarsinan.mapping.verification.verifier.mapping_verifier_hw import (
    clear_verify_hardware_config_memo,
)

from .bank_clustered_vehicles import (
    TWO_CORES,
    deployed_pass_count,
    softcores_of,
    token_graph,
)


def _verify(policy: str, softcores, **overrides):
    clear_verify_hardware_config_memo()
    caps = ChipCapabilities(
        allow_scheduling=True, schedule_policy=policy, **overrides
    )
    return verify_hardware_config(softcores, TWO_CORES, **caps.layout_kwargs())


class TestTheFlatPackFits:
    """Everything below is about a platform whose UNSCHEDULED pack already fits."""

    def test_the_pool_platform_reports_a_plain_feasible_pack(self):
        result = _verify("pool", softcores_of(token_graph(7)))
        assert result["feasible"] is True
        assert "schedule_info" not in result
        assert result["stats"]["schedule_pass_count"] == 0


class TestBankClusteredComposesOverAFittingPack:
    def test_the_verifier_reports_the_deployed_pass_structure(self):
        graph = token_graph(7)
        deployed = deployed_pass_count(graph, "bank_clustered")
        result = _verify("bank_clustered", softcores_of(graph))
        assert result["feasible"] is True
        # The pool platform above answers 0 here; this is the divergence closed.
        assert result["stats"]["schedule_pass_count"] == deployed == 4
        assert result["stats"]["schedule_sync_count"] == 3
        assert result["schedule_info"]["total_passes"] == deployed
        assert result["schedule_info"]["per_segment_passes"] == {0: deployed}

    def test_the_wizard_mini_view_plan_carries_the_same_structure(self):
        """``build_layout_plan`` is the wizard/snapshot front of the same call."""
        graph = token_graph(7)
        softcores = softcores_of(graph)

        class _Verification:
            def __init__(self, specs):
                self.softcores = specs
                self.layout_preview = None
                self.host_side_segment_count = 0

        clear_verify_hardware_config_memo()
        plan = build_layout_plan(
            _Verification(softcores), TWO_CORES,
            **ChipCapabilities(
                allow_scheduling=True, schedule_policy="bank_clustered",
            ).layout_kwargs(),
        )
        assert plan.feasible
        assert plan.stats.schedule_pass_count == deployed_pass_count(
            graph, "bank_clustered"
        )
        assert (plan.schedule_info or {}).get("total_passes") == 4

    def test_the_budget_is_part_of_the_answer(self):
        """A 1-pass budget needs 7 resident cores; the 2-core pool declines."""
        softcores = softcores_of(token_graph(7))
        tight = _verify("bank_clustered", softcores, max_schedule_passes=1)
        assert tight["stats"]["schedule_pass_count"] == 0
        assert "schedule_info" not in tight


class TestTheMemoIsKeyedOnTheWholeDeclaration:
    """A memo that forgot the scheduler would answer one platform with another's."""

    def test_a_pool_answer_is_never_replayed_for_a_clustered_platform(self):
        softcores = softcores_of(token_graph(7))
        clear_verify_hardware_config_memo()
        kwargs = dict(softcores=softcores, core_types=TWO_CORES)
        pool = verify_hardware_config(
            **kwargs, **ChipCapabilities(
                allow_scheduling=True, schedule_policy="pool").layout_kwargs()
        )
        clustered = verify_hardware_config(
            **kwargs, **ChipCapabilities(
                allow_scheduling=True, schedule_policy="bank_clustered").layout_kwargs()
        )
        assert pool["stats"]["schedule_pass_count"] == 0
        assert clustered["stats"]["schedule_pass_count"] == 4

    def test_a_narrower_budget_is_never_replayed_from_a_wider_one(self):
        softcores = softcores_of(token_graph(7))
        clear_verify_hardware_config_memo()
        kwargs = dict(softcores=softcores, core_types=TWO_CORES)
        wide = verify_hardware_config(
            **kwargs, **ChipCapabilities(
                allow_scheduling=True, schedule_policy="bank_clustered",
                max_schedule_passes=8).layout_kwargs()
        )
        tight = verify_hardware_config(
            **kwargs, **ChipCapabilities(
                allow_scheduling=True, schedule_policy="bank_clustered",
                max_schedule_passes=1).layout_kwargs()
        )
        assert wide["stats"]["schedule_pass_count"] == 4
        assert tight["stats"]["schedule_pass_count"] == 0

    def test_a_repeat_of_the_same_question_still_answers_identically(self):
        """The memo must be a cache, not a mutation: same key ⇒ same answer."""
        softcores = softcores_of(token_graph(7))
        clear_verify_hardware_config_memo()
        first = _verify("bank_clustered", softcores)
        second = verify_hardware_config(
            softcores, TWO_CORES,
            **ChipCapabilities(
                allow_scheduling=True, schedule_policy="bank_clustered").layout_kwargs(),
        )
        assert first["stats"] == second["stats"]
        assert first["schedule_info"]["total_passes"] == (
            second["schedule_info"]["total_passes"]
        )

"""The POOL search/deploy pass-count divergence, CLOSED — and pinned closed.

W5.2 closed the divergence for ``bank_clustered`` only, leaving a measured gap on a
fitting POOL platform: ``compute_mapping_stats`` short-circuited to the unscheduled
census (``schedule_pass_count == 0``) while ``allow_scheduling`` always routes the
builder through the scheduled build, which emits one pass PER SEGMENT. Searched 0 vs
deployed 6 — stated as a known limitation in ``docs/deployment_record_schema.md`` §9.4.

C4 closes it: with ``allow_scheduling`` on, the deployment builds a scheduled program,
so the searched census is the SCHEDULED one whatever the policy. The pass count is
now the structural equality the fidelity contract gates on, so this file asserts
agreement rather than measuring a gap.
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)

from .bank_clustered_vehicles import (
    deployed_pass_count,
    hard_core_types,
    multi_segment_graph,
    softcores_of,
)

# Eight cores for six one-core segments: the flat pack fits comfortably.
ROOMY_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 8}]
TIGHT_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 1}]
SEGMENTS = 6


def _searched(cores, *, allow_scheduling=True, policy="pool"):
    graph = multi_segment_graph(SEGMENTS)
    stats, error = compute_mapping_stats(
        softcores=softcores_of(graph), core_types=hard_core_types(cores),
        allow_scheduling=allow_scheduling, schedule_policy=policy,
    )
    assert error is None
    assert stats.feasible
    return stats


class TestAFittingPoolPlatformReportsThePassesItWillRun:
    def test_the_builder_emits_one_pass_per_segment(self):
        graph = multi_segment_graph(SEGMENTS)
        assert deployed_pass_count(graph, "pool", ROOMY_CORES) == SEGMENTS

    def test_the_searched_answer_agrees_with_the_deployed_one(self):
        """The structural equality the fidelity contract gates on."""
        graph = multi_segment_graph(SEGMENTS)
        assert _searched(ROOMY_CORES).schedule_pass_count == deployed_pass_count(
            graph, "pool", ROOMY_CORES
        )

    def test_a_tight_pool_agrees_too(self):
        """The case that already agreed must keep agreeing."""
        graph = multi_segment_graph(SEGMENTS)
        assert _searched(TIGHT_CORES).schedule_pass_count == deployed_pass_count(
            graph, "pool", TIGHT_CORES
        )

    def test_one_pass_per_segment_still_means_no_intra_segment_barrier(self):
        assert _searched(ROOMY_CORES).schedule_sync_count == 0


class TestSchedulingIsWhatDecidesNotThePolicy:
    @pytest.mark.parametrize("policy", ["pool", "bank_clustered"])
    def test_every_policy_reports_the_scheduled_census(self, policy):
        graph = multi_segment_graph(SEGMENTS)
        assert _searched(ROOMY_CORES, policy=policy).schedule_pass_count == (
            deployed_pass_count(graph, policy, ROOMY_CORES)
        )

    def test_an_unscheduled_platform_still_reports_the_flat_pack(self):
        """``allow_scheduling`` off means the builder runs no schedule at all, so
        the unscheduled census remains the honest answer."""
        assert _searched(
            ROOMY_CORES, allow_scheduling=False
        ).schedule_pass_count == 0

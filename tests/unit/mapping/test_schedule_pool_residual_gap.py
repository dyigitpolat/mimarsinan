"""The residual POOL divergence, measured — so the doc cannot outrun the code.

W5.2 closed the search/deploy divergence for ``bank_clustered``. It did NOT close
it for ``pool``: when the flat pack fits, ``compute_mapping_stats`` returns the
unscheduled census (``schedule_pass_count == 0``) while ``allow_scheduling`` always
routes the builder through the scheduled build, which emits one pass PER SEGMENT.
The two numbers therefore disagree by exactly the segment count on a fitting
pool platform.

That gap is stated in ``docs/deployment_record_schema.md`` §9 with these numbers;
this test is what keeps the statement true. If the gap is ever closed, this test
fails and the doc entry must go with it — a silent change to either is the thing
being prevented.
"""

from __future__ import annotations

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
SEGMENTS = 6


class TestAFittingPoolPlatformStillUnderReportsItsPasses:
    def test_the_builder_emits_one_pass_per_segment(self):
        graph = multi_segment_graph(SEGMENTS)
        assert deployed_pass_count(graph, "pool", ROOMY_CORES) == SEGMENTS

    def test_the_searched_answer_reports_none_of_them(self):
        graph = multi_segment_graph(SEGMENTS)
        stats, error = compute_mapping_stats(
            softcores=softcores_of(graph), core_types=hard_core_types(ROOMY_CORES),
            allow_scheduling=True, schedule_policy="pool",
        )
        assert error is None
        assert stats.feasible
        # The measured gap the doc states: searched 0 vs deployed 6.
        assert stats.schedule_pass_count == 0
        assert deployed_pass_count(graph, "pool", ROOMY_CORES) == SEGMENTS

    def test_the_barrier_count_is_the_half_that_does_agree(self):
        """One pass per segment means zero INTRA-segment barriers, so this is exact."""
        graph = multi_segment_graph(SEGMENTS)
        stats, _ = compute_mapping_stats(
            softcores=softcores_of(graph), core_types=hard_core_types(ROOMY_CORES),
            allow_scheduling=True, schedule_policy="pool",
        )
        assert stats.schedule_sync_count == 0

    def test_the_gap_is_the_flat_pack_shortcut_not_the_policy(self):
        """Shrink the pool until the pack fails and the two agree again."""
        tight = [{"max_axons": 32, "max_neurons": 32, "count": 1}]
        graph = multi_segment_graph(SEGMENTS)
        stats, error = compute_mapping_stats(
            softcores=softcores_of(graph), core_types=hard_core_types(tight),
            allow_scheduling=True, schedule_policy="pool",
        )
        assert error is None
        assert stats.schedule_pass_count == SEGMENTS
        assert deployed_pass_count(graph, "pool", tight) == SEGMENTS

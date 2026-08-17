"""The three guards that keep the composed answer a program the chip can run.

The residency composition is proven on ONE class of segment; each guard is a
boundary of that class, and each is behaviour-changing (every assertion here
fails when its guard is removed):

1. **Intra-segment dependency.** A segment whose cores feed each other is
   outside the streamed regime, so the planner must decline it — otherwise
   search and deployment are handed a weight-stationary chain that reorders a
   real dataflow dependency.
2. **Per-chunk packability.** A composition whose passes cannot be packed is
   not a program at all; it is declined in favour of the capacity split
   (the greedy packer places the tightest fit first, so a composition can be
   packable in principle and still be refused — the reported answer must be one
   the packer actually reaches).
3. **Per-ordinal stability.** A resident physical core carries ONE bank for the
   whole chain. A composition that would move a bank between ordinals is not the
   weight-stationary regime at all, and deployment's ``mark_bank_residency``
   raises on exactly that geometry — so the planner must decline it here.
"""

from __future__ import annotations

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.bank_clustered_law import (
    BankInstance,
    compose_bank_clustered_passes,
)
from mimarsinan.mapping.support.schedule.pass_planner import (
    bank_clustered_layout_passes,
    plan_segment_passes,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)

from .bank_clustered_vehicles import (
    deployed_pass_count,
    hard_core_types,
    softcores_of,
    two_layer_dependency_graph,
)

DEPENDENCY_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 3}]


def _specs_of(instances, names=None):
    """Layout specs carrying the same bank/extent facts as ``instances``."""
    return [
        LayoutSoftCoreSpec(
            input_count=inst.axons, output_count=inst.neurons,
            bank_id=inst.bank_id, latency_tag=0, segment_id=0,
            residency_class_id=index,
            name=(names[index] if names else f"i{index}"),
        )
        for index, inst in enumerate(instances)
    ]


class TestIntraSegmentDependency:
    """Layer 1 consumes layer 0 inside one segment: outside the proven class."""

    def test_the_composition_declines_a_segment_whose_cores_feed_each_other(self):
        softcores = softcores_of(two_layer_dependency_graph(3))
        assert {sc.latency_tag for sc in softcores} == {0, 1}
        assert bank_clustered_layout_passes(
            softcores, hard_core_types(DEPENDENCY_CORES), max_schedule_passes=8,
        ) is None

    def test_the_capacity_path_answers_and_says_so(self):
        softcores = softcores_of(two_layer_dependency_graph(3))
        plan = plan_segment_passes(
            softcores, 3, core_types=hard_core_types(DEPENDENCY_CORES),
            max_schedule_passes=8,
        )
        assert plan.residency_applied is False

    def test_the_searched_answer_is_the_one_the_builder_deploys(self):
        """Dropping the guard reports 3 passes / 2 barriers for a 1-stage program."""
        graph = two_layer_dependency_graph(3)
        softcores = softcores_of(graph)
        deployed = deployed_pass_count(graph, DEPENDENCY_CORES)
        assert deployed == 1

        stats, error = compute_mapping_stats(
            softcores=softcores,
            core_types=hard_core_types(DEPENDENCY_CORES),
            allow_scheduling=True,
        )
        assert error is None
        assert stats.feasible
        # [C4] A scheduled platform reports the ONE pass this single-stage
        # program runs, not the zero the flat pack would suggest.
        assert stats.schedule_pass_count == deployed
        assert stats.schedule_sync_count == 0


# Two core types, three bank-backed instances: the law composes ``[[a, c], [b]]``,
# and the greedy packer -- which places the tightest fit first -- puts ``c`` (3
# axons x 4 neurons) on the 4x4 core and then has nowhere for ``a`` (4x3).
UNPACKABLE_CORES = [
    {"max_axons": 3, "max_neurons": 8, "count": 1},
    {"max_axons": 4, "max_neurons": 4, "count": 1},
]
UNPACKABLE_SPECS = _specs_of(
    [
        BankInstance(bank_id=2, axons=4, neurons=3),
        BankInstance(bank_id=2, axons=4, neurons=2),
        BankInstance(bank_id=1, axons=3, neurons=4),
    ],
    names=["a", "b", "c"],
)


class TestAnUnpackableCompositionIsDeclined:
    def test_the_law_alone_would_propose_a_pass_that_cannot_be_packed(self):
        """Without this the test below could pass because nothing composes."""
        types = hard_core_types(UNPACKABLE_CORES)
        chunks = bank_clustered_layout_passes(
            UNPACKABLE_SPECS, types, max_schedule_passes=8,
        )
        assert chunks is not None
        assert [[sc.name for sc in chunk] for chunk in chunks] == [["a", "c"], ["b"]]
        assert not pack_layout(
            softcores=chunks[0], core_types=types,
            allow_neuron_splitting=False, allow_coalescing=False,
        ).feasible

    def test_every_pass_the_planner_returns_can_actually_be_packed(self):
        types = hard_core_types(UNPACKABLE_CORES)
        plan = plan_segment_passes(
            UNPACKABLE_SPECS, 2, core_types=types, max_schedule_passes=8,
        )
        assert plan.feasible
        for pass_list in plan.pass_lists:
            assert pack_layout(
                softcores=list(pass_list), core_types=types,
                allow_neuron_splitting=False, allow_coalescing=False,
            ).feasible, [sc.name for sc in pass_list]

    def test_it_falls_back_to_the_capacity_split_rather_than_reporting_it(self):
        types = hard_core_types(UNPACKABLE_CORES)
        plan = plan_segment_passes(
            UNPACKABLE_SPECS, 2, core_types=types, max_schedule_passes=8,
        )
        assert plan.residency_applied is False
        assert (
            plan.pass_count,
            [[sc.name for sc in chunk] for chunk in plan.pass_lists],
        ) == (3, [["a"], ["b"], ["c"]])


THREE_CORES = [{"max_axons": 8, "max_neurons": 8, "count": 3}]

# Four bank-0 instances and one bank-1 instance over three cores at a 3-pass
# budget: the floor grants bank 0 two cores and bank 1 one, so the emission is
# [b0, b1, b0] then [b0, b0] -- ordinal 1 would carry bank 1 in pass 0 and bank 0
# in pass 1, i.e. the "resident" weights would reprogram.
UNSTABLE_INSTANCES = [
    BankInstance(bank_id=0, axons=8, neurons=8),
    BankInstance(bank_id=1, axons=8, neurons=4),
    BankInstance(bank_id=0, axons=8, neurons=8),
    BankInstance(bank_id=0, axons=4, neurons=4),
    BankInstance(bank_id=0, axons=8, neurons=4),
]


class TestPerOrdinalStability:
    """A resident physical core must keep ONE bank for the whole pass chain."""

    def test_a_composition_that_would_move_a_bank_off_an_ordinal_is_refused(self):
        assert compose_bank_clustered_passes(
            UNSTABLE_INSTANCES, THREE_CORES, max_schedule_passes=3,
        ) is None

    def test_the_planner_declines_it_and_the_capacity_path_answers(self):
        specs = _specs_of(UNSTABLE_INSTANCES)
        types = hard_core_types(THREE_CORES)
        assert bank_clustered_layout_passes(
            specs, types, max_schedule_passes=3,
        ) is None
        plan = plan_segment_passes(
            specs, 3, core_types=types, max_schedule_passes=3,
        )
        assert plan.residency_applied is False
        assert plan.pass_count == 2

    def test_a_ragged_tail_that_only_SHRINKS_is_still_accepted(self):
        """The guard bans moving banks, not ragged tails — three of one bank,
        one of another: pass 1 is a prefix of pass 0's layout, so it stands."""
        instances = [BankInstance(bank_id=0, axons=4, neurons=4)] * 3
        instances.append(BankInstance(bank_id=1, axons=4, neurons=4))
        chunks = compose_bank_clustered_passes(
            instances, THREE_CORES, max_schedule_passes=8,
        )
        assert chunks is not None
        reference = [instances[i].bank_id for i in chunks[0]]
        assert reference == [0, 1, 0]
        for chunk in chunks[1:]:
            assert [instances[i].bank_id for i in chunk] == reference[:len(chunk)]

    def test_equal_banks_expand_in_lockstep_and_stay_aligned(self):
        instances = [
            BankInstance(bank_id=0, axons=4, neurons=4),
            BankInstance(bank_id=0, axons=4, neurons=4),
            BankInstance(bank_id=1, axons=4, neurons=4),
            BankInstance(bank_id=1, axons=4, neurons=4),
        ]
        chunks = compose_bank_clustered_passes(
            instances, THREE_CORES, max_schedule_passes=8,
        )
        assert [[instances[i].bank_id for i in chunk] for chunk in chunks] == [
            [0, 1], [0, 1],
        ]

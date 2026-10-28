"""[U-F4] The pipelined composition of the ONE pass planner: where one copy of every bank of consecutive levels packs
together, the levels share passes as a wavefront (an instance runs in the first pass after its producers and after its
copy's previous instance), the group's first pass loads its layout and every later pass runs on it.

Vehicle: a 1-D three-level chain of bank instances, level 0 (bank 0, 4 positions), level 1 (bank 1, 4 positions, each
reading the level-0 positions p - 1 to p + 1) and level 2 (bank 2, 2 positions at stride 2, each reading the level-1
positions 2q to 2q + 2). Position multiplexing (W02) runs it in 4 + 4 + 2 = 10 passes; the wavefront in 7."""

from __future__ import annotations

from typing import List

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.pipelined_multiplexing import pipelined_segment_passes
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes

CORES = [LayoutHardCoreType(32, 32, 16)]


def _level(bank: int, n: int, level: int, *, axons=10, neurons=6, segment=0) -> List[LayoutSoftCoreSpec]:
    return [LayoutSoftCoreSpec(axons, neurons, latency_tag=level, segment_id=segment, bank_id=bank,
                               name=f"b{bank}p{i}") for i in range(n)]


def chain(**kw):
    """The specs and, per spec (segment position), the positions it reads."""
    specs = _level(0, 4, 0, **kw) + _level(1, 4, 1, **kw) + _level(2, 2, 2, **kw)
    reads = [frozenset() for _ in range(4)]
    reads += [frozenset(q for q in (p - 1, p, p + 1) if 0 <= q < 4) for p in range(4)]
    reads += [frozenset(4 + r for r in (2 * q, 2 * q + 1, 2 * q + 2) if 0 <= r < 4) for q in range(2)]
    return specs, reads


def _plan(specs, reads, cores=CORES, passes=64, **kw):
    return plan_segment_passes(specs, sum(t.count for t in cores), core_types=cores, max_schedule_passes=passes,
                               position_multiplexing=True, pipelined_multiplexing=True, producers=reads, **kw)


def _names(plan):
    return [[s.name for s in chunk] for chunk in plan.pass_lists]


class TestTheWavefront:
    def test_levels_share_passes_as_a_wavefront(self):
        specs, reads = chain()
        plan = _plan(specs, reads)
        assert _names(plan) == [["b0p0"], ["b0p1"], ["b0p2", "b1p0"], ["b0p3", "b1p1"], ["b1p2"],
                                ["b1p3", "b2p0"], ["b2p1"]]
        assert plan.feasible

    def test_the_groups_first_pass_loads_and_every_later_pass_runs_on_its_layout(self):
        specs, reads = chain()
        plan = _plan(specs, reads)
        assert plan.run_heads == (True,) + (False,) * 6
        assert plan.resident_flags == (False,) + (True,) * 6
        group, = plan.wavefront.groups
        assert group.pipelined and group.levels == (0, 1, 2) and (group.first_pass, group.end_pass) == (0, 7)
        assert [specs[i].name for i in group.template] == ["b0p0", "b1p0", "b2p0"]

    def test_every_read_is_of_an_earlier_pass(self):
        specs, reads = chain()
        plan = _plan(specs, reads)
        pass_of = plan.wavefront.pass_of
        assert all(pass_of[p] < pass_of[c] for c, ps in enumerate(reads) for p in ps)

    def test_one_instance_of_a_copy_per_pass(self):
        specs, reads = chain()
        for chunk in _plan(specs, reads).pass_lists:
            banks = [s.bank_id for s in chunk]
            assert len(banks) == len(set(banks))

    def test_levels_whose_copies_do_not_pack_together_start_a_new_group(self):
        specs, reads = chain(axons=20, neurons=20)
        plan = _plan(specs, reads, cores=[LayoutHardCoreType(32, 32, 2)])
        assert [(g.levels, g.first_pass, g.end_pass, g.pipelined) for g in plan.wavefront.groups] == [
            ((0, 1), 0, 6, True), ((2,), 6, 8, True)]
        assert plan.run_heads == (True, False, False, False, False, False, True, False)

    def test_a_level_holding_owned_cores_is_planned_by_position_multiplexing(self):
        specs, reads = chain()
        owned = LayoutSoftCoreSpec(8, 4, latency_tag=1, segment_id=0, name="own")
        specs = specs[:4] + [owned] + specs[4:]
        reads = reads[:4] + [frozenset({0})] + [frozenset(i + (i >= 4) for i in r) for r in reads[4:]]
        plan = _plan(specs, reads)
        kinds = [(g.levels, g.pipelined) for g in plan.wavefront.groups]
        assert kinds == [((0,), True), ((1,), False), ((2,), True)]
        w02 = plan.wavefront.groups[1]
        assert w02.end_pass - w02.first_pass == 5    # the owned core's capacity pass, then bank 1's four
        assert all(plan.wavefront.pass_of[p] < plan.wavefront.pass_of[c] for c, ps in enumerate(reads) for p in ps)

    def test_the_fragments_of_a_coalescing_group_share_their_pass(self):
        specs = _level(0, 3, 0) + [LayoutSoftCoreSpec(10, 6, latency_tag=1, segment_id=0, bank_id=b, name=f"f{b}p{i}",
                                                      coalescing_group_id=100 + i)
                                   for i in range(3) for b in (5, 6)]
        reads = [frozenset()] * 3 + [frozenset({i}) for i in range(3) for _ in (5, 6)]
        reads[4] = frozenset({0, 1})     # one fragment of position 0 also reads position 1: its partner waits too
        plan = _plan(specs, reads)
        pass_of = plan.wavefront.pass_of
        for i in range(3):
            assert pass_of[3 + 2 * i] == pass_of[4 + 2 * i]

    def test_the_written_cells_are_one_copy_per_bank(self):
        specs, reads = chain()
        plan = _plan(specs, reads)
        group, = plan.wavefront.groups
        assert sum(specs[i].area for i in group.template) == 3 * 10 * 6


class TestRefusalsByName:
    def test_more_passes_than_the_budget_is_refused(self):
        specs, reads = chain()
        with pytest.raises(PositionMultiplexingError, match="7 passes.*above max_schedule_passes 6"):
            _plan(specs, reads, passes=6)

    def test_the_wavefront_fits_a_budget_position_multiplexing_exceeds(self):
        specs, reads = chain()
        with pytest.raises(PositionMultiplexingError, match="above max_schedule_passes 8"):
            plan_segment_passes(specs, 16, core_types=CORES, max_schedule_passes=8, position_multiplexing=True)
        assert len(_plan(specs, reads, passes=8).pass_lists) == 7

    def test_a_plan_without_producers_is_refused(self):
        specs, _reads = chain()
        with pytest.raises(PositionMultiplexingError, match="pipelined_multiplexing.*producers"):
            plan_segment_passes(specs, 16, core_types=CORES, max_schedule_passes=64, position_multiplexing=True,
                                pipelined_multiplexing=True)

    def test_a_producer_composed_after_its_consumer_is_refused(self):
        """Latency tags order every producer below its consumer; a read the tags do not order is a defect."""
        specs, reads = chain()
        reads[4] = reads[4] | {5}        # b1p0 reads b1p1, a later instance of its own level
        with pytest.raises(PositionMultiplexingError, match="pass transparency"):
            _plan(specs, reads)

    def test_a_copy_that_packs_on_no_core_is_refused(self):
        specs, reads = chain(axons=100, neurons=100)
        with pytest.raises(PositionMultiplexingError, match="does not pack on the declared cores"):
            _plan(specs, reads, cores=[LayoutHardCoreType(32, 32, 2)])


class TestOffByDefault:
    @pytest.mark.parametrize("value", [None, False])
    def test_absent_or_false_is_position_multiplexing_byte_for_byte(self, value):
        specs, reads = chain()
        w02 = plan_segment_passes(specs, 16, core_types=CORES, max_schedule_passes=64, position_multiplexing=True)
        same = plan_segment_passes(specs, 16, core_types=CORES, max_schedule_passes=64, position_multiplexing=True,
                                   pipelined_multiplexing=value, producers=reads)
        assert same == w02 and same.wavefront is None

    def test_the_program_plan_composes_every_segment_with_the_global_producers(self):
        a, reads_a = chain(segment=0)
        b, reads_b = chain(segment=1)
        reads = list(reads_a) + [frozenset(len(a) + p for p in r) for r in reads_b]
        plan = plan_program_passes(a + b, CORES, max_schedule_passes=64, position_multiplexing=True,
                                   pipelined_multiplexing=True, producers=reads)
        assert plan.per_segment_passes == {0: 7, 1: 7}
        assert plan.resident_flags == ((False,) + (True,) * 6) * 2

    def test_the_composition_is_one_function_of_both_planes(self):
        specs, reads = chain()
        direct = pipelined_segment_passes(specs, CORES, producers=reads, budget=16, allow_coalescing=False,
                                          allow_splitting=False, max_schedule_passes=64)
        assert direct == _plan(specs, reads).wavefront

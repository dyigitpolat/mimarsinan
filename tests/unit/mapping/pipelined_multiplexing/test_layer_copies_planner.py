"""[thru:w1] Layer copies in the ONE pass planner (PL/thru/design.md (c) 'Planner', (e) 'Planner'): under a declared
``layer_replication`` the cores a wavefront group's layout leaves free take more copies of its banks, one at a time,
each to the key with the most instances per copy, while the layout packs on the declared cores; each instance runs in
the first pass after its producers on its key's copy that is free earliest. ``wavefront_group_cores`` bounds the
cores one copy of a group's keys takes, so a group closes earlier and its free cores hold copies; a bound below one
copy of a level is refused by name.

Vehicle: chains of 20 x 20 bank instances on 32 x 32 cores, so every copy takes one whole core (two never share one)
and the copies' cores are hand-counted. Level 0 (bank 0, 8 positions), level 1 (bank 1, 4 positions, position q reads
the level-0 positions 2q and 2q + 1), level 2 (bank 2, 2 positions, q reads the level-1 positions 2q and 2q + 1)."""

from __future__ import annotations

from typing import List

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.pipelined_multiplexing import pipelined_segment_passes
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes


def _level(bank: int, n: int, level: int, *, segment=0) -> List[LayoutSoftCoreSpec]:
    return [LayoutSoftCoreSpec(20, 20, latency_tag=level, segment_id=segment, bank_id=bank, name=f"b{bank}p{i}")
            for i in range(n)]


def chain(levels: int = 2):
    """The specs and, per spec (segment position), the positions it reads."""
    specs = _level(0, 8, 0) + _level(1, 4, 1)
    reads = [frozenset() for _ in range(8)] + [frozenset({2 * q, 2 * q + 1}) for q in range(4)]
    if levels == 3:
        specs += _level(2, 2, 2)
        reads += [frozenset({8 + 2 * q, 8 + 2 * q + 1}) for q in range(2)]
    return specs, reads


def _cores(count: int):
    return [LayoutHardCoreType(32, 32, count)]


def _plan(specs, reads, count, passes=64, **kw):
    return plan_segment_passes(specs, count, core_types=_cores(count), max_schedule_passes=passes,
                               position_multiplexing=True, pipelined_multiplexing=True, producers=reads, **kw)


def _names(plan):
    return [[s.name for s in chunk] for chunk in plan.pass_lists]


class TestTheCopiesWaterFill:
    def test_the_hand_counted_two_level_chain_on_six_cores(self):
        specs, reads = chain()
        plan = _plan(specs, reads, 6, layer_replication=True)
        group, = plan.wavefront.groups
        assert group.levels == (0, 1) and group.copies == (4, 2)
        assert _names(plan) == [["b0p0", "b0p1", "b0p2", "b0p3"],
                                ["b0p4", "b0p5", "b0p6", "b0p7", "b1p0", "b1p1"],
                                ["b1p2", "b1p3"]]
        assert plan.run_heads == (True, False, False) and plan.feasible

    def test_one_copy_per_bank_needs_nine_passes_on_the_same_cores(self):
        specs, reads = chain()
        assert len(_plan(specs, reads, 6).pass_lists) == 9

    @pytest.mark.parametrize("count,copies", [
        (2, (1, 1)), (3, (2, 1)), (4, (3, 1)), (5, (3, 2)), (6, (4, 2)), (7, (5, 2)), (8, (5, 3)), (9, (6, 3)),
        (10, (7, 3)), (11, (7, 4)), (12, (8, 4)), (16, (8, 4))])
    def test_each_copy_goes_to_the_key_with_the_most_instances_per_copy(self, count, copies):
        """Hand-counted: ties go to the earlier key, and a key never takes more copies than it has instances."""
        specs, reads = chain()
        group, = _plan(specs, reads, count, layer_replication=True).wavefront.groups
        assert group.copies == copies

    @pytest.mark.parametrize("count", range(2, 17))
    def test_the_copies_never_pass_the_cores(self, count):
        specs, reads = chain(levels=3)
        plan = _plan(specs, reads, count, layer_replication=True)
        for group in plan.wavefront.groups:
            assert sum(group.copies) <= count
        for chunk in plan.pass_lists:
            assert len(chunk) <= count

    def test_every_instance_runs_on_its_key_s_copy_free_earliest(self):
        specs, reads = chain()
        wave = _plan(specs, reads, 6, layer_replication=True).wavefront
        assert wave.copy_of == tuple(i % 4 for i in range(8)) + tuple(q % 2 for q in range(4))
        seen = set()
        for i, q in enumerate(wave.pass_of):
            assert (wave.keys[i], wave.copy_of[i], q) not in seen    # one instance per copy per pass
            seen.add((wave.keys[i], wave.copy_of[i], q))

    def test_every_read_is_of_an_earlier_pass(self):
        specs, reads = chain(levels=3)
        for count in (3, 5, 8):
            pass_of = _plan(specs, reads, count, layer_replication=True).wavefront.pass_of
            assert all(pass_of[p] < pass_of[c] for c, ps in enumerate(reads) for p in ps)

    def test_a_chip_with_no_free_core_keeps_one_copy_and_the_one_copy_passes(self):
        specs, reads = chain()
        one, copies = _plan(specs, reads, 2), _plan(specs, reads, 2, layer_replication=True)
        assert copies.wavefront.groups[0].copies == (1, 1)
        assert copies.pass_lists == one.pass_lists and copies.run_heads == one.run_heads


class TestTheGroupLimit:
    def test_the_limit_closes_a_group_and_its_free_cores_take_copies(self):
        specs, reads = chain(levels=3)
        whole = _plan(specs, reads, 6, layer_replication=True)
        assert [(g.levels, g.copies) for g in whole.wavefront.groups] == [((0, 1, 2), (3, 2, 1))]
        assert len(whole.pass_lists) == 5
        capped = _plan(specs, reads, 6, layer_replication=True, wavefront_group_cores=2)
        assert [(g.levels, g.first_pass, g.end_pass, g.copies) for g in capped.wavefront.groups] == [
            ((0, 1), 0, 3, (4, 2)), ((2,), 3, 4, (2,))]
        assert _names(capped)[3] == ["b2p0", "b2p1"]
        assert capped.run_heads == (True, False, False, True)

    def test_a_limit_at_or_above_the_whole_layout_changes_nothing(self):
        specs, reads = chain(levels=3)
        whole = _plan(specs, reads, 6, layer_replication=True)
        for limit in (3, 6, 100):
            assert _plan(specs, reads, 6, layer_replication=True, wavefront_group_cores=limit) == whole

    def test_a_limit_below_one_copy_of_a_level_is_refused_by_name(self):
        specs = _level(0, 2, 0) + _level(1, 2, 0)          # one level holding two banks: one copy takes two cores
        reads = [frozenset()] * 4
        with pytest.raises(PositionMultiplexingError, match="wavefront_group_cores 1 .*one copy"):
            _plan(specs, reads, 6, layer_replication=True, wavefront_group_cores=1)

    @pytest.mark.parametrize("replication", [None, False])
    def test_a_limit_without_replication_is_refused_by_name(self, replication):
        specs, reads = chain()
        with pytest.raises(PositionMultiplexingError, match="wavefront_group_cores.*layer_replication"):
            _plan(specs, reads, 6, layer_replication=replication, wavefront_group_cores=2)

    def test_replication_without_the_wavefront_is_refused_by_name(self):
        specs, _reads = chain()
        with pytest.raises(PositionMultiplexingError, match="layer_replication.*pipelined_multiplexing"):
            plan_segment_passes(specs, 6, core_types=_cores(6), max_schedule_passes=64, position_multiplexing=True,
                                layer_replication=True)


class TestOneFunctionOfBothPlanes:
    def test_the_program_plan_composes_copies_in_every_segment(self):
        a, reads_a = chain()
        b = [LayoutSoftCoreSpec(20, 20, latency_tag=s.latency_tag, segment_id=1, bank_id=s.bank_id + 10, name=s.name)
             for s in a]
        reads = list(reads_a) + [frozenset(len(a) + p for p in r) for r in reads_a]
        plan = plan_program_passes(a + b, _cores(6), max_schedule_passes=64, position_multiplexing=True,
                                   pipelined_multiplexing=True, producers=reads, layer_replication=True)
        assert plan.per_segment_passes == {0: 3, 1: 3}
        assert plan.resident_flags == (False, True, True) * 2

    def test_the_planner_s_wavefront_is_the_composition_s(self):
        specs, reads = chain(levels=3)
        direct = pipelined_segment_passes(specs, _cores(6), producers=reads, budget=6, allow_coalescing=False,
                                          allow_splitting=False, max_schedule_passes=64, layer_replication=True,
                                          wavefront_group_cores=2)
        assert direct == _plan(specs, reads, 6, layer_replication=True, wavefront_group_cores=2).wavefront

    @pytest.mark.parametrize("value", [None, False])
    def test_absent_or_false_states_no_copies(self, value):
        specs, reads = chain()
        wave = _plan(specs, reads, 6, layer_replication=value).wavefront
        assert wave.copy_of is None and all(g.copies is None for g in wave.groups)
        assert wave == _plan(specs, reads, 6).wavefront

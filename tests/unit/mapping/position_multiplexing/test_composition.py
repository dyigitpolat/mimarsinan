"""[W02] The position-multiplexed composition on the shape-only plane: one copy per shared bank, instances in turn.

Every test reads the ONE planner (``plan_segment_passes`` / ``plan_program_passes``) the builder and the search
consume, so a composition pinned here is the program the chip runs.
"""

from __future__ import annotations

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.position_multiplexing import multiplexed_segment_passes, shares_a_bank
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes

CORES = [LayoutHardCoreType(32, 32, 16)]


def _bank(bank, n, axons=10, neurons=6, level=0, segment=0):
    return [LayoutSoftCoreSpec(axons, neurons, latency_tag=level, segment_id=segment, bank_id=bank, name=f"b{bank}p{i}")
            for i in range(n)]


def _owned(axons=8, neurons=4, level=0, name="own"):
    return LayoutSoftCoreSpec(axons, neurons, latency_tag=level, segment_id=0, name=name)


def _plan(specs, cores=CORES, passes=64, **kw):
    return plan_segment_passes(specs, sum(t.count for t in cores), core_types=cores, max_schedule_passes=passes,
                               position_multiplexing=True, **kw)


def _banks(chunk):
    return [s.bank_id for s in chunk]


class TestOneCopyPerBank:
    def test_one_bank_runs_one_instance_per_pass_on_one_copy(self):
        specs = _bank(0, 5)
        plan = _plan(specs)
        assert [list(chunk) for chunk in plan.pass_lists] == [[s] for s in specs]
        assert plan.resident_flags == (False, True, True, True, True)
        assert plan.feasible

    def test_banks_of_one_position_count_share_a_run(self):
        plan = _plan(_bank(0, 3) + _bank(1, 3))
        assert [_banks(c) for c in plan.pass_lists] == [[0, 1]] * 3
        assert plan.resident_flags == (False, True, True)

    def test_banks_of_different_counts_run_apart(self):
        plan = _plan(_bank(0, 3) + _bank(1, 2))
        assert [_banks(c) for c in plan.pass_lists] == [[0], [0], [0], [1], [1]]
        assert plan.resident_flags == (False, True, True, False, True)

    def test_a_run_splits_where_one_copy_of_each_bank_does_not_pack(self):
        cores = [LayoutHardCoreType(32, 32, 1)]
        plan = _plan(_bank(0, 2, axons=20, neurons=20) + _bank(1, 2, axons=20, neurons=20), cores=cores)
        assert [_banks(c) for c in plan.pass_lists] == [[0], [0], [1], [1]]
        assert plan.resident_flags == (False, True, False, True)

    def test_levels_run_in_dependency_order_with_owned_cores_programmed_every_pass(self):
        specs = _bank(0, 2, level=1) + [_owned(level=2)] + _bank(1, 2, level=0)
        plan = _plan(specs)
        assert [[(s.bank_id, s.latency_tag) for s in c] for c in plan.pass_lists] == [
            [(1, 0)], [(1, 0)], [(0, 1)], [(0, 1)], [(None, 2)]]
        assert plan.resident_flags == (False, True, False, True, False)

    def test_a_wide_many_neuron_instance_composes(self):
        """WI-5f: an instance wider and taller than a core packs (coalesced windows), so its bank multiplexes."""
        plan = _plan(_bank(0, 3, axons=70, neurons=40), allow_coalescing=True, allow_splitting=True)
        assert len(plan.pass_lists) == 3 and plan.resident_flags == (False, True, True)

    def test_instances_that_do_not_read_their_copy_are_owned(self):
        specs = _bank(0, 3)
        plan = _plan(specs, shares=[True, False, True])
        flat = [list(c) for c in plan.pass_lists]
        assert [specs[1]] in flat and plan.resident_flags.count(True) == 1


class TestRefusalsByName:
    def test_more_passes_than_the_budget_is_refused(self):
        with pytest.raises(PositionMultiplexingError, match="above max_schedule_passes 4"):
            _plan(_bank(0, 5), passes=4)

    def test_a_copy_that_packs_on_no_core_is_refused(self):
        with pytest.raises(PositionMultiplexingError, match="does not pack on the declared cores"):
            _plan(_bank(0, 2, axons=100, neurons=100), cores=[LayoutHardCoreType(32, 32, 2)])


class TestTheDecisionIsTakenOnlyWhereItChangesSomething:
    def test_a_segment_without_a_shared_instance_plans_as_undeclared(self):
        specs = [_owned(name="a"), _owned(name="b")] + _bank(3, 1)
        assert not shares_a_bank(specs)
        declared = _plan(specs)
        undeclared = plan_segment_passes(specs, 16, core_types=CORES, max_schedule_passes=64)
        assert declared == undeclared

    def test_undeclared_and_false_are_the_planner_byte_for_byte(self):
        specs = _bank(0, 5) + _bank(1, 5)
        base = plan_program_passes(specs, CORES, max_schedule_passes=64)
        assert plan_program_passes(specs, CORES, max_schedule_passes=64, position_multiplexing=False) == base
        assert base.segments[0][1].run_heads is None

    def test_the_program_plan_composes_every_segment(self):
        specs = _bank(0, 3, segment=0) + _bank(1, 2, segment=1)
        plan = plan_program_passes(specs, CORES, max_schedule_passes=64, position_multiplexing=True)
        assert plan.per_segment_passes == {0: 3, 1: 2}
        assert plan.resident_flags == (False, True, True, False, True)
        assert plan.sync_count == 3


class TestCommittedCells:
    def test_the_written_cells_are_one_copy_per_bank(self):
        specs = _bank(0, 6, axons=10, neurons=6) + _bank(1, 6, axons=12, neurons=4)
        mux = multiplexed_segment_passes(specs, CORES, budget=16, allow_coalescing=False, allow_splitting=False,
                                         max_schedule_passes=64)
        written = sum(s.area for chunk, head in zip(mux.pass_lists, mux.run_heads) if head for s in chunk)
        assert written == 10 * 6 + 12 * 4

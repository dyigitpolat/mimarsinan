"""[thru:w2] The batch pipeline in the ONE pass planner (PL/thru/design.md (c) 'Planner', (e) 'Planner'): under a
declared ``batch_pipeline_slots`` B a run of the program serves B inputs. The instances of input 1, then input 2, and so
on share each wavefront group's passes, each in the first pass after its own input's producers, on its key's copy that
is free earliest (one copy computes one instance per pass, of whichever input). A pass outside every wavefront group
serves one input, so it runs once per input, input by input. The pass budget is counted per run.

Vehicles: the copies' chain of ``test_layer_copies_planner`` (20 x 20 bank instances on 32 x 32 cores, every copy one
whole core, hand-counted), and the wavefront chain of ``test_composition`` with a level of owned cores (a W02 group)."""

from __future__ import annotations

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.pipelined_multiplexing import pipelined_segment_passes
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes, resident_passes

from . import test_composition as TC
from .test_layer_copies_planner import _cores, chain


def _plan(specs, reads, count, passes=64, slots=None, **kw):
    if slots is not None:
        kw.update(batch_pipeline_slots=slots, carry_input_slots=16)
    return plan_segment_passes(specs, count, core_types=_cores(count), max_schedule_passes=passes,
                               position_multiplexing=True, pipelined_multiplexing=True, producers=reads, **kw)


def _entries(plan):
    """Per pass: (input slot, spec name) of each entry, in the pass's order."""
    return [[(b, s.name) for s, b in zip(chunk, slots)] for chunk, slots in zip(plan.pass_lists, plan.input_slots)]


def _named(slot, names):
    return [(slot, n) for n in names]


class TestTheHandCountedChain:
    def test_one_copy_two_inputs_share_the_pass_where_the_first_ends(self):
        """One copy per bank on 6 cores: one input takes 9 passes, two inputs 17 (input 2 starts on bank 0's copy in
        the pass input 1's last level-1 instance runs), not 18."""
        specs, reads = chain()
        plan = _plan(specs, reads, 6, slots=2)
        one = [["b0p0"], ["b0p1"], ["b0p2", "b1p0"], ["b0p3"], ["b0p4", "b1p1"], ["b0p5"], ["b0p6", "b1p2"], ["b0p7"],
               ["b1p3"]]
        want = [_named(0, names) for names in one[:8]] + [_named(0, ["b1p3"]) + _named(1, ["b0p0"])]
        want += [_named(1, names) for names in one[1:]]
        assert _entries(plan) == want and len(plan.pass_lists) == 17
        assert plan.slots == 2 and plan.run_heads == (True,) + (False,) * 16 and plan.feasible

    def test_copies_two_inputs_take_five_passes_three_take_seven(self):
        specs, reads = chain()
        plan = _plan(specs, reads, 6, layer_replication=True, slots=2)
        b0 = [f"b0p{i}" for i in range(8)]
        b1 = [f"b1p{i}" for i in range(4)]
        assert _entries(plan) == [
            _named(0, b0[:4]), _named(0, b0[4:] + b1[:2]), _named(0, b1[2:]) + _named(1, b0[:4]),
            _named(1, b0[4:] + b1[:2]), _named(1, b1[2:])]
        assert plan.wavefront.groups[0].copies == (4, 2)
        assert plan.wavefront.entry_copies == ((0, 1, 2, 3), (0, 1, 2, 3, 0, 1), (0, 1, 0, 1, 2, 3), (0, 1, 2, 3, 0, 1),
                                               (0, 1))
        three = _plan(specs, reads, 6, layer_replication=True, slots=3)
        assert len(three.pass_lists) == 7 and [sorted(set(s)) for s in three.input_slots][4:] == [[1, 2], [2], [2]]

    def test_without_the_key_one_input_per_run(self):
        specs, reads = chain()
        for kw in ({}, {"layer_replication": True}):
            plan = _plan(specs, reads, 6, **kw)
            assert plan.slots is None and plan.input_slots is None and plan.wavefront.entry_slots is None
            assert plan == _plan(specs, reads, 6, carry_input_slots=4, **kw)


class TestTheRules:
    @pytest.mark.parametrize("slots", [2, 3, 5])
    @pytest.mark.parametrize("count,replication", [(2, None), (5, True), (8, True)])
    def test_every_read_is_of_an_earlier_pass_of_its_own_input(self, slots, count, replication):
        specs, reads = chain(levels=3)
        plan = _plan(specs, reads, count, layer_replication=replication, slots=slots)
        index = {id(s): i for i, s in enumerate(specs)}
        pass_of = {}
        for q, (chunk, inputs) in enumerate(zip(plan.pass_lists, plan.input_slots)):
            for spec, b in zip(chunk, inputs):
                assert (b, index[id(spec)]) not in pass_of     # each instance once per input
                pass_of[(b, index[id(spec)])] = q
        assert len(pass_of) == slots * len(specs)
        assert all(pass_of[(b, p)] < pass_of[(b, c)] for b in range(slots) for c, ps in enumerate(reads) for p in ps)

    @pytest.mark.parametrize("slots", [2, 3, 5])
    def test_one_instance_per_copy_per_pass_and_the_inputs_in_order_on_each_copy(self, slots):
        specs, reads = chain(levels=3)
        plan = _plan(specs, reads, 7, layer_replication=True, slots=slots)
        wave = plan.wavefront
        keys = {id(s): k for s, k in zip(specs, wave.keys)}
        last = {}
        for q, (chunk, inputs, copies) in enumerate(zip(plan.pass_lists, wave.entry_slots, wave.entry_copies)):
            used = [(keys[id(s)], c) for s, c in zip(chunk, copies)]
            assert len(used) == len(set(used))
            assert list(inputs) == sorted(inputs)              # a pass lists its inputs in order
            for copy_, b in zip(used, inputs):
                assert last.get(copy_, 0) <= b                  # a copy serves input 1, then input 2, ...
                last[copy_] = b

    def test_each_instance_runs_in_the_first_pass_its_input_and_its_copy_allow(self):
        """Re-derived greedily: input by input, unit by unit, the first pass after the instance's own input's
        producers and after its copy's previous instance."""
        specs, reads = chain(levels=3)
        plan = _plan(specs, reads, 6, layer_replication=True, slots=3)
        wave = plan.wavefront
        index = {id(s): i for i, s in enumerate(specs)}
        at = {}
        for q, (chunk, inputs, copies) in enumerate(zip(plan.pass_lists, wave.entry_slots, wave.entry_copies)):
            for spec, b, c in zip(chunk, inputs, copies):
                at[(b, index[id(spec)])] = (q, c)
        free = {}
        for b in range(3):
            for i in range(len(specs)):
                q, c = at[(b, i)]
                ready = max([at[(b, p)][0] + 1 for p in reads[i]] + [0])
                key = wave.keys[i]
                options = range(wave.groups[0].copies[wave.groups[0].keys.index(key)])
                c_want = min(options, key=lambda k: (free.get((key, k), 0), k))
                assert (q, c) == (max(ready, free.get((key, c_want), 0)), c_want)
                free[(key, c)] = q + 1

    def test_the_pass_budget_is_counted_per_run(self):
        specs, reads = chain()
        assert len(_plan(specs, reads, 6, passes=17, slots=2).pass_lists) == 17
        with pytest.raises(PositionMultiplexingError, match="needs 17 passes"):
            _plan(specs, reads, 6, passes=16, slots=2)


class TestPassesOutsideTheWavefront:
    def _owned(self):
        specs, reads = TC.chain()
        owned = LayoutSoftCoreSpec(8, 4, latency_tag=1, segment_id=0, name="own")
        specs = specs[:4] + [owned] + specs[4:]
        reads = reads[:4] + [frozenset({0})] + [frozenset(i + (i >= 4) for i in r) for r in reads[4:]]
        return specs, reads

    def _plan(self, specs, reads, slots=None):
        kw = {} if slots is None else {"batch_pipeline_slots": slots, "carry_input_slots": slots}
        return plan_segment_passes(specs, 16, core_types=TC.CORES, max_schedule_passes=64, position_multiplexing=True,
                                   pipelined_multiplexing=True, producers=reads, **kw)

    def test_a_w02_group_runs_once_per_input_input_by_input(self):
        specs, reads = self._owned()
        one, two = self._plan(specs, reads), self._plan(specs, reads, 2)
        w02, w02_two = one.wavefront.groups[1], two.wavefront.groups[1]
        assert (w02.pipelined, w02_two.pipelined) == (False, False)
        n = w02.end_pass - w02.first_pass
        assert w02_two.end_pass - w02_two.first_pass == 2 * n
        base = [[s.name for s in chunk] for chunk in one.pass_lists[w02.first_pass:w02.end_pass]]
        got = _entries(two)[w02_two.first_pass:w02_two.end_pass]
        assert got == [_named(0, names) for names in base] + [_named(1, names) for names in base]
        heads = one.run_heads[w02.first_pass:w02.end_pass]
        assert two.run_heads[w02_two.first_pass:w02_two.end_pass] == heads + heads
        assert len(two.pass_lists) == 22 and len(one.pass_lists) == 11

    def test_a_segment_without_a_shared_bank_runs_once_per_input(self):
        specs = [LayoutSoftCoreSpec(20, 20, latency_tag=lv, segment_id=0, name=f"own{lv}") for lv in range(3)]
        reads = [frozenset(), frozenset({0}), frozenset({1})]
        for count in (1, 3):
            kw = dict(core_types=_cores(count), max_schedule_passes=64, position_multiplexing=True,
                      pipelined_multiplexing=True, producers=reads)
            one = plan_segment_passes(specs, count, **kw)
            three = plan_segment_passes(specs, count, batch_pipeline_slots=3, carry_input_slots=3, **kw)
            assert one.wavefront is None and three.wavefront is None
            assert three.pass_lists == one.pass_lists * 3 and three.slots == 3
            assert three.input_slots == tuple((b,) * len(chunk) for b in range(3) for chunk in one.pass_lists)
            assert three.residency_applied == one.residency_applied
            assert three.resident_flags == resident_passes(3 * len(one.pass_lists),
                                                           policy_applied=one.residency_applied)


class TestOneFunctionOfBothPlanes:
    def test_the_program_plan_serves_the_inputs_in_every_segment(self):
        a, reads_a = chain()
        b = [LayoutSoftCoreSpec(20, 20, latency_tag=s.latency_tag, segment_id=1, bank_id=s.bank_id + 10, name=s.name)
             for s in a]
        reads = list(reads_a) + [frozenset(len(a) + p for p in r) for r in reads_a]
        plan = plan_program_passes(a + b, _cores(6), max_schedule_passes=64, position_multiplexing=True,
                                   pipelined_multiplexing=True, producers=reads, layer_replication=True,
                                   batch_pipeline_slots=2, carry_input_slots=2)
        assert plan.per_segment_passes == {0: 5, 1: 5}
        assert all(seg.slots == 2 for _, seg in plan.segments)

    def test_the_planner_s_wavefront_is_the_composition_s(self):
        specs, reads = chain(levels=3)
        direct = pipelined_segment_passes(specs, _cores(6), producers=reads, budget=6, allow_coalescing=False,
                                          allow_splitting=False, max_schedule_passes=64, layer_replication=True,
                                          batch_pipeline_slots=3, carry_input_slots=3)
        plan = _plan(specs, reads, 6, layer_replication=True, slots=3)
        assert direct == plan.wavefront and plan.input_slots == direct.entry_slots

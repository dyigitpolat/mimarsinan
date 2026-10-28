"""[U-F4] The deployment record of a pipelined program names its schedule and prices it as built.

The record states each wavefront group (segment, first pass, passes, levels, the layout cores its first pass loads),
each group head's dark cores (loaded, not evaluated) and each later pass's head ordinals, all declared-only, so a
record of any other program keeps its bytes. The record plane prices what the program runs: a wavefront pass is one
evaluation (its cores read carried inputs only), MAC events and core initializations count the evaluated cores (not
the dark ones), the writes are the heads' loads (every copy once per group), and the carry is the dense census of the
values the passes hand on, which the wavefront leaves unchanged against the one-level-per-pass program."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import mapping_program_passes, record_program_passes
from mimarsinan.deployment_record.quantities.value_domain import record_value_facts
from mimarsinan.deployment_record.schema import ScheduleRecord, SegmentRecord
from mimarsinan.deployment_record.schema.wavefront import WavefrontRecord
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.support.wavefront_stages import dark_cores, wavefront_head

from unit.mapping.pipelined_multiplexing import vehicles as V


@pytest.fixture(scope="module")
def block():
    return V.lane_program("block")


@pytest.fixture(scope="module")
def programs(block):
    return {p: V.build(block.ir, V.cores(3), p) for p in (None, True)}


def _schedule(hybrid):
    return schedule_record_from_mapping(hybrid, weight_bits=4, params_reloaded=0, timesteps=1, pass_transfer="collapse")


def _record(schedule):
    return SimpleNamespace(schedule=schedule, timing=SimpleNamespace(per_segment=(), s_global=1),
                           identity=SimpleNamespace(platform={"cores": V.cores(3), "core_semantics": "mvm",
                                                              "activation_bits": 4}),
                           utilization=SimpleNamespace(crossbar=SimpleNamespace(weight_bits=4)))


class TestTheRecordNamesTheSchedule:
    def test_every_wavefront_group_is_stated(self, programs):
        hybrid = programs[True]
        schedule = _schedule(hybrid)
        heads = [s for s in V.neural(hybrid) if wavefront_head(s) is not None]
        assert schedule.wavefronts is not None and len(schedule.wavefronts) == len(heads) >= 2
        for group, stage in zip(schedule.wavefronts, heads):
            assert isinstance(group, WavefrontRecord)
            assert (group.segment_index, group.first_pass) == (stage.schedule_segment_index, stage.schedule_pass_index)
            assert (group.passes, group.layout_cores, tuple(group.levels)) == (
                wavefront_head(stage).passes, wavefront_head(stage).layout_cores, wavefront_head(stage).levels)
        assert sum(g.passes for g in schedule.wavefronts) <= schedule.pass_count

    def test_heads_seal_their_dark_cores_and_later_passes_their_ordinals(self, programs):
        hybrid = programs[True]
        schedule = _schedule(hybrid)
        for segment, stage in zip(schedule.segments(), V.neural(hybrid)):
            assert segment.dark_core_indices == (tuple(sorted(dark_cores(stage))) or None)
            ordinals = getattr(stage, "schedule_core_ordinals", None)
            assert segment.core_ordinals == (None if ordinals is None else tuple(ordinals))
        assert any(s.dark_core_indices for s in schedule.segments())
        assert any(s.core_ordinals for s in schedule.segments())

    def test_the_fields_round_trip_and_stay_out_of_other_records(self, programs):
        schedule = _schedule(programs[True])
        assert ScheduleRecord.from_dict(schedule.to_dict()) == schedule
        for segment in schedule.segments():
            assert SegmentRecord.from_dict(segment.to_dict()) == segment
        other = _schedule(programs[None])
        assert other.wavefronts is None and "wavefronts" not in other.to_dict()
        assert all("dark_core_indices" not in s.to_dict() and "core_ordinals" not in s.to_dict()
                   for s in other.segments())

    def test_a_dark_core_index_outside_its_pass_is_refused(self, programs):
        segment = next(s for s in _schedule(programs[True]).segments() if s.dark_core_indices)
        with pytest.raises(ValueError, match="dark_core_indices"):
            replace(segment, dark_core_indices=(len(segment.cores),))


class TestTheRecordPricesWhatRuns:
    def test_a_wavefront_pass_is_one_evaluation(self, programs):
        schedule = _schedule(programs[True])
        facts = record_value_facts(_record(schedule))
        assert all(c.static_delay_levels == 0 for s in schedule.segments() if s.core_ordinals or s.dark_core_indices
                   for c in s.cores)
        assert facts.latency_steps == len(schedule.segments())

    def test_mac_events_count_the_evaluated_cores_only(self, programs):
        hybrid = programs[True]
        facts = record_value_facts(_record(_schedule(hybrid)))
        evaluated = sum(CoreOccupancy.from_hard_core(c).cells_used for s in V.neural(hybrid)
                        for i, c in enumerate(s.hard_core_mapping.cores) if i not in dark_cores(s))
        loaded = sum(CoreOccupancy.from_hard_core(c).cells_used for s in V.neural(hybrid)
                     for c in s.hard_core_mapping.cores)
        assert facts.mac_events == evaluated < loaded

    def test_the_writes_are_the_heads_loads_on_both_build_planes(self, programs):
        hybrid = programs[True]
        schedule = _schedule(hybrid)
        sealed, built = record_program_passes(_record(schedule)), mapping_program_passes(hybrid)
        assert [r.core_cells for r in sealed.rows] == [r.core_cells for r in built.rows]
        assert [r.resident for r in sealed.rows] == [r.resident for r in built.rows]
        written = [s for s in schedule.segments() if s.programming == "reprogram"]
        assert written and all(s.core_ordinals is None for s in written) and sum(s.params_bytes for s in written) > 0
        assert all(s.params_bytes == 0 for s in schedule.segments() if s.core_ordinals is not None)

    def test_the_carry_is_the_per_pass_programs(self, programs):
        mux, pipe = _schedule(programs[None]).carry, _schedule(programs[True]).carry
        assert pipe is not None and mux is not None
        assert (pipe.carried_wires, pipe.boundary_out_bytes) == (mux.carried_wires, mux.boundary_out_bytes)

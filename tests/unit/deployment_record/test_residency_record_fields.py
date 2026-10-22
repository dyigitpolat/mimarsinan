"""The residency seam: two declared-only record fields, their serde rule, and resident_core_indices."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json

import numpy as np
import pytest

from mimarsinan.deployment_record.schema import DeploymentRecord, ScheduleRecord, SegmentRecord
from mimarsinan.deployment_record.schema import serde
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.schema.schedule import (
    ResidencyPlanRecord,
    ResidencyRecordError,
    ResidencySegmentRecord,
)
from mimarsinan.mapping.noc.fragments import PassProgram
from mimarsinan.mapping.packing.hybrid_types import HybridStage
from mimarsinan.mapping.support.period_census import (
    RESIDENCY_STEP_RULES,
    SCHEDULE_RESIDENCY_PLAN,
    SCHEDULE_RESIDENT_CORES,
    PeriodCensusError,
    ResidencySegmentPlan,
    resident_core_indices,
)
from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_schedule,
    make_segment,
)

#: sha256 of ``json.dumps(x.to_dict(), sort_keys=True)`` on the Wave 0 base tree.
BASE_DIGESTS = {
    "full": "401f969c9f01ea108e15e1a473d540a9fe9805af100159a5646b84b9bd55a494",
    "schedule": "21bf8aff5e2db19afb97c637d3efda5c99fe320dad0f216757d081f19cb38f7b",
    "segment": "acf131bf327a3a61c9db63eccbf2895aec314dda1dad89723c57c0440836a290",
}


def _digest(fragment) -> str:
    return hashlib.sha256(json.dumps(fragment.to_dict(), sort_keys=True).encode()).hexdigest()


def _plan() -> ResidencyPlanRecord:
    return ResidencyPlanRecord(segments=(
        ResidencySegmentRecord(
            segment_index=0, step_rule="floor", step_index=0, rotating_tiles=3, passes=2,
            carried_elements=96, written_tiles=5, total_tiles=10, core_budget=8,
        ),
        ResidencySegmentRecord(
            segment_index=1, step_rule="floor", step_index=0, rotating_tiles=2, passes=3,
            carried_elements=40, written_tiles=4, total_tiles=10, core_budget=8,
        ),
    ))


class TestWaveZeroRecordsAreUnchanged:
    @pytest.mark.parametrize("name,make", [
        ("full", make_full_record), ("schedule", make_schedule), ("segment", make_segment),
    ])
    def test_to_dict_is_byte_identical(self, name, make):
        assert _digest(make()) == BASE_DIGESTS[name]

    def test_unset_fields_are_not_emitted(self):
        record = make_full_record()
        data = record.to_dict()
        assert "residency_plan" not in data["schedule"]
        for stage in data["schedule"]["stages"]:
            assert "resident_core_indices" not in stage

    def test_a_wave_zero_dict_loads(self):
        data = json.loads(json.dumps(make_full_record().to_dict()))
        record = DeploymentRecord.from_dict(data)
        assert record.schedule.residency_plan is None
        assert all(s.resident_core_indices is None for s in record.schedule.segments())


class TestDeclaredFields:
    def _schedule(self) -> ScheduleRecord:
        base = make_schedule()
        stages = tuple(
            dataclasses.replace(stage, resident_core_indices=(0,))
            if isinstance(stage, SegmentRecord) and stage.pass_index == 0 else stage
            for stage in base.stages
        )
        return dataclasses.replace(base, stages=stages, residency_plan=_plan())

    def test_both_fields_round_trip(self):
        schedule = self._schedule()
        wire = json.loads(json.dumps(schedule.to_dict()))
        assert wire["residency_plan"]["segments"][1]["passes"] == 3
        assert ScheduleRecord.from_dict(wire) == schedule
        record = dataclasses.replace(make_full_record(), schedule=schedule)
        assert DeploymentRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record

    def test_segment_field_is_emitted_when_set(self):
        segment = dataclasses.replace(make_segment(), resident_core_indices=(1, 0))
        assert segment.to_dict()["resident_core_indices"] == (1, 0)
        wire = json.loads(json.dumps(segment.to_dict()))
        assert SegmentRecord.from_dict(wire).resident_core_indices == (1, 0)

    def test_the_plan_record_validates_its_rule(self):
        with pytest.raises(ValueError, match="step_rule"):
            ResidencySegmentRecord(
                segment_index=0, step_rule="greedy", step_index=0, rotating_tiles=1, passes=1,
                carried_elements=0, written_tiles=0, total_tiles=1, core_budget=1,
            )
        with pytest.raises(ValueError, match="step_index"):
            ResidencySegmentRecord(
                segment_index=0, step_rule="stateless", step_index=0, rotating_tiles=1,
                passes=1, carried_elements=0, written_tiles=0, total_tiles=1, core_budget=1,
            )

    @pytest.mark.parametrize("rule,index", [("greedy", 0), ("stateless", 0), ("index", None)])
    def test_the_plan_record_refuses_by_name(self, rule, index):
        with pytest.raises(ResidencyRecordError) as caught:
            ResidencySegmentRecord(
                segment_index=0, step_rule=rule, step_index=index, rotating_tiles=1,
                passes=1, carried_elements=0, written_tiles=0, total_tiles=1, core_budget=1,
            )
        assert isinstance(caught.value, ToolchainRefusal)
        assert isinstance(caught.value, ValueError)

    _LEGAL_ROW = dict(segment_index=0, step_rule="index", step_index=1, rotating_tiles=1, passes=1,
                      carried_elements=0, written_tiles=0, total_tiles=1, core_budget=1)

    @pytest.mark.parametrize("field,value", [
        ("step_index", -3), ("step_index", -1), ("step_index", 1.0), ("step_index", True),
        ("segment_index", -1), ("rotating_tiles", -1), ("passes", -2), ("carried_elements", -1),
        ("written_tiles", -1), ("total_tiles", -5), ("core_budget", -1),
        ("passes", 2.5), ("core_budget", False), ("total_tiles", "4"),
    ])
    def test_the_plan_record_refuses_a_count_or_index_outside_the_naturals_by_name(self, field, value):
        """The record is as strict as the key it records (residency_plan.index >= 0) and as the
        census beside it (PassCensus: non-negative ints, bool excluded)."""
        with pytest.raises(ResidencyRecordError, match=f"ResidencySegmentRecord.{field}"):
            ResidencySegmentRecord(**{**self._LEGAL_ROW, field: value})

    def test_a_legal_row_and_its_zero_counts_construct(self):
        ResidencySegmentRecord(**self._LEGAL_ROW)
        ResidencySegmentRecord(**{**self._LEGAL_ROW, "step_index": 0, "rotating_tiles": 0})

    @pytest.mark.parametrize("marks", [(-1,), (0.0,), (True,), ("1",)])
    def test_the_segment_marks_refuse_a_non_index_by_name(self, marks):
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            dataclasses.replace(make_segment(), resident_core_indices=marks)
        wire = json.loads(json.dumps(make_segment().to_dict()))
        wire["resident_core_indices"] = list(marks)
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            SegmentRecord.from_dict(wire)

    def test_the_record_mirrors_the_executed_plan_field_by_field(self):
        record_fields = [f.name for f in dataclasses.fields(ResidencySegmentRecord)]
        plan_fields = [f.name for f in dataclasses.fields(ResidencySegmentPlan)]
        assert record_fields == plan_fields
        executed = tuple(
            ResidencySegmentPlan(**dataclasses.asdict(s)) for s in _plan().segments
        )
        assert ResidencyPlanRecord.from_plan(executed) == _plan()
        assert set(RESIDENCY_STEP_RULES) == {"floor", "stateless", "index"}

    def test_the_serde_rule_has_no_branch_on_a_field_name(self):
        tree = ast.parse(inspect.getsource(serde))
        constants = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        names = {
            f.name
            for cls in (SegmentRecord, ScheduleRecord, ResidencyPlanRecord, ResidencySegmentRecord)
            for f in dataclasses.fields(cls)
        }
        assert not names & constants, sorted(names & constants)


def _stage(**flags) -> HybridStage:
    return HybridStage(kind="neural", name="s", **flags)


class TestResidentCoreIndices:
    def test_the_attribute_names(self):
        assert SCHEDULE_RESIDENT_CORES == "schedule_resident_cores"
        assert SCHEDULE_RESIDENCY_PLAN == "schedule_residency_plan"

    def test_a_wholly_resident_stage_keeps_every_core(self):
        assert resident_core_indices(_stage(schedule_weights_resident=True), 3) == {0, 1, 2}

    def test_a_marked_stage_keeps_its_marks(self):
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset({0, 2}))
        assert resident_core_indices(stage, 3) == {0, 2}

    def test_an_unmarked_stage_keeps_none(self):
        assert resident_core_indices(_stage(), 3) == frozenset()

    def test_pass_programs_follow_the_same_rule(self):
        assert resident_core_indices(PassProgram(core_cells=(4, 4), resident=True), 2) == {0, 1}
        assert resident_core_indices(PassProgram(core_cells=(4, 4), resident=False), 2) == set()
        marked = PassProgram(core_cells=(4, 4, 4), resident=False)
        object.__setattr__(marked, SCHEDULE_RESIDENT_CORES, (1,))
        assert resident_core_indices(marked, 3) == {1}

    def test_a_mark_outside_the_pass_refuses_by_name(self):
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset({5}))
        with pytest.raises(PeriodCensusError, match="schedule_resident_cores"):
            resident_core_indices(stage, 3)

    def test_a_mark_equal_to_the_core_count_refuses_by_name(self):
        """A pass of 3 cores has indices 0, 1 and 2: mark 3 is one past the last core, on either plane."""
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset({3}))
        with pytest.raises(PeriodCensusError, match="schedule_resident_cores"):
            resident_core_indices(stage, 3)
        marked = PassProgram(core_cells=(4, 4, 4), resident=False)
        object.__setattr__(marked, SCHEDULE_RESIDENT_CORES, (0, 3))
        with pytest.raises(PeriodCensusError, match="schedule_resident_cores"):
            resident_core_indices(marked, 3)
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset({2}))
        assert resident_core_indices(stage, 3) == {2}

    @pytest.mark.parametrize("mark", [1.0, 1.7, True, "1", None])
    def test_a_mark_that_is_not_an_index_refuses_by_name(self, mark):
        """A float or bool mark is refused, never truncated to a core index."""
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, (0, mark))
        with pytest.raises(PeriodCensusError, match="schedule_resident_cores"):
            resident_core_indices(stage, 3)

    def test_an_integral_mark_of_any_int_type_is_an_index(self):
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, (np.int64(2), 0))
        assert resident_core_indices(stage, 3) == {0, 2}
        assert all(type(i) is int for i in resident_core_indices(stage, 3))

    def test_the_result_is_a_frozenset(self):
        assert isinstance(resident_core_indices(_stage(), 2), frozenset)

    @pytest.mark.parametrize("marks", [3, 2.0, True])
    def test_a_mark_set_that_is_not_a_collection_refuses_by_name(self, marks):
        stage = _stage()
        setattr(stage, SCHEDULE_RESIDENT_CORES, marks)
        with pytest.raises(PeriodCensusError, match="schedule_resident_cores"):
            resident_core_indices(stage, 4)

    @pytest.mark.parametrize("core_count", [3.0, -1, True, "3"])
    def test_a_core_count_outside_the_naturals_refuses_by_name(self, core_count):
        with pytest.raises(PeriodCensusError, match="core_count"):
            resident_core_indices(_stage(schedule_weights_resident=True), core_count)

    def test_a_core_count_of_any_int_type_counts_exactly(self):
        assert resident_core_indices(_stage(schedule_weights_resident=True), np.int64(3)) == {0, 1, 2}


_PLAN_ROW = dict(segment_index=0, step_rule="index", step_index=1, rotating_tiles=1, passes=2,
                 carried_elements=0, written_tiles=1, total_tiles=4, core_budget=3)


class TestResidencySegmentPlan:
    """The type G5 builds obeys the seam's one integer rule where it is created, not only when G3's
    record build copies it (ResidencyPlanRecord.from_plan)."""

    @pytest.mark.parametrize("field,value", [
        ("passes", -5), ("step_index", -1), ("rotating_tiles", 1.0), ("core_budget", 2.5),
        ("total_tiles", True), ("written_tiles", "1"), ("segment_index", None),
        ("carried_elements", np.float32(1)),
    ])
    def test_a_count_or_index_outside_the_naturals_refuses_by_name(self, field, value):
        with pytest.raises(PeriodCensusError, match=f"ResidencySegmentPlan.{field}"):
            ResidencySegmentPlan(**{**_PLAN_ROW, field: value})

    @pytest.mark.parametrize("rule,index", [("greedy", 0), ("stateless", 0), ("floor", None)])
    def test_the_step_rule_and_its_index_refuse_by_name(self, rule, index):
        with pytest.raises(PeriodCensusError, match="ResidencySegmentPlan.step_"):
            ResidencySegmentPlan(**{**_PLAN_ROW, "step_rule": rule, "step_index": index})

    def test_a_count_of_any_int_type_is_held_as_an_int(self):
        plan = ResidencySegmentPlan(**{**_PLAN_ROW, "passes": np.int64(2), "step_index": np.int32(1)})
        assert plan == ResidencySegmentPlan(**_PLAN_ROW)
        assert type(plan.passes) is int and type(plan.step_index) is int
        stateless = ResidencySegmentPlan(**{**_PLAN_ROW, "step_rule": "stateless", "step_index": None})
        assert stateless.step_index is None

    def test_the_record_of_a_numpy_built_plan_is_the_record_of_the_int_plan(self):
        """Records normalize exactly like the plan (never truncating), so from_plan copies a plan built
        with numpy counts into the same record, and to_dict stays JSON."""
        numpy_built = ResidencySegmentPlan(**{k: (np.int64(v) if isinstance(v, int) else v)
                                              for k, v in _PLAN_ROW.items()})
        record = ResidencyPlanRecord.from_plan((numpy_built,))
        assert record == ResidencyPlanRecord.from_plan((ResidencySegmentPlan(**_PLAN_ROW),))
        assert json.loads(json.dumps(record.to_dict()))["segments"][0]["passes"] == 2
        direct = ResidencySegmentRecord(**{**_PLAN_ROW, "core_budget": np.uint8(3)})
        assert type(direct.core_budget) is int and direct.core_budget == 3

    def test_segment_marks_of_any_int_type_are_held_as_ints(self):
        segment = dataclasses.replace(make_segment(), resident_core_indices=(np.int64(1), 0))
        assert segment.resident_core_indices == (1, 0)
        assert all(type(i) is int for i in segment.resident_core_indices)
        assert json.loads(json.dumps(segment.to_dict()))["resident_core_indices"] == [1, 0]

    @pytest.mark.parametrize("marks", [frozenset({0}), {0}, frozenset({1, 0})],
                             ids=["frozenset", "set", "frozenset-of-two"])
    def test_segment_marks_without_an_order_refuse_by_name(self, marks):
        """The record field takes a tuple or a list (the frozen interface): a set has no order a record
        could keep byte for byte, so G3 sorts the census's frozenset of marks before it builds the record."""
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            dataclasses.replace(make_segment(), resident_core_indices=marks)
        wire = json.loads(json.dumps(make_segment().to_dict()))
        wire["resident_core_indices"] = marks
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            SegmentRecord.from_dict(wire)
        sorted_marks = dataclasses.replace(make_segment(), resident_core_indices=tuple(sorted(marks)))
        assert sorted_marks.resident_core_indices == tuple(sorted(marks))

    @pytest.mark.parametrize("marks", [3, 1.0])
    def test_segment_marks_that_are_not_a_collection_refuse_by_name(self, marks):
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            dataclasses.replace(make_segment(), resident_core_indices=marks)
        wire = json.loads(json.dumps(make_segment().to_dict()))
        wire["resident_core_indices"] = marks
        with pytest.raises(ResidencyRecordError, match="SegmentRecord.resident_core_indices"):
            SegmentRecord.from_dict(wire)

"""Per-core residency marks: the record build fills S0's declared-only fields, and both planes'
period censuses keep the marked cores out of the rewritten set."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    mapping_program_passes,
    record_program_passes,
)
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema.schedule import ResidencyPlanRecord
from mimarsinan.mapping.support.period_census import (
    SCHEDULE_RESIDENCY_PLAN,
    SCHEDULE_RESIDENT_CORES,
    ResidencySegmentPlan,
)

from unit.deployment_record.g3_fixtures import HandPassProgram, built_chain, record_with


#: Three one-core passes on a two-core chip: one tile pinned, the other two rotate on the free core
#: (W = F - C + r = 3 - 2 + 1 = 2).
_PLAN = (ResidencySegmentPlan(segment_index=0, step_rule="floor", step_index=0, rotating_tiles=1, passes=3,
                              carried_elements=8, written_tiles=2, total_tiles=3, core_budget=2),)


def _marked():
    mapping = built_chain()
    neural = [s for s in mapping.stages if s.kind == "neural"]
    setattr(neural[1], SCHEDULE_RESIDENT_CORES, frozenset({0}))
    for stage in (neural[0], neural[2]):
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset())
    setattr(mapping, SCHEDULE_RESIDENCY_PLAN, _PLAN)
    return mapping


def _schedule(mapping):
    return schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0, timesteps=4,
                                        pass_transfer="collapse")


class TestTheRecordFill:
    def test_the_build_fills_both_declared_only_fields(self):
        schedule = _schedule(_marked())
        assert [s.resident_core_indices for s in schedule.segments()] == [(), (0,), ()]
        assert schedule.residency_plan == ResidencyPlanRecord.from_plan(_PLAN)
        data = schedule.to_dict()
        assert data["residency_plan"]["segments"][0]["step_rule"] == "floor"
        assert [s["resident_core_indices"] for s in data["stages"]] == [(), (0,), ()]

    def test_an_unmarked_build_emits_neither_field(self):
        data = _schedule(built_chain()).to_dict()
        assert "residency_plan" not in data
        assert not any("resident_core_indices" in stage for stage in data["stages"])

    def test_a_whole_pass_resident_stage_without_marks_seals_no_marks(self):
        mapping = built_chain()
        neural = [s for s in mapping.stages if s.kind == "neural"]
        neural[1].schedule_weights_resident = True
        assert [s.resident_core_indices for s in _schedule(mapping).segments()] == [None, None, None]


#: The chain's 8 x 8 core type, two of them: the three tiles exceed it, so every pass rewrites.
_TWO_CORES = {(8, 8): 2}


class TestTheCensusExcludesTheMarks:
    def test_the_built_and_sealed_planes_rewrite_only_the_unmarked_cores(self):
        mapping = _marked()
        built = period_census(mapping_program_passes(mapping), weight_bits=8, capacity=_TWO_CORES).census
        record = record_with(_schedule(mapping), cores=[{"max_axons": 8, "max_neurons": 8, "count": 1}])
        sealed = period_census(record_program_passes(record), weight_bits=8, capacity=_TWO_CORES).census
        assert [p.resident_cores for p in built.passes] == [0, 1, 0]
        assert built.passes[1].written_cells_backing_store == 0
        for a, b in zip(built.passes, sealed.passes):
            assert (a.resident_cores, a.written_cells_backing_store, a.payload_bytes) == \
                (b.resident_cores, b.written_cells_backing_store, b.payload_bytes)

    def test_the_period_rewrite_quantity_excludes_the_marked_core(self):
        cores = [{"max_axons": 8, "max_neurons": 8, "count": 2}]
        marked = from_record(record_with(_schedule(_marked()), cores=cores, inference_batch_K=4))
        unmarked = from_record(record_with(_schedule(built_chain()), cores=cores, inference_batch_K=4))
        per_core = unmarked.get("period_rewrite_bytes").value / 3
        assert marked.get("period_rewrite_bytes").value == pytest.approx(2 * per_core)
        # The pinned tile holds one core and the two others rotate on the second core, twice per period;
        # unmarked, every pass loads from the first core, three times per period.
        assert marked.get("max_writes_per_cell_per_inference").value == pytest.approx(2 / 4)
        assert unmarked.get("max_writes_per_cell_per_inference").value == pytest.approx(3 / 4)

    def test_a_pin_on_the_only_core_refuses_the_period_census_by_name(self):
        cores = [{"max_axons": 8, "max_neurons": 8, "count": 1}]
        values = from_record(record_with(_schedule(_marked()), cores=cores, inference_batch_K=4))
        assert not values.has("period_rewrite_bytes")

    def test_the_candidate_plane_reads_the_same_marks_from_pass_programs(self):
        softcores = [SimpleNamespace(segment_id=0, latency_tag=i, output_count=4) for i in range(3)]
        programs = [HandPassProgram(core_cells=(20,)) for _ in range(3)]       # 5 axons (bias) x 4 neurons
        programs[1] = replace(programs[1], schedule_resident_cores=frozenset({0}))
        noc = SimpleNamespace(pass_placements=(((0, 0),), ((1, 0),), ((2, 0),)), pass_programs=tuple(programs),
                              census=None)
        semantics = SimpleNamespace(retimed=False, is_cycle=False, is_cascade=False, latency_group_count=None,
                                    chip_latency=None)
        candidate = period_census(candidate_program_passes(softcores, noc, semantics=semantics, timesteps=4),
                                  weight_bits=8, capacity=_TWO_CORES).census
        built = period_census(mapping_program_passes(_marked()), weight_bits=8, capacity=_TWO_CORES).census
        assert [p.resident_cores for p in candidate.passes] == [p.resident_cores for p in built.passes]
        assert [p.written_cells_backing_store for p in candidate.passes] == \
            [p.written_cells_backing_store for p in built.passes]

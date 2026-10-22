"""[S0 staged] G5's executed residency plan through S0's seam: one set of names (S0's), S0's plan type on the
hybrid mapping and its record field for field, S0's reader of the pinned marks on both planes, and the key through
S0's resolver into the builder."""

from __future__ import annotations

import dataclasses

from mimarsinan.deployment_record.schema.schedule import ResidencyPlanRecord
from mimarsinan.mapping.packing import hybrid_build_pool, hybrid_build_scheduled
from mimarsinan.mapping.packing import schedule_residency_plan as executed_plan
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform import mapping_structure
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support import period_census as s0
from mimarsinan.mapping.support.schedule import residency_composition
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

from . import vehicles as V

C_TIGHT = 5          # F = 11 tiles on the chain: Q >= 3 and a pinned budget at the floor


class TestOneSetOfNames:
    MODULES = (mapping_structure, residency_composition, executed_plan, hybrid_build_pool, hybrid_build_scheduled)
    NAMES = ("RESIDENCY_STEP_RULES", "STEP_FLOOR", "STEP_STATELESS", "STEP_INDEX", "SCHEDULE_RESIDENT_CORES",
             "SCHEDULE_RESIDENCY_PLAN", "ResidencySegmentPlan")

    def test_every_g5_module_holds_s0s_objects(self):
        held = [(m.__name__, n) for m in self.MODULES for n in self.NAMES if n in vars(m)]
        assert held, "the G5 modules use the seam names"
        for module, name in held:
            assert vars(__import__(module, fromlist=[name]))[name] is getattr(s0, name), (module, name)

    def test_g5_defines_none_of_them(self):
        for module in self.MODULES:
            for name in self.NAMES:
                value = vars(module).get(name)
                assert value is None or value is getattr(s0, name), (module.__name__, name)


class TestAnExecutedPlanThroughTheSeam:
    def _built(self, plan=None):
        ir = V.ir_of(V.chain_repr(seed=0))
        return V.build(ir, C_TIGHT, plan), V.layout_plane(V.chain_repr(seed=0))

    def test_the_mapping_holds_s0_rows_that_its_record_copies_field_for_field(self):
        hybrid, _ = self._built({"step": "floor"})
        rows = getattr(hybrid, s0.SCHEDULE_RESIDENCY_PLAN)
        assert rows and all(type(row) is s0.ResidencySegmentPlan for row in rows)
        assert sum(row.passes for row in rows) >= 2
        record = ResidencyPlanRecord.from_plan(rows)
        assert [dataclasses.astuple(s) for s in record.segments] == [dataclasses.astuple(r) for r in rows]
        assert ResidencyPlanRecord.from_dict(record.to_dict()) == record

    def test_s0s_reader_returns_the_marks_on_both_planes(self):
        hybrid, (specs, census, program) = self._built({"step": "floor"})
        passes = V.neural_passes(hybrid)
        marks = [s0.resident_core_indices(s, len(s.hard_core_mapping.cores)) for s in passes]
        assert marks == [getattr(s, s0.SCHEDULE_RESIDENT_CORES) for s in passes]
        assert any(marks)
        frag = V.fragments(specs, census, C_TIGHT, {"step": "floor"}, program=program)
        assert [s0.resident_core_indices(p, len(p.core_cells)) for p in frag.pass_programs] == marks

    def test_without_the_key_the_reader_finds_no_marks(self):
        hybrid, (specs, census, program) = self._built(None)
        assert all(s0.resident_core_indices(s, len(s.hard_core_mapping.cores)) == frozenset()
                   for s in V.neural_passes(hybrid))
        frag = V.fragments(specs, census, C_TIGHT, None, program=program)
        assert all(s0.resident_core_indices(p, len(p.core_cells)) == frozenset() for p in frag.pass_programs)

    def test_the_key_rides_s0s_resolver_into_the_builder(self):
        config = {"cores": V.cores(C_TIGHT), "allow_scheduling": True, "residency_plan": {"step": "floor"}}
        platform = build_platform_constraints_resolved(config)
        assert platform["residency_plan"] == {"step": "floor"}
        assert "residency_plan" not in build_platform_constraints_resolved({**config, "residency_plan": None})
        def build(caps):
            return build_hybrid_hard_core_mapping(
                ir_graph=V.ir_of(V.chain_repr(seed=0)), cores_config=platform["cores"],
                strategy=MappingStrategy.resolve(caps), max_schedule_passes=caps.max_schedule_passes)

        resolved = getattr(build(ChipCapabilities.from_platform_constraints(platform)), s0.SCHEDULE_RESIDENCY_PLAN)
        assert resolved and resolved == getattr(build(V.capabilities({"step": "floor"})), s0.SCHEDULE_RESIDENCY_PLAN)

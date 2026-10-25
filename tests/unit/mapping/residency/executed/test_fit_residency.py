"""[W03] One residency planner on every plane: a program that fits is programmed once, and the floor runs wherever U1 runs.

The executed rotation stays the one period rule (finding 4, option (a)): a core is kept across periods only where the
executed plan pins it. The planner now pins every core of a program whose packed passes fit the declared cores side by
side (the floor at r = 0), with or without a declared ``residency_plan``, on the record plane (the build's marks) and
on the candidate plane (the fragments' marks). Regression at C = F (0 bytes per period) and at C = F - 1 (the floor
writes W = F - C + r* tiles, U1's rotation rewrites every core it writes).
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import candidate_program_passes, mapping_program_passes
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.residency_rules import fit_pins
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.mapping.support.schedule.residency_units import cut_nodes_from_census, pin_passes, residency_units
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

from . import vehicles as V

BITS = 8
#: A seeded chain (one segment) and a branchy graph of three IR-run segments (U1 runs four passes at C = F).
KINDS = ["chain", "deep_mismatch"]
#: Every branchy vehicle whose depth segments keep the reference edge classes (the key's segmentation rule).
ADMITTED = ["chain", "deep_mismatch", "skip", "skew3", "late_skip", "deep_skip", "balanced_skip", "wide_branches"]
PLANS = [None, {"step": "floor"}]


def _vehicle(kind):
    rep = V.chain_repr(seed=0) if kind == "chain" else V.branchy_repr(kind)
    specs, census, program = V.layout_plane(rep)
    return specs, census, program, V.ir_of(rep)


def _tiles(specs, census) -> int:
    """F: the tiles of the program's residency units."""
    units = residency_units(specs, cut_nodes_from_census(specs, census),
                            [LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, 1)],
                            segments=[int(s.segment_id or 0) for s in specs])
    return sum(units.tiles)


def _record(hybrid, count):
    return period_census(mapping_program_passes(hybrid), weight_bits=BITS,
                         capacity={(V.MAX_AXONS, V.MAX_NEURONS): count}).census


def _candidate(specs, frag, count):
    passes = candidate_program_passes(specs, frag, semantics=StageSemantics(), timesteps=V.T)
    return period_census(passes, weight_bits=BITS, capacity={(V.MAX_AXONS, V.MAX_NEURONS): count}).census


def _period_bytes(census) -> int:
    return sum(p.payload_bytes for p in census.passes)


def _rows(census):
    return [(p.cores, p.resident_cores, p.payload_bytes) for p in census.passes]


def _everything_pinned(stages) -> bool:
    return all(stage.schedule_resident_cores == frozenset(range(len(stage.hard_core_mapping.cores)))
               for stage in stages)


class TestAFitIsProgrammedOnce:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("extra", [0, 3])
    @pytest.mark.parametrize("plan", PLANS, ids=["absent", "floor"])
    def test_at_c_equal_f_and_above_both_planes_pin_every_core_and_rewrite_nothing(self, kind, extra, plan):
        specs, census, program, ir = _vehicle(kind)
        count = _tiles(specs, census) + extra
        hybrid = V.build(ir, count, plan)
        stages = V.neural_passes(hybrid)
        assert _everything_pinned(stages)
        assert _period_bytes(_record(hybrid, count)) == 0
        segments = schedule_record_from_mapping(hybrid, weight_bits=BITS, params_reloaded=0, timesteps=V.T,
                                                pass_transfer=VERBATIM).segments()
        assert all(s.resident_core_indices == tuple(range(len(s.cores))) for s in segments)
        frag = V.fragments(specs, census, count, plan, program=program)
        assert all(p.schedule_resident_cores == frozenset(range(p.cores)) for p in frag.pass_programs)
        assert _period_bytes(_candidate(specs, frag, count)) == 0

    def test_the_segmented_witness_runs_several_writing_passes_at_c_equal_f(self):
        """The regression bites: at C = F the three-segment program runs four passes, which the rotation alone would
        rewrite every period."""
        specs, census, program, ir = _vehicle("deep_mismatch")
        assert len(V.neural_passes(V.build(ir, _tiles(specs, census)))) >= 3

    @pytest.mark.parametrize("kind", KINDS)
    def test_at_c_equal_f_minus_one_the_floor_writes_w_tiles_on_both_planes(self, kind):
        specs, census, program, ir = _vehicle(kind)
        F = _tiles(specs, census)
        count = F - 1
        comp = V.shape_only(specs, census, program, count, {"step": "floor"})
        r = comp.step[0]
        assert comp.plan.F == F and r >= 1
        hybrid = V.build(ir, count, {"step": "floor"})
        record = _record(hybrid, count)
        candidate = _candidate(specs, V.fragments(specs, census, count, {"step": "floor"}, program=program), count)
        assert sum(c - k for c, k, _ in _rows(record)) == F - count + r                  # W = F - C + r*
        assert _rows(record) == _rows(candidate)
        assert _period_bytes(record) == comp.facts.programmed_bytes(lambda cells: params_bytes(cells, BITS)) > 0
        print(f"[W03] {kind} C = F - 1 = {count}: floor r* {r}, W {F - count + r} tiles, "
              f"{_period_bytes(record)} B per period")

    @pytest.mark.parametrize("kind", KINDS)
    def test_at_c_equal_f_minus_one_without_the_key_the_rotation_rewrites_every_core_it_writes(self, kind):
        specs, census, program, ir = _vehicle(kind)
        count = _tiles(specs, census) - 1
        hybrid = V.build(ir, count)
        stages = V.neural_passes(hybrid)
        assert len(stages) >= 2 and not any(hasattr(s, "schedule_resident_cores") for s in stages)
        every = sum(params_bytes(CoreOccupancy.from_hard_core(c).cells_used, BITS)
                    for s in stages if not s.schedule_weights_resident for c in s.hard_core_mapping.cores)
        assert _period_bytes(_record(hybrid, count)) == every
        frag = V.fragments(specs, census, count, program=program)
        assert all(p.schedule_resident_cores is None for p in frag.pass_programs)

    def test_an_unscheduled_program_of_several_segments_is_programmed_once(self):
        specs, census, program, ir = _vehicle("deep_mismatch")
        count = _tiles(specs, census)
        hybrid = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=V.cores(count),
                                                strategy=MappingStrategy.resolve(ChipCapabilities()))
        stages = V.neural_passes(hybrid)
        assert len(stages) >= 2 and _everything_pinned(stages)
        assert _period_bytes(_record(hybrid, count)) == 0
        frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS,
                                                                                     count)],
                                     census=census, program=program, **ChipCapabilities().layout_kwargs())
        assert [p.schedule_resident_cores for p in frag.pass_programs] == [frozenset(range(frag.pass_programs[0]
                                                                                           .cores))]
        assert len(candidate_program_passes(specs, frag, semantics=StageSemantics(), timesteps=V.T).rows) >= 2
        assert _period_bytes(_candidate(specs, frag, count)) == 0


class TestTheFitRule:
    """The fit is the floor's pinned rule at r = 0, judged physically on the declared pool (dominance matching)."""

    def test_it_pins_every_core_as_the_pinned_rule_at_r_zero(self):
        types = [[((8, 8),), ((8, 8),)], [((8, 8),)]]
        assert fit_pins(types, [False, False], {(8, 8): 3}) == (frozenset({0, 1}), frozenset({0})) == pin_passes(
            [[(64, 1), (64, 1)], [(64, 1)]], 0, 3)

    def test_one_core_short_is_no_fit(self):
        assert fit_pins([[((8, 8),), ((8, 8),)], [((8, 8),)]], [False, False], {(8, 8): 2}) is None

    def test_a_tile_fits_on_a_larger_core(self):
        assert fit_pins([[((8, 8),)], [((16, 8),)]], [False, False], {(16, 8): 2}) == (frozenset({0}),
                                                                                         frozenset({0}))
        assert fit_pins([[((16, 8),)], [((16, 8),)]], [False, False], {(16, 8): 1, (8, 8): 1}) is None

    def test_a_fused_core_occupies_its_components(self):
        assert fit_pins([[((8, 8), (8, 8))], [((8, 8),)]], [False, False], {(8, 8): 2}) is None
        assert fit_pins([[((8, 8), (8, 8))], [((8, 8),)]], [False, False], {(8, 8): 3}) is not None

    def test_a_wholly_resident_pass_runs_on_cores_already_pinned(self):
        """U1's bank-clustered passes after the first run on the first pass's cores: they occupy none of their own."""
        assert fit_pins([[((8, 8),)], [((8, 8),)]], [False, True], {(8, 8): 1}) == (frozenset({0}), frozenset({0}))


class TestTheFloorRunsWhereU1Runs:
    @pytest.mark.parametrize("kind", ADMITTED)
    def test_at_every_budget_u1_builds_the_floor_builds_on_both_planes(self, kind):
        specs, census, program, ir = _vehicle(kind)
        F = _tiles(specs, census)
        built = 0
        for count in range(1, F + 2):
            try:
                V.build(ir, count, max_schedule_passes=64)
            except ToolchainRefusal:
                continue
            built += 1
            hybrid = V.build(ir, count, {"step": "floor"}, max_schedule_passes=64)
            frag = V.fragments(specs, census, count, {"step": "floor"}, program=program, max_schedule_passes=64)
            assert len(V.neural_passes(hybrid)) == len(frag.pass_programs) >= 1
        assert built >= F - 1


class TestThePlannedProgramsRunExactly:
    """The certificate on the programs this planner plans: the pinned fit at C = F (key absent and floor; several
    passes on the three-segment vehicle) and the floor at C = F - 1 and at one core (multi-pass) run equal to the
    identity (SCM) run at atol 0, on four seeded batches; the marks change what is written, never what runs."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("point", ["fit_absent", "fit_floor", "below_floor", "one_group_floor"])
    def test_the_executed_program_equals_the_identity_run(self, kind, point):
        specs, census, program, ir = _vehicle(kind)
        F = _tiles(specs, census)
        count = {"fit_absent": F, "fit_floor": F, "below_floor": F - 1, "one_group_floor": 1}[point]
        plan = None if point == "fit_absent" else {"step": "floor"}
        mapping = V.build(ir, count, plan, max_schedule_passes=64)
        stages = V.neural_passes(mapping)
        segments = len({int(s.segment_id or 0) for s in specs})               # a fit runs one pass per segment
        assert len(stages) >= (min(2, segments) if point.startswith("fit") else 2)
        assert _everything_pinned(stages) == point.startswith("fit")
        identity, executed = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir)), V.lif_flow(mapping)
        outputs = []
        for batch in range(4):
            x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(700 + batch))
            with torch.no_grad():
                want = identity(x)
                assert torch.equal(executed(x), want)
            outputs.append(want)
        assert float(torch.cat(outputs).std(0).max()) > 0                   # the vehicle discriminates

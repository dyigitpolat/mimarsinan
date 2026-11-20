"""[thru:w2] The batch pipeline's two declared keys (PL/thru/design.md (b)): ``batch_pipeline_slots`` (an integer B of
2 or more: a run of the program serves B inputs, which share the passes of every wavefront group) and the chip
component ``carry_input_slots`` (an integer of 1 or more: how many inputs' carried values the chip holds at once).
B is cost-only, relevant and servable exactly where ``pipelined_multiplexing`` is true, deferred in the cost draws'
partition and passed through to the one pass planner on both planes. The chip states ``carry_input_slots``: a chip that
states none refuses a declared B by name, and a B above it is refused by name. Absent, every program is the paper
version's (the identity test, ``tests/identity``)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import mimarsinan.certification.randomized.cost_draws as cd
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import ADMISSIBILITY_BEARING, COST_ONLY
from mimarsinan.config_schema.registry.types import FieldType
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import (
    BatchPipelineError, ChipCapabilities, PositionMultiplexingError)
from mimarsinan.mapping.support.pipelined_multiplexing import pipelined_segment_passes
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier import verify_hardware_config

from .test_layer_copies_planner import chain

SLOTS, CARRY = "batch_pipeline_slots", "carry_input_slots"
KEYS = (SLOTS, CARRY)
PIPE = {"allow_scheduling": True, "position_multiplexing": True, "pipelined_multiplexing": True}
TARGETS = Path(__file__).resolve().parents[4] / "src" / "mimarsinan" / "mapping" / "platform" / "targets"


class TestTheRegistryEntries:
    def test_the_slots_are_a_cost_only_platform_key_of_the_mapping_strategy(self):
        entry = REGISTRY[SLOTS]
        assert str(entry.decision_class) == COST_ONLY
        assert (entry.section, entry.group, entry.passthrough) == ("platform_constraints", "mapping_strategy", True)
        assert entry.type is FieldType.INT and entry.bounds == (2, None)

    def test_the_carry_slots_are_an_admissibility_bearing_chip_component(self):
        """Like carry_buffer_bytes: the chip's statement changes no program, it only refuses a B above it by name."""
        entry = REGISTRY[CARRY]
        assert str(entry.decision_class) == ADMISSIBILITY_BEARING
        assert (entry.section, entry.group, entry.passthrough) == ("platform_constraints", "hardware", True)
        assert entry.type is FieldType.INT and entry.bounds == (1, None)

    @pytest.mark.parametrize("key", KEYS)
    def test_neither_has_a_default_so_an_absent_key_changes_nothing(self, key):
        from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET, DEFAULT_PLATFORM_CONSTRAINTS

        assert key not in DEFAULT_PLATFORM_CONSTRAINTS and key in CONFIG_KEYS_SET

    def test_the_slots_are_relevant_only_under_pipelined_multiplexing(self):
        relevant = REGISTRY[SLOTS].relevant
        assert relevant.evaluate(PIPE)
        assert not relevant.evaluate({**PIPE, "pipelined_multiplexing": False})

    def test_the_carry_slots_are_relevant_where_slots_are_declared(self):
        relevant = REGISTRY[CARRY].relevant
        assert relevant.evaluate({**PIPE, SLOTS: 4})
        assert not relevant.evaluate(PIPE)


class TestTheCostDrawsPartition:
    def test_the_slots_are_deferred_by_name_with_their_reason(self):
        assert SLOTS in cd.DEFERRED and cd.DEFERRED_REASONS[SLOTS]
        assert SLOTS in cd.cost_only_set() and CARRY not in cd.cost_only_set()

    def test_the_partition_still_is_the_registry_cost_only_set(self):
        assert {p for paths in cd.partition().values() for p in paths} == cd.cost_only_set()


class TestTheCapability:
    def test_undeclared_they_are_absent_from_every_served_dict(self):
        caps = ChipCapabilities.from_platform_constraints(PIPE)
        assert (caps.batch_pipeline_slots, caps.carry_input_slots) == (None, None)
        assert not set(KEYS) & (set(caps.layout_kwargs()) | set(caps.capability_bits()))

    def test_declared_they_are_read_and_forwarded(self):
        caps = ChipCapabilities.from_platform_constraints({**PIPE, SLOTS: 4, CARRY: 16})
        assert (caps.batch_pipeline_slots, caps.carry_input_slots) == (4, 16)
        assert caps.layout_kwargs()[SLOTS] == 4 and caps.layout_kwargs()[CARRY] == 16

    def test_the_chip_s_statement_alone_changes_nothing(self):
        assert ChipCapabilities(allow_scheduling=True, carry_input_slots=16).batch_pipeline_slots is None

    def test_the_slots_need_pipelined_multiplexing(self):
        with pytest.raises(BatchPipelineError, match="batch_pipeline_slots.*pipelined_multiplexing"):
            ChipCapabilities(allow_scheduling=True, position_multiplexing=True, batch_pipeline_slots=2,
                             carry_input_slots=4)

    def test_a_chip_stating_no_carry_slots_refuses_the_slots_by_name(self):
        with pytest.raises(BatchPipelineError, match="carry_input_slots.*refused"):
            ChipCapabilities(**PIPE, batch_pipeline_slots=2)

    def test_slots_above_the_chip_s_carry_slots_are_refused_by_name(self):
        with pytest.raises(BatchPipelineError, match="batch_pipeline_slots 5 .*carry_input_slots 4"):
            ChipCapabilities(**PIPE, batch_pipeline_slots=5, carry_input_slots=4)
        assert ChipCapabilities(**PIPE, batch_pipeline_slots=4, carry_input_slots=4).batch_pipeline_slots == 4

    @pytest.mark.parametrize("value", [1, 0, -2, True, 2.0, "2"])
    def test_slots_below_two_or_not_an_integer_are_refused(self, value):
        with pytest.raises(BatchPipelineError, match="batch_pipeline_slots must be an integer >= 2"):
            ChipCapabilities(**PIPE, batch_pipeline_slots=value, carry_input_slots=16)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0, -1, True, 1.0])
    def test_carry_slots_below_one_or_not_an_integer_are_refused(self, value):
        with pytest.raises(BatchPipelineError, match="carry_input_slots must be an integer >= 1"):
            ChipCapabilities(allow_scheduling=True, carry_input_slots=value)  # type: ignore[arg-type]

    def test_the_refusal_is_a_position_multiplexing_refusal(self):
        assert issubclass(BatchPipelineError, PositionMultiplexingError)


def test_every_layout_helper_takes_them():
    keys = set(ChipCapabilities(**PIPE, batch_pipeline_slots=2, carry_input_slots=2).layout_kwargs())
    assert set(KEYS) <= keys
    for helper in (compute_mapping_stats, verify_hardware_config, build_layout_plan, collect_noc_fragments,
                   plan_program_passes, plan_segment_passes, pipelined_segment_passes):
        assert set(KEYS) <= set(inspect.signature(helper).parameters), helper.__name__


class TestTheOnePlannerRefusesWhatTheChipCannotServe:
    def _plan(self, **kw):
        specs, reads = chain()
        return plan_segment_passes(specs, 6, core_types=[LayoutHardCoreType(32, 32, 6)], max_schedule_passes=64,
                                   position_multiplexing=True, pipelined_multiplexing=True, producers=reads, **kw)

    def test_slots_on_a_chip_stating_no_carry_slots(self):
        with pytest.raises(BatchPipelineError, match="carry_input_slots"):
            self._plan(batch_pipeline_slots=2)

    def test_slots_above_the_carry_slots(self):
        with pytest.raises(BatchPipelineError, match="batch_pipeline_slots 3 .*carry_input_slots 2"):
            self._plan(batch_pipeline_slots=3, carry_input_slots=2)

    def test_slots_without_the_wavefront(self):
        specs, _reads = chain()
        with pytest.raises(BatchPipelineError, match="batch_pipeline_slots.*pipelined_multiplexing"):
            plan_program_passes(specs, [LayoutHardCoreType(32, 32, 6)], position_multiplexing=True,
                                batch_pipeline_slots=2, carry_input_slots=2)

    def test_the_chip_s_statement_alone_plans_the_paper_version(self):
        assert self._plan(carry_input_slots=16) == self._plan()


def test_the_shape_only_plane_plans_the_slots():
    """A platform declaring them, through the candidate plane's calls (CL5's and W48's), reaches the planner: a run of
    two inputs takes fewer passes than two runs, every slot's instances placed on the plane's passes."""
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD
    from unit.mapping.lanes.lane_planes import ladder, network_of
    from unit.mapping.lanes.test_lane_shape_walk import FIXTURES, _fixture

    platform, program = _fixture("residual", ("L", "E"), multiplexing=True, axons=256, neurons=64, count=3,
                                 passes=1 << 16, coalescing=True, hardware_bias=True)
    network = network_of(program, FIXTURES["residual"][2])
    passes, placed = {}, {}
    for slots in (None, 2):
        declared = {**platform, "pipelined_multiplexing": True, "layer_replication": True, CARRY: 2}
        if slots:
            declared[SLOTS] = slots
        census = LD.shape_census(network, declared)
        scheduled = LD.schedule_program(network, census, declared, ladder().program_facts,
                                        tiles=LD.packed_tiles(census, declared))
        assert scheduled.refusal is None, scheduled.refusal
        passes[slots] = len(scheduled.noc.pass_programs)
        placed[slots] = sum(len(p) for p in scheduled.noc.pass_placements)
    assert passes[None] < passes[2] < 2 * passes[None]
    assert placed[2] == 2 * placed[None]


class TestTheStudyChipStatesItsCarrySlots:
    """The COMPASS-size study chip states 16, sourced to COMPASS's batch-16 runs, which load and store each image's
    intermediate values per partition (park2025compass Sec. II-B p. 2 with Fig. 2, and Sec. IV p. 6 with Fig. 9)."""

    def test_its_description_declares_sixteen_with_its_source(self):
        bundle = json.loads((TARGETS / "study_compass.json").read_text())
        assert bundle["keys"][CARRY] == 16
        assert "carry_input_slots" in bundle["notes"] and "park2025compass" in bundle["notes"]
        assert "Sec. II-B" in bundle["notes"] and "Fig. 9" in bundle["notes"]

    def test_no_other_description_declares_it(self):
        declaring = sorted(p.stem for p in TARGETS.glob("*.json") if CARRY in json.loads(p.read_text())["keys"])
        assert declaring == ["study_compass"]

    def test_the_bundle_loads_and_serves_it(self):
        from mimarsinan.mapping.platform.target_bundles import target_bundles

        bundle = target_bundles()["study_compass"]
        assert bundle.values[CARRY] == 16

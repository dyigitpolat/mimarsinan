"""[thru:k] The layer copies' two declared keys (PL/thru/design.md (b), (c)): ``layer_replication`` (more copies of a
wavefront group's layers on the cores its layout leaves free) and ``wavefront_group_cores`` (the cores one copy of a
group may take). Both are cost-only, legal exactly where ``pipelined_multiplexing`` is legal, relevant under the key
they refine, deferred in the cost draws' partition, and passed through to the one pass planner on both planes, which
plans the copies ([thru:w1], ``test_layer_copies_planner``) and refuses by name a group limit without
``layer_replication`` and ``layer_replication`` without the wavefront. Absent or false, every program is the paper
version's (the identity test, ``tests/identity``)."""

from __future__ import annotations

import inspect

import pytest

import mimarsinan.certification.randomized.cost_draws as cd
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.types import FieldType
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, PositionMultiplexingError
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes, plan_segment_passes
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier import verify_hardware_config

from . import vehicles as V

REPLICATION, GROUP_CORES = "layer_replication", "wavefront_group_cores"
KEYS = (REPLICATION, GROUP_CORES)
PIPE = {"allow_scheduling": True, "position_multiplexing": True, "pipelined_multiplexing": True}


@pytest.fixture(scope="module")
def block():
    return V.lane_program("block")


class TestTheRegistryEntries:
    @pytest.mark.parametrize("key", KEYS)
    def test_each_is_a_cost_only_platform_key_of_the_mapping_strategy(self, key):
        entry = REGISTRY[key]
        assert str(entry.decision_class) == COST_ONLY
        assert (entry.section, entry.group, entry.passthrough) == ("platform_constraints", "mapping_strategy", True)

    @pytest.mark.parametrize("key", KEYS)
    def test_neither_has_a_default_so_an_absent_key_changes_nothing(self, key):
        from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS

        assert key not in DEFAULT_PLATFORM_CONSTRAINTS

    def test_the_types_and_the_group_limit_s_bounds(self):
        assert REGISTRY[REPLICATION].type is FieldType.BOOL
        assert REGISTRY[GROUP_CORES].type is FieldType.INT and REGISTRY[GROUP_CORES].bounds == (1, None)

    def test_replication_is_relevant_only_under_pipelined_multiplexing(self):
        relevant = REGISTRY[REPLICATION].relevant
        assert relevant.evaluate(PIPE)
        assert not relevant.evaluate({**PIPE, "pipelined_multiplexing": False})

    def test_the_group_limit_is_relevant_only_under_replication(self):
        relevant = REGISTRY[GROUP_CORES].relevant
        assert relevant.evaluate({**PIPE, REPLICATION: True})
        assert not relevant.evaluate(PIPE) and not relevant.evaluate({**PIPE, REPLICATION: False})


#: States whose legal set of pipelined_multiplexing the registry's own tests pin (``test_declaration``): both values,
#: false only (synchronized LIF, cascaded TTFS with its relays, pruning), and the operand core type.
STATES = {
    "value": {"core_semantics": "mvm"},
    "streamed_lif": {"spiking_family": "lif", "spiking_variant": "streamed", "lif_depth_balancing_relays": False},
    "synchronized_lif": {"spiking_family": "lif", "spiking_variant": "synchronized",
                         "lif_depth_balancing_relays": False},
    "ttfs_quantized": {"spiking_family": "ttfs", "spiking_variant": "quantized"},
    "ttfs_cascaded": {"spiking_family": "ttfs", "spiking_variant": "cascaded"},
    "pruning": {"core_semantics": "mvm", "pruning": True},
    "operand": {"core_semantics": "mvm", "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2,
                                                    "operand": {"write_source": "host"}}]},
}


@pytest.mark.parametrize("state", sorted(STATES))
def test_replication_is_legal_exactly_where_pipelined_multiplexing_is(state):
    cfg = {"allow_scheduling": True, "position_multiplexing": True, **STATES[state]}
    assert tuple(REGISTRY[REPLICATION].legal_values(cfg)) == tuple(REGISTRY["pipelined_multiplexing"].legal_values(cfg))


def test_a_document_declaring_replication_with_pruning_is_a_keyed_error():
    from mimarsinan.config_schema.resolve import resolve_draft

    from unit.config_schema.test_legality import _document

    document = _document(pruning=True, pruning_fraction=0.5)
    document["platform_constraints"] = {"allow_scheduling": True, "position_multiplexing": False, REPLICATION: True}
    rows = [e for e in resolve_draft(document).errors if e["key"] == REPLICATION]
    assert [r["rule_id"] for r in rows] == ["legal_value_set"]
    document["platform_constraints"][REPLICATION] = False
    assert not [e for e in resolve_draft(document).errors if e["key"] == REPLICATION]


class TestTheCostDrawsPartition:
    @pytest.mark.parametrize("key", KEYS)
    def test_each_is_deferred_by_name_with_its_reason(self, key):
        assert key in cd.DEFERRED and cd.DEFERRED_REASONS[key]
        assert key in cd.cost_only_set()

    def test_the_partition_still_is_the_registry_cost_only_set(self):
        assert {p for paths in cd.partition().values() for p in paths} == cd.cost_only_set()


class TestTheCapability:
    def test_undeclared_they_are_absent_from_every_served_dict(self):
        caps = ChipCapabilities.from_platform_constraints(PIPE)
        assert (caps.layer_replication, caps.wavefront_group_cores) == (None, None)
        assert not set(KEYS) & (set(caps.layout_kwargs()) | set(caps.capability_bits()))

    @pytest.mark.parametrize("value", [True, False])
    def test_replication_declared_is_read_and_forwarded(self, value):
        caps = ChipCapabilities.from_platform_constraints({**PIPE, REPLICATION: value})
        assert caps.layer_replication is value and caps.layout_kwargs()[REPLICATION] is value

    def test_the_group_limit_declared_is_read_and_forwarded(self):
        caps = ChipCapabilities.from_platform_constraints({**PIPE, REPLICATION: True, GROUP_CORES: 6})
        assert caps.wavefront_group_cores == 6 and caps.layout_kwargs()[GROUP_CORES] == 6

    def test_replication_needs_pipelined_multiplexing(self):
        with pytest.raises(PositionMultiplexingError, match="layer_replication.*pipelined_multiplexing"):
            ChipCapabilities(allow_scheduling=True, position_multiplexing=True, layer_replication=True)

    def test_false_needs_nothing(self):
        assert ChipCapabilities(allow_scheduling=True, layer_replication=False).layer_replication is False

    @pytest.mark.parametrize("value", [1, "true"])
    def test_a_non_bool_replication_is_refused(self, value):
        with pytest.raises(PositionMultiplexingError, match="layer_replication must be a bool"):
            ChipCapabilities(**PIPE, layer_replication=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0, -2, True, 2.0])
    def test_a_group_limit_below_one_or_not_an_integer_is_refused(self, value):
        with pytest.raises(PositionMultiplexingError, match="wavefront_group_cores must be an integer"):
            ChipCapabilities(**PIPE, layer_replication=True, wavefront_group_cores=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("replication", [None, False])
    def test_a_group_limit_needs_replication(self, replication):
        with pytest.raises(PositionMultiplexingError, match="wavefront_group_cores.*layer_replication"):
            ChipCapabilities(**PIPE, layer_replication=replication, wavefront_group_cores=4)


def test_every_layout_helper_takes_them():
    keys = set(ChipCapabilities(**PIPE, layer_replication=True, wavefront_group_cores=2).layout_kwargs())
    assert set(KEYS) <= keys
    for helper in (compute_mapping_stats, verify_hardware_config, build_layout_plan, collect_noc_fragments,
                   plan_program_passes, plan_segment_passes):
        assert set(KEYS) <= set(inspect.signature(helper).parameters), helper.__name__


class TestTheOnePlannerPlansTheCopies:
    def test_the_builder_builds_a_declared_replication(self, block):
        one = len(V.neural(V.build(block.ir, V.cores(3), True)))
        for kw in ({}, {GROUP_CORES: 2}):
            assert len(V.neural(V.build(block.ir, V.cores(3), True, layer_replication=True, **kw))) <= one

    def test_the_shape_only_plane_plans_it(self):
        """A platform declaring it, through the candidate plane's calls (CL5's and W48's), reaches the planner."""
        from mimarsinan.deployment_record.introspection import scaling_ladder as LD
        from unit.mapping.lanes.lane_planes import ladder, network_of
        from unit.mapping.lanes.test_lane_shape_walk import FIXTURES, _fixture

        platform, program = _fixture("residual", ("L", "E"), multiplexing=True, axons=256, neurons=64, count=3,
                                     passes=1 << 16, coalescing=True, hardware_bias=True)
        network = network_of(program, FIXTURES["residual"][2])
        passes = {}
        for replication in (None, True):
            declared = {**platform, "pipelined_multiplexing": True}
            if replication:
                declared[REPLICATION] = True
            census = LD.shape_census(network, declared)
            scheduled = LD.schedule_program(network, census, declared, ladder().program_facts,
                                            tiles=LD.packed_tiles(census, declared))
            assert scheduled.refusal is None, scheduled.refusal
            passes[replication] = len(scheduled.noc.pass_programs)
        assert passes[True] <= passes[None]

    def test_the_planner_refuses_replication_without_the_wavefront(self, block):
        specs = list(block.ir.layout_softcores)
        with pytest.raises(PositionMultiplexingError, match="layer_replication.*pipelined_multiplexing"):
            plan_program_passes(specs, [LayoutHardCoreType(256, 64, 3)], position_multiplexing=True,
                                layer_replication=True)

    def test_the_planner_refuses_a_group_limit_without_replication(self, block):
        specs = list(block.ir.layout_softcores)
        with pytest.raises(PositionMultiplexingError, match="wavefront_group_cores.*layer_replication"):
            plan_segment_passes(specs, 3, core_types=[LayoutHardCoreType(256, 64, 3)], wavefront_group_cores=2)

    @pytest.mark.parametrize("count", [3, 1])
    def test_false_builds_the_absent_program(self, block, count):
        absent, false = (V.build(block.ir, V.cores(count), True, **kw) for kw in ({}, {REPLICATION: False}))
        assert V.levels_per_pass(false, block.ir) == V.levels_per_pass(absent, block.ir)
        assert [len(s.hard_core_mapping.cores) for s in V.neural(false)] == \
            [len(s.hard_core_mapping.cores) for s in V.neural(absent)]

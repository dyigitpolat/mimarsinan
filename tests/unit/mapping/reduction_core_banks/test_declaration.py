"""[U-L2, B-1] ``reduction_core_banks`` as a declared cost-only decision under position multiplexing: the registry entry,
the reader both planes call, the legal set and the doors. Off by default: an absent key leaves every served
dictionary and every program byte for byte (``test_banks``' controls).

The decision is value-domain: only the integer path runs lanes (U-L: the lane keys are value-domain keys), so a spiking
run declaring it is refused by the key's name at the document and run doors and held dormant by the wizard, one
fixture per firing mode (windowed, streamed and synchronized LIF, TTFS analytical and quantized): no spiking executor
(nevresim, SANA-FE, Loihi, the streamed and rate LIF executors, the TTFS executors) meets a banked reduction core."""

from __future__ import annotations

import copy

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError

from unit.config_schema.test_target_doors import PC, _load, _refused_by_name, _verdicts
from unit.mapping.packing.test_operand_sign_granularity import OPERAND, STATIC, _document
from unit.mapping.pool_lane_multiplexing.test_declaration import POOL_LANES

KEY = "reduction_core_banks"
MUX = {"allow_scheduling": True, "position_multiplexing": True}
STATE = {"allow_scheduling": True, "lif_depth_balancing_relays": False, "cores": [STATIC]}
#: One fixture per spiking firing mode the toolchain runs.
SPIKING = {"windowed_lif": "lif_deepmlp.json", "streamed_lif": "lif_lenet5_novena.json",
           "synchronized_lif": "sync_simplemlp.json", "ttfs_analytical": "ttfs_simplemlp_identity.json",
           "ttfs_quantized": "ttfsq_deepcnn.json"}


class TestTheRegistryEntry:
    def test_it_is_a_cost_only_value_domain_key_of_the_mapping_strategy(self):
        entry = REGISTRY[KEY]
        assert str(entry.decision_class) == COST_ONLY
        assert (entry.section, entry.group, entry.passthrough, entry.domain) == (
            "platform_constraints", "mapping_strategy", True, "value")

    def test_it_has_no_default_so_an_absent_key_changes_nothing(self):
        from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS

        assert KEY not in DEFAULT_PLATFORM_CONSTRAINTS

    def test_it_is_relevant_only_under_position_multiplexing_in_the_value_domain(self):
        entry = REGISTRY[KEY]
        assert entry.relevant.evaluate({**MUX, "core_semantics": "mvm"})
        assert not entry.relevant.evaluate({"allow_scheduling": True, "core_semantics": "mvm"})
        assert not entry.relevant.evaluate(MUX)  # the value domain's own relevance (its domain tag)


class TestTheReader:
    """``declared_reduction_banks``: the one reader of the fusion (builder plane) and the shape-only walk."""

    @staticmethod
    def _read(platform):
        from mimarsinan.mapping.lanes.ir.reduction_banks import declared_reduction_banks

        return declared_reduction_banks(platform)

    @pytest.mark.parametrize("platform,expected", [({}, False), ({**MUX, KEY: False}, False), ({**MUX, KEY: True}, True),
                                                   ({KEY: False}, False)], ids=["absent", "false", "true", "false_alone"])
    def test_it_reads_the_declaration(self, platform, expected):
        assert self._read(platform) is expected

    def test_true_needs_position_multiplexing(self):
        with pytest.raises(PositionMultiplexingError, match=f"{KEY}.*position_multiplexing"):
            self._read({"allow_scheduling": True, KEY: True})

    def test_a_non_bool_is_refused_by_name(self):
        with pytest.raises(PositionMultiplexingError, match=f"{KEY} must be a bool"):
            self._read({**MUX, KEY: 1})


class TestTheLegalSet:
    """True is legal exactly where position multiplexing may be true: the decision shapes what one copy serves."""

    def test_beside_executed_pool_lanes_it_takes_both_values(self):
        assert tuple(REGISTRY[KEY].legal_values({**STATE, **POOL_LANES, **MUX})) == (False, True)

    @pytest.mark.parametrize("extra", [{"pruning": True}, {"lif_depth_balancing_relays": True},
                                       {"cores": [STATIC, OPERAND]}], ids=["pruning", "relays", "operand_core_type"])
    def test_where_position_multiplexing_admits_only_false_so_does_it(self, extra):
        state = {**STATE, **POOL_LANES, **MUX, **extra}
        assert tuple(REGISTRY["position_multiplexing"].legal_values(state)) == (False,)
        assert tuple(REGISTRY[KEY].legal_values(state)) == (False,)


class TestTheDoors:
    @pytest.mark.parametrize("pipelined", [None, True], ids=["one_copy", "pipelined"])
    def test_a_pool_lane_chip_declaring_it_on_the_one_copy_arm_passes_every_door(self, tmp_path, pipelined):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **MUX, **POOL_LANES, KEY: True}
        if pipelined is not None:
            doc[PC]["pipelined_multiplexing"] = pipelined
        assert _verdicts(doc, tmp_path) == ([], [], None)
        from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

        from unit.config_schema.test_target_doors import _run

        resolved = build_platform_constraints_resolved(_run(doc, tmp_path))
        assert resolved.get(KEY) is True, "the run reads the declaration (Soft Core Mapping's platform)"

    def test_with_pruning_beside_it_every_door_refuses_it_by_name(self, tmp_path):
        doc = _document([STATIC])
        doc["deployment_parameters"] = {**doc["deployment_parameters"], "pruning": True, "pruning_fraction": 0.5}
        doc[PC] = {**doc[PC], **MUX, **POOL_LANES, KEY: True}
        _refused_by_name(doc, tmp_path, KEY)

    @pytest.mark.parametrize("fixture", sorted(SPIKING.values()), ids=sorted(SPIKING))
    def test_a_spiking_run_declaring_it_never_runs_it(self, tmp_path, fixture):
        from mimarsinan.config_schema.resolve import resolve_draft

        doc = _load(fixture)
        doc[PC] = {**doc[PC], KEY: True}
        document, _wizard, run = _verdicts(doc, tmp_path)
        assert any(m.startswith(KEY) for m in document), document
        assert isinstance(run, ToolchainRefusal) and KEY in str(run), repr(run)
        assert KEY in resolve_draft(copy.deepcopy(doc)).dormant

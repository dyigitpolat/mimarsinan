"""[U-F4] pipelined_multiplexing as a declared cost-only decision of the pass planner: registry, capability, legal set
and forwarding. Off by default: an absent key leaves every served dictionary and every program byte for byte."""

from __future__ import annotations

import inspect

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, PositionMultiplexingError
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier import verify_hardware_config

KEY = "pipelined_multiplexing"
MUX = {"allow_scheduling": True, "position_multiplexing": True}


class TestTheRegistryEntry:
    def test_it_is_a_cost_only_platform_key_of_the_mapping_strategy(self):
        entry = REGISTRY[KEY]
        assert str(entry.decision_class) == COST_ONLY
        assert (entry.section, entry.group, entry.passthrough) == ("platform_constraints", "mapping_strategy", True)

    def test_it_has_no_default_so_an_absent_key_changes_nothing(self):
        from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS

        assert KEY not in DEFAULT_PLATFORM_CONSTRAINTS

    def test_it_is_relevant_only_under_position_multiplexing(self):
        entry = REGISTRY[KEY]
        assert entry.relevant.evaluate(MUX)
        assert not entry.relevant.evaluate({"allow_scheduling": True})


class TestTheCapability:
    def test_undeclared_it_is_absent_from_every_served_dict(self):
        caps = ChipCapabilities.from_platform_constraints(MUX)
        assert caps.pipelined_multiplexing is None
        assert KEY not in caps.layout_kwargs()
        assert KEY not in caps.capability_bits()

    @pytest.mark.parametrize("value", [True, False])
    def test_declared_it_is_read_and_forwarded(self, value):
        caps = ChipCapabilities.from_platform_constraints({**MUX, KEY: value})
        assert caps.pipelined_multiplexing is value
        assert caps.layout_kwargs()[KEY] is value

    def test_it_needs_position_multiplexing(self):
        with pytest.raises(PositionMultiplexingError, match="pipelined_multiplexing.*position_multiplexing"):
            ChipCapabilities(allow_scheduling=True, pipelined_multiplexing=True)

    def test_false_needs_nothing(self):
        assert ChipCapabilities(allow_scheduling=True, pipelined_multiplexing=False).pipelined_multiplexing is False

    def test_a_non_bool_is_refused(self):
        with pytest.raises(PositionMultiplexingError, match="pipelined_multiplexing must be a bool"):
            ChipCapabilities(allow_scheduling=True, position_multiplexing=True,
                             pipelined_multiplexing=1)  # type: ignore[arg-type]


def test_every_layout_helper_takes_it():
    keys = set(ChipCapabilities(allow_scheduling=True, position_multiplexing=True,
                                pipelined_multiplexing=True).layout_kwargs())
    for helper in (compute_mapping_stats, verify_hardware_config, build_layout_plan, collect_noc_fragments):
        assert keys <= set(inspect.signature(helper).parameters), helper.__name__


class TestTheLegalSet:
    """True is legal only where position multiplexing itself is, and only under the semantics whose executors compose
    a wavefront pass: the value domain, streamed LIF and every TTFS variant (``test_programs``); windowed LIF, the
    synchronized variant, re-times each level as its own unit and refuses it by name, so the declaration admits only
    false there."""

    def test_the_value_domain_admits_both(self):
        assert tuple(REGISTRY[KEY].legal_values({**MUX, "core_semantics": "mvm"})) == (False, True)

    def test_streamed_lif_without_relays_admits_both(self):
        state = {**MUX, "spiking_family": "lif", "spiking_variant": "streamed", "lif_depth_balancing_relays": False}
        assert tuple(REGISTRY[KEY].legal_values(state)) == (False, True)

    def test_synchronized_lif_admits_only_false(self):
        """The windowed LIF semantics is the synchronized variant (lif_sync: per-hop re-timed level units)."""
        state = {**MUX, "spiking_family": "lif", "spiking_variant": "synchronized",
                 "lif_depth_balancing_relays": False}
        assert tuple(REGISTRY[KEY].legal_values(state)) == (False,)

    @pytest.mark.parametrize("variant", ["analytical", "quantized", "synchronized"])
    def test_every_ttfs_variant_admits_both(self, variant):
        state = {**MUX, "spiking_family": "ttfs", "spiking_variant": variant}
        assert tuple(REGISTRY[KEY].legal_values(state)) == (False, True)

    def test_the_cascaded_variant_admits_both_only_with_its_relays_declared_off(self):
        """[U-TGAUGE5] Its recipe arms the depth-balancing relays (a per-cycle cascade), which position multiplexing
        refuses."""
        state = {**MUX, "spiking_family": "ttfs", "spiking_variant": "cascaded"}
        assert tuple(REGISTRY[KEY].legal_values(state)) == (False,)
        assert tuple(REGISTRY[KEY].legal_values({**state, "lif_depth_balancing_relays": False})) == (False, True)

    def test_where_position_multiplexing_admits_only_false_so_does_it(self):
        assert tuple(REGISTRY[KEY].legal_values({**MUX, "core_semantics": "mvm", "pruning": True})) == (False,)

    def test_a_document_declaring_it_with_pruning_is_a_keyed_error(self):
        from mimarsinan.config_schema.resolve import resolve_draft

        from unit.config_schema.test_legality import _document

        document = _document(pruning=True, pruning_fraction=0.5)
        document["platform_constraints"] = {"allow_scheduling": True, "position_multiplexing": False, KEY: True}
        rows = [e for e in resolve_draft(document).errors if e["key"] == KEY]
        assert [r["rule_id"] for r in rows] == ["legal_value_set"]
        document["platform_constraints"][KEY] = False
        assert not [e for e in resolve_draft(document).errors if e["key"] == KEY]

"""[W02] position_multiplexing as a declared cost-only mapping decision: registry, capability and forwarding."""

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


class TestTheRegistryEntry:
    def test_it_is_a_cost_only_platform_key_of_the_mapping_strategy(self):
        entry = REGISTRY["position_multiplexing"]
        assert str(entry.decision_class) == COST_ONLY
        assert (entry.section, entry.group, entry.passthrough) == ("platform_constraints", "mapping_strategy", True)

    def test_it_has_no_default_so_an_absent_key_changes_nothing(self):
        from mimarsinan.config_schema.defaults import DEFAULT_PLATFORM_CONSTRAINTS

        assert "position_multiplexing" not in DEFAULT_PLATFORM_CONSTRAINTS


class TestTheCapability:
    def test_undeclared_it_is_absent_from_every_served_dict(self):
        caps = ChipCapabilities.from_platform_constraints({"allow_scheduling": True})
        assert caps.position_multiplexing is None
        assert "position_multiplexing" not in caps.layout_kwargs()
        assert "position_multiplexing" not in caps.capability_bits()

    @pytest.mark.parametrize("value", [True, False])
    def test_declared_it_is_read_and_forwarded(self, value):
        caps = ChipCapabilities.from_platform_constraints({"allow_scheduling": True, "position_multiplexing": value})
        assert caps.position_multiplexing is value
        assert caps.layout_kwargs()["position_multiplexing"] is value

    def test_it_needs_scheduling(self):
        with pytest.raises(PositionMultiplexingError, match="needs allow_scheduling"):
            ChipCapabilities(position_multiplexing=True)

    def test_it_is_not_planned_with_a_residency_plan(self):
        with pytest.raises(PositionMultiplexingError, match="residency_plan"):
            ChipCapabilities(allow_scheduling=True, position_multiplexing=True, residency_plan={"step": "floor"})

    def test_a_non_bool_is_refused(self):
        with pytest.raises(PositionMultiplexingError, match="must be a bool"):
            ChipCapabilities(allow_scheduling=True, position_multiplexing=1)  # type: ignore[arg-type]


def test_every_layout_helper_takes_it():
    keys = set(ChipCapabilities(allow_scheduling=True, position_multiplexing=True).layout_kwargs())
    for helper in (compute_mapping_stats, verify_hardware_config, build_layout_plan, collect_noc_fragments):
        assert keys <= set(inspect.signature(helper).parameters), helper.__name__


class TestTheCombinationsTheShapeOnlyPlaneCannotSee:
    """[W02 fix round, F2 and F3] The shape-only plane plans from the layout walk's specs, so it cannot see which bank
    instances IR elimination compacts into owned cores (they keep no shared copy on the executed plane) or the relay
    cores depth balancing inserts (they carry no layout record): position_multiplexing with either is refused by name
    at the declaration, so both planes refuse it, and by the builder on an IR that carries either."""

    @pytest.mark.parametrize("key", ["pruning", "lif_depth_balancing_relays"])
    def test_the_declaration_admits_only_false_with_it(self, key):
        entry = REGISTRY["position_multiplexing"]
        state = {"allow_scheduling": True, "lif_depth_balancing_relays": False}  # relays off unless the key is them
        assert tuple(entry.legal_values({**state, key: True})) == (False,)
        assert tuple(entry.legal_values({**state, key: False})) == (False, True)

    def test_a_document_declaring_it_with_pruning_is_a_keyed_error(self):
        from mimarsinan.config_schema.resolve import resolve_draft

        from unit.config_schema.test_legality import _document

        document = _document(pruning=True, pruning_fraction=0.5)
        document["platform_constraints"] = {"allow_scheduling": True, "position_multiplexing": True}
        rows = [e for e in resolve_draft(document).errors if e["key"] == "position_multiplexing"]
        assert [r["rule_id"] for r in rows] == ["legal_value_set"]
        document["platform_constraints"]["position_multiplexing"] = False
        assert not [e for e in resolve_draft(document).errors if e["key"] == "position_multiplexing"]

    @pytest.mark.parametrize("variant", ["streamed", "synchronized"])
    def test_a_lif_run_leaving_the_relays_to_its_recipe_admits_only_false(self, variant):
        """Carried minor N1 (pack verify): the legal set read the relays off the declarations alone, while a LIF run
        that leaves them undeclared takes its conversion recipe's value, which arms them (``recipe_fold``). Such a
        document was admitted with position_multiplexing true; the executed plane then refused by name where relays
        were inserted while the shape-only plane planned. The set now judges the relays the run derives."""
        entry = REGISTRY["position_multiplexing"]
        state = {"allow_scheduling": True, "spiking_family": "lif", "spiking_variant": variant}
        assert tuple(entry.legal_values(state)) == (False,)
        assert tuple(entry.legal_values({**state, "lif_depth_balancing_relays": False})) == (False, True)

    @pytest.mark.parametrize("variant", ["analytical", "quantized", "synchronized"])
    def test_a_ttfs_run_whose_recipe_arms_no_relays_admits_both(self, variant):
        entry = REGISTRY["position_multiplexing"]
        state = {"allow_scheduling": True, "spiking_family": "ttfs", "spiking_variant": variant}
        assert tuple(entry.legal_values(state)) == (False, True)

    def test_a_cascaded_ttfs_run_leaving_the_relays_to_its_recipe_admits_only_false(self):
        """[U-TGAUGE5] The cascaded schedule is a per-cycle cascade, whose recipe arms the relays."""
        entry = REGISTRY["position_multiplexing"]
        state = {"allow_scheduling": True, "spiking_family": "ttfs", "spiking_variant": "cascaded"}
        assert tuple(entry.legal_values(state)) == (False,)
        assert tuple(entry.legal_values({**state, "lif_depth_balancing_relays": False})) == (False, True)
        assert tuple(entry.legal_values({**state, "lif_depth_balancing_relays": True})) == (False,)

    def test_a_value_domain_run_folds_no_relays_and_admits_both(self):
        entry = REGISTRY["position_multiplexing"]
        assert tuple(entry.legal_values({"allow_scheduling": True, "core_semantics": "mvm"})) == (False, True)

    def test_a_lif_document_leaving_the_relays_undeclared_is_a_keyed_error(self):
        from mimarsinan.config_schema.resolve import resolve_draft

        from unit.config_schema.test_legality import _document

        document = _document(spiking_family="lif", spiking_variant="streamed")
        document["platform_constraints"] = {"allow_scheduling": True, "position_multiplexing": True}
        rows = [e for e in resolve_draft(document).errors if e["key"] == "position_multiplexing"]
        assert [r["rule_id"] for r in rows] == ["legal_value_set"]
        document["deployment_parameters"]["lif_depth_balancing_relays"] = False
        assert not [e for e in resolve_draft(document).errors if e["key"] == "position_multiplexing"]

    def test_the_builder_refuses_it_on_an_ir_whose_bank_instances_elimination_compacted(self):
        from unit.chip_simulation.value_run.integer_programs import _randomize_bn, _ResNet, build_integer_program

        from . import vehicles as V

        cores = [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}]
        pruned = build_integer_program(_randomize_bn(_ResNet((8, 16), blocks=1)), (3, 8, 8), num_classes=10,
                                       prune={0: [0, 3], 2: [1, 5, 6]})
        V.build(pruned.ir, cores, False)  # replicated: builds
        with pytest.raises(PositionMultiplexingError, match="elimination"):
            V.build(pruned.ir, cores, True)

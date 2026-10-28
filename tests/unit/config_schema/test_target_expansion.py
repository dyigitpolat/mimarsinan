"""The target expansion (plan CH-B0, acceptance 1, 3 and 10): bundle form against hand form, conflicts,
overrides with their faithfulness rule, unknown names, idempotency."""

import copy
import dataclasses
import json
import os

import numpy as np
import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.mapping.platform.target_bundles import (
    DP,
    PC,
    UnknownTargetError,
    load_bundles,
    target_bundles,
    target_departures,
)
from mimarsinan.mapping.platform.target_expansion import (
    TargetConflictError,
    TargetDeclarationError,
    TargetOverrideError,
    expand_target,
    expand_target_document,
    same_declaration,
    target_assignments,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs")
#: A spiking and a value-domain fixture: each bundle is declared over the one of its core semantics.
FIXTURE_FOR = {"spiking": "sync_simplemlp.json", "mvm": "mvm_aq_lenet5.json"}
#: The fixture keys a bundle's geometry replaces: the core grid, its weight width and the two derived maxima.
GEOMETRY_KEYS = ("cores", "weight_bits", "max_axons", "max_neurons")
#: The keys only the bundle form carries.
TARGET_KEYS = ("target", "target_claim_class", "target_faithfulness_class")
#: The five hand-written declarations, written out: the registry geometries (``imc_platforms_literature.py``:
#: truenorth_like, loihi_dense_equiv_128x1024, isaac_like), ODIN's from frenkel2019odin (256 x 256 physical
#: rows with one sign per axon row, CL-SG; 4-bit synapses, no bias lane, no coalescing or neuron splitting, as
#: the odin-fpga-infra declaration states them, f08c8efa), the 22 nm point's from its profile's validity text (256 x 256, 8-bit; 1000 cores, the
#: default grid's count), TrueNorth's floorplan from its profile's validity text (4096 cores, 64 x 64 mesh), and
#: each profile's core semantics.
HAND = {
    "odin": {DP: {"core_semantics": "spiking"}, PC: {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False,
                   "weight_sign_granularity": "per_axon"}], "weight_bits": 4,
        "allow_coalescing": False, "allow_neuron_splitting": False, "platform_physics_profile": "odin"}},
    "truenorth": {DP: {"core_semantics": "spiking"}, PC: {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 4096, "has_bias": True}], "weight_bits": 1,
        "platform_physics_profile": "truenorth", "cores_per_tile": 1, "tile_grid_rows": 64, "tile_grid_cols": 64,
        "allow_coalescing": False, "allow_neuron_splitting": False}},
    "loihi": {DP: {"core_semantics": "spiking"}, PC: {
        "cores": [{"max_axons": 128, "max_neurons": 1024, "count": 128, "has_bias": False}], "weight_bits": 8,
        "platform_physics_profile": "loihi", "allow_coalescing": False, "allow_neuron_splitting": True}},
    "isaac_like": {DP: {"core_semantics": "mvm"}, PC: {
        "cores": [{"max_axons": 128, "max_neurons": 128, "count": 16128, "has_bias": False}], "weight_bits": 16,
        "platform_physics_profile": "isaac_like", "allow_coalescing": True, "allow_neuron_splitting": True,
        "activation_bits": 16,  # [U-C C-5] the chip's input width (16 voltage levels in sequence)
        # Wave 2 W08: the value-domain bundles declare the lane subset with its lane block (one lane per 64 columns).
        "onchip_arithmetic": {"classes": ["L", "E", "P"], "lanes_per_core": 2, "lane_block": "ng45_os_lane"}}},
    "generic_estimated_22nm": {DP: {"core_semantics": "mvm"}, PC: {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}], "weight_bits": 8,
        "platform_physics_profile": "generic_estimated_22nm", "allow_coalescing": True,
        "allow_neuron_splitting": True,
        "onchip_arithmetic": {"classes": ["L", "E", "P"], "lanes_per_core": 4, "lane_block": "ng45_os_lane"}}},
}
PHYSICS = tuple(sorted(HAND))
ODIN_CORES = {"value": [{"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": False,
                         "weight_sign_granularity": "per_axon"}],
              "note": "a 64-core array of the ODIN core for a pool-sized study"}


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _forms(name):
    """``(hand, bundle)``: a fixture with the literal hand declaration ``HAND[name]``, and with ``target`` instead."""
    doc = _load(FIXTURE_FOR[HAND[name][DP]["core_semantics"]])
    written = {key for section in HAND[name].values() for key in section}
    dp = {k: v for k, v in doc["deployment_parameters"].items() if k not in written}
    pc = {k: v for k, v in doc["platform_constraints"].items() if k not in GEOMETRY_KEYS and k not in written}
    bits = target_bundles()[name].values.get("weight_bits")
    if bits is not None and bits < REGISTRY["weight_bits"].applied_schema.low:
        dp["weight_quantization"] = False  # key amendment K1: no nonzero level under weight quantization
    hand = {DP: {**dp, **copy.deepcopy(HAND[name][DP])}, PC: {**pc, **copy.deepcopy(HAND[name][PC])}}
    return hand, {DP: dp, PC: {**pc, "target": name}}, doc


def _flat(form):
    return merge_pipeline_config(copy.deepcopy(form[DP]), copy.deepcopy(form[PC]))


def _plan(flat, doc):
    config = dict(flat)
    config["input_shape"], config["num_classes"] = (1, 28, 28), 10
    config["input_size"], config["device"] = int(np.prod(config["input_shape"])), "cpu"
    assert doc["data_provider_name"] == "MNIST_DataProvider"
    return DeploymentPlan.resolve(config)


def _without(mapping, keys=TARGET_KEYS):
    return {k: v for k, v in mapping.items() if k not in keys}


class TestBundleFormEqualsHandForm:
    """Acceptance 1: resolved platform and DeploymentPlan equal the hand-written declaration's."""

    @pytest.mark.parametrize("name", PHYSICS)
    def test_the_bundle_sets_exactly_the_hand_written_keys(self, name):
        values = target_bundles()[name].values
        assert set(values) == {key for section in HAND[name].values() for key in section}
        for section in (DP, PC):
            for key, value in HAND[name][section].items():
                assert REGISTRY[key].section == section and same_declaration(values[key], value), key

    @pytest.mark.parametrize("name", PHYSICS)
    def test_resolved_platform_and_plan(self, name):
        hand, bundled, doc = _forms(name)
        flat_hand, flat_bundle = _flat(hand), _flat(bundled)
        assert _without(flat_bundle) == flat_hand
        pcfg_hand = build_platform_constraints_resolved(flat_hand)
        pcfg_bundle = build_platform_constraints_resolved(flat_bundle)
        assert _without(pcfg_bundle) == pcfg_hand
        assert pcfg_bundle["platform_physics_resolved"] == pcfg_hand["platform_physics_resolved"] is not None
        # Key amendment K2 (review round 1): a float run applies no weight grid, which the label names; the key's
        # every-run readers (payload sizes, pricing, the record) read the bundle's weight_bits.
        inert = ["weight grid not applied"] if bundled[DP].get("weight_quantization") is False else []
        assert {k: pcfg_bundle[k] for k in TARGET_KEYS} == {
            "target": name, "target_claim_class": target_bundles()[name].claim_class,
            "target_faithfulness_class": target_bundles()[name].faithfulness.with_departures(inert).label()}
        plan_hand, plan_bundle = _plan(flat_hand, doc), _plan(flat_bundle, doc)
        for field in dataclasses.fields(DeploymentPlan):
            a, b = getattr(plan_hand, field.name), getattr(plan_bundle, field.name)
            assert (a == _without(b)) if field.name == "config" else (a == b), field.name


class TestConflictsAndOverrides:
    """Acceptance 3 and 10."""

    def test_a_differing_explicit_key_refuses_by_name(self):
        with pytest.raises(TargetConflictError, match=r"weight_bits=8 differs from target='odin'.*weight_bits=4"):
            expand_target({}, {"target": "odin", "weight_bits": 8}, write_derived=False)
        with pytest.raises(TargetConflictError, match="core_semantics='mvm'"):
            expand_target({"core_semantics": "mvm"}, {"target": "loihi"}, write_derived=False)

    def test_an_equal_explicit_value_is_accepted(self):
        once = expand_target({}, {"target": "odin"}, write_derived=True)
        assert expand_target({"core_semantics": "spiking"}, {"target": "odin", "weight_bits": 4},
                             write_derived=True) == once

    def test_an_equal_explicit_value_is_kept_as_declared(self):
        """Nothing is written for a key the declarations hold: the registry judges the value as written."""
        written = target_assignments({"core_semantics": "spiking"}, {"target": "odin", "weight_bits": 4},
                                     write_derived=False)
        assert "weight_bits" not in written[PC] and "core_semantics" not in written[DP]
        assert written[PC]["cores"] == target_bundles()["odin"].values["cores"]

    @pytest.mark.parametrize("value", [4.0, True, "4"])
    def test_a_value_of_another_type_is_another_declaration(self, value):
        with pytest.raises(TargetConflictError, match=rf"weight_bits={value!r} differs from target='odin'"):
            expand_target({}, {"target": "odin", "weight_bits": value}, write_derived=False)

    def test_same_declaration_is_the_json_value(self):
        assert same_declaration([{"a": 1, "b": False}], [{"b": False, "a": 1}])
        assert not same_declaration(4, 4.0) and not same_declaration(1, True) and not same_declaration(0, False)
        assert not same_declaration([{"count": 1}], [{"count": 1.0}])

    def test_an_override_that_departs_from_nothing_refuses_by_name(self):
        overrides = {"weight_bits": {"value": 4, "note": "the bundle's own width"}}
        with pytest.raises(TargetOverrideError, match=r"target_overrides\.weight_bits\.value equals the weight_bits"):
            expand_target({}, {"target": "odin", "target_overrides": overrides}, write_derived=False)
        overrides["weight_bits"]["value"] = 4.0  # another JSON value: a departure the registry then judges
        assert expand_target({}, {"target": "odin", "target_overrides": overrides}, write_derived=False)[1][
            "weight_bits"] == 4.0

    @pytest.mark.parametrize("dp,pc,named", [
        ({"target": "odin"}, {}, "deployment_parameters.target"),
        ({"target_overrides": {}}, {"target": "odin"}, "deployment_parameters.target_overrides"),
        ({"target_claim_class": "synthetic"}, {}, "deployment_parameters.target_claim_class"),
        ({}, {"target": "odin", "core_semantics": "spiking"}, "platform_constraints.core_semantics"),
    ], ids=lambda x: str(x)[:40])
    def test_a_key_outside_its_section_refuses_by_name(self, dp, pc, named):
        with pytest.raises(TargetDeclarationError, match=named):
            expand_target(dp, pc, write_derived=False)

    def test_a_bundle_key_no_reader_reads_is_the_judge_s_to_refuse(self):
        """A deployment-parameters copy of a platform key the platform default layer replaces is read by no reader
        (the expansion reads the run's layout, so it never takes it for a declaration): the one judge refuses it by
        name at the merge, with the message the document door states (review round 1)."""
        assert expand_target({"weight_bits": 4}, {"target": "odin"}, write_derived=False)[1]["weight_bits"] == 4
        with pytest.raises(ToolchainRefusal, match=r"deployment_parameters\.weight_bits: weight_bits is a "
                                                   r"platform_constraints key, and nothing reads it here"):
            merge_pipeline_config({"weight_bits": 4}, {"target": "odin"})

    def test_a_bundle_key_outside_its_section_is_the_bundle_s_concern_only(self):
        """Without a target, a misplaced key is the base's reading (the run's flat merge), not the expansion's."""
        assert expand_target({"weight_bits": 4}, {}, write_derived=False) == ({"weight_bits": 4}, {})

    @pytest.mark.parametrize("pc,named", [
        ({"target_claim_class": "synthetic"}, "no target is declared"),
        ({"target": "odin", "target_claim_class": "synthetic"}, "'quarantined', which target='odin' derives"),
        ({"target": "odin", "target_faithfulness_class": "exact"}, "target_faithfulness_class='exact' differs"),
    ], ids=lambda x: str(x)[:40])
    def test_a_declared_label_refuses_where_labels_are_written(self, pc, named):
        with pytest.raises(TargetDeclarationError, match=named):
            expand_target({}, pc, write_derived=True)
        expand_target({}, pc, write_derived=False)  # a document's labels are the document door's to refuse

    def test_an_override_with_a_note_is_accepted_and_reported_beside_the_bundle_value(self):
        dp, pc = expand_target({}, {"target": "odin", "target_overrides": {"cores": ODIN_CORES}},
                               write_derived=True)
        assert pc["cores"] == ODIN_CORES["value"]
        assert pc["target_faithfulness_class"] == (
            "approximate: per-cycle soma law; unbounded membrane for the 8-bit saturating one; cores declared")
        pcfg = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
        assert pcfg["target_overrides"] == {"cores": ODIN_CORES}
        assert target_departures(pcfg) == ({
            "key": "cores", "bundle": target_bundles()["odin"].values["cores"],
            "declared": ODIN_CORES["value"], "note": ODIN_CORES["note"]},)

    def test_an_override_lowers_an_exact_bundle_to_approximate_naming_the_key(self, tmp_path):
        (tmp_path / "exact_chip.json").write_text(json.dumps({
            "name": "exact_chip", "geometry": "recom_64x64_256", "keys": {}, "faithfulness": {"class": "exact"}}))
        bundles = load_bundles(tmp_path)
        _, plain = expand_target({}, {"target": "exact_chip"}, write_derived=True, bundles=bundles)
        assert plain["target_faithfulness_class"] == "exact"
        overrides = {"cores": {"value": [{"max_axons": 64, "max_neurons": 64, "count": 512}], "note": "twice"}}
        _, declared = expand_target({}, {"target": "exact_chip", "target_overrides": overrides},
                                    write_derived=True, bundles=bundles)
        assert declared["target_faithfulness_class"] == "approximate: cores declared"

    def test_an_explicit_key_must_equal_its_override(self):
        cores = copy.deepcopy(ODIN_CORES["value"])
        pc = {"target": "odin", "target_overrides": {"cores": ODIN_CORES}, "cores": cores}
        assert expand_target({}, pc, write_derived=False)[1]["cores"] == cores
        pc["cores"] = target_bundles()["odin"].values["cores"]
        with pytest.raises(TargetConflictError, match=r"target_overrides\.cores\.value"):
            expand_target({}, pc, write_derived=False)

    @pytest.mark.parametrize("pc,error,named", [
        ({"target": "odin", "target_overrides": {"simulation_steps": {"value": 4, "note": "n"}}},
         TargetOverrideError, "sets no such key"),
        ({"target_overrides": {"cores": ODIN_CORES}}, TargetOverrideError, "without a target"),
        ({"target": "odin", "target_overrides": {"cores": {"value": []}}}, DeclarationValueError,
         r"target_overrides\.cores\.note"),
        ({"target": 7}, DeclarationValueError, "target"),
        ({"target": "no_such_chip"}, UnknownTargetError, "no_such_chip"),
    ], ids=lambda x: str(x)[:30])
    def test_each_defect_refuses_by_name(self, pc, error, named):
        with pytest.raises(error, match=named) as raised:
            expand_target({}, pc, write_derived=False)
        assert isinstance(raised.value, ToolchainRefusal)

    def test_a_key_the_declarations_hand_to_another_concern_refuses(self):
        """Under a hardware search the co-search provides ``cores`` (its registry relevance), so a bundle
        setting them would name a chip the run does not deploy."""
        with pytest.raises(TargetConflictError, match=r"target='odin' sets cores, which co_search provides"):
            expand_target({"hw_config_mode": "search"}, {"target": "odin"}, write_derived=False)
        assert expand_target({"hw_config_mode": "fixed"}, {"target": "odin"}, write_derived=False)[1]["cores"]

    def test_an_unknown_name_lists_the_known_names(self):
        with pytest.raises(UnknownTargetError) as raised:
            expand_target({}, {"target": "odin2"}, write_derived=True)
        for name in target_bundles():
            assert repr(name) in str(raised.value)

    def test_the_labels_are_written_on_request_only(self):
        assert "target_claim_class" not in expand_target({}, {"target": "loihi"}, write_derived=False)[1]
        assert expand_target({}, {"target": "loihi"}, write_derived=True)[1]["target_claim_class"] == "curve-only"


class TestIdempotentAndInert:
    @pytest.mark.parametrize("name", sorted(target_bundles()))
    @pytest.mark.parametrize("write_derived", [False, True])
    def test_twice_equals_once(self, name, write_derived):
        pcs = [{"target": name}]
        if name == "odin":
            pcs.append({"target": name, "target_overrides": {"cores": ODIN_CORES}})
        for pc in pcs:
            once = expand_target({}, pc, write_derived=write_derived)
            assert expand_target(*once, write_derived=write_derived) == once
            assert expand_target(*once, write_derived=True) == expand_target({}, pc, write_derived=True)

    def test_a_document_expands_in_its_flat_platform(self):
        _, bundled, doc = _forms("odin")
        document = {**doc, DP: bundled[DP], PC: dict(bundled[PC])}
        expanded = expand_target_document(document)
        assert expanded[PC]["cores"] == target_bundles()["odin"].values["cores"]
        assert expanded[DP]["core_semantics"] == "spiking"
        assert "target_claim_class" not in expanded[PC]
        assert expand_target_document(expanded) == expanded
        assert document[PC] == bundled[PC]  # the input is not changed

    @pytest.mark.parametrize("shape,path", [
        ({"mode": "user", "user": {"target": "odin"}}, "platform_constraints.user.target"),
        ({"mode": "auto", "auto": {"fixed": {"target": "odin"}}}, "platform_constraints.auto.fixed.target"),
        ({"mode": "user", "user": {"target_overrides": {"cores": ODIN_CORES}}},
         "platform_constraints.user.target_overrides"),
        ({"mode": "user", "user": {"target_faithfulness_class": "exact"}},
         "platform_constraints.user.target_faithfulness_class"),
        ({"mode": "user", "user": {}, "target": "odin"}, "platform_constraints.target"),
    ], ids=["user", "auto_fixed", "overrides", "label", "beside_the_mode"])
    def test_a_nested_platform_declaration_refuses_by_name(self, shape, path):
        """The run door does not read the nested user/auto shapes, so neither does any other door: the document
        reading and the section reading (the run door's, the merge's) refuse the same path."""
        for expand in (lambda: expand_target_document({DP: {}, PC: copy.deepcopy(shape)}),
                       lambda: expand_target({}, copy.deepcopy(shape), write_derived=False),
                       lambda: expand_target({}, copy.deepcopy(shape), write_derived=True)):
            with pytest.raises(TargetDeclarationError, match=rf"{path}: \w+ is read in the flat platform_constraints") \
                    as raised:
                expand()
            remedy = "remove it, the expansion derives it" if path.endswith("_class") else "declare it there"
            assert str(raised.value).endswith(remedy), str(raised.value)

    def test_without_a_target_nothing_moves(self):
        doc = _load("mvm_aq_lenet5.json")
        before = copy.deepcopy(doc)
        assert expand_target_document(doc) is doc
        dp, pc = expand_target(doc[DP], doc[PC], write_derived=True)
        assert (dp, pc) == (doc[DP], doc[PC]) and doc == before
        assert not set(TARGET_KEYS) & set(merge_pipeline_config(doc[DP], doc[PC]))

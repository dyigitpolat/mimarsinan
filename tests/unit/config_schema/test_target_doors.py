"""Every door gives one verdict on a target document (plan CH-B0, review round 1): the document door
(``validate_deployment_config``), the wizard (``resolve_draft``) and the run door (``parse_deployment_config``, then
``merge_pipeline_config``, then ``build_platform_constraints_resolved``)."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

import mimarsinan.mapping.platform.target_bundles as bundle_module
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY, Relevance
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.mapping.platform.target_bundles import load_bundles, target_bundles
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import identity_from_pipeline
from mimarsinan.pipelining.session import parse_deployment_config
from unit.mapping.platform.hand_descriptions import described

DP, PC = "deployment_parameters", "platform_constraints"
_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs")
SPIKING, VALUE = "sync_simplemlp.json", "mvm_aq_lenet5.json"
ODIN_CORE = {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False,
             "weight_sign_granularity": "per_axon"}
READOUT_LAW = {"adc_bits": 6, "range": "fixed", "range_value": 4.0, "mult_bits": 16}
LABELS = ("target_claim_class", "target_faithfulness_class")


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _owned(name):
    """The platform keys a bundle sets and the keys derived from them (the core maxima)."""
    keys = set(target_bundles()[name].values)
    return keys | {key for key, entry in REGISTRY.items() if set(entry.derived_from) & keys}


def _plain(fixture=SPIKING):
    """A fixture without its core grid: a document a target can be declared on."""
    doc = _load(fixture)
    doc[PC] = {k: v for k, v in doc[PC].items() if k not in ("cores", "weight_bits", "max_axons", "max_neurons")}
    return doc


def _document(name, fixture=SPIKING):
    """A fixture with ``target: <name>`` in place of every key the bundle sets ([U-CHIPKEYS] in both sections: a
    description now states its neuron law's deployment-parameter keys too)."""
    doc = _load(fixture)
    doc[DP] = {k: v for k, v in doc[DP].items() if k == "core_semantics" or k not in _gives_way(name)}
    doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in _owned(name)}, "target": name}
    return doc


def _gives_way(name):
    """The fixture's deployment keys a target replaces: the keys its bundle sets and, where the bundle states a
    spiking family, the fixture's whole neuron law (its spiking-group keys, such as a TTFS input encoding)."""
    owned = set(target_bundles()[name].values)
    if "spiking_family" in owned:
        owned |= {k for k, e in REGISTRY.items() if e.group == "spiking" and k != "core_semantics"}
    return owned


def _run(doc, tmp_path):
    """The run door's flat config of a document (the refusal propagates)."""
    parsed = parse_deployment_config({**copy.deepcopy(doc), "_working_directory": str(tmp_path)},
                                     data_provider_factory=object())
    dp = dict(parsed.deployment_parameters)
    DeploymentPipeline.apply_preset(parsed.pipeline_mode, dp)
    flat = merge_pipeline_config(dp, copy.deepcopy(parsed.platform_constraints))
    build_platform_constraints_resolved(flat)
    return flat


def _verdicts(doc, tmp_path):
    """``(document messages, wizard messages, run refusal or None)`` of one document."""
    document = validate_deployment_config(copy.deepcopy(doc))
    wizard = [row["message"] for row in resolve_draft(copy.deepcopy(doc)).errors]
    try:
        _run(doc, tmp_path)
        run = None
    except ValueError as exc:
        run = exc
    return document, wizard, run


def _refused_by_name(doc, tmp_path, named):
    document, wizard, run = _verdicts(doc, tmp_path)
    assert any(named in m for m in document), document
    assert any(named in m for m in wizard), wizard
    assert isinstance(run, ToolchainRefusal) and named in str(run), repr(run)
    return document, wizard, run


class TestTypeStrictDeclarations:
    """A declaration equals the bundle's only as the same JSON value: the registry's INT and BOOL rules, which
    refuse 4.0 for an int and 0 for a bool, are not bypassed by Python's 4.0 == 4 and 0 == False."""

    @pytest.mark.parametrize("name,key,value", [
        ("odin", "weight_bits", 4.0),
        ("truenorth", "weight_bits", True),
        ("odin", "cores", [{**ODIN_CORE, "has_bias": 0}]),
        ("odin", "cores", [{**ODIN_CORE, "max_axons": 256.0, "count": 1.0}]),
    ], ids=["int_as_float", "int_as_bool", "bool_as_int", "ints_as_floats"])
    def test_a_type_differing_value_refuses_by_name_at_every_door(self, tmp_path, name, key, value):
        doc = _document(name)
        doc[PC][key] = value
        _refused_by_name(doc, tmp_path, key)

    def test_the_same_values_are_accepted_at_every_door(self, tmp_path):
        doc = _document("odin")
        doc[PC].update(weight_bits=4, cores=[dict(ODIN_CORE)], allow_coalescing=False)
        doc[DP]["core_semantics"] = "spiking"
        assert _verdicts(doc, tmp_path) == ([], [], None)


class TestTargetKeysAreReadInTheirSection:
    """``target``, ``target_overrides`` and the labels are platform keys, and a bundle's keys are read in their
    registry sections: a declaration in the other section refuses by name at every door."""

    @staticmethod
    def _cases():
        x1 = _plain()
        x1[DP]["target"] = "odin"
        x1b = _plain()
        x1b[DP]["target"] = "odin"
        x1b[PC].update(weight_bits=8, cores=[{"max_axons": 256, "max_neurons": 256, "count": 10}])
        x1c = _plain(VALUE)
        del x1c[DP]["core_semantics"]
        x1c[DP]["target"] = "isaac_like"
        x1c[PC]["readout_law"] = dict(READOUT_LAW)
        x3 = _document("odin")
        x3[DP]["target_overrides"] = {"cores": {"value": [{**ODIN_CORE, "count": 64}], "note": "a pool"}}
        label = _plain()
        label[DP]["target_claim_class"] = "headline-eligible"
        return {"target": (x1, "deployment_parameters.target"),
                "target_with_hand_keys": (x1b, "deployment_parameters.target"),
                "target_value_domain": (x1c, "deployment_parameters.target"),
                "overrides": (x3, "deployment_parameters.target_overrides"),
                "label": (label, "deployment_parameters.target_claim_class")}

    @pytest.mark.parametrize("case", ["target", "target_with_hand_keys", "target_value_domain", "overrides", "label"])
    def test_a_target_key_in_deployment_parameters_refuses_at_every_door(self, tmp_path, case):
        doc, path = self._cases()[case]
        document, wizard, run = _refused_by_name(doc, tmp_path, path)
        assert {m for m in document if path in m} == {m for m in wizard if path in m} == {str(run)}

    @pytest.mark.parametrize("key,section,value", [
        ("core_semantics", PC, "mvm"), ("core_semantics", PC, "spiking"), ("weight_bits", DP, 8)])
    def test_a_bundle_key_in_the_other_section_refuses_at_every_door(self, tmp_path, key, section, value):
        doc = _document("odin")
        doc[section][key] = value
        document, wizard, run = _refused_by_name(doc, tmp_path, f"{section}.{key}")
        assert {m for m in document if f"{section}.{key}" in m} == {str(run)}


class TestTargetIsReadInTheFlatPlatform:
    """``target``, ``target_overrides`` and the labels are read in the flat ``platform_constraints`` at every door.
    The run door does not read the wizard's nested user/auto platform shapes, so a declaration in one, in its body
    or beside its mode, refuses by name at every door, and the wizard previews no refused target (review round 2:
    R15, R16, N and A)."""

    @staticmethod
    def _nest(doc, shape, **beside):
        body = doc[PC]
        doc[PC] = {"mode": "user", "user": body} if shape == "user" else {"mode": "auto", "auto": {"fixed": body}}
        doc[PC].update(beside)
        return doc

    @staticmethod
    def _value_witness():
        """A target-only value-domain draft: an MVM bundle, a readout law and activation bits, no core semantics."""
        doc = _document("isaac_like", VALUE)
        del doc[DP]["core_semantics"]
        doc[PC]["readout_law"] = dict(READOUT_LAW)
        return doc

    @classmethod
    def _cases(cls):
        conflicting = _document("odin")
        conflicting[PC]["weight_bits"] = 8
        overridden = _document("odin")
        overridden[PC]["target_overrides"] = {"cores": {"value": [{**ODIN_CORE, "count": 64}], "note": "a pool"}}
        labelled = _load(SPIKING)
        labelled[PC]["target_claim_class"] = "headline-eligible"
        return {
            "R15_user": (cls._nest(_document("odin"), "user"), ["platform_constraints.user.target"]),
            "R16_user_conflict": (cls._nest(conflicting, "user"), ["platform_constraints.user.target"]),
            "N_user_value_domain": (cls._nest(cls._value_witness(), "user"), ["platform_constraints.user.target"]),
            "A_auto_fixed": (cls._nest(_document("odin"), "auto"), ["platform_constraints.auto.fixed.target"]),
            "overrides_user": (cls._nest(overridden, "user"),
                               ["platform_constraints.user.target", "platform_constraints.user.target_overrides"]),
            "label_user": (cls._nest(labelled, "user"), ["platform_constraints.user.target_claim_class"]),
            "beside_the_mode": (cls._nest(_plain(), "user", target="odin"), ["platform_constraints.target"]),
        }

    @pytest.mark.parametrize("case", ["R15_user", "R16_user_conflict", "N_user_value_domain", "A_auto_fixed",
                                      "overrides_user", "label_user", "beside_the_mode"])
    def test_a_nested_declaration_refuses_by_name_at_every_door(self, tmp_path, case):
        doc, paths = self._cases()[case]
        document, wizard, run = _refused_by_name(doc, tmp_path, paths[0])
        for path in paths:
            assert f"{path}: " in str(run), str(run)
        assert {m for m in document if paths[0] in m} == {m for m in wizard if paths[0] in m} == {str(run)}
        assert resolve_draft(copy.deepcopy(doc)).resolved == {}

    def test_the_flat_declaration_of_each_case_resolves(self, tmp_path):
        """The same declarations in the flat platform: the witness and ODIN are accepted at every door."""
        assert _verdicts(self._value_witness(), tmp_path) == ([], [], None)
        assert _verdicts(_document("odin"), tmp_path) == ([], [], None)


class TestDerivedLabelsAreNeverDeclared:
    """The run door refuses a hand-written label, as the document door and the wizard do, so the sealed record
    never states a claim class no bundle derived."""

    @staticmethod
    def _sealed_platform(doc, tmp_path):
        """The run door into the record: merge, resolved platform, then ``identity_from_pipeline``."""
        flat = _run(doc, tmp_path)
        pcfg = build_platform_constraints_resolved(flat)
        config = {**flat, "input_shape": (1, 28, 28), "num_classes": 10, "input_size": 784, "device": "cpu"}
        pipeline = SimpleNamespace(working_directory=str(tmp_path), config=config,
                                   data_provider_factory=SimpleNamespace(_name="MNIST_DataProvider"))
        return identity_from_pipeline(pipeline, DeploymentPlan.resolve(config), pcfg).platform

    def test_labels_without_a_target_refuse_at_every_door(self, tmp_path):
        doc = _load(SPIKING)
        doc[PC].update(target_claim_class="headline-eligible", target_faithfulness_class="exact")
        for label in LABELS:
            _refused_by_name(doc, tmp_path, label)
        with pytest.raises(ToolchainRefusal, match="target_claim_class='headline-eligible'"):
            self._sealed_platform(doc, tmp_path)

    def test_a_label_beside_a_target_that_derives_another_refuses_at_every_door(self, tmp_path):
        doc = _document("odin")
        doc[PC]["target_claim_class"] = "headline-eligible"
        _refused_by_name(doc, tmp_path, "target_claim_class")
        with pytest.raises(ToolchainRefusal, match="'quarantined', which target='odin' derives"):
            self._sealed_platform(doc, tmp_path)

    def test_the_sealed_record_carries_the_derived_labels(self, tmp_path):
        platform = self._sealed_platform(_document("odin"), tmp_path)
        assert platform["target_claim_class"] == "quarantined"
        assert platform["target_faithfulness_class"] == target_bundles()["odin"].faithfulness.label()

    @pytest.mark.parametrize("label", LABELS)
    def test_a_null_label_without_a_target_refuses_at_every_door(self, tmp_path, label):
        """A label is judged by presence at the run door too: a declared null is a declaration (review round 2, L1)."""
        doc = _load(SPIKING)
        doc[PC][label] = None
        _refused_by_name(doc, tmp_path, label)
        with pytest.raises(ToolchainRefusal, match=f"{label}=None is derived by the target expansion"):
            self._sealed_platform(doc, tmp_path)

    @pytest.mark.parametrize("label", LABELS)
    def test_a_null_label_beside_a_target_refuses_at_every_door(self, tmp_path, label):
        """The derived label never overwrites a declared null (review round 2, L2)."""
        doc = _document("odin")
        doc[PC][label] = None
        _refused_by_name(doc, tmp_path, label)
        with pytest.raises(ToolchainRefusal, match=f"{label}=None differs from {label}="):
            self._sealed_platform(doc, tmp_path)

    def test_a_flat_config_re_merges_to_itself(self, tmp_path):
        """A flat config laid back into its two registry sections merges again, its equal labels kept."""
        flat = _run(_document("odin"), tmp_path)
        dp = {k: v for k, v in flat.items() if k in REGISTRY and REGISTRY[k].section == DP}
        again = merge_pipeline_config(dp, {k: v for k, v in flat.items() if k not in dp})
        assert {k: again[k] for k in ("target", *LABELS)} == {k: flat[k] for k in ("target", *LABELS)}


class TestOverridesDepart:
    def test_an_override_equal_to_the_bundle_value_refuses_by_name_at_every_door(self, tmp_path):
        doc = _document("odin")
        doc[PC]["target_overrides"] = {"weight_bits": {"value": 4, "note": "the bundle's own width"}}
        _refused_by_name(doc, tmp_path, "target_overrides.weight_bits")

    def test_an_override_of_another_type_departs(self, tmp_path):
        doc = _document("odin")
        doc[PC]["target_overrides"] = {"allow_coalescing": {"value": True, "note": "a departure"}}
        assert _verdicts(doc, tmp_path) == ([], [], None)
        # [U-CHIPKEYS] the exact ODIN description becomes approximate by the one override, named
        # [U-CHIPKEYS2] the label keeps the configuration the exact class rested on
        assert _run(doc, tmp_path)["target_faithfulness_class"] == (
            'approximate, given "leakage and SDSP learning off by the deployment": allow_coalescing declared')


class TestConsumerReadDeclarations:
    """A spelling the consumer reads as the bundle's value declares the bundle's value, at every door, whichever
    order a door reads and expands in (a hand bundle: no shipped bundle sets a consumer-read key)."""

    @pytest.fixture
    def uniform_chip(self, tmp_path, monkeypatch):
        directory = tmp_path / "bundles"
        directory.mkdir()
        (directory / "uniform_chip.json").write_text(json.dumps(described({  # [U-CHIPKEYS] a hand chip under the rule
            "name": "uniform_chip", "geometry": "recom_64x64_256", "keys": {"s_allocation": "uniform"},
            "faithfulness": {"class": "exact"}}, semantics="spiking")))
        shipped = dict(target_bundles())
        bundles = {**shipped, **load_bundles(directory)}
        monkeypatch.setattr(bundle_module, "target_bundles", lambda: bundles)
        return "uniform_chip"

    def test_an_uppercase_spelling_is_accepted_at_every_door(self, tmp_path, uniform_chip):
        doc = _plain()
        doc[PC]["target"] = uniform_chip
        doc[DP]["s_allocation"] = "UNIFORM"
        assert _verdicts(doc, tmp_path) == ([], [], None)
        flat = _run(doc, tmp_path)
        # [U-CHIPKEYS4] 3(f) the hand chip states no pricing: its label names the pricing the run resolves (none)
        assert flat["s_allocation"] == "uniform" and flat["target_faithfulness_class"] == "approximate: no physics profile"

    def test_a_reserved_mode_still_conflicts_at_every_door(self, tmp_path, uniform_chip):
        doc = _plain()
        doc[PC]["target"] = uniform_chip
        doc[DP]["s_allocation"] = "budget"
        _refused_by_name(doc, tmp_path, "s_allocation")


class TestTheOdinCell:
    def test_the_odin_bundle_carries_the_chip_s_permissions(self):
        values = target_bundles()["odin"].values
        assert values["allow_coalescing"] is False and values["allow_neuron_splitting"] is False

    def test_the_cell_names_every_permission_it_overrides(self):
        path = os.path.join(_FIXTURES, "..", "..", "..", "templates", "tier_0", "t0_61_lifs_simplemlp_wq_s4_odin.json")
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        dp = dict(doc[DP])
        flat = merge_pipeline_config(dp, copy.deepcopy(doc[PC]))
        bundle = target_bundles()["odin"]
        departed = sorted(key for key in bundle.values
                          if json.dumps(flat[key], sort_keys=True) != json.dumps(bundle.values[key], sort_keys=True))
        # [U-CHIPKEYS] the cell's own soma law (per-cycle compare, unbounded membrane, subtractive reset, strict
        # compare) departs from ODIN's stated one, each key named
        assert departed == ["allow_coalescing", "cores", "firing_granularity", "firing_mode", "membrane_bits",
                            "thresholding_mode"]
        for key in departed:
            assert f"{key} declared" in flat["target_faithfulness_class"]


class TestOverridesNeedATarget:
    """``target_overrides`` declared without a target (none, or a null one) refuses by name at every door: the
    document door's rule (``_document_assignments``) and the run door's (``target_assignments``) state one message
    (G4 final review, overrides without a target; probes O1 and O3)."""

    @pytest.mark.parametrize("target", ["absent", None], ids=["O1_no_target", "O3_null_target"])
    def test_overrides_without_a_target_refuse_at_every_door(self, tmp_path, target):
        doc = _load(SPIKING)
        if target != "absent":
            doc[PC]["target"] = target
        doc[PC]["target_overrides"] = {"weight_bits": {"value": 6, "note": "a narrower grid"}}
        document, wizard, run = _refused_by_name(doc, tmp_path, "declared without a target")
        assert {m for m in document if "without a target" in m} == {m for m in wizard if "without a target" in m} \
            == {str(run)}


class TestTargetWhereNothingReadsIt:
    """``target`` (and every key the expansion reads) declared where no door reads it refuses by name at every
    door: at the document's top level (P1), or inside an object the parser does not read, the nested bias object
    of either section included, at any depth (P7); before, every door ignored it and the run deployed the default
    grid (G4 final review, the placement rule)."""

    @staticmethod
    def _cases():
        top = _plain()
        top["target"] = "odin"
        nested = _plain()
        nested[PC]["platform_constraints"] = {"target": "odin"}
        nested_bias = _plain()
        nested_bias[PC]["platform_constraints"] = {"has_bias": False, "target": "odin"}
        deep = _plain()
        deep[DP]["platform_constraints"] = {"extra": {"target_overrides": {}}}
        label = _plain()
        label["target_claim_class"] = "headline-eligible"
        return {"P1_top_level": (top, "target: target"),
                "P7_nested_bias_object": (nested, "platform_constraints.platform_constraints.target"),
                "P7b_beside_a_bias": (nested_bias, "platform_constraints.platform_constraints.target"),
                "deep_in_the_deployment_parameters": (
                    deep, "deployment_parameters.platform_constraints.extra.target_overrides"),
                "label_at_the_top_level": (label, "target_claim_class: target_claim_class")}

    @pytest.mark.parametrize("case", ["P1_top_level", "P7_nested_bias_object", "P7b_beside_a_bias",
                                      "deep_in_the_deployment_parameters", "label_at_the_top_level"])
    def test_an_unread_declaration_refuses_by_name_at_every_door(self, tmp_path, case):
        doc, named = self._cases()[case]
        document, wizard, run = _refused_by_name(doc, tmp_path, named)
        assert "nothing reads it here" in str(run)
        # The one judge refuses it (review round 1: every registry key no reader reads, one rule), beside any other
        # violation the document holds (P7b's nested bias disagrees with the flat one).
        assert {m for m in document if named in m} == {m for m in wizard if named in m} == {
            part for part in str(run).split("; ") if named in part}
        assert set(str(run).split("; ")) <= set(document)
        assert resolve_draft(copy.deepcopy(doc)).resolved == {}

    def test_the_run_door_of_the_two_sections_refuses_a_nested_one(self):
        """The merge reads the two sections: a target in either section's unread object refuses there too."""
        with pytest.raises(ToolchainRefusal, match="platform_constraints.platform_constraints.target"):
            merge_pipeline_config({}, {"platform_constraints": {"target": "odin"}})
        with pytest.raises(ToolchainRefusal, match="deployment_parameters.platform_constraints.target"):
            merge_pipeline_config({"platform_constraints": {"target": "odin"}}, {})


class TestAWeightGridWithoutALevel:
    """Key amendment K1: under weight quantization the signed weight grid of 1 bit has no nonzero level (q_max =
    0), so ``weight_bits`` below 2 refuses by name at every door through its applied schema, the bundle's value
    and the hand form alike; under float weights it is not applied and passes (key amendment K2 names it)."""

    MESSAGE = "weight_bits=1 is below the minimum 2"

    def test_truenorth_under_weight_quantization_refuses_at_every_door(self, tmp_path):
        doc = _document("truenorth")
        assert doc[DP]["weight_quantization"] is True and target_bundles()["truenorth"].values["weight_bits"] == 1
        document, wizard, run = _refused_by_name(doc, tmp_path, self.MESSAGE)
        assert isinstance(run, DeclarationValueError)

    def test_the_hand_form_refuses_alike(self, tmp_path):
        doc = _load(SPIKING)
        doc[PC]["weight_bits"] = 1
        _refused_by_name(doc, tmp_path, self.MESSAGE)

    def test_truenorth_under_float_weights_passes_every_door(self, tmp_path):
        doc = _document("truenorth")
        doc["pipeline_mode"], doc[DP]["weight_quantization"] = "vanilla", False
        assert _verdicts(doc, tmp_path) == ([], [], None)
        assert _run(doc, tmp_path)["target_faithfulness_class"].endswith("; weight grid not applied")

    def test_two_bits_pass_under_weight_quantization(self, tmp_path):
        doc = _load(SPIKING)
        doc[PC]["weight_bits"] = 2
        assert _verdicts(doc, tmp_path) == ([], [], None)


class TestNotApplied:
    """Key amendment K2, as review round 1 corrected it: ``weight_bits``' consumers read it in every run (payload
    sizes, pricing, the record, the mapping's register bound), so its relevance is ``always``; the weight grid
    applies it only under weight quantization (``applied_when``). A bundle key whose relevance fails is named
    "<key> not applied", and one whose ``applied_when`` fails "<applied_by> not applied"; an applied key is never
    named."""

    def test_weight_bits_is_relevant_and_its_grid_is_conditional(self):
        """The grid applies where neither ``pipeline_mode: vanilla`` nor ``weight_quantization: false`` holds (absent
        reads as on), through the relevance language's ``not``; the key itself stays relevant in every run."""
        entry = REGISTRY["weight_bits"]
        assert entry.relevant == Relevance.always() and entry.applied_by == "weight grid"
        when = entry.applied_when
        assert when.evaluate({}) and when.evaluate({"pipeline_mode": "phased", "weight_quantization": True})
        assert not when.evaluate({"weight_quantization": False}) and not when.evaluate({"pipeline_mode": "vanilla"})
        assert Relevance.from_json(when.to_json()) == when and {item["op"] for item in when.to_json()["items"]} == {"not"}
        assert Relevance.not_(Relevance.when_true("x")).evaluate({}) and not Relevance.not_(Relevance.always()).evaluate({})

    def test_a_quantized_run_applies_the_grid(self, tmp_path):
        assert "not applied" not in _run(_document("odin"), tmp_path)["target_faithfulness_class"]

    def test_the_float_truenorth_cell_names_the_grid(self):
        path = os.path.join(_FIXTURES, "..", "..", "..", "templates", "tier_0", "t0_62_lifs_simplemlp_fp_s8_truenorth.json")
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        flat = merge_pipeline_config(dict(doc[DP]), copy.deepcopy(doc[PC]))
        assert flat["target_faithfulness_class"] == (  # [U-CHIPKEYS] the cell's strict compare departs, named
            # [U-CHIPKEYS3] the weight grid's departure as its statement states it (the report's candidate m-4)
            # [U-CHIPKEYS6] the statement's computation and value assumed as quoted fields
            # [U-CHIPKEYS7] the computation reworded to the alphabet
            'approximate: weight_bits departs from its source\'s computation "binary synapse times a 9-bit signed '
            'integer of its neuron for the type of the axon", assumed "1"; allow_coalescing declared; thresholding_mode '
            'declared; weight grid not applied')


class TestAPerAxonCoreUnderFloatWeights:
    """[MN4] CL-SG design section 4 step A, door strictness "Strict, stated": a per-axon core type stores each weight
    in sign pairs, exact only on the integer grid, so under float weights (``pipeline_mode: vanilla`` or
    ``weight_quantization: false``, the derivation's reading) it refuses by name at every door, before any training
    spend, where the split's ``SignSplitError`` would name it only at hard-core mapping."""

    NAMED = "weight_sign_granularity='per_axon' needs weights on the integer grid"

    @staticmethod
    def _per_axon(doc):
        doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in ("cores", "max_axons", "max_neurons")},
                   "cores": [dict(ODIN_CORE)]}
        return doc

    @pytest.mark.parametrize("explicit_false", [True, False], ids=["vanilla_and_false", "vanilla_alone"])
    def test_a_per_axon_core_type_under_float_weights_refuses_at_every_door(self, tmp_path, explicit_false):
        """The float-weight mechanism (``pipeline_mode: vanilla``, with or without the explicit
        ``weight_quantization: false`` the derivation writes)."""
        doc = self._per_axon(_load(SPIKING))
        doc["pipeline_mode"] = "vanilla"
        if explicit_false:
            doc[DP]["weight_quantization"] = False
        else:
            doc[DP].pop("weight_quantization")
        _refused_by_name(doc, tmp_path, self.NAMED)

    def test_the_odin_target_under_float_weights_refuses_alike(self, tmp_path):
        doc = _document("odin")
        doc["pipeline_mode"], doc[DP]["weight_quantization"] = "vanilla", False
        _refused_by_name(doc, tmp_path, self.NAMED)

    def test_a_per_axon_core_type_under_weight_quantization_passes_every_door(self, tmp_path):
        doc = self._per_axon(_load(SPIKING))
        assert doc[DP]["weight_quantization"] is True
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_a_per_synapse_core_type_under_float_weights_passes(self, tmp_path):
        doc = self._per_axon(_load(SPIKING))
        doc[PC]["cores"] = [{k: v for k, v in ODIN_CORE.items() if k != "weight_sign_granularity"}]
        doc["pipeline_mode"], doc[DP]["weight_quantization"] = "vanilla", False
        assert not any(self.NAMED in m for m in _verdicts(doc, tmp_path)[0])

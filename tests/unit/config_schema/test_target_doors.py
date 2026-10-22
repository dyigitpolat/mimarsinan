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
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.mapping.platform.target_bundles import load_bundles, target_bundles
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import identity_from_pipeline
from mimarsinan.pipelining.session import parse_deployment_config

DP, PC = "deployment_parameters", "platform_constraints"
_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs")
SPIKING, VALUE = "sync_simplemlp.json", "mvm_aq_lenet5.json"
ODIN_CORE = {"max_axons": 128, "max_neurons": 256, "count": 1, "has_bias": False}
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
    """A fixture with ``target: <name>`` in place of every key the bundle sets."""
    doc = _load(fixture)
    doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in _owned(name)}, "target": name}
    return doc


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
        ("odin", "cores", [{**ODIN_CORE, "max_axons": 128.0, "count": 1.0}]),
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
        assert _run(doc, tmp_path)["target_faithfulness_class"].endswith("; allow_coalescing declared")


class TestConsumerReadDeclarations:
    """A spelling the consumer reads as the bundle's value declares the bundle's value, at every door, whichever
    order a door reads and expands in (a hand bundle: no shipped bundle sets a consumer-read key)."""

    @pytest.fixture
    def uniform_chip(self, tmp_path, monkeypatch):
        directory = tmp_path / "bundles"
        directory.mkdir()
        (directory / "uniform_chip.json").write_text(json.dumps({
            "name": "uniform_chip", "geometry": "recom_64x64_256", "keys": {"s_allocation": "uniform"},
            "faithfulness": {"class": "exact"}}))
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
        assert flat["s_allocation"] == "uniform" and flat["target_faithfulness_class"] == "exact"

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
        assert departed == ["allow_coalescing", "cores"]
        for key in departed:
            assert f"{key} declared" in flat["target_faithfulness_class"]

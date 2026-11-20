"""Every config entry point expands a target one way (plan CH-B0, acceptance 2 and 4), and a run's flat config
carries the value every consumer reads (S0 interface note: one ``read_like_consumers`` call in the merge)."""

import copy
import dataclasses
import functools
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mimarsinan.config_schema.registry import REGISTRY, parse_deployment_document
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.config_schema.display_view import build_config_display_view
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.mapping.platform.target_bundles import DP, PC, target_bundles
from mimarsinan.mapping.platform.target_expansion import (
    expand_target_document,
    same_declaration,
    target_core_semantics,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import identity_from_pipeline
from mimarsinan.pipelining.session import parse_deployment_config
from unit.deployment_record.record_fixtures import make_full_record

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
_FIXTURES = os.path.join(_ROOT, "tests", "fixtures", "deployment_configs")
FIXTURE_FOR = {"spiking": "sync_simplemlp.json", "mvm": "mvm_aq_lenet5.json"}
GEOMETRY_KEYS = ("cores", "weight_bits", "max_axons", "max_neurons")
LABELS = ("target_claim_class", "target_faithfulness_class")
READOUT_LAW = {"adc_bits": 6, "range": "fixed", "range_value": 4.0, "mult_bits": 16}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _generator():
    """The tier-0 template generator, loaded from its file as the template checks load it."""
    spec = importlib.util.spec_from_file_location("tier_generate_entry_points",
                                                  os.path.join(_ROOT, "templates", "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owned(name):
    """The keys a bundle sets and the keys derived from them (the core maxima)."""
    keys = set(target_bundles()[name].values)
    return keys | {key for key, entry in REGISTRY.items() if set(entry.derived_from) & keys}


def _document(name):
    """A shipped fixture of the bundle's core semantics with every key the bundle sets replaced by ``target``; the
    float assembly where the bundle's signed weight grid has no nonzero level under weight quantization (key
    amendment K1 refuses that by name, ``TestAWeightGridWithoutALevel``)."""
    doc = _load(os.path.join(_FIXTURES, FIXTURE_FOR[target_bundles()[name].values.get("core_semantics", "spiking")]))
    # [U-CHIPKEYS] a description states its neuron law's deployment-parameter keys too: the fixture's give way (its
    # whole neuron law where the bundle states a spiking family; an equal core semantics stays, as before)
    owned = set(target_bundles()[name].values)
    if "spiking_family" in owned:
        owned |= {k for k, e in REGISTRY.items() if e.group == "spiking" and k != "core_semantics"}
    doc[DP] = {k: v for k, v in doc[DP].items() if k == "core_semantics" or k not in owned}
    doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in GEOMETRY_KEYS and k not in _owned(name)},
               "target": name}
    bits = target_bundles()[name].values.get("weight_bits")
    if bits is not None and bits < REGISTRY["weight_bits"].applied_schema.low:
        doc["pipeline_mode"], doc[DP]["weight_quantization"] = "vanilla", False
    return doc


def _value_witness():
    """A target-only value-domain draft: an MVM bundle and a readout law, no core semantics; [U-C C-5] its input
    width is the chip description's (the bundle's activation_bits 16), which the draft does not author."""
    doc = _document("isaac_like")
    del doc[DP]["core_semantics"]
    doc[PC]["readout_law"] = dict(READOUT_LAW)
    assert "activation_bits" not in doc[PC]
    return doc


def _merged(doc):
    dp = dict(doc[DP])
    dp.setdefault("seed", int(doc.get("seed", 0)))
    DeploymentPipeline.apply_preset(doc.get("pipeline_mode", "phased"), dp)
    return merge_pipeline_config(dp, copy.deepcopy(doc[PC]))


def _projection(flat, name):
    keys = set(target_bundles()[name].values) | {"target", *LABELS}
    return {key: flat.get(key) for key in sorted(keys)}


def _no_labels(doc):
    sections = [doc] + [doc.get(s) for s in (DP, PC) if isinstance(doc.get(s), dict)]
    return not any(label in section for section in sections for label in LABELS)


class TestEveryEntryPoint:
    @pytest.mark.parametrize("name", sorted(target_bundles()))
    def test_equal_expanded_configs(self, name, tmp_path):
        doc = _document(name)
        merged = _merged(doc)
        assert merged["target"] == name and all(merged[label] for label in LABELS)
        parsed = parse_deployment_config({**copy.deepcopy(doc), "_working_directory": str(tmp_path)},
                                         data_provider_factory=object())
        assert _no_labels({DP: parsed.deployment_parameters, PC: parsed.platform_constraints})
        assert parsed.platform_constraints["cores"] == target_bundles()[name].values["cores"]
        dp = dict(parsed.deployment_parameters)
        DeploymentPipeline.apply_preset(parsed.pipeline_mode, dp)
        assert merge_pipeline_config(dp, parsed.platform_constraints) == merged
        flat = build_flat_pipeline_config(doc[DP], doc[PC], pipeline_mode=doc["pipeline_mode"])
        assert _projection(flat, name) == _projection(merged, name)
        assert validate_deployment_config(copy.deepcopy(doc)) == []
        expanded = expand_target_document(doc, write_derived=False)
        assert validate_deployment_config(copy.deepcopy(dict(expanded))) == [] and _no_labels(expanded)
        resolution = resolve_draft(copy.deepcopy(doc))
        assert resolution.errors == []
        assert _projection(resolution.resolved, name) == _projection(merged, name)
        emitted = emit_deployment_config(copy.deepcopy(doc))
        assert _no_labels(emitted) and emitted[PC]["target"] == name
        assert set(emitted[DP]) | set(emitted[PC]) <= set(doc[DP]) | set(doc[PC])
        again = resolve_draft(emitted)
        assert again.errors == [] and _projection(again.resolved, name) == _projection(merged, name)

    def test_the_monitor_view_of_a_target_run_reads_its_saved_document(self):
        """The run monitor's config view merges the run's saved document, so it shows the target's keys and labels."""
        doc = _document("odin")
        merged = _merged(doc)
        resolved = build_config_display_view(merged, saved_config=doc)["raw_resolved"]
        assert _projection(resolved, "odin") == _projection(merged, "odin")

    @pytest.mark.parametrize("name", sorted(target_bundles()))
    def test_the_monitor_s_flat_fallback_view_of_a_target_run(self, name):
        """Without a saved run document the monitor splits the flat config back into its two sections by the
        registry, so the target, its labels and its keys re-merge as the run laid them (G4 final review, the
        display fallback; ``evidence/cl9_proposals/``)."""
        merged = _merged(_document(name))
        resolved = build_config_display_view(copy.deepcopy(merged))["raw_resolved"]
        assert _projection(resolved, name) == _projection(merged, name)

    def test_emit_reads_a_draft_the_parser_cannot_unpack(self):
        """``emit_deployment_config`` writes nothing and never refuses: a structural container that is no object
        is an unknown path to the parser, and the emitted document keeps the draft's own declarations (G4 final
        review, the emit regression on a malformed draft)."""
        draft = {"platform_constraints": {"mode": "auto", "auto": "x"},
                 "deployment_parameters": {"core_semantics": "mvm"}}
        assert "platform_constraints.auto" in parse_deployment_document(draft).unknown
        emitted = emit_deployment_config(copy.deepcopy(draft))
        assert emitted[PC] == {"mode": "auto", "auto": "x"} and emitted[DP] == {"core_semantics": "mvm"}
        assert target_core_semantics(draft) == "mvm"
        assert target_core_semantics({"deployment_parameters": "x"}) is None

    #: Values a draft holds while a person types: none is an object of the shape a door reads.
    MALFORMED = ("x", 7, 7.5, None, True, [1], {"a": 1}, [], {})

    @pytest.mark.parametrize("value", MALFORMED, ids=repr)
    @pytest.mark.parametrize("where", ["dp", "pc", "core_semantics", "target", "target_overrides", "override_entry",
                                       "user", "auto", "mode", "label", "top_pipeline_mode"])
    def test_emit_never_raises_on_a_malformed_draft(self, where, value):
        """The emit's core-semantics read parses any draft (a container that is no object is an unknown path) and
        catches only the expansion's refusals, so no draft a person types raises from it (G4 final review, the emit
        regression on a malformed draft; closure mutant X7)."""
        target = {"target": "odin"}
        draft = {
            "dp": {DP: value, PC: target},
            "pc": {PC: value},
            "core_semantics": {DP: {"core_semantics": value}, PC: target},
            "target": {PC: {"target": value}},
            "target_overrides": {PC: {**target, "target_overrides": value}},
            "override_entry": {PC: {**target, "target_overrides": {"weight_bits": value}}},
            "user": {PC: {**target, "mode": "user", "user": value}},
            "auto": {PC: {**target, "mode": "auto", "auto": value}},
            "mode": {PC: {**target, "mode": value}},
            "label": {PC: {**target, "target_claim_class": value}},
            "top_pipeline_mode": {"pipeline_mode": value, PC: target},
        }[where]
        semantics = target_core_semantics(draft)
        assert semantics is None or isinstance(semantics, str)
        emit_deployment_config(copy.deepcopy(draft))

    def test_a_derived_label_in_a_document_is_refused_at_both_wizard_doors(self):
        doc = _document("odin")
        doc[PC]["target_claim_class"] = "headline-eligible"
        assert any("target_claim_class is not declarable" in m for m in validate_deployment_config(doc))
        assert any(row["key"] == "target_claim_class" for row in resolve_draft(doc).errors)
        assert "target_claim_class" not in emit_deployment_config(doc)[PC]

    def test_a_refused_target_is_one_row_at_each_door(self):
        doc = _document("odin")
        doc[PC]["target"] = "odin2"
        messages = validate_deployment_config(doc)
        assert sum("names no target bundle" in m for m in messages) == 1
        rows = [row for row in resolve_draft(doc).errors if "names no target bundle" in row["message"]]
        assert len(rows) == 1 and rows[0]["key"] == "target"
        assert emit_deployment_config(doc)[PC]["target"] == "odin2"


    def test_a_schema_refusal_of_the_overrides_is_stated_once(self):
        doc = _document("odin")
        doc[PC]["target_overrides"] = {"cores": {"value": []}}
        messages = validate_deployment_config(doc)
        assert messages.count("target_overrides.cores.note is required") == 1, messages

    def test_the_bundle_s_weight_bits_reach_the_quantization_contract(self, tmp_path):
        """The run door expands before the assembly contract: float weights under a phased pipeline contradict
        the bundle's declared weight width, as they do the same width written by hand."""
        doc = _document("loihi")
        doc[DP]["weight_quantization"] = False
        with pytest.raises(ValueError, match="bits-driven"):
            parse_deployment_config({**doc, "_working_directory": str(tmp_path)}, data_provider_factory=object())


class TestValueDomainWitness:
    """A target-only MVM draft keeps its value-domain keys at every door (no dormant-domain stripping)."""

    def test_resolve_and_emit(self):
        doc = _value_witness()
        resolution = resolve_draft(copy.deepcopy(doc))
        assert resolution.errors == [] and resolution.dormant == []
        assert resolution.resolved["readout_law"] == READOUT_LAW and resolution.resolved["activation_bits"] == 16
        assert resolution.resolved["core_semantics"] == "mvm"
        emitted = emit_deployment_config(copy.deepcopy(doc))
        assert emitted[PC]["readout_law"] == READOUT_LAW
        assert emitted[PC]["target"] == "isaac_like" and "core_semantics" not in emitted[DP]
        assert not {"cores", "weight_bits", "platform_physics_profile", "activation_bits"} & set(emitted[PC])
        assert validate_deployment_config(copy.deepcopy(doc)) == []

    def test_emit_reads_no_nested_target(self):
        """Emit takes the target's core semantics from the flat platform only, as every door reads it: a target in
        the wizard's nested user shape, refused at every door, selects nothing emit keeps or drops (review round 2)."""
        flat = _value_witness()
        flat[DP]["lif_membrane_readout"] = True  # a declarable event-domain key: dormant under the target's mvm
        nested = copy.deepcopy(flat)
        nested[PC] = {"mode": "user", "user": nested[PC]}
        assert target_core_semantics(flat) == "mvm" and target_core_semantics(nested) is None
        assert "lif_membrane_readout" not in emit_deployment_config(copy.deepcopy(flat))[DP]
        without_target = copy.deepcopy(nested)
        del without_target[PC]["user"]["target"]
        emitted = emit_deployment_config(copy.deepcopy(nested))
        assert emitted[DP] == emit_deployment_config(without_target)[DP]
        assert emitted[DP]["lif_membrane_readout"] is True

    def test_the_run_door(self):
        pcfg = build_platform_constraints_resolved(_merged(_value_witness()))
        assert pcfg["readout_law"] == READOUT_LAW and pcfg["core_semantics"] == "mvm"
        assert pcfg["activation_bits"] == 16 and pcfg["target"] == "isaac_like"


def _target_cells():
    directory = os.path.join(_ROOT, "templates", "tier_0")
    cells = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.startswith("t0_")]
    return [path for path in cells if "target" in _load(path)[PC]]


class TestTargetCells:
    """One tier-0 cell per physics bundle: it authors no key its bundle sets and passes every door."""

    def test_one_cell_per_physics_bundle(self):
        named = sorted(_load(path)[PC]["target"] for path in _target_cells())
        assert named == sorted(n for n, b in target_bundles().items() if "platform_physics_profile" in b.values)

    @pytest.mark.parametrize("path", _target_cells(), ids=os.path.basename)
    def test_a_cell_passes_every_door(self, path):
        doc = _load(path)
        bundle = target_bundles()[doc[PC]["target"]]
        derived = {k for k, e in REGISTRY.items() if e.section == PC and set(e.derived_from) & set(bundle.values)}
        assert not (set(bundle.values) | derived) & set(doc[PC])
        # [composed head, merge rule 23] an equal key of the cell's kind may be declared (rule 8's CELL_KIND_KEYS)
        assert not (set(bundle.values) - set(_generator().CELL_KIND_KEYS)) & set(doc[DP])
        assert validate_deployment_config(copy.deepcopy(doc)) == []
        assert resolve_draft(copy.deepcopy(doc)).errors == []
        pcfg = build_platform_constraints_resolved(_merged(doc))
        assert pcfg["platform_physics_resolved"]["name"] == bundle.values["platform_physics_profile"]
        assert pcfg["target_faithfulness_class"].startswith("approximate")

    #: The chip permissions a mapping reads (``ChipCapabilities``): a target cell authors none, its bundle states the
    #: chip's (G4 final review, the TrueNorth and Loihi permissions).
    PERMISSIONS = ("allow_coalescing", "allow_neuron_splitting")

    @pytest.mark.parametrize("path", _target_cells(), ids=os.path.basename)
    def test_a_cell_s_faithfulness_names_every_bundle_key_it_departs_from(self, path):
        """Each bundle key a cell runs at another value is a declared override, named in its faithfulness; the
        cell's own permission keys come from its bundle or from such an override, never from an ordinary key the
        label would not name; and each bundle key no reader applies is named "<key> not applied", one the consumer
        ``applied_by`` names does not apply "<applied_by> not applied" (K2, review round 1)."""
        doc = _load(path)
        bundle = target_bundles()[doc[PC]["target"]]
        flat = _merged(doc)
        departed = {key for key in bundle.values if not same_declaration(flat[key], bundle.values[key])}
        assert departed == set(doc[PC].get("target_overrides", {}))
        named = flat["target_faithfulness_class"].split(": ", 1)[1].split("; ")
        assert [item for item in named if item.endswith(" declared")] == [f"{key} declared" for key in sorted(departed)]
        assert set(self.PERMISSIONS) <= set(bundle.values) and not set(self.PERMISSIONS) & set(doc[PC])
        inert = [f"{key} not applied" if not REGISTRY[key].relevant.evaluate(flat) else
                 f"{REGISTRY[key].applied_by} not applied" for key in bundle.values
                 if not REGISTRY[key].relevant.evaluate(flat)
                 or (REGISTRY[key].applied_when is not None and not REGISTRY[key].applied_when.evaluate(flat))]
        assert [item for item in named if item.endswith(" not applied")] == inert

    @pytest.mark.parametrize("name", sorted(n for n, b in target_bundles().items()
                                            if b.values.get("core_semantics") == "spiking"))
    def test_a_target_row_whose_law_departs_names_it_or_refuses(self, name):
        """[U-CHIPKEYS] the class swept to the generator: a target row authors no deployment key its bundle sets, in
        either section; one equal to the bundle's is dropped, one the row's overrides name departs there, any other
        departure refuses by name (a lif row's subtractive reset against a chip that states its reset law)."""
        gen = _generator()
        bundle = target_bundles()[name].values
        row = {"n": 99, "mode": "lifs", "quant": "wq", "s": 4, "vehicle": "simplemlp", "target": name}
        mode = gen.MODES["lifs"]
        departs = sorted(k for k in ("spiking_family", "spiking_variant", "firing_mode", "thresholding_mode")
                         if k in bundle and bundle[k] != mode[k])
        if departs:
            with pytest.raises(ValueError, match=rf"{departs[0]}=.* departs from target='{name}'"):
                gen._deployment(0, row, gen.VEHICLES, "MNIST")
            row["target_overrides"] = {k: {"value": mode[k], "note": "a departure"} for k in departs}
        dp = gen._deployment(0, row, gen.VEHICLES, "MNIST")
        assert not (set(bundle) - set(gen.CELL_KIND_KEYS)) & set(dp)  # [composed head, merge rule 23]

    @pytest.mark.parametrize("name", sorted(target_bundles()))
    def test_a_target_row_authors_no_permission_key(self, name):
        """The generator's permission defaults describe the study grid: a row of any target authors no permission
        key, whether its bundle states the chip's (the bundle-key rule strips those) or states none (closure
        mutant G1)."""
        gen = _generator()
        plain = {"n": 1, "mode": "lif", "quant": "fp", "s": 8, "vehicle": "simplemlp"}
        assert set(self.PERMISSIONS) <= set(gen._platform(plain, gen.VEHICLES))
        assert not set(self.PERMISSIONS) & set(gen._platform({**plain, "target": name}, gen.VEHICLES))


class TestRecordIdentity:
    """Acceptance 4: the two classes ride the resolved surface into ``record.identity.platform``."""

    def test_the_classes_reach_the_record(self):
        config = _merged(_document("odin"))
        pcfg = build_platform_constraints_resolved(config)
        config.update(input_shape=(1, 28, 28), num_classes=10, input_size=784, device="cpu")
        pipeline = SimpleNamespace(working_directory="run", config=config,
                                   data_provider_factory=SimpleNamespace(_name="MNIST_DataProvider"))
        identity = identity_from_pipeline(pipeline, DeploymentPlan.resolve(config), pcfg)
        record = dataclasses.replace(make_full_record(), identity=identity)
        loaded = DeploymentRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        platform = loaded.identity.platform
        assert platform["target"] == "odin" and platform["target_claim_class"] == "quarantined"
        # [U-CHIPKEYS] ODIN states its RTL's soma law: the description is exact
        # [U-CHIPKEYS2] the label states the configuration the exact class rests on
        assert platform["target_faithfulness_class"] == 'exact, given "leakage and SDSP learning off by the deployment"'


class TestRunFlatConfigReadsLikeConsumers:
    """The run's flat config carries the canonical value of the two keys a consumer reads (S0 interface note)."""

    def test_a_document_spelled_uppercase_carries_the_canonical_value(self):
        doc = _load(os.path.join(_FIXTURES, "sync_simplemlp.json"))
        canonical = _merged(doc)
        doc[DP]["s_allocation"], doc[DP]["nevresim_connectivity_mode"] = "UNIFORM", " runtime "
        flat = _merged(doc)
        assert flat["s_allocation"] == "uniform" and flat["nevresim_connectivity_mode"] == "runtime"
        assert flat == canonical

    def test_the_per_layer_s_template_runs_as_the_uniform_declaration(self):
        doc = _load(os.path.join(_ROOT, "templates", "mnist_mmixcore_per_layer_s_uniform.json"))
        assert doc[DP]["s_allocation"] == "explicit"
        flat = _merged(doc)
        assert flat["s_allocation"] == "uniform" and flat["s_allocation_explicit"] == [4, 4, 4]
        flat.update(input_shape=(1, 28, 28), num_classes=10, input_size=int(np.prod((1, 28, 28))), device="cpu")
        assert DeploymentPlan.resolve(flat).s_allocation == "uniform"

    def test_a_value_the_consumer_refuses_is_refused_by_name_in_the_merge(self):
        """A value the consumer's reading refuses stays as written for the one judge, which the merge calls before
        the derivation (G4 interface note 1): it refuses there by name, with the run door's message."""
        doc = _load(os.path.join(_FIXTURES, "sync_simplemlp.json"))
        doc[DP]["nevresim_connectivity_mode"] = "codegen"
        message = "nevresim_connectivity_mode='codegen' is not a valid option; choose one of ['compile_time', 'runtime']"
        with pytest.raises(DeclarationValueError) as refusal:
            _merged(doc)
        assert str(refusal.value) == message and message in validate_deployment_config(doc)

    def test_the_bias_declarations_stay_as_written(self):
        """The platform bias mode is the resolver's reading (flat, then nested), not a consumer reading."""
        doc = _load(os.path.join(_FIXTURES, "sync_simplemlp.json"))
        doc[PC]["platform_constraints"] = {"has_bias": doc[PC]["has_bias"]}
        flat = _merged(doc)
        assert flat["has_bias"] == doc[PC]["has_bias"]
        assert flat["platform_constraints"] == {"has_bias": doc[PC]["has_bias"]}
        assert REGISTRY["has_bias"].consumer_reading is None

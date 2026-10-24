"""A declaration has one reading, the derivation's included (the S0-B and G4 joint closure, review round 2).

A run derives its flat config from the judged declarations of the sections it takes (``runtime.run_config``: the
target expansion, then ``derive_run_config``: the core maxima and the Tq | S contract, the quantization flags and the
pipeline mode). Every reader takes that one derivation:

- a declared derived key the derivation would replace is refused by name at every door (the core maxima's
  contradiction included), and the session's pipeline mode is the derived one, so its working directory and run
  information name the mode the run executes;
- the wizard previews the sections the run takes (``run_sections``) through the run's own derivation, and states the
  builder registration's refusals wherever the run reads the declaration;
- an unread declaration equal to the value the run reads is refused like any other unread declaration (a boundary
  finding: the base ran it exactly);
- the recipes, a null non-declarable key, an empty path, a structural platform body and the nested bias object are
  judged as their consumers read them.
"""

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import pytest

from mimarsinan.common.workload_profile import ModelWorkloadProfile
from mimarsinan.config_schema.deployment_derivation import derive_platform_constraints, legality_bearing_keys
from mimarsinan.config_schema.registry import REGISTRY, parse_deployment_document
from mimarsinan.config_schema.registry.entries_write import RECIPE_SCHEMA
import mimarsinan.config_schema.resolve as resolve_module
from mimarsinan.config_schema.resolve import effective_view, resolve_draft
from mimarsinan.config_schema.runtime import build_flat_pipeline_config, run_sections, unset
from mimarsinan.config_schema.validation import legality_errors, validate_deployment_config
from mimarsinan.gui.wizard.pretrained_panel import (
    PRETRAINED_LEGALITY_KEYS, effective_with_builder, pretrained_legality_errors,
)
from mimarsinan.gui.wizard.schema_api import resolve_payload
from mimarsinan.model_training.training_recipe import TrainingRecipe
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.session import parse_deployment_config
import mimarsinan.pipelining.session as session_module


def _sibling():
    """The four-door harness of ``test_one_run_reading`` (its helpers only; its tests are collected there)."""
    spec = importlib.util.spec_from_file_location(
        "_one_run_reading_doors", Path(__file__).with_name("test_one_run_reading.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DOORS = _sibling()
_load, _session, _cli, _merge, _wizard, _parts = (
    _DOORS._load, _DOORS._session, _DOORS._cli, _DOORS._merge, _DOORS._wizard, _DOORS._parts)
_admitted_everywhere = _DOORS._admitted_everywhere


def _states(message, messages):
    """``message`` is among ``messages``, each split at ``; `` as the run joins its refusals (a message may hold one)."""
    parts = _parts(messages)
    return all(part in parts for part in message.split("; "))


def _refused_everywhere(document, message, tmp_path, *, merge=True):
    """``message`` is stated by the document door, the wizard, the CLI path and (``merge``) the run door."""
    assert message in validate_deployment_config(copy.deepcopy(document))
    assert message in _wizard(document)
    assert _states(message, _cli(document, tmp_path))
    if merge:
        assert _states(message, _merge(document))
DP, PC, SEARCH_SPACE = _DOORS.DP, _DOORS.PC, _DOORS.SEARCH_SPACE
SPIKING, LIF = "sync_simplemlp.json", "lif_mmixcore.json"
_FIXTURES = _DOORS._FIXTURES


def _float_weights(mode_at, wq_at, mode="phased"):
    """lif_mmixcore with ``weight_quantization: false`` at ``wq_at``, no ``weight_bits``, and ``pipeline_mode``
    declared at ``mode_at`` (``top``, a section, or None for undeclared)."""
    document = _load(LIF)
    document.pop("pipeline_mode")
    document[DP].pop("weight_quantization")
    document[PC].pop("weight_bits")
    if wq_at is not None:
        document[wq_at]["weight_quantization"] = False
    if mode_at is not None:
        (document if mode_at == "top" else document[mode_at])["pipeline_mode"] = mode
    return document


REPLACED_MODE = ("pipeline_mode='phased' is declared, and the run derives pipeline_mode='vanilla' from "
                 "weight_quantization=False: declare pipeline_mode='vanilla' or drop one of them")


class TestADeclaredDerivedKeyTheDerivationWouldReplace:
    """Review round 2, blocker: the derivation replaced a declared ``pipeline_mode`` (``phased`` under
    ``weight_quantization: false`` ran ``vanilla``) while the session kept the declared mode for the working
    directory and the run information. Each declarable derived key is judged against what the derivation derives
    from the judged declarations, and a declaration it would replace is refused by name at every door."""

    @pytest.mark.parametrize("mode_at", ["top", DP, PC])
    @pytest.mark.parametrize("wq_at", [DP, PC])
    def test_a_phased_mode_over_float_weights_refuses_at_every_door(self, mode_at, wq_at, tmp_path):
        """The merge-only run door holds no top level, so a top-level mode is the other doors' alone."""
        _refused_everywhere(_float_weights(mode_at, wq_at), REPLACED_MODE, tmp_path, merge=mode_at != "top")

    @pytest.mark.parametrize("wq_at", [DP, PC])
    def test_an_undeclared_mode_is_the_derived_one_at_the_session(self, wq_at, tmp_path, monkeypatch):
        """No mode declared and float weights: the session's mode, its working directory and the wizard's
        preview name the mode the run executes."""
        document = _float_weights(None, wq_at)
        _admitted_everywhere(document, tmp_path)
        session = _session(document, tmp_path / "s")
        assert session.parsed.pipeline_mode == session.pipeline.config["pipeline_mode"] == "vanilla"
        monkeypatch.chdir(tmp_path)
        assert parse_deployment_config(copy.deepcopy(document)).working_directory.endswith(
            "_vanilla_deployment_run")
        assert resolve_draft(copy.deepcopy(document)).resolved["pipeline_mode"] == "vanilla"

    @pytest.mark.parametrize("mode_at", ["top", DP, PC])
    def test_a_declared_vanilla_mode_alone_is_the_session_s(self, mode_at, tmp_path, monkeypatch):
        """Review round 2, minor (mutants V15, V11): a vanilla mode declared in either section, without a top-level
        one, is the session's mode and working directory, and the wizard preview's mode."""
        document = _float_weights(mode_at, None, mode="vanilla")
        _admitted_everywhere(document, tmp_path)
        session = _session(document, tmp_path / "s")
        assert session.parsed.pipeline_mode == session.pipeline.config["pipeline_mode"] == "vanilla"
        monkeypatch.chdir(tmp_path)
        parsed = parse_deployment_config(copy.deepcopy(document))
        assert os.path.basename(parsed.working_directory) == f"{document['experiment_name']}_vanilla_deployment_run"
        assert resolve_draft(copy.deepcopy(document)).resolved["pipeline_mode"] == "vanilla"

    def test_a_declared_mode_the_derivation_keeps_is_admitted(self, tmp_path):
        document = _float_weights(DP, DP, mode="vanilla")
        _admitted_everywhere(document, tmp_path)
        assert _session(document, tmp_path / "s").parsed.pipeline_mode == "vanilla"

    def test_the_rule_is_every_declarable_derived_key_s(self, tmp_path):
        """The same rule names a declared ``weight_quantization: true`` the vanilla mode replaces at the merge-only
        run door, which never applied the session's quantization assembly contract (the other doors state the
        contract first)."""
        document = _float_weights(DP, DP, mode="vanilla")
        document[DP]["weight_quantization"] = True
        message = ("weight_quantization=True is declared, and the run derives weight_quantization=False from "
                   "pipeline_mode='vanilla': declare weight_quantization=False or drop one of them")
        assert _merge(document) == [message]
        (contract,) = validate_deployment_config(copy.deepcopy(document))
        assert contract.startswith("quantization contract violation: explicit weight_quantization=true")
        assert _states(contract, _cli(document, tmp_path))

    def test_a_mode_laid_beneath_a_flat_build_is_no_declaration(self):
        """``build_flat_pipeline_config``'s mode is the mode of a document that declares none: the derivation may
        derive another from the declarations, as the run does."""
        flat = build_flat_pipeline_config({"weight_quantization": False}, {}, pipeline_mode="phased")
        assert flat["pipeline_mode"] == "vanilla" and flat["weight_quantization"] is False


CONTRADICTING_AXONS = ("max_axons=1000 contradicts the cores-derived value 512 (the largest per-core value across the "
                       "declared core types; the mapping uses the derived value). Drop max_axons to accept the "
                       "derivation, or fix the core grid.")
UNTILED_WINDOW = ("target_tq=6 must divide simulation_steps=32 (the QAT activation-quantization level must tile the "
                  "deployment spike window; the tier configs set them equal). Fix target_tq or simulation_steps so "
                  "simulation_steps % target_tq == 0.")


class TestThePlatformDerivationAtEveryDoor:
    """Review round 2, major: the platform derivation (the core maxima from the declared cores, the Tq | S
    contract) ran in the wizard's preview only, so the document door, the merge and the CLI path ran a declared
    maximum the mapping and the pricing ignore. Every door now derives it, as the wizard did."""

    def test_a_contradicting_core_maximum_refuses_with_the_wizard_s_message(self, tmp_path):
        document = _load(LIF)
        document[PC]["max_axons"] = 1000
        _refused_everywhere(document, CONTRADICTING_AXONS, tmp_path)

    def test_an_absent_core_maximum_is_derived_by_the_run(self, tmp_path):
        document = _load(LIF)
        del document[PC]["max_axons"], document[PC]["max_neurons"]
        _admitted_everywhere(document, tmp_path)
        flat = merge_pipeline_config(copy.deepcopy(document[DP]), copy.deepcopy(document[PC]))
        assert (flat["max_axons"], flat["max_neurons"]) == (512, 512)
        assert _session(document, tmp_path / "s").pipeline.config["max_axons"] == 512

    def test_a_window_the_level_does_not_tile_refuses_at_every_door(self, tmp_path):
        document = _load(LIF)
        document[PC]["target_tq"] = 6
        _refused_everywhere(document, UNTILED_WINDOW, tmp_path)


def _shipped_documents():
    root = _FIXTURES.parents[2]
    for path in sorted({*root.glob("templates/**/*.json"), *root.glob("examples/**/*.json"),
                        *root.glob("tests/fixtures/**/*.json")}):
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and isinstance(document.get(PC), dict):
            yield str(path.relative_to(root)), document


class TestTheShippedCoreMaxima:
    """[M1, owner sign-off of 2026-10-22, option (a)] The platform derivation refuses a declared core maximum that
    contradicts the cores at every door. The shipped documents' stale scalars are dropped in one named edit (the run
    derives them from the cores), so the three examples the base ran run again."""

    def test_no_shipped_document_declares_a_core_maximum_its_cores_contradict(self):
        refused = {}
        for name, document in _shipped_documents():
            platform = copy.deepcopy(document[PC])
            try:
                derive_platform_constraints(platform, cores_declared="cores" in platform)
            except ValueError as refusal:
                refused[name] = str(refusal)
        assert refused == {}

    @pytest.mark.parametrize("name", ["cifar100_run.json", "cifar10_run.json", "mnist_run.json"])
    def test_the_examples_the_base_ran_are_admitted_at_every_door(self, name, tmp_path):
        document = json.loads((_FIXTURES.parents[2] / "examples" / name).read_text(encoding="utf-8"))
        _admitted_everywhere(document, tmp_path)
        cores = document[PC]["cores"]
        flat = merge_pipeline_config(copy.deepcopy(document[DP]), copy.deepcopy(document[PC]))
        assert (flat["max_axons"], flat["max_neurons"]) == (max(c["max_axons"] for c in cores),
                                                            max(c["max_neurons"] for c in cores))


def _lif(edit):
    def build():
        document = _load(LIF)
        edit(document)
        return document
    return build


def _move(key, value, section=PC):
    def edit(document):
        document[DP].pop(key, None)
        document[section][key] = value
    return edit


def _top_seed_in(section):
    def edit(document):
        document.pop("seed")
        document[section]["seed"] = 9
    return edit


def _platform_search(document):
    document[DP].pop("hw_config_mode")
    document[PC]["hw_config_mode"] = "search"
    document[DP]["arch_search"] = {"pop_size": 4, "generations": 1}
    document[PC]["search_space"] = dict(SEARCH_SPACE)


def _platform_float_weights(document):
    document.pop("pipeline_mode")
    document[DP].pop("weight_quantization")
    document[PC].pop("weight_bits")
    document[PC]["weight_quantization"] = False


def _nested_bias(name, at):
    """``name`` with no flat ``has_bias``, cores that declare none, and the nested bias object ``{"has_bias": False}``
    in the section ``at`` (the run reads it: ``PLATFORM_BIAS_PATHS``)."""
    def build():
        document = _load(name)
        document[PC].pop("has_bias", None)
        for core in document[PC]["cores"]:
            core.pop("has_bias", None)
        document[at]["platform_constraints"] = {"has_bias": False}
        return document
    return build


_NESTED_BIAS = {f"nested_bias_in_{at}_{name}": _nested_bias(name, at) for name in (SPIKING, LIF) for at in (DP, PC)}


#: A declaration the run reads in the other section, and the keys it decides.
_PREVIEWS = {
    "platform_lr": (_lif(_move("lr", 0.01)), ("lr",)),
    "platform_batch_size": (_lif(_move("batch_size", 64)), ("batch_size",)),
    "platform_hardware_search": (_lif(_platform_search), ("hw_config_mode", "arch_search")),
    "deployment_parameters_seed": (_lif(_top_seed_in(DP)), ("seed",)),
    "platform_seed": (_lif(_top_seed_in(PC)), ("seed",)),
    "platform_float_weights": (_lif(_platform_float_weights), ("pipeline_mode", "weight_quantization")),
}


class TestTheWizardPreviewsTheSectionsTheRunTakes:
    """Review round 2, blocker: the wizard built its preview from each section's own registry keys, so a
    declaration the run reads in the other section, or one the run lays from a section, previewed otherwise than it
    ran (a quantized preview of a float run among them). The preview now takes the run's sections and derivation."""

    @pytest.mark.parametrize("case", sorted(_PREVIEWS))
    def test_the_payload_resolves_and_assembles_as_the_cli_session(self, case, tmp_path):
        build, keys = _PREVIEWS[case]
        document = build()
        payload = resolve_payload(copy.deepcopy(document))
        assert payload["ok"], payload["errors"]
        session = _session(document, tmp_path)
        for key in keys:
            assert payload["resolved"][key] == session.pipeline.config[key], key
        assert payload["pipeline"]["steps"] == [name for name, _ in session.pipeline.steps]

    @pytest.mark.parametrize("name", sorted([*(path.name for path in _FIXTURES.glob("*.json")), *_NESTED_BIAS]))
    def test_the_preview_is_the_run_s_flat_config(self, name):
        """Every fixture, and a nested bias object in either section (review round 3, blocker 2): the wizard's
        resolved config is the merge's flat config of the sections the session takes, the containers the run reads
        included, plus the top-level keys the wizard shows beside it."""
        document = _NESTED_BIAS[name]() if name in _NESTED_BIAS else _load(name)
        resolution = resolve_draft(copy.deepcopy(document))
        assert resolution.ok, resolution.errors
        dp, pc, _ = run_sections(document)
        flat = merge_pipeline_config(dp, pc)
        assert {key: value for key, value in resolution.resolved.items() if key in flat} == flat
        assert set(resolution.resolved) - set(flat) <= set(document)


class TestTheWizardStatesTheBuilderRegistrationWhereverTheRunReadsIt:
    """Review round 2, major: the wizard left the builder registration's refusals to its pretrained panel, which
    reads the deployment parameters only, so a platform-constraints declaration was admitted by the wizard and
    refused by every other door."""

    @pytest.mark.parametrize("key,value", [("preload_weights", True), ("pretrained_weight_set", "x")])
    def test_a_platform_declaration_gets_the_document_door_s_message_keyed(self, key, value):
        document = _load(SPIKING)
        document[PC][key] = value
        (message,) = validate_deployment_config(copy.deepcopy(document))
        payload = resolve_payload(copy.deepcopy(document))
        assert not payload["ok"]
        rows = [row for row in payload["errors"] if row["message"] == message]
        assert len(rows) == 1 and rows[0]["key"] == key

    @pytest.mark.parametrize("key,value", [("preload_weights", True), ("pretrained_weight_set", "x")])
    def test_a_deployment_parameters_declaration_is_stated_once(self, key, value):
        document = _load(SPIKING)
        document[DP][key] = value
        (message,) = validate_deployment_config(copy.deepcopy(document))
        messages = [row["message"] for row in resolve_payload(copy.deepcopy(document))["errors"]]
        assert messages.count(message) == 1

    @pytest.mark.parametrize("edits", [{}, {"preload_weights": True}, {"firing_mode": "TTFS"},
                                       {"preload_weights": True, "firing_mode": "TTFS"}], ids=repr)
    def test_the_rows_left_to_the_panel_are_the_panel_s(self, edits):
        """The wizard drops a document-door row only where the panel states it: a deployment-parameters legality row
        the registration does not decide (``firing_mode: TTFS`` on a LIF document) stays the wizard's own."""
        document = _load(LIF)
        document[DP].update(edits)
        panel = {row["message"] for row in pretrained_legality_errors(effective_with_builder(document), document)}
        assert resolve_module._panel_messages(parse_deployment_document(document).dp) == panel

    def test_every_key_the_registration_decides_is_one_the_panel_states(self):
        """The wizard leaves to the panel the rows the builder registration adds to the deployment parameters'
        declarations; every legality-bearing key whose legal set the registration changes is a panel key."""
        weights = {"id": "w", "label": "W", "task": "image classification", "dataset": "MNIST",
                   "input_shape": [1, 28, 28], "num_classes": 10, "source": "x"}
        decided, plain = set(), effective_view(_load(SPIKING)[DP])
        for sets in ([], [weights]):  # a builder that registers none, and one that registers a set
            enriched = {**plain, **ModelWorkloadProfile().config_updates(), "pretrained_weight_sets": sets}
            for key in legality_bearing_keys():
                for value in (True, False, "w", "x"):
                    if legality_errors(plain, {key: value}) != legality_errors(enriched, {key: value}):
                        decided.add(key)
        assert decided and decided <= set(PRETRAINED_LEGALITY_KEYS)


#: One unread declaration per class, equal to the value the run reads: (the refused path, the place it is declared
#: in, the place the run reads the key).
_EQUAL_UNREAD = {
    "section_key_at_the_top_level": ("batch_size", (), (DP,)),
    "top_level_key_in_a_section": ("deployment_parameters.data_provider_name", (DP,), ()),
    "under_the_platform_defaults": ("deployment_parameters.weight_bits", (DP,), (PC,)),
    "inside_an_unread_object": ("extra.layer.lr", ("extra", "layer"), (DP,)),
}


def _at(document, place):
    for segment in place:
        document = document.setdefault(segment, {})
    return document


class TestAnUnreadDeclarationEqualToTheValueTheRunReads:
    """Review round 2, major: a boundary finding for the owner's sign-off. An unread declaration equal to the value
    the run reads ran exactly on the base (the base CLI flat config equals that of the document without it); the
    generic unread rule now refuses it by name at every door, as it refuses an unread declaration of another value.
    The alternative, admitting an unread declaration equal to what the run reads, is recorded and not taken: the
    rule judges where a declaration is, and a declaration nothing reads states nothing the run runs."""

    @pytest.mark.parametrize("case", sorted(_EQUAL_UNREAD))
    def test_it_refuses_by_name_although_the_run_reads_an_equal_value(self, case, tmp_path):
        path, declared_at, read_at = _EQUAL_UNREAD[case]
        key = path.rsplit(".", 1)[-1]
        clean = _load(SPIKING)
        assert _cli(clean, tmp_path / "clean") == []
        document = copy.deepcopy(clean)
        _at(document, declared_at)[key] = _at(clean, read_at)[key]  # the value the run reads, copied where none reads
        (message,) = [m for m in validate_deployment_config(copy.deepcopy(document)) if m.startswith(f"{path}: ")]
        _refused_everywhere(document, message, tmp_path, merge=path.startswith((DP, PC)))


class TestTwoEqualCopiesInAnotherOrder:
    def test_mapping_order_does_not_count(self, tmp_path):
        """Review round 2, minor (mutant V26): two equal objects whose fields are in another order are one
        reading at every door."""
        document = _load(SPIKING)
        document[DP]["preprocessing"] = {"interpolation": "bilinear", "resize_to": 32}
        document[PC]["preprocessing"] = {"resize_to": 32, "interpolation": "bilinear"}
        _admitted_everywhere(document, tmp_path)


class TestTheRecipesAreJudgedByTheirConsumersFields:
    """Review round 2, minor: a recipe field the consumer (``TrainingRecipe``) does not read was dropped silently at
    the training and tuning steps, and a recipe of ``[]`` read as no recipe; both doors now judge a recipe by the
    consumer's fields and their types."""

    @pytest.mark.parametrize("key,value,start", [
        ("training_recipe", {"optimiser": "sgd"}, "training_recipe.optimiser is not a declared field"),
        ("tuning_recipe", {"lr_decay": 0.5}, "tuning_recipe.lr_decay is not a declared field"),
        ("training_recipe", [], "training_recipe must be an object, got list ([])"),
        ("tuning_recipe", {"optimizer": 7}, "tuning_recipe.optimizer must be str, got int (7)"),
        ("training_recipe", {"betas": ["a", 0.9]}, "training_recipe.betas[0] must be float, got str ('a')"),
    ], ids=lambda x: repr(x)[:40])
    def test_a_field_the_consumer_does_not_read_or_a_non_object_refuses(self, key, value, start, tmp_path):
        document = _load(SPIKING)
        document[DP][key] = value
        (message,) = [m for m in validate_deployment_config(copy.deepcopy(document)) if m.startswith(start)]
        _refused_everywhere(document, message, tmp_path)

    def test_the_schema_s_fields_are_the_consumer_s(self):
        assert set(RECIPE_SCHEMA.fields) == {field.name for field in dataclasses.fields(TrainingRecipe)}
        assert REGISTRY["training_recipe"].value_schema is REGISTRY["tuning_recipe"].value_schema is RECIPE_SCHEMA

    def test_every_field_the_consumer_reads_is_admitted(self, tmp_path):
        document = _load(SPIKING)
        document[DP]["training_recipe"].update(momentum=0.8, no_decay_keywords=["bias"], label_smoothing=None)
        document[DP]["tuning_recipe"] = {}
        _admitted_everywhere(document, tmp_path)
        assert not any(path.startswith(f"{DP}.training_recipe") for path in resolve_draft(document).unknown_keys)


class TestANullNonDeclarableKey:
    """Review round 2, minor: a null declaration is unset at every door, so the not-declarable rule refuses only a
    value (a runtime fact at the top level, a derivation-owned key in either section)."""

    @pytest.mark.parametrize("place,key", [
        ("top", "device"), (DP, "activation_quantization"), (PC, "activation_quantization"),
        (PC, "pretrained_weight_sets"), (PC, "cycle_accurate_lif_forward"),
    ])
    def test_it_is_unset(self, place, key, tmp_path):
        document = _load(SPIKING)
        (document if place == "top" else document[place])[key] = None
        _admitted_everywhere(document, tmp_path)


class _Recorder:
    """The data provider factory's arguments as the session passes them (dataset I/O replaced by the stub)."""

    seen = {}

    def __init__(self, name, datasets_path, **kwargs):
        _Recorder.seen = {"name": name, "datasets_path": datasets_path, **kwargs}
        self._provider = _DOORS._Provider({DP: {k: kwargs.get(k) for k in ("preprocessing", "batch_size")}, PC: {}})

    def create(self):
        return self._provider.create()


class TestAnEmptyPath:
    """Review round 2, minor: an empty path names no directory, so a run reads it as unset (``runtime.unset``):
    ``datasets_path`` takes the meaning its registry states ('./datasets'), and ``generated_files_path``, which a
    run reads with no value of its own, is missing at every door."""

    def test_an_empty_datasets_path_reads_its_stated_meaning(self, tmp_path, monkeypatch):
        document = _load(SPIKING)
        document["datasets_path"] = ""
        assert validate_deployment_config(copy.deepcopy(document)) == []
        monkeypatch.setattr(session_module, "BasicDataProviderFactory", _Recorder)
        parse_deployment_config({**document, "_working_directory": str(tmp_path)})
        assert _Recorder.seen["datasets_path"] == "./datasets"

    def test_only_an_empty_path_is_unset(self):
        """The rule is the PATH type's: an empty path names no directory, while an empty string of another type is a
        value its schema judges."""
        assert unset("datasets_path", "") and unset("generated_files_path", "") and unset("datasets_path", None)
        assert not unset("experiment_name", "") and not unset("data_provider_name", "") and not unset("datasets_path", "x")

    def test_an_empty_generated_files_path_is_missing(self, tmp_path):
        document = _load(SPIKING)
        document["generated_files_path"] = ""
        message = "Missing top-level key: generated_files_path"
        assert message in validate_deployment_config(copy.deepcopy(document))
        assert message in _wizard(document)
        assert message in _parts(_cli(document, tmp_path))
        assert list(tmp_path.iterdir()) == []  # refused before the session writes a working directory


K1 = ("weight_bits=1 is below the minimum 2: the signed weight grid (TensorQuantization) of 1 bit has no nonzero level "
      "(q_max = 2^0 - 1 = 0), so weight quantization would divide by zero: declare weight_bits >= 2, or float "
      "weights (pipeline_mode 'vanilla' or weight_quantization false)")


class TestAStructuralPlatformBody:
    def test_the_document_door_judges_the_applied_schema_of_its_body(self, tmp_path):
        """Review round 2, minor: the wizard's structural platform body (``mode`` user) is judged by the schemas
        its consumers apply, at the document door and in the wizard; the CLI path does not unpack the shape (the
        carried structural-shape boundary, G4/CL9), so it runs the default weight_bits."""
        document = _load(SPIKING)
        body = {**document[PC], "weight_bits": 1}
        document[PC] = {"mode": "user", "user": body}
        assert K1 in validate_deployment_config(copy.deepcopy(document))
        assert K1 in _wizard(document)

    def test_the_wizard_previews_the_parsed_body(self):
        """The wizard's preview takes the parser's flat body for a structural platform shape (the run's sections
        hold the shape itself, which only the wizard nests)."""
        document = _load(SPIKING)
        document[PC] = {"mode": "user", "user": {**document[PC], "simulation_steps": 16, "target_tq": 16}}
        resolution = resolve_draft(copy.deepcopy(document))
        assert resolution.ok, resolution.errors
        assert (resolution.resolved["simulation_steps"], resolution.resolved["target_tq"]) == (16, 16)


class TestANestedBiasObjectThatIsNoObject:
    """Review round 2, minor: the one nested object the bias rule reads is refused by name when it is no object,
    in either section, never silently ignored."""

    @pytest.mark.parametrize("section", [DP, PC])
    @pytest.mark.parametrize("value", ["x", 7, [1], True], ids=repr)
    def test_it_refuses_by_name_at_every_door(self, section, value, tmp_path):
        document = _load(SPIKING)
        document[section]["platform_constraints"] = value
        message = (f"{section}.platform_constraints must be an object holding has_bias, "
                   f"got {type(value).__name__} ({value!r})")
        _refused_everywhere(document, message, tmp_path)

    def test_one_a_platform_object_replaces_refuses_too(self, tmp_path):
        document = _load(SPIKING)
        document[DP]["platform_constraints"] = "x"
        document[PC]["platform_constraints"] = {"has_bias": True}
        _refused_everywhere(document, f"{DP}.platform_constraints must be an object holding has_bias, got str ('x')",
                            tmp_path)

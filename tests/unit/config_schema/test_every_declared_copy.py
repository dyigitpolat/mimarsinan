"""Every copy of a declaration is judged, and the preview takes every container the run reads (the S0-B and G4
joint closure, review round 3).

- **A shadowed copy** (review round 3, blocker 1): a key declared in ``deployment_parameters`` while
  ``platform_constraints`` declares it too is read from the platform copy; a deployment-parameters copy its own
  schema refuses was judged by no rule (the two-readings rule dropped it, ``section_declarations`` judges a key in its
  own section only, and no default layer replaces these keys). It is now refused by its schema, prefixed with its
  place, at every door.
- **The nested bias object** (review round 3, blocker 2): the wizard's preview kept only each section's registry
  keys, so it dropped ``<section>.platform_constraints``, the nested bias object the run reads
  (``PLATFORM_BIAS_PATHS``). The preview now takes each section as the run takes it.
- **The round-2 rules' unpinned parts** (review round 3, minor): K1 under an ``auto`` body and the deployment
  parameters it is judged under, a null nested bias container, an ``auto`` body's preview.
- **A null retired key** (review round 3, minor) is unset at every door, as every other rule reads a null.
- **The recipes** (review round 3, minor): a null field the consumer reads as a value it refuses, and the bounds and
  lengths the consumer applies, are refused by name; an empty recipe is no recipe, as its registry entry states.
"""

import copy
import importlib.util
from pathlib import Path

import pytest

from mimarsinan.config_schema.defaults import get_default_platform_constraints
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.declaration_judge import declared_sections_errors
from mimarsinan.config_schema.registry.entries_write import RECIPE_SCHEMA, platform_bias_mode
from mimarsinan.config_schema.registry.retired_keys import RETIREMENT_SCOPES, retired_keys_in_scope
from mimarsinan.config_schema.registry.value_schema import describe, entry_value_errors, schema_of
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.gui.wizard.schema_api import resolve_payload
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved


def _sibling():
    """The four-door harness of ``test_one_run_reading`` (its helpers only; its tests are collected there)."""
    spec = importlib.util.spec_from_file_location(
        "_one_run_reading_doors_r3", Path(__file__).with_name("test_one_run_reading.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DOORS = _sibling()
_load, _session, _cli, _merge, _wizard, _parts = (
    _DOORS._load, _DOORS._session, _DOORS._cli, _DOORS._merge, _DOORS._wizard, _DOORS._parts)
_refused_everywhere, _admitted_everywhere = _DOORS._refused_everywhere, _DOORS._admitted_everywhere
DP, PC, SPIKING, LIF = _DOORS.DP, _DOORS.PC, _DOORS.SPIKING, _DOORS.LIF
MVM = "mvm_lenet5.json"

#: Values a run's schemas refuse for most keys; each case takes the first its key's schema refuses.
_REFUSED = ("false", 7.5, [], True, {})
#: Candidates for a value a key's schema admits, after the fixture's own declaration and the key's options (the
#: object-valued keys' from ``test_wave1_declarations.LEGAL``).
_ADMITTED = (True, 8, 784, 0.5, "x", {}, [1], {"mode": "serial"}, {"step": "index", "index": 1},
             {"adc_bits": 6, "range": "fixed", "range_value": 4.0, "mult_bits": 16},
             {"file": "/tmp/state.json", "sha256": "ab" * 32},  # Wave 2 WI-5a: quantizer_state
             {"classes": ["L", "E", "P"]},  # Wave 2 WI-5e: onchip_arithmetic
             [{"layers": ["fc"], "weight_bits": 2}])  # Wave 2 WI-5g: layer_core_types


def _shadowable_keys():
    """Every key the run reads in ``deployment_parameters`` that only a platform declaration of it shadows, outside
    its own section: a platform key no platform default replaces, and a top-level key the run lays beneath the
    deployment parameters."""
    defaults = get_default_platform_constraints()
    return sorted(key for key, entry in REGISTRY.items()
                  if (entry.section == PC and key not in defaults) or entry.laid_beneath is not None)


def _shadowed_keys():
    """``_shadowable_keys`` a value schema judges (a key no value schema judges is its consumer's: a differing copy
    is the two-readings rule's, ``test_a_key_no_schema_judges_is_the_two_readings_rule_s``)."""
    return [key for key in _shadowable_keys() if schema_of(REGISTRY[key]) is not None]


def _fixture_of(key):
    """A fixture of the key's own domain (the wizard keeps another domain's keys dormant)."""
    return MVM if REGISTRY[key].domain == "value" else SPIKING


def _admitted_value(key, document):
    entry = REGISTRY[key]
    candidates = (document[PC].get(key), document.get(key), *(entry.resolved_options() or ()), *_ADMITTED)
    return next(value for value in candidates if value is not None and not entry_value_errors(entry, value))


def _shadow(name, key, bad):
    """``name`` with ``key`` declared in the platform constraints by a value its schema admits (the fixture's own,
    else another), and in the deployment parameters by ``bad``; no top-level copy."""
    document = _load(name)
    legal = _admitted_value(key, document)
    document.pop(key, None)
    document[PC][key] = legal
    document[DP][key] = bad
    return document


def _shadow_message(key, bad):
    return f"{DP}.{key}: " + "; ".join(entry_value_errors(REGISTRY[key], bad))


class TestACopyThePlatformDeclarationShadows:
    """Review round 3, blocker 1: a deployment-parameters copy its schema refuses, shadowed by a platform
    declaration of the key, was silently ignored at every door (``has_bias: "false"`` ran ``True``)."""

    @pytest.mark.parametrize("name", [SPIKING, LIF, MVM])
    @pytest.mark.parametrize("key", ["has_bias", "max_axons", "max_neurons"])
    @pytest.mark.parametrize("bad", ["false", 7.5, [], True], ids=repr)
    def test_it_refuses_by_its_schema_at_every_door(self, name, key, bad, tmp_path):
        document = _shadow(name, key, bad)
        if not entry_value_errors(REGISTRY[key], bad):  # has_bias: True is a legal copy, the same reading
            assert document[PC][key] is True
            _admitted_everywhere(document, tmp_path)
            return
        _refused_everywhere(document, _shadow_message(key, bad), tmp_path)

    @pytest.mark.parametrize("key", _shadowed_keys())
    def test_every_shadowed_key_refuses_by_name(self, key, tmp_path):
        """The generic form: every platform key no platform default replaces, and every top-level key a run lays,
        declared in the deployment parameters with a value its schema refuses under a platform declaration. The
        document door and the wizard state the schema's message with the place; the CLI path and the run door state
        it too, unless the target expansion refuses the same copy by name at its place first (a target key, which
        it reads in the platform constraints only)."""
        bad = next(value for value in _REFUSED if entry_value_errors(REGISTRY[key], value))
        document = _shadow(_fixture_of(key), key, bad)
        message = _shadow_message(key, bad)
        assert message in validate_deployment_config(copy.deepcopy(document))
        assert message in _wizard(document)
        for messages in (_cli(document, tmp_path), _merge(document)):
            parts = _parts(messages)
            assert (all(part in parts for part in message.split("; "))
                    or any(part.startswith(f"{DP}.{key}: ") for part in parts)), messages

    @pytest.mark.parametrize("key", [key for key in _shadowable_keys() if key not in _shadowed_keys()])
    def test_a_key_no_schema_judges_is_the_two_readings_rule_s(self, key, tmp_path):
        """A key no value schema judges admits every value, so a differing deployment-parameters copy under a
        platform declaration is two readings, refused by name at the document door and the run door."""
        document = _shadow(SPIKING, key, {"shadowed": 1})
        expected = f"{key} is declared in {DP} as"
        assert any(message.startswith(expected) for message in validate_deployment_config(copy.deepcopy(document)))
        assert any(part.startswith(expected) for part in _merge(document))

    def test_the_merge_s_judge_states_it_on_the_two_sections(self):
        message = _shadow_message("has_bias", "false")
        assert message in declared_sections_errors({"has_bias": "false"}, {"has_bias": True})
        assert message not in declared_sections_errors({}, {"has_bias": True})

    @pytest.mark.parametrize("key,bad", [("has_bias", "false"), ("max_axons", 7.5)])
    def test_a_copy_judged_elsewhere_is_stated_once(self, key, bad, tmp_path):
        """The layout's own copy (no platform declaration) and a key's copy in its own section keep the one
        unprefixed message their judgements state; the place prefix names only a copy no other judgement reads."""
        document = _load(SPIKING)
        document[PC].pop(key)
        document[DP][key] = bad
        messages = validate_deployment_config(copy.deepcopy(document))
        (unprefixed,) = entry_value_errors(REGISTRY[key], bad)
        assert unprefixed in messages and not any(message.startswith(f"{DP}.{key}: ") for message in messages)
        own = _load(SPIKING)
        own[PC][key] = bad
        own[DP][key] = True if key == "has_bias" else 784
        messages = validate_deployment_config(copy.deepcopy(own))
        assert messages.count(unprefixed) == 1
        assert not any(message.startswith(f"{PC}.{key}: ") for message in messages)

    def test_a_top_level_copy_keeps_the_top_level_judgement_s_message(self, tmp_path):
        """A top-level copy a section's copy shadows is the session's top-level judgement's, stated once."""
        document = _load(SPIKING)
        document["seed"] = 7.5
        document[DP]["seed"] = 3
        (message,) = entry_value_errors(REGISTRY["seed"], 7.5)
        assert validate_deployment_config(copy.deepcopy(document)) == [message]
        assert _parts(_cli(document, tmp_path)) == [message]

    def test_a_deployment_parameters_key_shadowed_in_its_own_section_keeps_its_message(self, tmp_path):
        """A deployment-parameters key's own copy is judged as written (``section_declarations``), unprefixed."""
        document = _load(SPIKING)
        document[PC]["lr"] = 0.01
        document[DP]["lr"] = "x"
        (message,) = entry_value_errors(REGISTRY["lr"], "x")
        _refused_everywhere(document, message, tmp_path)
        assert not any(m.startswith(f"{DP}.lr: ") for m in validate_deployment_config(copy.deepcopy(document)))


def _nested_bias(name, at):
    """``name`` with no flat ``has_bias`` and cores that declare none, and the nested bias object
    ``{"has_bias": False}`` in the section ``at``."""
    document = _load(name)
    document[PC].pop("has_bias", None)
    for core in document[PC]["cores"]:
        core.pop("has_bias", None)
    document[at]["platform_constraints"] = {"has_bias": False}
    return document


class TestThePreviewTakesTheNestedBiasObject:
    """Review round 3, blocker 2: the preview kept each section's registry keys only, so it dropped the nested bias
    object the run reads: the preview showed bias lanes, the run built none."""

    @pytest.mark.parametrize("name", [SPIKING, LIF])
    @pytest.mark.parametrize("at", [DP, PC])
    def test_the_resolved_config_reads_the_cli_s_bias(self, name, at, tmp_path):
        document = _nested_bias(name, at)
        _admitted_everywhere(document, tmp_path)
        resolved = resolve_draft(copy.deepcopy(document)).resolved
        config = _session(document, tmp_path / "s").pipeline.config
        assert platform_bias_mode(config) is False
        assert platform_bias_mode(resolved) is platform_bias_mode(config)
        cores = [core["has_bias"] for core in build_platform_constraints_resolved(dict(resolved))["cores"]]
        assert cores == [core["has_bias"] for core in build_platform_constraints_resolved(dict(config))["cores"]]
        assert cores and not any(cores)

    @pytest.mark.parametrize("at", [DP, PC])
    def test_the_payload_derives_from_the_run_s_reading(self, at, tmp_path):
        """The payload's views derive from the resolution (its pipeline steps, its derived rows); its ``resolved``
        projection onto registry keys is the frontend's field values (carried, G4/CL9)."""
        document = _nested_bias(LIF, at)
        payload = resolve_payload(copy.deepcopy(document))
        assert payload["ok"], payload["errors"]
        session = _session(document, tmp_path)
        assert payload["pipeline"]["steps"] == [name for name, _ in session.pipeline.steps]


class TestTheRoundTwoRulesPinned:
    """Review round 3, minor (mutants W4, W5, W6, W9): the parts of the round-2 rules no test pinned."""

    K1 = ("weight_bits=1 is below the minimum 2: the signed weight grid (TensorQuantization) of 1 bit has no nonzero "
          "level (q_max = 2^0 - 1 = 0), so weight quantization would divide by zero: declare weight_bits >= 2, or "
          "float weights (pipeline_mode 'vanilla' or weight_quantization false)")

    def test_k1_judges_an_auto_body(self):
        """W4: the structural K1 judges an ``auto`` body as it judges a ``user`` one."""
        document = _load(SPIKING)
        document[PC] = {"mode": "auto", "auto": {"fixed": {**document[PC], "weight_bits": 1}}}
        assert self.K1 in validate_deployment_config(copy.deepcopy(document))
        assert self.K1 in _wizard(document)

    @pytest.mark.parametrize("mode", [None, "vanilla"])
    def test_k1_reads_the_deployment_parameters_weight_quantization(self, mode):
        """W5: a structural body's K1 is judged under the deployment parameters' declarations: float weights
        declared there apply no weight grid, so a 1-bit body is admitted."""
        document = _load(SPIKING)
        document.pop("pipeline_mode")
        if mode is not None:
            document["pipeline_mode"] = mode
        document[DP]["weight_quantization"] = False
        document[PC] = {"mode": "user", "user": {**document[PC], "weight_bits": 1}}
        assert validate_deployment_config(copy.deepcopy(document)) == []
        assert _wizard(document) == []

    @pytest.mark.parametrize("section", [DP, PC])
    def test_a_null_nested_bias_container_is_unset(self, section, tmp_path):
        """W6: a null ``platform_constraints`` nested in a section is unset, as every null declaration is."""
        document = _load(SPIKING)
        document[section]["platform_constraints"] = None
        _admitted_everywhere(document, tmp_path)

    def test_the_preview_takes_an_auto_body(self):
        """W9: the wizard previews an ``auto`` body's fixed declarations, as it previews a ``user`` body."""
        document = _load(SPIKING)
        document[PC] = {"mode": "auto", "auto": {"fixed": {**document[PC], "simulation_steps": 16,
                                                           "target_tq": 16}}}
        resolution = resolve_draft(copy.deepcopy(document))
        assert resolution.ok, resolution.errors
        assert (resolution.resolved["simulation_steps"], resolution.resolved["target_tq"]) == (16, 16)


_RETIRED = sorted((scope, key) for scope in RETIREMENT_SCOPES for key in retired_keys_in_scope(scope))


class TestANullRetiredKey:
    """Review round 3, minor: the retired-key rule refused a null declaration (``spiking_mode: null`` with
    "unknown spiking_mode 'None'"), which every other rule and the CLI read as unset."""

    @pytest.mark.parametrize("scope,key", _RETIRED)
    @pytest.mark.parametrize("own", [True, False], ids=["own_section", "other_section"])
    def test_it_is_unset_at_every_door(self, scope, key, own, tmp_path):
        document = _load(LIF)
        section = scope if own else next(name for name in RETIREMENT_SCOPES if name != scope)
        document[section][key] = None
        _admitted_everywhere(document, tmp_path)

    def test_a_retired_value_still_refuses(self):
        document = _load(LIF)
        document[DP]["lif_per_hop_retiming"] = False
        assert any(message.startswith("lif_per_hop_retiming is retired")
                   for message in validate_deployment_config(copy.deepcopy(document)))


class TestTheRecipesAreJudgedAsTheirConsumerReadsThem:
    """Review round 3, minor: a null recipe field reaches ``TrainingRecipe`` as None, which the trainer refuses only
    at training (AttributeError, TypeError), and the consumer's bounds and lengths were not judged."""

    @pytest.mark.parametrize("key", ["training_recipe", "tuning_recipe"])
    @pytest.mark.parametrize("field,value,message", [
        ("optimizer", None, "{key}.optimizer must be str, got NoneType (None)"),
        ("weight_decay", None, "{key}.weight_decay must be float, got NoneType (None)"),
        ("betas", None, "{key}.betas must be a list, got NoneType (None)"),
        ("no_decay_keywords", None, "{key}.no_decay_keywords must be a list, got NoneType (None)"),
        ("betas", [0.9], "{key}.betas must be a list of at least 2 item(s)"),
        ("betas", [0.9, 0.99, 0.5], "{key}.betas must be a list of at most 2 item(s)"),
        ("betas", [0.9, 1.0], "{key}.betas[1]=1.0 must be below 1.0 (exclusive bound)"),
        ("label_smoothing", 2.0, "{key}.label_smoothing=2.0 is above the maximum 1.0"),
        ("warmup_ratio", 7.0, "{key}.warmup_ratio=7.0 is above the maximum 1.0"),
        ("weight_decay", -1.0, "{key}.weight_decay=-1.0 is below the minimum 0.0"),
        ("momentum", -0.5, "{key}.momentum=-0.5 is below the minimum 0.0"),
        ("grad_clip_norm", -1.0, "{key}.grad_clip_norm=-1.0 is below the minimum 0.0"),
    ], ids=lambda x: repr(x)[:30])
    def test_a_value_the_consumer_refuses_refuses_at_both_doors(self, key, field, value, message, tmp_path):
        document = _load(SPIKING)
        document[DP][key] = {**document[DP][key], field: value}
        _refused_everywhere(document, message.format(key=key), tmp_path)

    @pytest.mark.parametrize("key", ["training_recipe", "tuning_recipe"])
    def test_a_null_label_smoothing_is_the_consumer_s_none(self, key, tmp_path):
        """``TrainingRecipe.label_smoothing`` is Optional: a null is the consumer's own value (no smoothing)."""
        document = _load(SPIKING)
        document[DP][key] = {**document[DP][key], "label_smoothing": None}
        _admitted_everywhere(document, tmp_path)
        assert build_recipe(document[DP], key=key).label_smoothing is None

    @pytest.mark.parametrize("key", ["training_recipe", "tuning_recipe"])
    def test_an_empty_recipe_is_no_recipe_as_its_entry_states(self, key, tmp_path):
        """``{}`` is the one declaration of the trainer's legacy path (``build_recipe`` reads it as no recipe; an
        absent recipe is the default one): admitted, and stated in the registry entry."""
        document = _load(SPIKING)
        document[DP][key] = {}
        _admitted_everywhere(document, tmp_path)
        assert build_recipe(document[DP], key=key) is None
        assert "An empty recipe ({}) is no recipe" in REGISTRY[key].doc

    def test_every_field_but_label_smoothing_reads_null_as_a_value(self):
        assert {name for name, schema in RECIPE_SCHEMA.fields.items() if schema.null_unset} == {"label_smoothing"}

    def test_the_declaration_table_states_the_length(self):
        """The declaration table's cell of a list of fixed length names it (``describe``)."""
        assert describe(RECIPE_SCHEMA.fields["betas"]) == "list of FLOAT in [0, 1) (2 items)"

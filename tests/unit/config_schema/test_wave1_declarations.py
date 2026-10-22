"""Wave 1 declarations: registry entries, both judges, the resolver passthrough and the census type."""

import ast
import copy
import inspect
import json
import os
import textwrap
from typing import get_args

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.nevresim import connectivity
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import (
    REGISTRY,
    Category,
    parse_deployment_document,
    serialize_registry,
)
from mimarsinan.config_schema import validation as validation_module
from mimarsinan.config_schema.registry import consumer_reading as consumer_reading_module
from mimarsinan.config_schema.registry.build import passthrough_keys
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry import declaration_judge as declaration_judge_module
from mimarsinan.config_schema.registry.declaration_judge import declared_sections_errors, sections_run_layout
from mimarsinan.config_schema.registry import entries_write as entries_write_module
from mimarsinan.config_schema.registry.entries_write import (
    conflicting_platform_bias_error,
    platform_bias_declarations,
    platform_bias_mode,
)
from mimarsinan.config_schema.registry.parse import CORE_FIELDS
from mimarsinan.config_schema.registry.value_schema import (
    DeclarationValueError,
    schema_of,
    value_errors,
)
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config, validate_merged_config
from mimarsinan.deployment_record.build.from_mapping import (
    placement_record_from_mapping,
    schedule_record_from_mapping,
)
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.layout_types import (
    LayoutSoftCoreSpec,
    core_types_from_declaration,
)
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.period_census import (
    PassBoundaryCensus,
    PassCensus,
    PassPeriodCensus,
    PeriodCensusError,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core import platform_constraints_resolver as resolver_module
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
    judged_passthrough,
    resolve_bias_mode,
)
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_MODES,
    S_ALLOCATION_SUPPORTED_MODES,
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
    unsupported_s_allocation_error,
)

_FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs"
)
_PER_LAYER_S_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "templates", "mnist_mmixcore_per_layer_s_uniform.json"
)

WAVE1_KEYS = (
    "decompose_leaf_modules", "target", "target_overrides", "target_claim_class",
    "target_faithfulness_class", "readout_law", "inference_batch_K", "write_overlap",
    "write_ports", "inference_rate_hz", "min_lifetime_hours", "residency_plan",
)
PLATFORM_KEYS = tuple(k for k in WAVE1_KEYS if k != "decompose_leaf_modules")
DERIVED_KEYS = ("target_claim_class", "target_faithfulness_class")

LEGAL = {
    "readout_law": {"adc_bits": 6, "range": "fixed", "range_value": 4.0, "mult_bits": 16},
    "inference_batch_K": 16,
    "write_overlap": {"mode": "overlapped", "shadow_capacity_fraction": 0.25},
    "write_ports": 2,
    "inference_rate_hz": 30.0,
    "min_lifetime_hours": 43800.0,
    "residency_plan": {"step": "index", "index": 1},
}
WRITE_FIELDS = {"write_granularity": "row", "write_units_in_parallel": 4}


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _every_declarable_key_config():
    config = _load("mvm_aq_lenet5.json")
    config["deployment_parameters"]["decompose_leaf_modules"] = True
    config["deployment_parameters"]["allow_scheduling"] = True
    pc = config["platform_constraints"]
    pc.update(copy.deepcopy(LEGAL))
    for core in pc["cores"]:
        core.update(WRITE_FIELDS)
    return config


def _flat(config, **pc_extra):
    pc = {**config["platform_constraints"], **pc_extra}
    return merge_pipeline_config(dict(config["deployment_parameters"]), pc)


def _resolve(config, **pc_extra):
    return build_platform_constraints_resolved(_flat(config, **pc_extra))


class TestRegistryEntries:
    def test_every_wave1_key_is_registered_with_no_default(self):
        for key in WAVE1_KEYS:
            assert key in REGISTRY, key
            assert not REGISTRY[key].has_default(), key

    def test_sections_and_groups_follow_the_key_contract(self):
        sections = {
            "decompose_leaf_modules": ("deployment_parameters", "mapping_strategy"),
            "target": ("platform_constraints", "hardware"),
            "target_overrides": ("platform_constraints", "hardware"),
            "readout_law": ("platform_constraints", "hardware"),
            "inference_batch_K": ("platform_constraints", "mapping_strategy"),
            "write_overlap": ("platform_constraints", "hardware"),
            "write_ports": ("platform_constraints", "hardware"),
            "inference_rate_hz": ("platform_constraints", "mapping_strategy"),
            "min_lifetime_hours": ("platform_constraints", "hardware"),
            "residency_plan": ("platform_constraints", "mapping_strategy"),
        }
        for key, (section, group) in sections.items():
            assert (REGISTRY[key].section, REGISTRY[key].group) == (section, group), key
        assert REGISTRY["readout_law"].domain == "value"

    def test_the_passthrough_set_is_the_wave1_platform_keys(self):
        assert set(passthrough_keys()) == set(PLATFORM_KEYS)

    def test_derived_keys_are_not_declarable(self):
        for key in DERIVED_KEYS:
            entry = REGISTRY[key]
            assert entry.category is Category.DERIVED and not entry.declarable, key

    def test_core_type_fields_include_the_write_fields(self):
        assert CORE_FIELDS == {
            "max_axons", "max_neurons", "count", "has_bias",
            "write_granularity", "write_units_in_parallel",
        }


class TestRoundTrip:
    """Acceptance 1, after test_a_physics_declaring_config_is_representable."""

    def test_every_declarable_key_round_trips(self):
        config = _every_declarable_key_config()
        parsed = parse_deployment_document(config)
        assert parsed.unknown == []
        emitted = emit_deployment_config(config)
        assert json.dumps(emitted, sort_keys=True) == json.dumps(config, sort_keys=True)
        resolution = resolve_draft(config)
        assert resolution.errors == []
        assert resolution.unknown_keys == []
        assert validate_deployment_config(config) == []

    @pytest.mark.parametrize("key", DERIVED_KEYS)
    def test_a_derived_key_in_a_document_is_refused_by_name(self, key):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"][key] = "anything"
        assert any(key in m for m in validate_deployment_config(config))
        resolution = resolve_draft(config)
        assert any(row["key"] == key for row in resolution.errors)
        assert key not in emit_deployment_config(config)["platform_constraints"]

    def test_an_unknown_core_type_field_is_still_flagged(self):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"]["cores"][0]["wat"] = 1
        assert parse_deployment_document(config).unknown == [
            "platform_constraints.cores[0].wat"
        ]


#: A 400-digit integer, as json.loads reads it: an int too large for a float.
HUGE_JSON_INT = json.loads("1" + "0" * 400)

ILLEGAL = (
    ("readout_law", {"adc_bits": 1, "range": "calibrated", "mult_bits": 8}, "readout_law.adc_bits"),
    ("inference_batch_K", 0, "inference_batch_K"),
    ("write_overlap", {"mode": "overlapped", "shadow_capacity_fraction": 1.0},
     "write_overlap.shadow_capacity_fraction"),
    ("readout_law", {"adc_bits": 4, "range": "fixed", "range_value": 0, "mult_bits": 8},
     "readout_law.range_value"),
    ("readout_law", {"adc_bits": 4, "range": "fixed", "mult_bits": 8}, "readout_law.range_value"),
    ("target_overrides", {"weight_bits": {"value": 6}}, "target_overrides.weight_bits.note"),
    ("inference_rate_hz", 0, "inference_rate_hz"),
    ("residency_plan", {"step": "greedy"}, "residency_plan.step"),
    ("residency_plan", {"step": "index"}, "residency_plan.index"),
    ("residency_plan", {"step": "index", "index": -1}, "residency_plan.index"),
    # Non-finite floats: every comparison with NaN is False, and json.load reads NaN and Infinity.
    ("inference_rate_hz", float("nan"), "inference_rate_hz"),
    ("min_lifetime_hours", float("inf"), "min_lifetime_hours"),
    ("write_overlap", {"mode": "overlapped", "shadow_capacity_fraction": float("nan")},
     "write_overlap.shadow_capacity_fraction"),
    ("readout_law", {"adc_bits": 4, "range": "fixed", "range_value": float("nan"), "mult_bits": 8},
     "readout_law.range_value"),
    # JSON integers beyond float range (json.loads reads them as int): judged exactly, by name.
    ("write_overlap", {"mode": "overlapped", "shadow_capacity_fraction": HUGE_JSON_INT},
     "write_overlap.shadow_capacity_fraction"),
    ("inference_rate_hz", -HUGE_JSON_INT, "inference_rate_hz"),
)


class TestBothJudges:
    """Acceptance 2 and 7: every illegal value refuses by name at the document and the run door."""

    @pytest.mark.parametrize("key,value,named", ILLEGAL, ids=lambda x: str(x)[:40])
    def test_document_door(self, key, value, named):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"][key] = value
        assert any(m.startswith(named) for m in validate_deployment_config(config)), named

    @pytest.mark.parametrize("key,value,named", ILLEGAL, ids=lambda x: str(x)[:40])
    def test_resolver_door(self, key, value, named):
        with pytest.raises(DeclarationValueError, match=named.replace(".", r"\.")):
            _resolve(_load("mvm_aq_lenet5.json"), **{key: value})

    def test_write_granularity_refuses_at_both_doors(self):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"]["cores"][1]["write_granularity"] = "diagonal"
        assert any(
            m.startswith("cores[1].write_granularity")
            for m in validate_deployment_config(config)
        )
        with pytest.raises(DeclarationValueError, match=r"cores\[1\]\.write_granularity"):
            _resolve(config)

    def test_a_value_domain_key_under_spiking_refuses_at_both_doors(self):
        config = _load("sync_simplemlp.json")
        config["platform_constraints"]["readout_law"] = LEGAL["readout_law"]
        assert any(m.startswith("readout_law") for m in validate_deployment_config(config))
        with pytest.raises(DeclarationValueError, match="readout_law"):
            _resolve(config)

    def test_the_refusal_is_a_toolchain_refusal(self):
        with pytest.raises(ToolchainRefusal):
            _resolve(_load("mvm_aq_lenet5.json"), inference_batch_K=0)

    def test_a_non_finite_value_read_from_json_refuses_at_both_doors(self):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"]["inference_rate_hz"] = float("nan")
        wire = json.loads(json.dumps(config))
        assert any("must be a finite number" in m for m in validate_deployment_config(wire))
        with pytest.raises(DeclarationValueError, match="finite"):
            build_platform_constraints_resolved(
                merge_pipeline_config(wire["deployment_parameters"], wire["platform_constraints"]))

    def test_an_integer_beyond_float_range_read_from_json_is_judged_at_both_doors(self):
        """A legal one passes both doors and an illegal one refuses by name (no OverflowError)."""
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"]["inference_rate_hz"] = HUGE_JSON_INT
        wire = json.loads(json.dumps(config))
        assert wire["platform_constraints"]["inference_rate_hz"] == HUGE_JSON_INT
        assert validate_deployment_config(wire) == []
        pcfg = build_platform_constraints_resolved(
            merge_pipeline_config(wire["deployment_parameters"], wire["platform_constraints"]))
        assert pcfg["inference_rate_hz"] == HUGE_JSON_INT
        wire["platform_constraints"]["write_overlap"] = {
            "mode": "overlapped", "shadow_capacity_fraction": HUGE_JSON_INT}
        assert any(m.startswith("write_overlap.shadow_capacity_fraction")
                   for m in validate_deployment_config(wire))
        with pytest.raises(DeclarationValueError, match=r"write_overlap\.shadow_capacity_fraction"):
            build_platform_constraints_resolved(
                merge_pipeline_config(wire["deployment_parameters"], wire["platform_constraints"]))


#: Keys whose registry legal set is taken from their consumer (plan section 20 item 3): the key, the set
#: its consumer accepts and runs, and the values the consumer names but no run implements (reserved).
CONSUMER_LEGAL_SETS = (
    # connectivity._coerce refuses everything outside _VALID (connectivity.py:11-16).
    ("nevresim_connectivity_mode", frozenset(connectivity._VALID), frozenset()),
    # resolve_s_allocation_mode names three modes; only uniform is threaded, the rest run as uniform.
    ("s_allocation", frozenset(S_ALLOCATION_SUPPORTED_MODES),
     frozenset(S_ALLOCATION_MODES) - frozenset(S_ALLOCATION_SUPPORTED_MODES)),
)

#: Values their consumer refuses (no option) or does not run (reserved), and the message both doors give.
CONSUMER_REFUSED = (
    ("nevresim_connectivity_mode", "codegen",
     "nevresim_connectivity_mode='codegen' is not a valid option; choose one of "
     "['compile_time', 'runtime']"),
    ("s_allocation", "explicit", unsupported_s_allocation_error("explicit")),
    ("s_allocation", "budget", unsupported_s_allocation_error("budget")),
)

#: Spellings the consumers normalize (``resolve_s_allocation_mode`` lowercases,
#: ``resolve_nevresim_connectivity_mode`` strips spaces): the key, the value, the consumer and what it
#: reads. Both doors read the document as the consumer does (owner decision 6), so each resolves to the
#: canonical value.
NON_CANONICAL_SPELLINGS = (
    ("s_allocation", "UNIFORM", resolve_s_allocation_mode, "uniform"),
    ("nevresim_connectivity_mode", " runtime ", connectivity.resolve_nevresim_connectivity_mode,
     "runtime"),
    ("nevresim_connectivity_mode", "compile_time\n", connectivity.resolve_nevresim_connectivity_mode,
     "compile_time"),
)

#: The audited keys whose consumer normalizes what it reads (evidence/consumer_normalization_audit_owner_r1.txt):
#: every other normalization in src/ reads no declared registry value, reads a nested JSON field neither door
#: judges, or reads a derivation-owned key a document cannot declare.
NORMALIZING_CONSUMERS = frozenset({"s_allocation", "nevresim_connectivity_mode"})

_NORMALIZATIONS = frozenset({"lower", "upper", "casefold", "strip", "lstrip", "rstrip"})


def _attribute_calls(source):
    tree = ast.parse(textwrap.dedent(source))
    return {node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}


class TestConsumerLegalSets:
    """One rule: the registry legal set of a key is what its consumer accepts and runs. A value the
    consumer refuses is no option; a value it names but does not run is reserved; both doors refuse
    either by name, through the same value schema."""

    @pytest.mark.parametrize("key,accepted,reserved", CONSUMER_LEGAL_SETS, ids=lambda x: str(x)[:30])
    def test_the_registry_legal_set_is_the_consumer_s(self, key, accepted, reserved):
        schema = schema_of(REGISTRY[key])
        assert schema is not None and set(schema.options) == accepted
        assert set(schema.reserved) == reserved
        # The wizard payload names the legal and the reserved values, nothing else.
        assert set(serialize_registry()["keys"][key]["options"]) == accepted | reserved
        assert value_errors(schema, REGISTRY[key].default, key) == []

    def test_the_connectivity_consumer_accepts_exactly_the_registry_legal_set(self):
        legal = schema_of(REGISTRY["nevresim_connectivity_mode"]).options
        assert legal == get_args(connectivity.ConnectivityMode)
        for mode in legal:
            assert connectivity.resolve_nevresim_connectivity_mode(
                {"nevresim_connectivity_mode": mode}) == mode
        with pytest.raises(ValueError, match="compile_time or runtime"):
            connectivity.resolve_nevresim_connectivity_mode({"nevresim_connectivity_mode": "codegen"})

    @pytest.mark.parametrize("key,value,message", CONSUMER_REFUSED, ids=lambda x: str(x)[:30])
    def test_both_doors_refuse_by_name(self, key, value, message):
        config = _load("sync_simplemlp.json")
        config["deployment_parameters"][key] = value
        assert message in validate_deployment_config(config)
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == message

    @pytest.mark.parametrize("key,value", [
        ("nevresim_connectivity_mode", "compile_time"), ("nevresim_connectivity_mode", "runtime"),
        ("s_allocation", "uniform"),
    ])
    def test_a_legal_value_passes_both_doors_and_resolves_unchanged(self, key, value):
        config = _load("sync_simplemlp.json")
        before = _resolve(config)
        config["deployment_parameters"][key] = value
        assert validate_deployment_config(config) == []
        assert _resolve(config) == before

    @pytest.mark.parametrize("key,value,consumer,canonical", NON_CANONICAL_SPELLINGS,
                             ids=["s_allocation-UNIFORM", "nevresim_connectivity_mode-space-runtime-space",
                                  "nevresim_connectivity_mode-compile_time-newline"])
    def test_a_spelling_the_consumer_normalizes_resolves_to_the_canonical_value_at_both_doors(
            self, key, value, consumer, canonical):
        """Both doors read the declaration as its consumer does, before the one judging loop: no door
        refuses, and the parsed document, the wizard's resolution and the run door's surface are those of
        the canonical document."""
        assert consumer({key: value}) == canonical
        config = _load("sync_simplemlp.json")
        canonical_config = copy.deepcopy(config)
        config["deployment_parameters"][key] = value
        canonical_config["deployment_parameters"][key] = canonical
        assert validate_deployment_config(config) == []
        assert parse_deployment_document(config).dp[key] == canonical
        resolution, reference = resolve_draft(config), resolve_draft(canonical_config)
        assert resolution.resolved[key] == canonical
        assert resolution.resolved == reference.resolved and resolution.errors == reference.errors
        assert _resolve(config) == _resolve(canonical_config)

    def test_a_spelling_of_a_reserved_value_refuses_as_the_value_the_consumer_reads(self):
        """``Explicit`` is the reserved ``explicit`` to its consumer: both doors refuse it by that name."""
        config = _load("sync_simplemlp.json")
        config["deployment_parameters"]["s_allocation"] = "Explicit"
        reserved = unsupported_s_allocation_error("explicit")
        assert validate_deployment_config(config) == [
            reserved,
            "s_allocation='explicit' is not legal here: the current config admits {'uniform'}. "
            "Remove s_allocation to accept the derived value."]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == reserved

    @pytest.mark.parametrize("key,value,run", [
        ("nevresim_connectivity_mode", "codegen",
         "nevresim_connectivity_mode='codegen' is not a valid option; choose one of "
         "['compile_time', 'runtime']"),
        ("s_allocation", "magic", "s_allocation='magic' is not a valid option; choose one of ['uniform']"),
        ("s_allocation", 3, "s_allocation must be enum, got int (3)"),
    ])
    def test_a_value_the_consumer_refuses_is_read_as_written_and_refused_by_name(self, key, value, run):
        assert read_like_consumers({key: value}) == {key: value}
        config = _load("sync_simplemlp.json")
        config["deployment_parameters"][key] = value
        assert parse_deployment_document(config).dp[key] == value
        assert any(repr(value) in message for message in validate_deployment_config(config))
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == run

    def test_a_defaulted_key_is_judged_by_value_where_its_presence_is_no_declaration(self):
        """Every flat config carries the schema defaults, the event-domain s_allocation included: the run
        door judges a defaulted key's value and leaves its domain to the document door, so a value-domain
        run resolves with the default, and refuses a reserved value by name."""
        flat = _flat(_load("mvm_aq_lenet5.json"))
        assert flat["s_allocation"] == "uniform" and REGISTRY["s_allocation"].domain == "event"
        build_platform_constraints_resolved(flat)
        flat["s_allocation"] = "explicit"
        with pytest.raises(DeclarationValueError, match="s_allocation='explicit' is reserved"):
            build_platform_constraints_resolved(flat)

    def test_the_run_door_has_no_branch_on_a_key_name(self):
        """The run door's judging loop names no registry key: which keys it judges is read from the
        registry (every present key since S0 amendment B; the domain tag where a value schema or the passthrough
        flag is declared), never listed."""
        for function in (judged_passthrough, resolver_module.declaration_errors):
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            constants = {node.value for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            assert not (set(REGISTRY) & constants), sorted(set(REGISTRY) & constants)


class TestConsumerReading:
    """One generic mechanism (owner decisions 6 and 7): a key whose consumer normalizes its value declares
    the consumer's own reading (registry ``consumer_reading``, a reference, never a copy), and both doors
    read every declaration through it before the one judging loop."""

    def test_the_normalizing_consumers_are_the_audited_set(self):
        readers = {key: entry.consumer_reading for key, entry in REGISTRY.items()
                   if entry.consumer_reading is not None}
        assert set(readers) == NORMALIZING_CONSUMERS
        assert readers["nevresim_connectivity_mode"] is connectivity.resolve_nevresim_connectivity_mode

    def test_the_allocation_reading_is_the_consumer_s(self):
        """The s_allocation reading calls the consumer's mode reading and its per-depth resolver, so every
        mode spelling reads as ``resolve_s_allocation_mode`` reads it."""
        reading = REGISTRY["s_allocation"].consumer_reading
        names = {node.id for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(reading))))
                 if isinstance(node, ast.Name)}
        assert {"resolve_s_allocation_mode", "TemporalAllocationResolver"} <= names
        for raw in ("uniform", "UNIFORM", "Explicit", "BUDGET"):
            assert reading({"s_allocation": raw, "simulation_steps": 4}) == resolve_s_allocation_mode(
                {"s_allocation": raw})

    def test_no_door_copies_a_consumer_normalization(self):
        """The reading loop, both validators, the resolver and the allocation reading spell no string
        normalization: each normalization lives in its consumer and is reached by reference."""
        sources = [inspect.getsource(module) for module in (
            consumer_reading_module, validation_module, resolver_module)]
        sources.append(inspect.getsource(REGISTRY["s_allocation"].consumer_reading))
        assert not set().union(*map(_attribute_calls, sources)) & _NORMALIZATIONS

    def test_the_reading_loop_names_no_key(self):
        tree = ast.parse(inspect.getsource(consumer_reading_module))
        constants = {node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not (set(REGISTRY) & constants), sorted(set(REGISTRY) & constants)

    def test_a_reading_replaces_a_value_and_never_a_key(self):
        declared = {"s_allocation": "UNIFORM", "nevresim_connectivity_mode": " runtime ", "lr": 0.1}
        read = read_like_consumers(declared)
        assert read == {"s_allocation": "uniform", "nevresim_connectivity_mode": "runtime", "lr": 0.1}
        assert declared["s_allocation"] == "UNIFORM"
        assert read_like_consumers({}) == {} and read_like_consumers({"s_allocation": None}) == {
            "s_allocation": None}

    def test_the_merged_validator_reads_like_the_consumers(self):
        """``validate_merged_config`` judges the runtime flat config as its consumers read it, as both doors
        do: a spelling the consumer normalizes gives the canonical config's verdict, and a spelling of a
        reserved value gives the reserved value's."""
        flat = _flat(_load("sync_simplemlp.json"))
        spelled = {**flat, "s_allocation": "UNIFORM", "nevresim_connectivity_mode": " runtime "}
        assert validate_merged_config(spelled) == validate_merged_config(flat) == []
        assert spelled["s_allocation"] == "UNIFORM" and spelled["nevresim_connectivity_mode"] == " runtime "
        reserved = validate_merged_config({**flat, "s_allocation": "explicit"})
        assert reserved and validate_merged_config({**flat, "s_allocation": "Explicit"}) == reserved


def _value_schema_statements(config):
    """The document door's refusals but its legal-value-set rows: at the run door ``merge_pipeline_config`` applies
    that law, and the resolver states every value-schema refusal (S0 amendment B)."""
    dp = config["deployment_parameters"]
    law = {row["message"] for row in validation_module.legality_errors(dp, dp)}
    return [m for m in validate_deployment_config(config) if m not in law]


def _per_layer_s_document(mode, explicit):
    config = _load("sync_simplemlp.json")
    config["deployment_parameters"].update(s_allocation=mode, s_allocation_explicit=explicit)
    return config


def _uniform_document():
    config = _load("sync_simplemlp.json")
    config["deployment_parameters"]["s_allocation"] = "uniform"
    return config


class TestPerLayerSMap:
    """Owner decision 7: an explicit per-layer S map whose every entry equals simulation_steps is the uniform
    declaration, read so at both doors (the consumer's allocation of that map is uniform,
    ``TemporalAllocation.is_uniform``). The doors never know the cascade depth (mapping does), so the map is
    read at its own length: a uniform declaration runs simulation_steps at every depth whatever the depth,
    and the consumer's one-S-per-depth rule applies only to a map that stays explicit, which both doors
    refuse as reserved. The map itself is judged first by its own value schema, the consumer's legal set
    (a non-empty list of positive ints), so a map the consumer's int() cast would change refuses by name."""

    _S = 32  # simulation_steps of sync_simplemlp.json

    @pytest.mark.parametrize("mode", ["explicit", "Explicit"])
    @pytest.mark.parametrize("length", [1, 3, 7])
    def test_a_map_of_simulation_steps_at_every_entry_resolves_as_uniform_at_both_doors(self, mode, length):
        config = _per_layer_s_document(mode, [self._S] * length)
        assert validate_deployment_config(config) == []
        parsed = parse_deployment_document(config)
        assert parsed.dp["s_allocation"] == "uniform"
        assert parsed.dp["s_allocation_explicit"] == [self._S] * length
        resolution = resolve_draft(config)
        assert resolution.resolved["s_allocation"] == "uniform"
        assert _resolve(config) == _resolve(_uniform_document())

    @pytest.mark.parametrize("depth", [1, 3, 5])
    def test_the_uniform_reading_runs_simulation_steps_at_every_depth_whatever_the_map_length(self, depth):
        read = read_like_consumers(_flat(_per_layer_s_document("explicit", [self._S] * 3)))
        allocation = TemporalAllocationResolver.from_config(read).resolve(depth=depth)
        assert allocation.mode == "uniform" and allocation.per_depth_steps == (self._S,) * depth

    @pytest.mark.parametrize("explicit", [[32, 32, 16], [16, 16], [32, 0], "32", [], None],
                             ids=["one-entry-differs", "all-equal-to-another-S", "non-positive",
                                  "a-string", "empty", "no-map"])
    def test_a_map_that_is_not_uniform_or_not_readable_keeps_the_declared_mode(self, explicit):
        config = _per_layer_s_document("explicit", explicit)
        reserved = unsupported_s_allocation_error("explicit")
        assert parse_deployment_document(config).dp["s_allocation"] == "explicit"
        stated = _value_schema_statements(config)
        assert reserved in stated
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == "; ".join(stated)

    @pytest.mark.parametrize("steps", [float("inf"), "four", [32]], ids=["infinity", "a-string", "a-list"])
    def test_a_simulation_steps_the_consumer_cannot_read_leaves_the_mode_and_refuses_by_name(self, steps):
        """The reading never raises out of a door: a window the consumer cannot read (``int(inf)`` raises
        OverflowError) leaves the declared mode, and the judging loop names both values."""
        config = _per_layer_s_document("explicit", [self._S] * 3)
        config["platform_constraints"]["simulation_steps"] = steps
        assert read_like_consumers({"s_allocation": "explicit", "s_allocation_explicit": [self._S] * 3,
                                    "simulation_steps": steps})["s_allocation"] == "explicit"
        messages = validate_deployment_config(config)
        assert unsupported_s_allocation_error("explicit") in messages
        assert any(m.startswith("simulation_steps must be int") for m in messages)

    def test_the_map_schema_is_the_consumer_s_legal_set(self):
        """The map's value schema is the legal set its consumer states (``_parse_explicit``: a non-empty
        per-depth list of positive ints), judged without the consumer's int() cast: every map the schema
        admits, the consumer reads unchanged, and every map the consumer refuses, the schema refuses."""
        schema = schema_of(REGISTRY["s_allocation_explicit"])
        assert schema is not None and schema.kind == "int_list" and schema.min_length == 1
        assert schema.item is not None and (schema.item.kind, schema.item.low) == ("int", 1)
        for raw in ([1], [4, 4, 4], [32, 0], [], [-1], [1.5], [32.0], ["3"], [True], "32", {"a": 1}, [[1]]):
            try:
                consumer = TemporalAllocationResolver.from_config(
                    {"s_allocation_explicit": raw, "simulation_steps": 4}).explicit
            except (TypeError, ValueError):
                consumer = None
            admitted = value_errors(schema, raw, "s_allocation_explicit") == []
            assert admitted == (isinstance(raw, list) and bool(raw)
                                and all(type(v) is int and v >= 1 for v in raw)), raw
            if admitted:
                assert consumer == tuple(raw), raw
            if consumer is None:
                assert not admitted, raw

    #: Maps no entry of which is a JSON int equal to simulation_steps, though the consumer's int() cast reads
    #: each as simulation_steps at every entry (``int(32.9) == 32``, ``int("32") == 32``, ``int(32.0) == 32``):
    #: the map's value schema refuses each by name. 32.0 equals simulation_steps in value (the cast loses
    #: nothing, and the base ran that map exactly as uniform); it is refused by the registry's INT rule, as
    #: ``simulation_steps: 32.0`` is (a boundary finding, owner_review.md Registry Legal Sets).
    CAST_MAPS = (
        ([32.9, 32.1], "s_allocation_explicit[0] must be int, got float (32.9); "
                       "s_allocation_explicit[1] must be int, got float (32.1)"),
        (["32", "32"], "s_allocation_explicit[0] must be int, got str ('32'); "
                       "s_allocation_explicit[1] must be int, got str ('32')"),
        ([32.0, 32.0], "s_allocation_explicit[0] must be int, got float (32.0); "
                       "s_allocation_explicit[1] must be int, got float (32.0)"),
    )

    @pytest.mark.parametrize("explicit,message", CAST_MAPS, ids=["fractional", "strings", "integral-floats"])
    def test_a_map_the_consumer_s_cast_reads_as_uniform_is_refused_by_name_at_both_doors(self, explicit, message):
        """No entry of such a map is a JSON int equal to simulation_steps, though the consumer's cast reads it
        as uniform (lossily for 32.9 and "32", exactly for the integral float 32.0): both doors judge the map
        by its own value schema, whose items follow the registry's INT rule, and refuse it by name, so no
        admitted document is read through the cast."""
        assert TemporalAllocationResolver.from_config(
            {"s_allocation": "explicit", "s_allocation_explicit": explicit, "simulation_steps": self._S}
        ).resolve(depth=len(explicit)).is_uniform
        config = _per_layer_s_document("explicit", explicit)
        assert validate_deployment_config(config) == [message]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == message

    @pytest.mark.parametrize("explicit,message", [
        ([], "s_allocation_explicit must be a list of at least 1 item(s)"),
        ([32, 0], "s_allocation_explicit[1]=0 is below the minimum 1"),
        ([32.5], "s_allocation_explicit[0] must be int, got float (32.5)"),
        ("32", "s_allocation_explicit must be a list, got str ('32')"),
    ], ids=["empty", "non-positive", "a-float", "a-string"])
    def test_an_illegal_map_is_refused_by_name_at_both_doors_under_the_uniform_mode_too(self, explicit, message):
        """The value check is relevance-independent at both doors, as for every other key: a map outside
        the consumer's legal set refuses by name even where the uniform mode leaves it unread."""
        config = _per_layer_s_document("uniform", explicit)
        assert validate_deployment_config(config) == [message]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == message

    @pytest.mark.parametrize("length", [1, 3])
    def test_a_document_without_simulation_steps_reads_its_map_against_the_schema_default_at_both_doors(
            self, length):
        """A run merges the schema defaults beneath every document (``merge_pipeline_config``), and the
        reading sees them beneath the declarations at both doors: a document that omits simulation_steps
        reads its map against the default window the run door reads, so the doors agree."""
        default = REGISTRY["simulation_steps"].default
        assert default == self._S
        config = _per_layer_s_document("explicit", [default] * length)
        del config["platform_constraints"]["simulation_steps"]
        uniform = _uniform_document()
        del uniform["platform_constraints"]["simulation_steps"]
        assert validate_deployment_config(config) == []
        assert parse_deployment_document(config).dp["s_allocation"] == "uniform"
        assert _flat(config)["simulation_steps"] == default
        assert _resolve(config) == _resolve(uniform) == _resolve(_uniform_document())

    def test_a_document_without_simulation_steps_and_another_map_stays_reserved_at_both_doors(self):
        config = _per_layer_s_document("explicit", [4, 4])
        del config["platform_constraints"]["simulation_steps"]
        reserved = unsupported_s_allocation_error("explicit")
        assert parse_deployment_document(config).dp["s_allocation"] == "explicit"
        stated = _value_schema_statements(config)
        assert reserved in stated
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == "; ".join(stated)

    def test_a_budget_declaration_stays_reserved(self):
        config = _per_layer_s_document("budget", [self._S] * 3)
        assert unsupported_s_allocation_error("budget") in validate_deployment_config(config)
        with pytest.raises(DeclarationValueError, match="s_allocation='budget' is reserved"):
            _resolve(config)

    def test_the_per_layer_s_template_is_admitted_at_the_document_door_and_resolves_at_the_run_door(self):
        """The shipped template declares explicit [4, 4, 4] with simulation_steps 4: both doors read it as
        uniform, the document door admits it (for the first time: the base document door refused it as the
        reserved explicit mode), and the run door resolves it, as on the base, exactly as the uniform
        document."""
        with open(_PER_LAYER_S_TEMPLATE, encoding="utf-8") as f:
            template = json.load(f)
        assert template["deployment_parameters"]["s_allocation"] == "explicit"
        assert validate_deployment_config(copy.deepcopy(template)) == []
        uniform = copy.deepcopy(template)
        uniform["deployment_parameters"]["s_allocation"] = "uniform"
        assert _resolve(template) == _resolve(uniform)


class TestPlatformBiasMode:
    """The platform bias mode is read like the consumer at both doors (owner's S0 review): the flat
    has_bias first, then the nested platform_constraints.has_bias, else a bias lane; every core type
    without its own has_bias takes it, and two declarations that disagree refuse by name."""

    _BARE = {"max_axons": 8, "max_neurons": 8, "count": 1}

    def _resolve(self, cores, **pc):
        return build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": cores, **pc}))

    @staticmethod
    def _document(**pc):
        config = _load("sync_simplemlp.json")
        config["platform_constraints"].update(pc)
        return config

    def test_a_flat_bias_reaches_every_core_type_without_its_own(self):
        pcfg = self._resolve([dict(self._BARE), {**self._BARE, "has_bias": True}], has_bias=False)
        assert [ct["has_bias"] for ct in pcfg["cores"]] == [False, True]

    def test_a_flat_bias_over_the_default_cores_reaches_them_at_both_doors(self):
        config = self._document(has_bias=False)
        del config["platform_constraints"]["cores"]
        assert validate_deployment_config(config) == []
        pcfg = _resolve(config)
        assert pcfg["cores"] and all(ct["has_bias"] is False for ct in pcfg["cores"])
        assert resolve_bias_mode(_flat(config)) == "param_encoded"

    def test_the_document_door_admits_the_flat_bias_the_run_door_reads(self):
        config = self._document(has_bias=False, cores=[dict(self._BARE), {**self._BARE, "has_bias": True}])
        assert validate_deployment_config(config) == []
        assert [ct["has_bias"] for ct in _resolve(config)["cores"]] == [False, True]

    @pytest.mark.parametrize("flat,nested,expected", [
        (None, None, True), (False, None, False), (True, None, True), (None, False, False),
        (False, False, False), (True, True, True),
    ])
    def test_the_flat_bias_comes_first_then_the_nested_one_then_a_bias_lane(self, flat, nested, expected):
        config = {"cores": [dict(self._BARE)]}
        if flat is not None:
            config["has_bias"] = flat
        if nested is not None:
            config["platform_constraints"] = {"has_bias": nested}
        assert platform_bias_mode(config) is expected
        assert build_platform_constraints_resolved(config)["cores"][0]["has_bias"] is expected

    @pytest.mark.parametrize("flat,nested", [(False, True), (True, False)])
    @pytest.mark.parametrize("cores", [[_BARE], [{**_BARE, "has_bias": True}]], ids=["bare", "declared"])
    def test_a_flat_and_a_nested_bias_that_disagree_refuse_by_name(self, flat, nested, cores):
        config = {"cores": [dict(c) for c in cores], "has_bias": flat,
                  "platform_constraints": {"has_bias": nested}}
        message = (f"has_bias={flat!r} and platform_constraints.has_bias={nested!r} declare the platform "
                   f"bias mode twice and disagree: declare it once")
        assert conflicting_platform_bias_error(config) == message
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved(config)
        assert isinstance(refusal.value, ToolchainRefusal) and str(refusal.value) == message

    @pytest.mark.parametrize("door", [declaration_judge_module, validation_module], ids=["run-door", "document-door"])
    def test_both_doors_call_the_one_bias_rule(self, door):
        """A flat config may carry the nested form too, and the same rule judges both, declaration by
        declaration and then their agreement (each door's behaviour is pinned by the tests below). The run
        door's judge is the registry's ``declaration_judge`` since S0 amendment B round 1, which the resolver
        calls."""
        assert resolver_module.run_door_errors is declaration_judge_module.run_door_errors
        tree = ast.parse(textwrap.dedent(inspect.getsource(door)))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        called = names | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        # The document door calls the rule itself, or through the registry judge (G4 interface note 3).
        assert "platform_bias_errors" in called or "document_declaration_errors" in called
        assert not {"conflicting_platform_bias_error", "dropped_platform_bias_error"} & names

    @pytest.mark.parametrize("flat,nested,expected", [
        (None, None, []), (False, None, []), (None, True, []), (True, True, []),
        ("false", None, ["has_bias must be bool, got str ('false')"]),
        (None, "false", ["platform_constraints.has_bias must be bool, got str ('false')"]),
        (False, "false", ["platform_constraints.has_bias must be bool, got str ('false')"]),
        ("no", 0, ["has_bias must be bool, got str ('no')",
                   "platform_constraints.has_bias must be bool, got int (0)"]),
        (False, True, ["has_bias=False and platform_constraints.has_bias=True declare the platform bias mode "
                       "twice and disagree: declare it once"]),
    ])
    def test_the_one_bias_rule_judges_each_declaration_then_their_agreement(self, flat, nested, expected):
        """``platform_bias_errors``: each declaration, flat and nested, by the bias entry's BOOL schema at its
        own path, then a flat and a nested boolean that disagree; the domain tag is not the rule's to judge,
        because has_bias is universal (a tag on it would need both doors to judge it here)."""
        config = {}
        if flat is not None:
            config["has_bias"] = flat
        if nested is not None:
            config["platform_constraints"] = {"has_bias": nested}
        assert entries_write_module.platform_bias_errors(config, REGISTRY["has_bias"]) == expected
        assert REGISTRY["has_bias"].domain == "universal"

    @pytest.mark.parametrize("section", ["platform_constraints", "deployment_parameters"])
    @pytest.mark.parametrize("flat,nested", [(False, True), (True, False)])
    def test_a_document_nesting_a_disagreeing_bias_is_refused_by_name_at_both_doors(self, section, flat, nested):
        """A run lays the document's two sections flat (``merge_pipeline_config``: the deployment
        parameters, then the platform constraints), so a ``platform_constraints`` object nested in either
        section, an unknown key the parser reports, reaches the run door as the nested bias declaration.
        Both doors apply the one rule to that layout and refuse a disagreement by name."""
        config = self._document(has_bias=flat)
        config[section]["platform_constraints"] = {"has_bias": nested}
        assert f"{section}.platform_constraints" in parse_deployment_document(config).unknown
        message = (f"has_bias={flat!r} and platform_constraints.has_bias={nested!r} declare the platform "
                   f"bias mode twice and disagree: declare it once")
        assert message in validate_deployment_config(config)
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == message

    @pytest.mark.parametrize("section", ["platform_constraints", "deployment_parameters"])
    def test_a_document_nesting_an_agreeing_bias_passes_both_doors(self, section):
        config = self._document(has_bias=False, cores=[dict(self._BARE)])
        config[section]["platform_constraints"] = {"has_bias": False}
        assert validate_deployment_config(config) == []
        assert [ct["has_bias"] for ct in _resolve(config)["cores"]] == [False]

    @pytest.mark.parametrize("cores", [[_BARE], [{**_BARE, "has_bias": True}]])
    def test_a_flat_bias_that_is_not_a_bool_refuses_by_name_at_both_doors(self, cores):
        """A string 'false' is truthy: judged by the registry's BOOL schema, never compared. The document
        door states the refusal once (the registry loop and the bias rule judge the same declaration)."""
        config = self._document(has_bias="false", cores=[dict(c) for c in cores])
        assert validate_deployment_config(config) == ["has_bias must be bool, got str ('false')"]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == "has_bias must be bool, got str ('false')"

    def test_a_nested_bias_that_is_not_a_bool_refuses_by_name(self):
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved({
                "cores": [dict(self._BARE)], "platform_constraints": {"has_bias": "false"}})
        assert str(refusal.value) == "platform_constraints.has_bias must be bool, got str ('false')"

    def _bare_document(self, cores=(_BARE,)):
        config = self._document(cores=[dict(c) for c in cores])
        del config["platform_constraints"]["has_bias"]
        return config

    #: The core types a platform bias declaration may reach: a bare core takes it; a core declaring its own
    #: has_bias reads none, so there the declaration is inert (it ran exactly on the base).
    _CORES = pytest.mark.parametrize("cores", [(_BARE,), ({**_BARE, "has_bias": True},)],
                                     ids=["bare-core", "every-core-declares-its-own"])

    @_CORES
    @pytest.mark.parametrize("section", ["platform_constraints", "deployment_parameters"])
    @pytest.mark.parametrize("flat", [None, False], ids=["alone", "beside-a-flat-false"])
    def test_a_document_nesting_a_non_boolean_bias_is_refused_by_name_at_both_doors(self, section, flat, cores):
        """The run door judges the nested declaration by the BOOL schema, so the document door does too, over
        the layout a run gives the two sections: a nested string 'false' (truthy, which the base deployed
        onto every bare core) refuses by name at both doors, alone or beside a flat boolean.

        Boundary finding (S0 amendment B): where every core type declares its own has_bias no core reads the
        platform declaration, and the base ran such a document exactly; both doors refuse it too, by the
        relevance-independent value check every key follows (plan section 21 item 7, "Strict, stated").
        Alternative (a), judging the bias only where a bare core reads it, is recorded and not taken."""
        config = self._bare_document(cores)
        if flat is not None:
            config["platform_constraints"]["has_bias"] = flat
        config[section]["platform_constraints"] = {"has_bias": "false"}
        message = "platform_constraints.has_bias must be bool, got str ('false')"
        assert validate_deployment_config(config) == [message]
        assert any(row["message"] == message for row in resolve_draft(config).errors)
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert isinstance(refusal.value, ToolchainRefusal) and str(refusal.value) == message

    @_CORES
    def test_a_non_boolean_flat_bias_in_the_deployment_parameters_is_refused_by_name_at_both_doors(self, cores):
        """A run reads a has_bias the deployment parameters declare when the platform constraints declare
        none (``merge_pipeline_config`` lays both sections flat); the parser reports it as an unknown key, and
        the bias rule over the run layout judges it as the run door does, whether a bare core reads it or
        every core type declares its own (the boundary finding above)."""
        config = self._bare_document(cores)
        config["deployment_parameters"]["has_bias"] = "no"
        message = "has_bias must be bool, got str ('no')"
        assert "deployment_parameters.has_bias" in parse_deployment_document(config).unknown
        assert validate_deployment_config(config) == [message]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == message

    #: Where a document declares the two bias forms: the flat has_bias and a nested platform_constraints
    #: object, in neither section, one, or both (the deployment parameters' value differs from the platform
    #: constraints', so the order of the layout shows).
    _PLACEMENTS = ((), ("deployment_parameters",), ("platform_constraints",),
                   ("deployment_parameters", "platform_constraints"))
    _SECTION_BIAS = {"deployment_parameters": True, "platform_constraints": False}

    @pytest.mark.parametrize("flat_in", _PLACEMENTS, ids=["flat-none", "flat-dp", "flat-pc", "flat-both"])
    @pytest.mark.parametrize("nested_in", _PLACEMENTS,
                             ids=["nested-none", "nested-dp", "nested-pc", "nested-both"])
    def test_the_document_door_s_run_layout_is_the_run_s(self, flat_in, nested_in):
        """The document door lays the two sections out by hand (``document_run_layout``), because
        ``merge_pipeline_config`` is the run's: the bias declarations the layout carries equal the merged
        run config's (the default layers beneath hold neither form), and both doors give the same verdict,
        with the same message."""
        config = self._bare_document()
        for section in flat_in:
            config[section]["has_bias"] = self._SECTION_BIAS[section]
        for section in nested_in:
            config[section]["platform_constraints"] = {"has_bias": self._SECTION_BIAS[section]}
        self._assert_the_doors_agree(config)

    def _assert_the_doors_agree(self, config):
        """Both doors, and the registry judge both can call (``declaration_judge``, S0 amendment B round 1), over
        the run's layout of the two sections: the same bias declarations as the merged run config, the same
        verdict and the same message."""
        sections = (copy.deepcopy(config["deployment_parameters"]), copy.deepcopy(config["platform_constraints"]))
        try:
            merged = merge_pipeline_config(*copy.deepcopy(sections))
        except DeclarationValueError:  # the merge judges the declared sections first (G4 interface note 1)
            merged = None
        if merged is not None:
            bias = platform_bias_declarations(merged)
            assert platform_bias_declarations(validation_module.document_run_layout(copy.deepcopy(config))) == bias
            assert platform_bias_declarations(sections_run_layout(*sections)) == bias
        try:
            build_platform_constraints_resolved(merge_pipeline_config(*copy.deepcopy(sections)))
            run = []
        except DeclarationValueError as refusal:
            run = [str(refusal)]
        assert validate_deployment_config(config) == run
        judged = declared_sections_errors(*sections)
        assert (["; ".join(judged)] if judged else []) == run

    #: What the later section (the platform constraints) may carry over an earlier section's bias declaration in
    #: the run's layout, one shallow update per section: an explicit null flat bias, and a nested object
    #: without a bias, with another field, or no object at all.
    _SHADOWS = (("has_bias", None), ("platform_constraints", {}), ("platform_constraints", {"foo": 1}),
                ("platform_constraints", "x"), ("platform_constraints", None))

    @pytest.mark.parametrize("earlier", ["flat", "nested"])
    @pytest.mark.parametrize("key,shadow", _SHADOWS,
                             ids=["null-flat", "empty-nested", "nested-without-bias", "nested-str", "nested-null"])
    def test_a_later_section_shadows_an_earlier_bias_alike_at_both_doors(self, earlier, key, shadow):
        """The run's layout replaces an earlier section's value whole (``merge_pipeline_config``'s
        ``dict.update``): no deep merge of the nested objects, no skipped null, no skipped non-object. The
        document door's layout carries the merged run config's bias declarations, and both doors give the same
        verdict and message, where the platform constraints shadow a non-boolean bias the deployment
        parameters declare flat or nested."""
        config = self._bare_document()
        if earlier == "flat":
            config["deployment_parameters"]["has_bias"] = "false"
        else:
            config["deployment_parameters"]["platform_constraints"] = {"has_bias": "false"}
        config["platform_constraints"][key] = shadow
        self._assert_the_doors_agree(config)

    @pytest.mark.parametrize("flat_section", ["platform_constraints", "deployment_parameters"])
    @pytest.mark.parametrize("nested_section", ["platform_constraints", "deployment_parameters"])
    def test_two_invalid_bias_declarations_are_both_refused_by_name_at_both_doors(self, flat_section,
                                                                                    nested_section):
        """A flat and a nested declaration that are both non-boolean: the document door states both messages,
        and the run door joins the same two (each once; the wizard's keyed rows are G4's)."""
        config = self._bare_document()
        config[flat_section]["has_bias"] = "no"
        config[nested_section]["platform_constraints"] = {"has_bias": 0}
        flat_message = "has_bias must be bool, got str ('no')"
        nested_message = "platform_constraints.has_bias must be bool, got int (0)"
        assert validate_deployment_config(config) == [flat_message, nested_message]
        with pytest.raises(DeclarationValueError) as refusal:
            _resolve(config)
        assert str(refusal.value) == "; ".join([flat_message, nested_message])


class TestPassthrough:
    """Acceptance 5 and 7: declared keys ride verbatim; absent keys stay absent."""

    def test_each_declared_key_rides_verbatim(self):
        pcfg = _resolve(_load("mvm_aq_lenet5.json"), **copy.deepcopy(LEGAL))
        for key, value in LEGAL.items():
            assert pcfg[key] == value, key

    def test_target_keys_ride_the_passthrough(self):
        overrides = {"weight_bits": {"value": 6, "note": "the bundle's 8 bits narrowed"}}
        # The resolver alone, on a flat config past the merge: the merge expands a target (G4), which
        # refuses any name this test could pick.
        pcfg = build_platform_constraints_resolved({
            **_flat(_load("sync_simplemlp.json")), "target": "some_bundle", "target_overrides": overrides,
            "target_claim_class": "headline", "target_faithfulness_class": "exact",
        })
        assert pcfg["target"] == "some_bundle"
        assert pcfg["target_overrides"] == overrides
        assert pcfg["target_claim_class"] == "headline"
        assert pcfg["target_faithfulness_class"] == "exact"

    @pytest.mark.parametrize("name", ["sync_simplemlp.json", "mvm_aq_lenet5.json"])
    def test_absent_keys_do_not_appear(self, name):
        pcfg = _resolve(_load(name))
        assert not set(PLATFORM_KEYS) & set(pcfg)

    def test_an_undeclared_bias_resolves_to_the_stated_absent_meaning(self):
        """has_bias states its absence (a bias lane); the resolver gives it to every core type
        that declares none."""
        pcfg = build_platform_constraints_resolved({"cores": [{"max_axons": 8, "max_neurons": 8}]})
        assert [core["has_bias"] for core in pcfg["cores"]] == [True]
        assert REGISTRY["has_bias"].empty_means.startswith("true")

    def test_a_passthrough_value_is_a_copy(self):
        law = copy.deepcopy(LEGAL["readout_law"])
        flat = _flat(_load("mvm_aq_lenet5.json"), readout_law=law)
        pcfg = build_platform_constraints_resolved(flat)
        pcfg["readout_law"]["adc_bits"] = 99
        assert flat["readout_law"]["adc_bits"] == 6


class TestValueDomainSurface:
    """Acceptance 4 and 5: the resolved surface names the value domain, and only it."""

    def test_a_value_domain_run_carries_its_semantics_and_grid(self):
        pcfg = _resolve(_load("mvm_aq_lenet5.json"))
        assert pcfg["core_semantics"] == "mvm"
        assert pcfg["activation_bits"] == 8

    def test_a_value_domain_run_without_a_grid_carries_only_its_semantics(self):
        pcfg = _resolve(_load("mvm_lenet5.json"))
        assert pcfg["core_semantics"] == "mvm"
        assert "activation_bits" not in pcfg

    def test_a_spiking_run_carries_neither(self):
        pcfg = _resolve(_load("sync_simplemlp.json"))
        assert "core_semantics" not in pcfg
        assert "activation_bits" not in pcfg

    def test_a_minimal_declaration_is_spiking(self):
        pcfg = build_platform_constraints_resolved({"cores": [{"max_axons": 8, "max_neurons": 8}]})
        assert "core_semantics" not in pcfg

    def test_resolution_without_new_keys_differs_only_by_the_surface(self):
        config = _load("mvm_aq_lenet5.json")
        pcfg = _resolve(config)
        stripped = {k: v for k, v in pcfg.items() if k not in ("core_semantics", "activation_bits")}
        spiking = copy.deepcopy(config)
        spiking["deployment_parameters"]["core_semantics"] = "spiking"
        del spiking["platform_constraints"]["activation_bits"]
        spiking_pcfg = _resolve(spiking)
        assert set(stripped) == set(spiking_pcfg)


def _two_layer_graph():
    rng = np.random.default_rng(3)
    first = NeuralCore(
        id=0, name="l0",
        input_sources=np.array([IRSource(-2, i) for i in range(6)] + [IRSource(-3, 0)],
                               dtype=object),
        core_matrix=rng.normal(size=(7, 5)).astype(np.float32), latency=0,
    )
    second = NeuralCore(
        id=1, name="l1",
        input_sources=np.array([IRSource(0, i) for i in range(5)] + [IRSource(-3, 0)],
                               dtype=object),
        core_matrix=rng.normal(size=(6, 3)).astype(np.float32), latency=1,
    )
    outputs = np.array([IRSource(1, j) for j in range(3)], dtype=object)
    return IRGraph(nodes=[first, second], output_sources=outputs)


def _canonical(obj, seen=None):
    """A content digest of a built mapping (pickle bytes embed torch storage addresses)."""
    seen = set() if seen is None else seen
    if isinstance(obj, (str, int, float, bool, type(None))):
        return repr(obj)
    if isinstance(obj, torch.Tensor):
        return ("tensor", str(obj.dtype), tuple(obj.shape), obj.detach().cpu().numpy().tobytes())
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return ("objects", obj.shape, tuple(_canonical(x, seen) for x in obj.flat))
        return ("array", obj.dtype.str, obj.shape, obj.tobytes())
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_canonical(x, seen) for x in obj))
    if isinstance(obj, dict):
        return ("dict", tuple(sorted((repr(k), _canonical(v, seen)) for k, v in obj.items())))
    if id(obj) in seen:
        return ("seen", type(obj).__name__)
    seen.add(id(obj))
    state = getattr(obj, "__dict__", None)
    if state is None:
        slots = [n for c in type(obj).__mro__ for n in getattr(c, "__slots__", ())]
        state = {n: getattr(obj, n) for n in slots if hasattr(obj, n)}
    return (type(obj).__name__, tuple(sorted(
        (k, _canonical(v, seen)) for k, v in state.items() if k != "build_token"
    )))


class TestWriteFieldsAreInertForMapping:
    """Acceptance 5: cores with the two write fields map, pack and build identically."""

    CORES = [{"max_axons": 8, "max_neurons": 8, "count": 4, "has_bias": True}]

    def _with_write_fields(self):
        return [{**core, **WRITE_FIELDS} for core in self.CORES]

    def test_resolution_keeps_the_fields_and_nothing_else_moves(self):
        plain = build_platform_constraints_resolved({"cores": self.CORES})
        written = build_platform_constraints_resolved({"cores": self._with_write_fields()})
        assert written["cores"][0]["write_granularity"] == "row"
        written["cores"] = [
            {k: v for k, v in core.items() if k not in WRITE_FIELDS} for core in written["cores"]
        ]
        assert written == plain

    def test_capabilities_and_packing_are_identical(self):
        plain, written = self.CORES, self._with_write_fields()
        assert (ChipCapabilities.from_platform_constraints({"cores": plain})
                == ChipCapabilities.from_platform_constraints({"cores": written}))
        softcores = [LayoutSoftCoreSpec(input_count=7, output_count=5),
                     LayoutSoftCoreSpec(input_count=6, output_count=3)]
        stats_plain = compute_mapping_stats(softcores, core_types_from_declaration(plain))
        stats_written = compute_mapping_stats(softcores, core_types_from_declaration(written))
        assert stats_plain == stats_written

    def test_the_hybrid_build_is_byte_identical(self):
        def build(cores):
            return build_hybrid_hard_core_mapping(
                ir_graph=_two_layer_graph(), cores_config=cores,
                strategy=MappingStrategy.resolve(ChipCapabilities()),
            )
        plain, written = build(self.CORES), build(self._with_write_fields())
        assert _canonical(plain) == _canonical(written)
        assert (schedule_record_from_mapping(plain, weight_bits=8, params_reloaded=0).to_dict()
                == schedule_record_from_mapping(written, weight_bits=8,
                                                params_reloaded=0).to_dict())
        assert (placement_record_from_mapping(plain).to_dict()
                == placement_record_from_mapping(written).to_dict())


def _two_pass_census():
    return PassPeriodCensus(
        passes=(
            PassCensus(position=0, segment_index=0, resident=False, cores=3,
                       resident_cores=1, written_cells_backing_store=2048,
                       written_cells_producer=0, payload_bytes=2048, compute_steps=5),
            PassCensus(position=1, segment_index=0, resident=False, cores=2,
                       resident_cores=0, written_cells_backing_store=1024,
                       written_cells_producer=64, payload_bytes=1024, compute_steps=3),
        ),
        boundaries=(PassBoundaryCensus(after_pass=0, carry_out_bytes=40, carry_in_bytes=40),),
        label="hand census",
    )


class TestPassPeriodCensus:
    """Acceptance 5: the census holds the carried bytes per boundary and direction."""

    def test_a_hand_census_round_trips(self):
        census = _two_pass_census()
        wire = json.loads(json.dumps(census.to_dict()))
        assert PassPeriodCensus.from_dict(wire) == census
        (boundary,) = census.boundaries
        assert (boundary.carry_out_bytes, boundary.carry_in_bytes) == (40, 40)
        assert census.carry_out_bytes == 40 and census.carry_in_bytes == 40

    def test_unknown_fields_and_bad_shapes_refuse_by_name(self):
        wire = _two_pass_census().to_dict()
        wire["passes"][0]["wat"] = 1
        with pytest.raises(PeriodCensusError):
            PassPeriodCensus.from_dict(wire)
        with pytest.raises(PeriodCensusError, match="position"):
            PassPeriodCensus(passes=(_two_pass_census().passes[1],), boundaries=())
        with pytest.raises(PeriodCensusError, match="after_pass"):
            PassPeriodCensus(
                passes=_two_pass_census().passes,
                boundaries=(PassBoundaryCensus(after_pass=1, carry_out_bytes=1, carry_in_bytes=1),),
            )
        with pytest.raises(PeriodCensusError, match="resident_cores"):
            PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=2,
                       written_cells_backing_store=0, written_cells_producer=0,
                       payload_bytes=0, compute_steps=0)

    def test_a_boundary_never_crosses_a_segment(self):
        first, second = _two_pass_census().passes
        other_segment = PassCensus(**{**second.__dict__, "segment_index": 1})
        with pytest.raises(PeriodCensusError, match="segment"):
            PassPeriodCensus(
                passes=(first, other_segment),
                boundaries=(PassBoundaryCensus(after_pass=0, carry_out_bytes=1, carry_in_bytes=1),),
            )

    def test_a_two_segment_census_counts_positions_across_segments(self):
        """``position`` runs over the whole program; a record's ``SegmentRecord.pass_index`` restarts at
        0 in every segment, so copying it across would refuse every multi-segment program."""
        first, second = _two_pass_census().passes
        third = PassCensus(position=2, segment_index=1, resident=True, cores=2, resident_cores=2,
                           written_cells_backing_store=0, written_cells_producer=0,
                           payload_bytes=0, compute_steps=4)
        census = PassPeriodCensus(passes=(first, second, third), boundaries=_two_pass_census().boundaries)
        assert [p.position for p in census.passes] == [0, 1, 2]
        assert [p.segment_index for p in census.passes] == [0, 0, 1]
        assert PassPeriodCensus.from_dict(json.loads(json.dumps(census.to_dict()))) == census
        restarted = PassCensus(**{**third.__dict__, "position": 0})
        with pytest.raises(PeriodCensusError, match="position 0 at 2"):
            PassPeriodCensus(passes=(first, second, restarted))

    def test_the_error_is_a_toolchain_refusal(self):
        assert issubclass(PeriodCensusError, ToolchainRefusal)
        assert issubclass(PeriodCensusError, ValueError)

    def test_a_count_of_any_int_type_is_held_as_an_int(self):
        """The seam's one integer rule: any integral type (numpy included), normalized exactly."""
        first = _two_pass_census().passes[0]
        census = PassCensus(**{**first.__dict__, "cores": np.int64(3), "compute_steps": np.int32(5)})
        assert census == first and type(census.cores) is int and type(census.compute_steps) is int
        boundary = PassBoundaryCensus(after_pass=np.int64(0), carry_out_bytes=np.uint16(40),
                                      carry_in_bytes=40)
        assert boundary == _two_pass_census().boundaries[0]
        assert type(boundary.carry_out_bytes) is int

    @pytest.mark.parametrize("value", [-1, 2.0, 2.5, True, "2", None, np.float64(2.0)])
    def test_a_count_outside_the_naturals_refuses_by_name(self, value):
        first = _two_pass_census().passes[0]
        with pytest.raises(PeriodCensusError, match="PassCensus.payload_bytes"):
            PassCensus(**{**first.__dict__, "payload_bytes": value})
        with pytest.raises(PeriodCensusError, match="PassBoundaryCensus.carry_in_bytes"):
            PassBoundaryCensus(after_pass=0, carry_out_bytes=1, carry_in_bytes=value)

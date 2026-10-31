"""S0 amendment B (owner decision 8): the run door judges every present key's schema, as the document door does.

A run never calls the document validator, so the resolver (``build_platform_constraints_resolved``) is the run
door. Its one loop judges every present key by the schema the document door's registry loop judges it with
(``schema_of``), one row per key with the document door's message, except the runtime facts the run itself writes
(no document declares them); the domain tag of each declaration that carries a value schema or the passthrough flag;
and the platform bias rule. A refusal is a ``DeclarationValueError`` (a ``ToolchainRefusal``) naming every
violation. The legal-value-set law is the run's in ``merge_pipeline_config`` (``derive_pipeline_runtime_parameters``
raises the document door's message), which every document run passes before the resolver.

The 85 closed or bounded keys the run door did not judge were audited against their consumers: a key's legal set is
what its consumer accepts and runs (plan section 20 item 3). ``AUDITED_LEGAL_SETS`` pins the outcome, so a new closed
key or a moved legal set fails here until it is audited.

The judge lives in the registry (``declaration_judge``), so both doors call one function. A run reads some
declarations before its flat config exists (the derivation re-derives or casts them, the session casts the top-level
seed and reads some deployment parameters from the section itself), so the both-doors test judges a document through
the CLI path (``PipelineSession.from_config``). On the base and through S0 amendment B the path read those cases
before any judge (``RUN_PATH_GAPS``, ``SHADOWED_SECTION_READS``, ``MISPLACED``, ``NULL_MISREADS``, pinned as strict
xfails); the joint closure of S0-B and G4 lands G4 interface notes 1 to 3, so every entry point judges a document's
declarations as written before anything reads one, and each case is a plain passing test. The judge of a document
judges the run's layout of its two sections and each section's own declarations as written, by their schemas, the
document domain rule, the legal-value-set law and the schemas consumers apply, so it states every refusal of the
document door's registry loop, legality and domain rules and of the run's layout (review round 2, the closure).
"""

import ast
import copy
import importlib.util
import inspect
import json
import math
import os
import textwrap
from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from conftest import MockDataProviderFactory, MockPipeline, default_config
from mimarsinan.chip_simulation.execution_bounds import resolve_simulation_step_timeout_s
from mimarsinan.chip_simulation.nevresim import connectivity
from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import floorplan_config_errors, resolve_floorplan
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
    derive_pipeline_runtime_parameters,
    legal_value_error,
    legal_values_for,
)
from mimarsinan.config_schema.registry import (
    REGISTRY,
    Category,
    mvm_document_errors,
    parse_deployment_document,
    retired_key_errors,
)
from mimarsinan.config_schema.registry import build as build_module
from mimarsinan.config_schema.registry import declaration_judge, entries_platform
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.declaration_judge import document_declaration_errors
from mimarsinan.config_schema.registry.entries_write import INJECTED_DECLARATIONS
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.config_schema.runtime import DerivationRefusal
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.config_schema.registry.relevance import Relevance
from mimarsinan.config_schema.registry.types import ConfigKeySchema
from mimarsinan.config_schema.registry.value_schema import (
    DeclarationValueError,
    ValueSchema,
    describe,
    entry_value_errors,
    schema_of,
)
from mimarsinan.config_schema.validation import (
    legality_errors,
    non_declarable_key_errors,
    validate_deployment_config,
)
from mimarsinan.data_handling.ffcv.spec_builder import _PIL_INTERP, _PILResize
from mimarsinan.data_handling.preprocessing import PreprocessingSpec, interpolation_mode_names, resolve_preprocessing
from mimarsinan.pipelining.core import platform_constraints_resolver as resolver_module
from mimarsinan.pipelining.core import simulation_factory
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.determinism import isolated_rng_stream
from mimarsinan.pipelining.session import PipelineSession
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.transformations.pruning.seed_generators import SeedContext, partial_column_group_seed_masks
from mimarsinan.tuning.learning_rate_explorer import LRRangeFinder
from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss

_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs")


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _base(entry):
    """A legal document of the key's own domain (a value-domain key lives in a value-domain document)."""
    return _load("mvm_lenet5.json" if entry.domain == "value" else "sync_simplemlp.json")


def _placed(document, entry, value):
    document = copy.deepcopy(document)
    container = document if entry.section == "top" else document[entry.section]
    container[entry.flat_key] = value
    return document


def _flat(document):
    return merge_pipeline_config(copy.deepcopy(document["deployment_parameters"]),
                                 copy.deepcopy(document["platform_constraints"]))


def _run(flat):
    """The run door's verdict: [] when it resolves, else the refusal's messages."""
    try:
        build_platform_constraints_resolved(flat)
    except DeclarationValueError as refusal:
        assert isinstance(refusal, ToolchainRefusal)
        return str(refusal).split("; ")
    return []


def _parts(messages):
    return sorted(part for message in messages for part in message.split("; "))


def _golden_module():
    path = os.path.join(os.path.dirname(__file__), "test_golden_resolution_snapshot.py")
    spec = importlib.util.spec_from_file_location("_golden_resolution_for_run_door", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GOLDEN = _golden_module()


class _StubProviderFactory:
    """The golden snapshot's provider stub: the registered provider class with dataset I/O replaced by facts."""

    def __init__(self, document):
        self._name = document.get("data_provider_name")
        self._dp = document.get("deployment_parameters") or {}

    def create(self):
        return _GOLDEN.stub_provider(self._name, self._dp)


def _through_the_run(document, workdir):
    """The run's verdict on a document through the CLI path, as ``src/main.py`` builds it
    (``PipelineSession.from_config``: ``parse_deployment_config``, the preset, ``merge_pipeline_config``, the
    provider facts, the workload profiles, the plan and the session's determinism), then the run door that Model
    Configuration calls on the session's config: [] when it resolves, else the refusal's messages. Any other
    exception propagates. The session's global determinism (RNG seeds, torch flags) is put back afterwards."""
    document = {**copy.deepcopy(document), "_working_directory": str(workdir)}
    flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        with isolated_rng_stream():
            session = PipelineSession.from_config(document, data_provider_factory=_StubProviderFactory(document))
            build_platform_constraints_resolved(session.pipeline.config)
    except DeclarationValueError as refusal:
        assert isinstance(refusal, ToolchainRefusal)
        return str(refusal).split("; ")
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.benchmark = flags[2]
        torch.backends.cuda.matmul.allow_tf32 = flags[3]
        torch.backends.cudnn.allow_tf32 = flags[4]
        torch.set_float32_matmul_precision(flags[5])
    return []


def _document_only(document):
    """The document door's refusals no run states: of the document as a document, a key a document may not
    declare (``non_declarable_key_errors``) and a retired key (``retired_key_errors``), which the run reads through
    the derivation or the retirement bridge. The domain rule, the legal-value-set law and the floorplan pairing
    are the one judge's since the joint closure, so both doors state them."""
    return set(non_declarable_key_errors(document)) | {row["message"] for row in retired_key_errors(document)}


_LEGALITY = frozenset(key for key, entry in REGISTRY.items() if entry.legal_values is not None)
_RUNTIME = frozenset(key for key, entry in REGISTRY.items() if entry.category is Category.RUNTIME)

#: Every key judged by a scalar schema (type, options, bounds) and by no legal-value rule: the 85 closed or bounded
#: keys of owner decision 8 among them, and every typed key beside them (a bool, a string, an unbounded number).
SCALAR_KEYS = tuple(sorted(
    key for key, entry in REGISTRY.items()
    if schema_of(entry) is not None and entry.value_schema is None
    and key not in _LEGALITY and key not in _RUNTIME))


def _illegal_values(entry):
    """A value of another type, and a value outside the options or the bounds (where the schema has any)."""
    schema = schema_of(entry)
    values = {"int": [7.5], "float": ["0.5"], "bool": ["false"]}.get(schema.kind, [7])
    if schema.options is not None:
        values.append(f"{schema.options[0]}_unknown")
    if schema.low is not None:
        values.append(schema.low if schema.low_exclusive else schema.low - 1)
    if schema.high is not None:
        values.append(schema.high if schema.high_exclusive else schema.high + 1)
    return values


SCALAR_CASES = [(key, value) for key in SCALAR_KEYS for value in _illegal_values(REGISTRY[key])]

#: G4 interface note 1 (``merge_pipeline_config``, G4's, landed by the joint closure): the merge calls
#: ``declared_sections_errors`` on the two sections and raises its refusal before ``derive_deployment_parameters``
#: re-derives or casts a declaration (``bool("false")`` is true; an explicit activation_quantization is replaced by
#: the derived one) or raises a plain ValueError that misreads it.
G4_MERGE = ("G4 interface note 1: merge_pipeline_config judges the declared sections (declared_sections_errors) "
            "before derive_deployment_parameters reads them")
#: G4 interface note 2 (``parse_deployment_config``, G4's, landed): the session calls ``document_declaration_errors``
#: on the document before it reads one (the ``int(seed)`` cast, the data provider, the working directory, the preset,
#: the target metric override, and the deployment parameters it hands the data provider or branches on).
G4_SESSION = ("G4 interface note 2: parse_deployment_config judges the document (document_declaration_errors) "
              "before it reads a declaration")

#: Every case of the both-doors test that the CLI path read before any judge on the base and through S0 amendment B
#: (evidence/run_path_gaps_fix1.txt), with the G4 interface note that closed it; each was a strict xfail until the
#: joint closure and is a plain passing case since.
RUN_PATH_GAPS = {
    ("activation_quantization", "false"): G4_MERGE,
    ("enable_loihi_simulation", "false"): G4_MERGE,
    ("enable_nevresim_simulation", "false"): G4_MERGE,
    ("enable_sanafe_simulation", "false"): G4_MERGE,
    ("lif_exact_qat", "false"): G4_MERGE,
    ("lif_exact_qat_kd", "false"): G4_MERGE,
    ("mirror_training_recipe", "false"): G4_MERGE,
    ("spiking_mode", 7): G4_MERGE,
    ("spiking_mode", "lif_unknown"): G4_MERGE,
    ("ttfs_cycle_schedule", 7): G4_MERGE,
    ("ttfs_cycle_schedule", "cascaded_unknown"): G4_MERGE,
    ("weight_quantization", "false"): G4_MERGE,
    ("data_provider_name", 7): G4_SESSION,
    ("datasets_path", 7): G4_SESSION,
    ("experiment_name", 7): G4_SESSION,
    ("generated_files_path", 7): G4_SESSION,
    ("pipeline_mode", 7): G4_SESSION,
    ("pipeline_mode", "vanilla_unknown"): G4_SESSION,
    ("seed", 7.5): G4_SESSION,
    ("seed", -1): G4_SESSION,
    ("start_step", 7): G4_SESSION,
    ("stop_step", 7): G4_SESSION,
    ("target_metric_override", "0.5"): G4_SESSION,
}


SCALAR_PARAMS = [pytest.param(key, value, id=f"{key}-{value!r}"[:48]) for key, value in SCALAR_CASES]


def _document_door(document):
    """The document door's statements, or the refusal it raises (a crash in a document rule is no statement:
    an unknown core_semantics raises in the domain rules, a G4 interface note)."""
    try:
        return validate_deployment_config(document)
    except ValueError as refusal:
        return [str(refusal)]


class TestEveryPresentSchemaAtTheRunDoor:
    """Owner decision 8: the run door judges every present key's schema, with the document door's verdict and
    message."""

    @pytest.mark.parametrize("key,value", SCALAR_PARAMS)
    def test_a_value_outside_its_schema_refuses_by_name_at_both_doors(self, key, value, tmp_path):
        """The value is declared in the document and the run side takes the document through the CLI path
        (``_through_the_run``), so a declaration the derivation or the session read before the run door would
        show (``RUN_PATH_GAPS``, closed by G4 interface notes 1 and 2)."""
        entry = REGISTRY[key]
        base = _base(entry)
        assert validate_deployment_config(base) == [] and _through_the_run(base, tmp_path / "base") == []
        row = "; ".join(entry_value_errors(entry, value))
        assert row
        document = _placed(base, entry, value)
        stated = [m for m in validate_deployment_config(document) if m not in _document_only(document)]
        assert row in stated
        assert _parts(_through_the_run(document, tmp_path / "run")) == _parts(stated)

    def test_every_run_path_gap_is_a_case_of_the_both_doors_test(self):
        """A closed gap whose case left the parametrization would drop its pin silently."""
        assert set(RUN_PATH_GAPS) <= set(SCALAR_CASES)

    @pytest.mark.parametrize("key,value", SCALAR_CASES, ids=lambda x: str(x)[:40])
    def test_the_declaration_judge_states_the_document_door_s_verdict(self, key, value):
        """The judge both doors can call (``document_declaration_errors``) states, before any reading, what the
        document door states of every case, those the CLI path reads first included."""
        entry = REGISTRY[key]
        document = _placed(_base(entry), entry, value)
        stated = [m for m in validate_deployment_config(document) if m not in _document_only(document)]
        assert _parts(document_declaration_errors(document)) == _parts(stated)

    @pytest.mark.parametrize("key", sorted(_LEGALITY))
    def test_the_resolver_refuses_a_legality_bearing_value_outside_its_schema_by_name(self, key):
        """Every present key's schema is judged at the run door, a legality-bearing key's too (its legal set is a
        subset of its options), on a flat config no merge built (a search overlay, a hand config). Where the key's
        legality does not apply (``preload_weights: 7`` on a model with no weight sets) both doors judge its schema
        since the joint closure (``validate_against_registry`` judges every key's schema, and the legal-value-set
        law only a value the schema admits)."""
        entry = REGISTRY[key]
        value = 7 if entry.type.value in ("bool", "str", "enum") else "not_a_legal_value"
        flat = _flat(_base(entry))
        flat[key] = value
        row = "; ".join(entry_value_errors(entry, value))
        assert row and row in _run(flat)

    @pytest.mark.parametrize("key", sorted(_LEGALITY - {"s_allocation"}))
    def test_a_document_run_refuses_a_value_outside_the_legal_set_at_both_doors(self, key):
        """Where its legality applies, a legality-bearing key outside the legal set refuses at the document door
        (``legality_errors``) and at the run door in ``merge_pipeline_config`` (the same message where the
        derivation judges the key, ``derive_pipeline_runtime_parameters``; an unknown core_semantics or family
        raises in the derivation, and in the document door's domain rules, by name)."""
        entry = REGISTRY[key]
        base = _base(entry)
        legal = legal_values_for(key, _flat(base))
        if legal is None:
            pytest.skip(f"{key}: legality does not apply in the base document's state")
        value = "not_a_legal_value"
        document = _placed(base, entry, value)
        stated = _document_door(document)
        assert any(key in m and repr(value) in m for m in stated), stated
        with pytest.raises(ValueError) as merge_refusal:
            _flat(document)
        merge_message = str(merge_refusal.value)
        assert key in merge_message and repr(value) in merge_message
        if merge_message == str(legal_value_error(key, value, list(legal))):
            assert merge_message in stated

    def test_simulation_steps_is_judged_at_the_run_door(self):
        """``simulation_steps`` is judged at both doors since this amendment: a non-integral window refuses by
        name where the base run door ran it at its int() cast."""
        base = _load("sync_simplemlp.json")
        for value, message in ((32.5, "simulation_steps must be int, got float (32.5)"),
                               ("32", "simulation_steps must be int, got str ('32')"),
                               (0, "simulation_steps=0 is below the minimum 1")):
            document = copy.deepcopy(base)
            document["platform_constraints"]["simulation_steps"] = value
            assert message in validate_deployment_config(document)
            with pytest.raises(DeclarationValueError) as refusal:
                build_platform_constraints_resolved(_flat(document))
            assert str(refusal.value) == message

    def test_an_integral_float_window_is_refused_at_both_doors_strict_stated(self):
        """Boundary finding (owner decision 'Strict, stated', plan section 21 item 7): 32.0 equals the integer
        window in value and the base ran it exactly at the int() cast; both doors refuse it by the INT rule."""
        document = _load("sync_simplemlp.json")
        document["platform_constraints"]["simulation_steps"] = 32.0
        message = "simulation_steps must be int, got float (32.0)"
        assert message in validate_deployment_config(document)
        with pytest.raises(DeclarationValueError, match=r"simulation_steps must be int, got float \(32\.0\)"):
            build_platform_constraints_resolved(_flat(document))

    @pytest.mark.parametrize("key,value,consumer_reads", [
        # spiking_semantics.lif_execution_synchronized reads every discipline but 'synchronized' as streaming.
        ("lif_execution_discipline", "synchronised", "streaming"),
        # torch_mapping/encoding_layers.mark_encoding_layers refuses any placement but subsume or offload.
        ("encoding_layer_placement", "Offload", "a ValueError at the mapping step"),
        # DeploymentPlan reads the pruning switch through bool(): the string 'false' is true.
        ("pruning", "false", "true"),
    ])
    def test_the_scouted_run_door_admissions_are_refused_by_name(self, key, value, consumer_reads):
        """The polish-round scout found these resolving at the run door with the flat config holding them as
        declared (``evidence/run_door_scope_polish.txt``); the consumer misreads or refuses each."""
        flat = _flat(_load("sync_simplemlp.json"))
        flat[key] = value
        with pytest.raises(DeclarationValueError, match=key):
            build_platform_constraints_resolved(flat)

    def test_a_runtime_fact_is_no_declaration(self):
        """The run writes the device, the input shape and the class count; no document declares them, so the
        run door never judges them (a torch.device is no STR declaration)."""
        flat = _flat(_load("sync_simplemlp.json"))
        flat.update({"device": torch.device("cpu"), "input_shape": (1, 28, 28), "input_size": 784,
                     "num_classes": 10})
        assert REGISTRY["device"].category is Category.RUNTIME
        build_platform_constraints_resolved(flat)

    def test_an_unknown_core_semantics_names_no_domain(self):
        """An unknown core_semantics names no run domain, so the run door states its value schema's refusal alone
        and judges no domain tag: a value-domain declaration beside it is judged by its value only."""
        entry = REGISTRY["readout_law"]
        assert entry.domain == "value" and not entry.has_default() and entry.value_schema is not None
        assert _message("readout_law", _LEGAL_READOUT_LAW) == ""
        flat = _flat(_load("sync_simplemlp.json"))
        flat.update({"core_semantics": "bogus", "readout_law": _LEGAL_READOUT_LAW})
        assert _parts(_run(flat)) == _parts([_message("core_semantics", "bogus")])

    def test_every_violation_is_stated_once(self):
        """Every present key outside its schema is named in the one refusal, each once (a passthrough key and a
        key judged by its scalar schema alike)."""
        flat = _flat(_load("mvm_lenet5.json"))
        flat.update({"lr": -1.0, "inference_batch_K": 0, "pruning": "no"})
        messages = _run(flat)
        assert sorted(messages) == sorted([
            "lr=-1.0 is below the minimum 0.0", "inference_batch_K=0 is below the minimum 1",
            "pruning must be bool, got str ('no')"])

    def test_a_value_the_floorplan_cannot_cast_is_refused_by_its_schema_first(self):
        """The floorplan resolution casts the grid with int(); the run door judges the value schema first, so a
        value the cast cannot read refuses by name (the document door raises in its floorplan pairing, a G4
        interface note)."""
        flat = _flat(_load("sync_simplemlp.json"))
        flat["tile_grid_rows"] = "x"
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved(flat)
        assert str(refusal.value) == "tile_grid_rows must be int, got str ('x')"

    def test_the_run_door_judges_by_the_registry_schema_and_names_no_key(self):
        """One loop over the registry, judging ``entry_value_errors`` (``schema_of``) for every present key: which
        keys it judges is read from the registry, never listed; the bias rule takes the bias entry by name. The
        loop is the registry's (``declaration_judge``), and the resolver calls it."""
        assert resolver_module.run_door_errors is declaration_judge.run_door_errors
        assert resolver_module.declaration_errors is declaration_judge.declaration_errors
        loop = ast.parse(textwrap.dedent(inspect.getsource(declaration_judge.declaration_errors)))
        names = {node.id for node in ast.walk(loop) if isinstance(node, ast.Name)}
        assert {"REGISTRY", "entry_value_errors", "domain_errors"} <= names
        layout = ast.parse(textwrap.dedent(inspect.getsource(declaration_judge._layout_errors)))
        assert {"declaration_errors", "platform_bias_errors", "floorplan_config_errors"} <= {
            node.id for node in ast.walk(layout) if isinstance(node, ast.Name)}
        door = ast.parse(textwrap.dedent(inspect.getsource(declaration_judge.run_door_errors)))
        assert "_layout_errors" in {node.id for node in ast.walk(door) if isinstance(node, ast.Name)}
        trees = [(loop, set()), (layout, {"has_bias"}), (door, set())] + [
            (ast.parse(textwrap.dedent(inspect.getsource(function))), set())
            for function in (declaration_judge.sections_run_layout, declaration_judge.section_declarations,
                             declaration_judge.legality_rows, declaration_judge.applied_errors,
                             declaration_judge._domain_rule_errors, declaration_judge._sections_view,
                             declaration_judge.declared_legality_rows, declaration_judge.declared_sections_errors,
                             declaration_judge.document_declaration_errors)]
        for tree, allowed in trees:
            constants = {node.value for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            assert not (set(REGISTRY) & constants) - allowed, sorted(set(REGISTRY) & constants)
        source = inspect.getsource(resolver_module.build_platform_constraints_resolved)
        assert "run_door_errors(pipeline_config)" in source


#: G4 interface note 3 (``validate_deployment_config``, G4's, landed): the document door judges
#: ``document_declaration_errors(config)`` beside ``validate_against_registry`` over the parser's known flat keys (which
#: skipped a legality-bearing key and misses a key declared in the other section), in place of the bias rule over
#: ``document_run_layout``, so it judges every declaration where, and as, the run judges it, each message once.
G4_DOCUMENT = ("G4 interface note 3: validate_deployment_config judges document_declaration_errors(config), the run "
               "layout with legality-bearing keys included")

#: The reviewed misreads (review round 1, ``review_1/probes/session_probe_rows.txt``): a declaration the CLI path reads
#: before any judge, by its section and fixture (lif_mmixcore, or the TTFS fixture where lif_exact_qat's pairing
#: raises), with the G4 note that closes it. On lif_mmixcore the run door refuses ``lif_exact_qat: "false"`` by
#: name, because the pairing reads it as true without raising and the flat config keeps it as written.
SCOUTED_MISREADS = [
    ("deployment_parameters", "weight_quantization", "false", "lif_mmixcore.json", G4_MERGE),
    ("deployment_parameters", "weight_quantization", "false", "sync_simplemlp.json", G4_MERGE),
    ("deployment_parameters", "enable_nevresim_simulation", "false", "lif_mmixcore.json", G4_MERGE),
    ("deployment_parameters", "enable_loihi_simulation", "false", "lif_mmixcore.json", G4_MERGE),
    ("deployment_parameters", "enable_sanafe_simulation", "false", "lif_mmixcore.json", G4_MERGE),
    ("deployment_parameters", "lif_exact_qat", "false", "sync_simplemlp.json", G4_MERGE),
    ("deployment_parameters", "mirror_training_recipe", "false", "lif_mmixcore.json", G4_MERGE),
    ("top", "seed", 7.5, "lif_mmixcore.json", G4_SESSION),
    ("top", "seed", "7", "lif_mmixcore.json", G4_SESSION),
    ("top", "pipeline_mode", "Vanilla", "lif_mmixcore.json", G4_SESSION),
    ("top", "target_metric_override", "0.5", "lif_mmixcore.json", G4_SESSION),
]

#: A key declared in the other section: the parser lists it as unknown and the document door judged nothing, while
#: the run lays both sections flat and reads it (review round 1, ``misplaced_probe.txt``). The last two are
#: derivation-read keys (S0-B final review, the major): the derivation rewrote them before the run door saw them, so
#: both doors admitted them and the run misread them (``bool("false")`` ran quantized, ``"false" is not False`` armed
#: the simulator; ``review_3/probes/misplaced_derived_*.jsonl``, identical on the base).
MISPLACED = [
    ("platform_constraints", "lr", -1.0),
    ("platform_constraints", "lif_execution_discipline", "synchronised"),
    ("platform_constraints", "encoding_layer_placement", "Offload"),
    ("platform_constraints", "weight_quantization", "false"),
    ("platform_constraints", "enable_nevresim_simulation", "false"),
]

#: A legality-bearing key outside its schema where its legality does not apply (the model registers no weight sets):
#: the document door leaves it to ``legality_errors``, which skips it (review round 1, ``docprobe_after.json``).
INAPPLICABLE_LEGALITY = [("preload_weights", "false"), ("preload_weights", 0), ("preload_weights", 1),
                         ("pretrained_weight_set", 7)]

#: A null declaration the run reads differently from an absent one (``evidence/null_scan_fix1.txt``): both doors read a
#: null as unset, ``merge_pipeline_config`` lays it over the defaults as None, and the derivation reads it
#: (``bool(None)`` is false, ``str(None)`` is no firing mode, ``fold_spiking_axes`` refuses a None family).
NULL_MISREADS = [("sync_simplemlp.json", "weight_quantization"), ("lif_mmixcore.json", "weight_quantization"),
                 ("mvm_lenet5.json", "weight_quantization"), ("lif_mmixcore.json", "firing_mode"),
                 ("lif_mmixcore.json", "lif_exact_qat"), ("lif_mmixcore.json", "spiking_family"),
                 ("sync_simplemlp.json", "spiking_family")]
G4_NULL = ("G4 interface note 1(c): merge_pipeline_config lays a null as unset (run_layers lays nothing for it) before "
           "derive_deployment_parameters, as both doors read a null declaration")

#: A deployment-parameters declaration the session reads from the section itself, shadowed in the run's layout by a
#: platform-constraints copy of the key (review round 2, ``review_2/probes/shadow_dp_reads.txt``,
#: ``hw_mode_shadow.txt``): ``parse_deployment_config`` hands the data provider the section's ``preprocessing`` and
#: ``batch_size`` and branches on its ``hw_config_mode`` (the fixed branch skips the search-space merge), while the
#: flat config carries the copy, which the run door admits. The fixture, the key, the section's value, the copy.
SHADOWED_SECTION_READS = [
    ("sync_simplemlp.json", "preprocessing", {"resize_to": 32, "interpolation": "Bilinear"},
     {"resize_to": 32, "interpolation": "bilinear"}),
    ("sync_simplemlp.json", "batch_size", 0, 64),
    ("sync_simplemlp.json", "batch_size", 7.5, 64),
    ("lif_mmixcore.json", "hw_config_mode", "Search", "search"),
]

#: A legal value-domain readout law (a key with a value schema and no schema default, whose domain tag is judged).
_LEGAL_READOUT_LAW = {"adc_bits": 6, "range": "fixed", "range_value": 4.0, "mult_bits": 16}


def _shadowed(name, key, value, shadow):
    document = _load(name)
    document["deployment_parameters"][key] = value
    document["platform_constraints"][key] = shadow
    return document


def _declared(section, key, value, name="lif_mmixcore.json"):
    document = _load(name)
    (document if section == "top" else document[section])[key] = value
    return document


def _message(key, value):
    return "; ".join(entry_value_errors(REGISTRY[key], value))


def _merged_or_refusal(document):
    """The merged run config, or the merge's refusal (type and message)."""
    try:
        return merge_pipeline_config(copy.deepcopy(document["deployment_parameters"]),
                                     copy.deepcopy(document["platform_constraints"]))
    except ValueError as refusal:
        return f"{type(refusal).__name__}: {refusal}"


def _absent_and_null(name, key):
    """The merged run config (or refusal) with ``key`` absent and with it declared null, each without the key."""
    entry = REGISTRY[key]
    absent = _load(name)
    absent[entry.section].pop(key, None)
    null = copy.deepcopy(absent)
    null[entry.section][key] = None
    out = []
    for document in (absent, null):
        merged = _merged_or_refusal(document)
        out.append({k: v for k, v in merged.items() if k != key} if isinstance(merged, dict) else merged)
    return out


class TestTheDeclarationJudge:
    """The judge both doors can call before anything reads a declaration (``declaration_judge``): the run door's
    verdict on a document's declarations as written, in the layout a run gives them, with the top-level keys the
    session reads. Interface notes 1 to 3 hand it to G4's merge, session parse and document door."""

    @pytest.mark.parametrize("section,key,value,name,note", SCOUTED_MISREADS, ids=lambda x: str(x)[:30])
    def test_the_judge_refuses_each_scouted_misread_by_name(self, section, key, value, name, note):
        document = _declared(section, key, value, name)
        assert _message(key, value) in validate_deployment_config(document)
        assert document_declaration_errors(document) == [_message(key, value)]
        if section != "top":
            assert declaration_judge.declared_sections_errors(
                document["deployment_parameters"], document["platform_constraints"]) == [_message(key, value)]

    @pytest.mark.parametrize("section,key,value", MISPLACED, ids=lambda x: str(x)[:30])
    def test_a_key_declared_in_the_other_section_is_judged_where_the_run_lays_it(self, section, key, value):
        document = _declared(section, key, value)
        assert f"{section}.{key}" in parse_deployment_document(document).unknown
        assert document_declaration_errors(document) == [_message(key, value)]

    @pytest.mark.parametrize("key,value", INAPPLICABLE_LEGALITY, ids=lambda x: str(x)[:30])
    def test_a_legality_bearing_key_is_judged_by_its_schema_where_its_legality_does_not_apply(self, key, value):
        document = _declared("deployment_parameters", key, value)
        assert legal_values_for(key, _flat(_load("lif_mmixcore.json"))) is None
        assert document_declaration_errors(document) == [_message(key, value)]

    def test_a_declaration_in_the_other_section_a_later_one_replaces_is_refused_by_name(self):
        """A platform-constraints key declared in the deployment parameters (an unknown key to the parser, so no
        section's own declaration) is replaced in the run's layout by the platform default layer and the platform
        constraints' own window, so no reader reads it: the judge refuses it by name at both doors and at the merge
        (review round 1; before, neither door judged it and the run read the window). A key in its own section is
        judged as written there, a later copy or not
        (``test_a_section_s_own_declaration_is_judged_as_written_under_a_later_copy``)."""
        document = _declared("deployment_parameters", "simulation_steps", 32.5, name="sync_simplemlp.json")
        assert REGISTRY["simulation_steps"].section == "platform_constraints"
        assert isinstance(document["platform_constraints"]["simulation_steps"], int)
        message = ("deployment_parameters.simulation_steps: simulation_steps is a platform_constraints key, and "
                   "nothing reads it here (the platform default layer replaces it): declare it in platform_constraints")
        assert document_declaration_errors(document) == validate_deployment_config(document) == [message]
        with pytest.raises(DeclarationValueError, match=r"deployment_parameters\.simulation_steps: "):
            _flat(document)

    def test_a_top_level_key_is_judged_where_the_session_reads_it(self):
        """The session seeds the data provider with the top-level seed even where the deployment parameters
        declare one (which the merged config carries), so the top-level value is judged on its own."""
        document = _declared("top", "seed", 7.5)
        document["deployment_parameters"]["seed"] = 3
        assert _message("seed", 7.5) in document_declaration_errors(document)

    def test_a_refusal_two_declarations_state_is_stated_once(self):
        """The top-level seed is judged on its own and the deployment parameters' seed where the run lays it; the
        one violation both state is stated once, as the document door states it."""
        document = _declared("top", "seed", 7.5)
        document["deployment_parameters"]["seed"] = 7.5
        assert document_declaration_errors(document) == validate_deployment_config(document) == [
            _message("seed", 7.5)]

    @pytest.mark.parametrize("name,key,value,shadow", SHADOWED_SECTION_READS, ids=lambda x: str(x)[:30])
    def test_a_section_s_own_declaration_is_judged_as_written_under_a_later_copy(self, name, key, value, shadow):
        """The judge judges each section's own declarations as written beside the run's layout
        (``section_declarations``, the document door's view of the two sections): the session reads the deployment
        parameters' declaration from the section itself, while the flat config carries the platform constraints'
        copy, so the layout alone admits what the session misreads (``Bilinear`` runs as bicubic, a batch size of
        0 or 7.5 reaches the data provider, ``Search`` takes the fixed branch under a merged ``search``). The
        judge and the document door refuse it by name, with one message, and so does the merge since G4 interface
        note 1 landed (``TestRunPathGaps`` pins the CLI path)."""
        document = _shadowed(name, key, value, shadow)
        dp, pc = document["deployment_parameters"], document["platform_constraints"]
        assert REGISTRY[key].section == "deployment_parameters" and _message(key, shadow) == ""
        layout = declaration_judge.sections_run_layout(dp, pc)
        assert layout[key] == shadow and declaration_judge.run_door_errors(read_like_consumers(layout)) == []
        message = _message(key, value)
        # A copy that turns the hardware search on brings the document door's presence rule over the layout too.
        assert validate_deployment_config(document)[:1] == [message]
        assert declaration_judge.declared_sections_errors(dp, pc) == [message]
        assert document_declaration_errors(document) == [message]

    def test_the_document_door_judges_a_structural_platform_body(self):
        """The wizard's structural platform-constraints shapes (``mode`` user or auto) nest the body the document
        door's parser reads; ``merge_pipeline_config`` lays their containers as flat keys, so the judge, which judges
        what a run lays, sees no body. G4 interface note 3 therefore adds the judge beside the parser's registry
        loop, never in place of it: in place of it, the document door would stop refusing a body value outside its
        schema (review round 2, ``evidence/shadow_probe_fix2.txt``, W1)."""
        document = _load("sync_simplemlp.json")
        body = dict(document["platform_constraints"], simulation_steps=32.5)
        document["platform_constraints"] = {"mode": "user", "user": body}
        assert _message("simulation_steps", 32.5) in validate_deployment_config(document)

    def test_the_own_declarations_are_read_over_the_other_section_s(self):
        """Each section's own declarations are read like their consumers together, as the document door reads its
        parser's view of the two sections: an explicit S map whose every entry equals the platform constraints'
        window reads as the uniform declaration at both doors and in the judge, and one whose entries differ
        stays the reserved explicit mode, refused by name by both."""
        document = _load("lif_mmixcore.json")
        document["deployment_parameters"].update({"s_allocation": "explicit", "s_allocation_explicit": [4, 4, 4]})
        document["platform_constraints"].update({"simulation_steps": 4, "target_tq": 4})
        assert validate_deployment_config(document) == document_declaration_errors(document) == []
        document["platform_constraints"]["simulation_steps"] = 8
        reserved = _message("s_allocation", "explicit")
        assert reserved and reserved in validate_deployment_config(document)
        assert document_declaration_errors(document) == [reserved]

    def test_a_null_declaration_is_unset(self):
        """Both doors read a null as unset, so the judge states of a null declaration exactly what it states of the
        key undeclared (``NULL_MISREADS``: the run reads it alike, G4 interface note 1(c); on the TTFS fixture an
        undeclared spiking family is the lif default, under which its TTFS firing and spike modes are illegal)."""
        for name, key in NULL_MISREADS:
            entry = REGISTRY[key]
            absent = _load(name)
            absent[entry.section].pop(key, None)
            null = copy.deepcopy(absent)
            null[entry.section][key] = None
            assert document_declaration_errors(null) == document_declaration_errors(absent), (name, key)
            assert validate_deployment_config(null) == validate_deployment_config(absent), (name, key)

    def test_the_judge_reads_like_the_consumers(self):
        """A spelling the consumer normalizes is judged as the consumer reads it (``read_like_consumers``), as at
        both doors."""
        document = _declared("deployment_parameters", "s_allocation", "UNIFORM")
        assert document_declaration_errors(document) == validate_deployment_config(document) == []

    @pytest.mark.parametrize("name", sorted(os.listdir(_FIXTURES)))
    def test_every_fixture_passes_the_judge(self, name):
        document = _load(name)
        assert document_declaration_errors(document) == []

    def test_a_laid_null_reads_as_the_key_undeclared(self):
        """``run_layers`` (G4 interface note 1(c); review round 1: a null lays nothing): over the merge's layers, a
        null declaration reads as the key undeclared, the default layer's value or absent, for every misread of
        ``NULL_MISREADS`` and for keys the default layers hold (``lr``, a value; ``sanafe_custom_arch_path``,
        None); and the merge, which lays the sections so, derives the null document as it derives the absent
        one."""
        defaults = {**get_default_deployment_parameters(), **get_default_platform_constraints()}
        assert defaults["lr"] is not None and defaults["sanafe_custom_arch_path"] is None
        defaulted = [("lif_mmixcore.json", "lr"), ("sync_simplemlp.json", "sanafe_custom_arch_path")]

        def laid(document):
            return declaration_judge.run_layers(document["deployment_parameters"], document["platform_constraints"],
                                                defaults=True)

        for name, key in [*NULL_MISREADS, *defaulted]:
            entry = REGISTRY[key]
            absent = _load(name)
            absent[entry.section].pop(key, None)
            null = copy.deepcopy(absent)
            null[entry.section][key] = None
            assert laid(null) == laid(absent), (name, key)
            assert _merged_or_refusal(null) == _merged_or_refusal(absent), (name, key)

    def test_the_layout_is_the_merge_s_layers(self):
        """``sections_run_layout``: the deployment parameters, then the platform constraints' default layer (a
        deployment-parameters declaration of a key it holds is read by no run), then the platform constraints, each
        replacing a value whole; a null declaration is unset, so it lays nothing (review round 1)."""
        assert "weight_bits" in get_default_platform_constraints()
        layout = declaration_judge.sections_run_layout(
            {"weight_bits": 64, "a": 1, "nested": {"x": 1}, "b": "keep"}, {"a": None, "nested": {}})
        assert layout == {"a": 1, "nested": {}, "b": "keep"}
        assert declaration_judge.sections_run_layout("no object", {"lr": 0.5}) == {"lr": 0.5}

    def test_a_declaration_the_default_layer_replaces_is_refused_by_name(self, tmp_path):
        """A deployment-parameters ``activity_factor`` is replaced by the platform constraints' default layer, so the
        run read the default and ignored it, where both doors and the judge admitted it: no reader reads it, and the
        judge refuses it by name at both doors and on the CLI path (review round 1)."""
        document = _declared("deployment_parameters", "activity_factor", 5.0, name="sync_simplemlp.json")
        assert "activity_factor" not in document["platform_constraints"]
        assert "activity_factor" in get_default_platform_constraints() and _message("activity_factor", 5.0)
        message = ("deployment_parameters.activity_factor: activity_factor is a platform_constraints key, and "
                   "nothing reads it here (the platform default layer replaces it): declare it in platform_constraints")
        assert document_declaration_errors(document) == validate_deployment_config(document) == [message]
        assert _through_the_run(document, tmp_path) == [message]


class TestRunPathGaps:
    """Each case the CLI path read before any judge on the base and through S0 amendment B (the run door judged the
    flat config, which the derivation and the session had already rewritten), now refused by name on the CLI path:
    the joint closure lands the G4 interface note each case names (strict xfails until then)."""

    @pytest.mark.parametrize("section,key,value,name,note", [
        pytest.param(*case, id=f"{case[3][:-5]}-{case[1]}-{case[2]!r}") for case in SCOUTED_MISREADS])
    def test_a_scouted_misread_is_refused_by_name_on_the_cli_path(self, section, key, value, name, note, tmp_path):
        assert _parts(_through_the_run(_declared(section, key, value, name), tmp_path)) == _parts(
            [_message(key, value)])

    def test_lif_exact_qat_false_on_a_lif_document_is_refused_by_name_at_the_run_door(self, tmp_path):
        """The pairing reads ``"false"`` as true without raising on a lif document, and the flat config keeps
        the declaration as written, so the run door refuses it by name (the misread is the derivation's
        ``lif_per_hop_retiming``, which no deployment reaches)."""
        document = _declared("deployment_parameters", "lif_exact_qat", "false")
        assert _parts(_through_the_run(document, tmp_path)) == _parts([_message("lif_exact_qat", "false")])

    @pytest.mark.parametrize("name,key,value,shadow", [
        pytest.param(*case, id=f"{case[1]}-{case[2]!r}"[:40]) for case in SHADOWED_SECTION_READS])
    def test_a_shadowed_section_read_is_refused_by_name_on_the_cli_path(self, name, key, value, shadow, tmp_path):
        """The session reads the deployment parameters' declaration before any judge, and the run door judges the
        flat config, which carries the platform constraints' copy (``SHADOWED_SECTION_READS``)."""
        assert _parts(_through_the_run(_shadowed(name, key, value, shadow), tmp_path)) == _parts(
            [_message(key, value)])

    @pytest.mark.parametrize("name,key", [pytest.param(name, key, id=f"{name}-{key}") for name, key in NULL_MISREADS])
    def test_a_null_declaration_runs_as_an_absent_one(self, name, key):
        absent, null = _absent_and_null(name, key)
        assert null == absent

    def test_no_other_null_declaration_moves_the_run(self):
        """Every other declarable key of the three fixtures runs alike declared null and absent (beside the key
        itself), so ``NULL_MISREADS`` is the whole set."""
        moved = []
        for name in ("sync_simplemlp.json", "lif_mmixcore.json", "mvm_lenet5.json"):
            for key, entry in sorted(REGISTRY.items()):
                if (entry.section == "top" or entry.category is Category.RUNTIME or not entry.declarable
                        or (name, key) in NULL_MISREADS):
                    continue
                absent, null = _absent_and_null(name, key)
                if null != absent:
                    moved.append((name, key))
        assert moved == []

    @pytest.mark.parametrize("section,key,value", [
        pytest.param(*case, id=f"{case[0]}.{case[1]}-{case[2]!r}") for case in MISPLACED])
    def test_a_key_declared_in_the_other_section_is_refused_alike_at_both_doors(self, section, key, value,
                                                                               tmp_path):
        document = _declared(section, key, value)
        assert _parts(_through_the_run(document, tmp_path)) == _parts([_message(key, value)])
        assert _message(key, value) in validate_deployment_config(document)

    @pytest.mark.parametrize("key,value", [
        pytest.param(*case, id=f"{case[0]}-{case[1]!r}") for case in INAPPLICABLE_LEGALITY])
    def test_a_legality_bearing_key_outside_its_schema_is_refused_alike_at_both_doors(self, key, value, tmp_path):
        """Where the key's legality does not apply, the document door judges nothing; the CLI path reads
        ``preload_weights: "false"`` as true in the workload-profile fold before the run door."""
        document = _declared("deployment_parameters", key, value)
        assert validate_deployment_config(document) == [_message(key, value)]
        assert _parts(_through_the_run(document, tmp_path)) == _parts([_message(key, value)])


#: The legal set of every closed or bounded key (an option set or a bound), read from its consumers: owner decision
#: 8's audit of the 85 keys the run door did not judge, beside the four passthrough keys it did
#: (``RA/10_wave1/s0_declarations/amendment_b/evidence/consumer_audit.txt``). A new closed key, or a legal set
#: that moves, fails until it is audited against its consumers and re-pinned here.
AUDITED_LEGAL_SETS = {
    "aa_endpoint_recovery_steps": "INT >= 0",
    "activation_analysis_batch_size": "INT >= 1",
    "activation_bits": "INT >= 2",
    "activation_scale_policy": "ENUM count_quantile | min_distortion | percentile_norm | max_norm",
    "activation_scale_quantile": "FLOAT in [0, 1]",
    # Wave 2 WI-5b, WI-5c: the consumers' vocabularies (mapping/support/grid_declarations.py), audited there.
    "bias_resolution": "ENUM weight_grid | law",
    "activity_factor": "FLOAT in [0, 1]",
    "batch_size": "INT >= 1",
    # [U-F4] audited against its one consumer, deployment_record/quantities/value_domain.py onchip_carry: a declared
    # on-chip carry buffer's capacity in bytes, compared with each wavefront group's live carry (above it refuses).
    "carry_buffer_bytes": "INT >= 1",
    "conversion_draws": "INT >= 1",
    "core_semantics": "ENUM spiking | mvm",
    "cores_per_tile": "INT >= 0",
    "degradation_tolerance": "FLOAT in [0, 1]",
    "elimination_constant_folding": "ENUM full | off",
    "elimination_propagation": "ENUM masked | closure | cascade",
    "encoding_layer_placement": "ENUM subsume | offload",
    "endpoint_floor_lr": "FLOAT >= 0",
    "endpoint_floor_min_cover_steps": "INT >= 0",
    "endpoint_floor_steps": "INT >= 0",
    "endpoint_target_floor": "FLOAT in [0, 1]",
    "endpoint_target_margin": "FLOAT in [0, 1]",
    "eval_max_samples": "INT >= 0",
    "eval_subsample_target": "INT >= 1",
    "event_census_samples": "INT >= 0",
    "fast_lr_scale": "FLOAT >= 0",
    "finetune_epochs": "INT >= 0",
    "finetune_lr": "FLOAT in [0, 1]",
    "firing_mode": "ENUM Default | Novena | TTFS",
    # [U-PS] the declared host-kernel evaluation: the consumer's vocabulary (mapping/support/host_kernel_evaluation).
    "host_kernel_evaluation": "ENUM batched | per_sample",
    "hw_config_mode": "ENUM fixed | search",
    "inference_batch_K": "INT >= 1",
    "inference_rate_hz": "FLOAT > 0",
    "input_data_scale": "FLOAT >= 0",
    "kd_ce_alpha": "FLOAT in [0, 1]",
    "kd_temperature": "FLOAT > 0",
    "lif_execution_discipline": "ENUM streaming | synchronized",
    "loihi_parity_sample_index": "INT >= 0",
    "lr": "FLOAT in [0, 1]",
    "lr_range_max": "FLOAT > 0",
    "lr_range_min": "FLOAT > 0",
    "max_schedule_passes": "INT >= 1",
    "max_simulation_samples": "INT >= 0",
    "min_lifetime_hours": "FLOAT > 0",
    "model_config_mode": "ENUM user | search",
    "nf_scm_parity_atol": "FLOAT >= 0",
    "nf_scm_parity_max_mismatch_fraction": "FLOAT in [0, 1]",
    "nf_scm_parity_min_agreement": "FLOAT in [0, 1]",
    "nf_scm_parity_samples": "INT >= 0",
    "num_workers": "INT >= 0",
    "onchip_majority_fraction": "FLOAT in [0, 1]",
    "onchip_min_fraction": "FLOAT in [0, 1]",
    "paired_confirm_batches": "INT >= 0",
    "pass_buffer_capacity_bytes": "INT >= 0",
    "pipeline_mode": "ENUM vanilla | phased",
    "prefix_stage_lr": "FLOAT >= 0",
    "pretrain_floor_chance_multiple": "FLOAT >= 0",
    "proven_recovery_depth": "INT >= 1",
    "prune_criterion": "ENUM row_col_l1 | activation | partial_column_group",
    "prune_group_size": "INT in [1, 4096]",
    "prune_sparsity": "FLOAT in [0, 1]",
    "pruning_fraction": "FLOAT in [0, 1]",
    # N2 (Wave 2) key amendment: readout_ranges, its consumer's legal set (design §2.9).
    "readout_ranges": "ENUM per_perceptron | per_consumer_group",
    "sanafe_arch_preset": "ENUM loihi | truenorth | custom",
    "sanafe_sample_count": "INT >= 1",
    "scale_migration_clip_ratio": "FLOAT >= 1",
    "scheduling_latency_weight": "FLOAT >= 0",
    "scm_degradation_tolerance": "FLOAT in [0, 1]",
    "scm_torch_sim_parity_min_agreement": "FLOAT in [0, 1]",
    "scm_torch_sim_parity_samples": "INT >= 0",
    "seed": "INT in [0, 4294967295]",
    "simulation_batch_count": "INT >= 1",
    "simulation_batch_size": "INT >= 1",
    "simulation_step_timeout_s": "FLOAT > 0",
    "simulation_steps": "INT >= 1",
    "spike_count_parity_samples": "INT >= 0",
    "spike_generation_mode": "ENUM Uniform | Deterministic | Stochastic | TTFS",
    "spiking_family": "ENUM lif | ttfs",
    "spiking_mode": "ENUM lif | ttfs | ttfs_quantized | ttfs_cycle_based",
    "spiking_variant": "ENUM streamed | synchronized | analytical | quantized | cascaded",
    "target_tq": "INT >= 1",
    "thresholding_mode": "ENUM < | <=",
    "tile_grid_cols": "INT >= 0",
    "tile_grid_rows": "INT >= 0",
    "training_epochs": "INT >= 0",
    "ttfs_cycle_schedule": "ENUM cascaded | synchronized",
    "ttfs_genuine_blend_ce_alpha": "FLOAT in [0, 1]",
    "tuner_target_floor_ratio": "FLOAT in [0, 1]",
    "tuning_batch_size": "INT >= 1",
    "tuning_budget_scale": "FLOAT >= 0",
    "tuning_step_cap_epochs": "FLOAT >= 0",
    "value_parity_samples": "INT >= 0",
    "weight_bits": "INT in [1, 32]",
    "wq_endpoint_recovery_steps": "INT >= 0",
    "write_ports": "INT >= 1",
    "weight_scale_rule": "ENUM integer | fractional",  # Wave 2 WI-5b
    # [U-H] the FPGA backend's keys, each the set its consumer accepts and runs: the soma point (SomaLaw.resolve; the
    # torch executors run every width and both rails, a generator refuses a width it cannot emit by name, which is
    # that consumer's own refusal), the bias-row split (weight_quantization_step; k=0 is "undeclared"), the ODIN
    # fabric sign claim (chip_selection), the device step's transport, card index and samples, the bundle's shipped
    # and certified samples (odin_hacc/artifact.campaign_indices refuses certified > shipped by name).
    "bias_row_splitting": "ENUM off | auto | fixed",
    "bias_rows_per_perceptron": "INT >= 0",
    "firing_granularity": "ENUM per_cycle | per_event",
    "membrane_arithmetic": "ENUM unbounded | saturating_unsigned | saturating_signed",
    "membrane_bits": "INT in [0, 64]",
    "odin_fpga_device_index": "INT >= 0",
    "odin_fpga_sample_count": "INT >= 1",
    "odin_fpga_transport": "ENUM rtl_cosim | xrt",
    "odin_hacc_bundle_samples": "INT >= 1",
    "odin_hacc_certification_samples": "INT >= 1",
    "weight_sign_granularity": "ENUM per_synapse | per_axon",
    # [U-HV] the ODIN steps' declared sample split: the reader's two splits (sample_loader.SAMPLE_SPLITS); any other
    # name is refused by name at the door and by the reader (require_sample_split).
    "odin_sample_split": "ENUM validation | test",
}


def _closed_or_bounded():
    out = {}
    for key, entry in REGISTRY.items():
        schema = schema_of(entry)
        if schema is None or entry.value_schema is not None:
            continue
        if schema.options is not None or schema.low is not None or schema.high is not None:
            out[key] = describe(schema)
    return out


def _admitted_at_both_doors(key, value):
    entry = REGISTRY[key]
    document = _placed(_base(entry), entry, value)
    flat = _flat(_base(entry))
    flat[key] = value
    return validate_deployment_config(document) == [] and _run(flat) == []


class TestAuditedLegalSets:
    """Owner decision 8's audit: each legal set is what its consumer accepts and runs. A value the consumer gives
    a meaning of its own is legal (the set widens); a value the consumer refuses or cannot compute is no option
    (the set narrows); a value the consumer reads only through a cast is judged as written (Strict, stated)."""

    def test_every_closed_or_bounded_key_is_the_audited_set(self):
        assert _closed_or_bounded() == AUDITED_LEGAL_SETS

    def test_an_uncapped_evaluation_is_legal(self):
        """eval_max_samples=0: both consumers read 0 as no cap, the full test set (``PipelineStep._capped_test``
        at ``if not plan.eval_max_samples``, ``simulation_factory`` at ``if cap else existing``)."""
        assert _admitted_at_both_doors("eval_max_samples", 0)
        assert not _admitted_at_both_doors("eval_max_samples", -1)

    def test_a_disarmed_spike_count_certificate_is_legal(self):
        """spike_count_parity_samples=0 disarms the gate (``certificate_gate_armed``: ``> 0``), as 0 disarms
        the other three sample-count gates."""
        assert _admitted_at_both_doors("spike_count_parity_samples", 0)
        for sibling in ("nf_scm_parity_samples", "scm_torch_sim_parity_samples", "value_parity_samples"):
            assert schema_of(REGISTRY[sibling]).low == 0

    @pytest.mark.parametrize("key", ["lr", "finetune_lr"])
    def test_a_learning_rate_above_one_is_refused_strict_stated(self, key):
        """Boundary finding (Strict, stated): the optimizer runs every learning rate >= 0 (torch refuses only
        lr < 0), but the registry range is also the search's option-axis range and the wizard's range (one
        declaration, three readers; ``test_a_search_axis_over_a_key_takes_its_registry_range``), so both doors
        refuse a rate above 1 by name, as the base document door did (the base run door ran it). Alternative
        recorded, not taken: widen the set and have the search accept a declared narrowing on a half-open range
        (``search/option_axes.py`` is read only in Wave 1)."""
        params = [torch.nn.Parameter(torch.zeros(1))]
        torch.optim.SGD(params, lr=2.0)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            torch.optim.SGD(params, lr=-1.0)
        assert not _admitted_at_both_doors(key, 2.0)
        assert not _admitted_at_both_doors(key, -1.0)
        assert _admitted_at_both_doors(key, 1.0)

    def test_a_sub_second_simulator_timeout_is_legal(self):
        """``resolve_simulation_step_timeout_s`` refuses a cap <= 0 and runs every positive one."""
        assert resolve_simulation_step_timeout_s(0.5) == 0.5
        with pytest.raises(ValueError, match="simulation_step_timeout_s"):
            resolve_simulation_step_timeout_s(0.0)
        assert _admitted_at_both_doors("simulation_step_timeout_s", 0.5)
        assert not _admitted_at_both_doors("simulation_step_timeout_s", 0.0)

    def test_a_pruning_group_above_4096_rows_is_refused_strict_stated(self):
        """Boundary finding (Strict, stated): ``partial_column_group_seed_masks`` refuses a group size below 1 and
        runs every other (a group larger than a layer's rows is one group per column), but the registry range is
        also the search's option-axis range, so both doors refuse a group above 4096 rows by name, as the base
        document door did (the base run door ran it)."""
        layer = SimpleNamespace(layer=torch.nn.Linear(3, 4))
        (seed,) = partial_column_group_seed_masks([layer], 0.5, SeedContext(group_size=8192))
        assert seed.element_pruned.shape == (4, 3)
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            partial_column_group_seed_masks([layer], 0.5, SeedContext(group_size=0))
        assert not _admitted_at_both_doors("prune_group_size", 8192)
        assert not _admitted_at_both_doors("prune_group_size", 0)
        assert _admitted_at_both_doors("prune_group_size", 4096)

    @pytest.mark.parametrize("key,whole,narrowing", [
        ("lr", (0.0, 1.0), (0.001, 0.1)),
        ("finetune_lr", (0.0, 1.0), (0.001, 0.1)),
        ("prune_group_size", (1.0, 4096.0), (8.0, 64.0)),
    ])
    def test_a_search_axis_over_a_key_takes_its_registry_range(self, key, whole, narrowing):
        """``build_option_axes`` reads the registry bounds as the axis range and refuses a half-open one, so an
        ``arch_search.option_axes`` over these keys, which the base ran, keeps its axis: the whole range for a
        listed key, a declared narrowing inside it."""
        (axis,) = build_option_axes([key])
        assert axis.bounds == whole
        (axis,) = build_option_axes({key: {"bounds": list(narrowing)}})
        assert axis.bounds == narrowing

    def test_a_negative_simulator_cap_is_refused_strict_stated(self):
        """Boundary finding (Strict, stated): with full evaluation off, the deployed-metric read takes min(cap,
        eval_max_samples), so a negative cap is no positive cap and reads the whole test set past
        eval_max_samples (``simulation_factory.run_trainer_metric``), which no legal value gives there (0 keeps
        the eval_max_samples cap); no branch names a negative (only 0 is special-cased), every negative reads
        alike, and the simulator runner reads it as no cap. Both doors refuse it by name, as the sentinel rule
        pins for the three sample-count sentinels (``test_field_validation.py::TestSentinelHonesty``: 0 legal,
        -1 refused). The base run door ran it: the repo's own ``tests/conftest.py::default_config`` declared -1,
        re-pinned to 0, which reads alike on every test set at or below eval_max_samples (every test set of the
        suite). Alternative recorded, not taken: INT >= -1 with -1 named "the full test set past
        eval_max_samples"."""
        class _Counting(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen, self.linear = 0, torch.nn.Linear(64, 4)

            def forward(self, x):
                self.seen += int(x.shape[0])
                return self.linear(x.flatten(1))

        def seen(max_simulation_samples):
            config = {**default_config(), "eval_max_samples": 100, "deployment_metric_full_eval": False,
                      "max_simulation_samples": max_simulation_samples}
            model = _Counting()
            simulation_factory.run_trainer_metric(
                MockPipeline(config=config, data_provider_factory=MockDataProviderFactory(size=300)), model,
                device="cpu")
            return model.seen
        assert (seen(-1), seen(-2), seen(0), seen(50)) == (300, 300, 100, 50)
        for value in (-1, -2):
            assert not _admitted_at_both_doors("max_simulation_samples", value)
        assert _admitted_at_both_doors("max_simulation_samples", 0)

    def test_a_zero_distillation_temperature_is_no_option(self):
        """``KDClassificationLoss`` divides the logits by the temperature: at 0 the loss is not a number."""
        model = torch.nn.Linear(2, 3)
        loss = KDClassificationLoss(torch.nn.Linear(2, 3), temperature=0.0)(
            model, torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
        assert math.isnan(float(loss))
        assert not _admitted_at_both_doors("kd_temperature", 0.0)
        assert _admitted_at_both_doors("kd_temperature", 0.5)

    def test_a_zero_learning_rate_range_end_is_no_option(self):
        """``LRRangeFinder`` probes lr_min * (lr_max / lr_min) ** t: lr_min = 0 divides by zero, and lr_max = 0
        probes a zero learning rate at every point past the first."""
        def finder(lr_min, lr_max):
            return LRRangeFinder(trainer=None, clone_state=None, restore_state=None, lr_min=lr_min,
                                 lr_max=lr_max, num_probes=4, steps_per_probe=1, validate_fn=None)
        with pytest.raises(ZeroDivisionError):
            finder(0.0, 0.1)._probe_lr(1)
        assert [finder(1e-5, 0.0)._probe_lr(i) for i in (1, 2, 3)] == [0.0, 0.0, 0.0]
        for key in ("lr_range_min", "lr_range_max"):
            assert not _admitted_at_both_doors(key, 0.0)
            assert _admitted_at_both_doors(key, 1e-3)

    def test_a_platform_of_no_cores_is_no_option(self):
        """The floorplan resolution refuses a platform of no cores (``resolve_floorplan``: ``declared_core_capacity
        must be >= 1``) and every core type counts at least one core (``count`` INT >= 1), so the legal core list
        is non-empty: both doors refuse ``cores: []`` by name, where the document door admitted it and the run door
        raised the floorplan's plain ValueError."""
        with pytest.raises(ValueError, match="declared_core_capacity must be >= 1"):
            resolve_floorplan(0, "loihi", 0, 0, 0)
        assert REGISTRY["cores"].value_schema.item.fields["count"].low == 1
        document = _load("sync_simplemlp.json")
        document["platform_constraints"]["cores"] = []
        message = "cores must be a list of at least 1 item(s)"
        assert validate_deployment_config(document) == [message]
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved(_flat(document))
        assert str(refusal.value) == message


class TestNestedFieldConsumersAgree:
    """``preprocessing.interpolation`` has two consumers that read a non-canonical spelling differently (the
    FFCV writer lowercases ``Bilinear`` to bilinear, the torchvision transform falls back to bicubic,
    ``data_handling/preprocessing.py``), so its legal set is the spellings both read alike, judged as written."""

    def test_the_interpolation_legal_set_is_the_spellings_both_consumers_read_alike(self):
        schema = REGISTRY["preprocessing"].value_schema
        assert schema is not None and not schema.closed
        assert schema.fields["interpolation"].options == interpolation_mode_names()

    @pytest.mark.parametrize("value", ["Bilinear", "BICUBIC", 3])
    def test_a_spelling_the_consumers_read_differently_refuses_by_name_at_both_doors(self, value):
        document = _load("sync_simplemlp.json")
        document["deployment_parameters"]["preprocessing"] = {"interpolation": value}
        stated = [m for m in validate_deployment_config(document) if m.startswith("preprocessing.interpolation")]
        assert len(stated) == 1
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved(_flat(document))
        assert str(refusal.value) == stated[0]

    def test_the_absent_interpolation_is_the_consumers_default(self):
        """The field's absent meaning is read from its consumers, never written by hand: the spec's default, the
        document reading's fallback and the FFCV writer's fallback agree."""
        absent = REGISTRY["preprocessing"].value_schema.fields["interpolation"].absent_means
        assert absent == PreprocessingSpec().interpolation
        assert resolve_preprocessing({"resize_to": 32}).interpolation == absent
        assert _PILResize(None, 32, None)._interp == _PIL_INTERP[absent]

    @pytest.mark.parametrize("value", interpolation_mode_names())
    def test_a_canonical_spelling_passes_both_doors(self, value):
        document = _load("sync_simplemlp.json")
        document["deployment_parameters"]["preprocessing"] = {"resize_to": 32, "interpolation": value}
        assert validate_deployment_config(document) == []
        build_platform_constraints_resolved(_flat(document))


def _raw_entries():
    return {entry.flat_key: entry for entry in (
        build_module._RUN + build_module._MODEL + build_module._SEMANTICS + build_module._CONVERSION
        + build_module._PRUNING + build_module._TUNING + build_module._ENDPOINT + build_module._EXECUTION
        + build_module._PLATFORM + build_module._PHYSICS + build_module._LOWERING + build_module._TARGET
        + build_module._WRITE + build_module._READOUT + build_module._RESIDENCY)}


class TestNoDeadDeclaration:
    """An attribute the builder injects never replaces a different value its entry declares: such a value is read
    by nothing (the base's ``nevresim_connectivity_mode`` options ``runtime, codegen`` and
    ``simulation_batch_size``'s absent meaning ``8`` were)."""

    def test_an_injected_attribute_never_replaces_a_declared_one(self):
        defaults = {f.name: f.default for f in ConfigKeySchema.__dataclass_fields__.values()}
        raw = _raw_entries()
        for key, attributes in INJECTED_DECLARATIONS.items():
            for name, value in attributes.items():
                declared = getattr(raw[key], name)
                assert declared == defaults[name] or declared == value, (key, name, declared)

    def test_the_connectivity_entry_declares_its_consumer_s_set_and_reading(self):
        (entry,) = [e for e in entries_platform.ENTRIES if e.flat_key == "nevresim_connectivity_mode"]
        assert entry.resolved_options() == get_args(connectivity.ConnectivityMode)
        assert entry.consumer_reading is connectivity.resolve_nevresim_connectivity_mode
        assert REGISTRY["nevresim_connectivity_mode"].value_schema == entry.value_schema


def _cli_refusal(document, workdir):
    """The CLI path's refusal of a document as ``_through_the_run`` takes it (``PipelineSession.from_config``, then
    the run door), the exception itself (any ``ToolchainRefusal``), or None when it runs."""
    document = {**copy.deepcopy(document), "_working_directory": str(workdir)}
    flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        with isolated_rng_stream():
            session = PipelineSession.from_config(document, data_provider_factory=_StubProviderFactory(document))
            build_platform_constraints_resolved(session.pipeline.config)
    except ToolchainRefusal as refusal:
        return refusal
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.benchmark = flags[2]
        torch.backends.cuda.matmul.allow_tf32 = flags[3]
        torch.backends.cudnn.allow_tf32 = flags[4]
        torch.set_float32_matmul_precision(flags[5])
    return None


class TestTheClosedRunPath:
    """The joint closure of S0-B and G4: every entry point judges a document's declarations as written before
    anything reads one (G4 interface notes 1 to 3), by the one judge, and the document door states the run's own
    refusal where the judge admits, so the two doors and the CLI path give one verdict, each violation stated
    once."""

    def test_a_legality_bearing_value_outside_its_schema_is_named_once(self, tmp_path):
        """S0-B final review (note 3): ``legality_errors`` and the judge named one violation twice; the legal-value
        set now judges only a value its schema admits, so both doors and the wizard state the schema's message."""
        document = _declared("deployment_parameters", "firing_mode", 7)
        message = _message("firing_mode", 7)
        assert validate_deployment_config(document) == document_declaration_errors(document) == [message]
        rows = [row for row in resolve_draft(document).errors if "firing_mode" in row["message"]]
        assert [(row["key"], row["rule_id"], row["message"]) for row in rows] == [("firing_mode", "field_domain", message)]
        assert _parts(_through_the_run(document, tmp_path)) == [message]

    def test_a_section_the_judge_cannot_read_is_no_declaration(self):
        """Mutant R1 (S0-B final review): the judge reads only object sections; a document without them, or with a
        non-object section, declares nothing to it, and the document door states its shape refusals."""
        for document in ({}, {"deployment_parameters": "x", "platform_constraints": [1]}):
            assert document_declaration_errors(document) == []
            assert "deployment_parameters must be a dict" in validate_deployment_config(document) or document == {}
        assert "Missing top-level key: deployment_parameters" in validate_deployment_config({})

    def test_a_value_domain_key_is_judged_by_the_document_s_domain(self, tmp_path):
        """Mutant R5 (S0-B final review): the domain is the document's (its core semantics); a legal readout law
        passes a value-domain document at both doors, and a spiking one refuses it by one domain message at both
        doors and on the CLI path."""
        legal = _declared("platform_constraints", "readout_law", _LEGAL_READOUT_LAW, name="mvm_lenet5.json")
        assert document_declaration_errors(legal) == validate_deployment_config(legal) == []
        spiking = _declared("platform_constraints", "readout_law", _LEGAL_READOUT_LAW, name="sync_simplemlp.json")
        (message,) = document_declaration_errors(spiking)
        assert message.startswith("readout_law declares a value-domain boundary grid")
        assert validate_deployment_config(spiking) == [message]
        assert str(_cli_refusal(spiking, tmp_path)) == message

    def test_a_key_of_the_other_domain_refuses_by_name_on_the_cli_path(self, tmp_path):
        """The document domain rule is the judge's, so a run refuses an other-domain declaration with the document
        door's message where it ran it (``activation_bits`` on a spiking document); a null one is unset."""
        document = _declared("platform_constraints", "activation_bits", 8, name="sync_simplemlp.json")
        (message,) = validate_deployment_config(document)
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DeclarationValueError) and str(refusal) == message
        document["platform_constraints"]["activation_bits"] = None
        assert validate_deployment_config(document) == [] and _through_the_run(document, tmp_path / "null") == []

    @pytest.mark.parametrize("placement", ["other_section", "shadowed_own"])
    def test_the_legal_value_set_law_judges_the_run_s_layout(self, placement, tmp_path):
        """The legal-value-set law is the judge's over the run's layout and each section's own declarations, under
        the run's state: a legality-bearing key declared in the other section, or one a later copy shadows, refuses
        by name at both doors and on the CLI path (the document door admitted the first, the run the second)."""
        document = _load("sync_simplemlp.json")
        if placement == "other_section":
            document["deployment_parameters"].pop("firing_mode", None)
            document["platform_constraints"]["firing_mode"] = "Default"
        else:
            document["deployment_parameters"]["firing_mode"] = "Default"
            document["platform_constraints"]["firing_mode"] = "TTFS"
        message = "firing_mode='Default' is not legal here: the current config admits {'TTFS'}. Remove firing_mode to accept the derived value."
        # A shadowed copy is a second reading too (review round 1), stated beside its legality.
        twice = ("firing_mode is declared in deployment_parameters as 'Default' and in platform_constraints as "
                 "'TTFS', and a run reads one firing_mode: declare it once")
        stated = [message] if placement == "other_section" else [twice, message]
        assert validate_deployment_config(document) == stated
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DeclarationValueError) and str(refusal) == "; ".join(stated)
        rows = [row for row in resolve_draft(document).errors if row["message"] == message]
        assert [(row["key"], row["rule_id"]) for row in rows] == [("firing_mode", "legal_value_set")]
        assert rows[0]["remedies"] and resolve_draft(document).resolved == {}

    @pytest.mark.parametrize("section", ["deployment_parameters", "platform_constraints"])
    def test_the_document_door_states_the_run_s_derivation_refusal(self, section, tmp_path):
        """A pairing the derivation refuses (a simulator the mode cannot run) is stated by the document door, which
        admitted it, and refused on the CLI path as a ``DerivationRefusal`` (a ``ToolchainRefusal``), one message."""
        document = _load("sync_simplemlp.json")
        document["deployment_parameters"].pop("enable_loihi_simulation", None)
        document[section]["enable_loihi_simulation"] = True
        (message,) = validate_deployment_config(document)
        assert message.startswith("enable_loihi_simulation=true contradicts the mode capability")
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DerivationRefusal) and str(refusal) == message

    def test_the_document_door_states_the_quantization_assembly_contract(self, tmp_path):
        """The contract reads the pipeline mode and the weight-quantization declaration the run reads; the document
        door states it as the CLI path refuses it (a deployment-parameters float declaration under bits)."""
        document = _declared("deployment_parameters", "weight_quantization", False, name="sync_simplemlp.json")
        (message,) = validate_deployment_config(document)
        assert message.startswith("quantization contract violation: weight quantization is bits-driven")
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DerivationRefusal) and str(refusal) == message

    def test_the_top_level_pipeline_mode_reaches_the_derivation(self, tmp_path):
        """A top-level ``pipeline_mode: vanilla`` without a weight-quantization declaration ran quantized on the
        CLI path (the session read the mode for its working directory only), while the wizard ran it float; the
        session lays it beneath the deployment parameters, as the wizard's builder does."""
        document = _load("sync_simplemlp.json")
        document["pipeline_mode"] = "vanilla"
        document["deployment_parameters"].pop("weight_quantization", None)
        document["platform_constraints"].pop("weight_bits", None)
        assert validate_deployment_config(document) == []
        session = PipelineSession.from_config({**copy.deepcopy(document), "_working_directory": str(tmp_path)},
                                              data_provider_factory=_StubProviderFactory(document))
        assert session.pipeline.plan.weight_quantization is False
        assert resolve_draft(document).resolved["weight_quantization"] is False

    def test_a_null_top_level_declaration_is_unset_on_the_cli_path(self, tmp_path):
        """``seed: null`` raised a TypeError at the session's seed cast; it reads as the absent seed, as both doors
        read a null."""
        document = _load("sync_simplemlp.json")
        document["seed"] = None
        assert validate_deployment_config(document) == [] and _through_the_run(document, tmp_path) == []

    def test_the_floorplan_pairing_refuses_by_name_at_the_run_door(self):
        """A half-declared tile grid: the run door's resolver raised ``resolve_floorplan``'s plain ValueError; the
        pairing is the judge's (over the declarations their schemas admit), so the run refuses by name, with the
        document door's message, once."""
        document = _declared("platform_constraints", "tile_grid_rows", 4, name="sync_simplemlp.json")
        (message,) = validate_deployment_config(document)
        assert message.startswith("tile_grid_rows and tile_grid_cols must be declared together")
        with pytest.raises(DeclarationValueError) as refusal:
            build_platform_constraints_resolved({**_flat(_load("sync_simplemlp.json")), "tile_grid_rows": 4})
        assert str(refusal.value) == message
        with pytest.raises(DeclarationValueError, match="must be declared together"):
            _flat(document)

    def test_the_seed_s_legal_set_is_its_consumer_s(self, tmp_path):
        """numpy's seed (``apply_determinism``) refuses a seed of 2^32 or more, which the CLI path raised as a
        plain ValueError while both doors admitted it; the legal set is the consumer's (plan section 20 item 3)."""
        top = 2 ** 32 - 1
        assert describe(schema_of(REGISTRY["seed"])) == f"INT in [0, {top}]"
        refused = _declared("top", "seed", top + 1, name="sync_simplemlp.json")
        assert validate_deployment_config(refused) == [f"seed={top + 1} is above the maximum {top}"]
        assert _parts(_through_the_run(refused, tmp_path)) == [f"seed={top + 1} is above the maximum {top}"]
        assert _through_the_run(_declared("top", "seed", top, name="sync_simplemlp.json"), tmp_path / "top") == []

    def test_a_declared_label_is_judged_and_never_overwritten(self, tmp_path):
        """The generic rule for a derived key (item 3 of the closure): the document door refuses its declaration
        (it is not declarable); the run judges it by its schema first (``target_claim_class: 7``, the both-doors
        test), then its owner refuses a value it does not derive by name, and keeps one equal to what it derives,
        so a declared label is never silently overwritten (as the quantization contract does for
        ``activation_quantization``). Boundary, stated: an equal label under its target is refused by the
        document door only, as an equal ``activation_quantization`` is."""
        typed = _declared("platform_constraints", "target_claim_class", 7, name="sync_simplemlp.json")
        assert _merged_or_refusal(typed) == f"DeclarationValueError: {_message('target_claim_class', 7)}"
        refused = _declared("platform_constraints", "target_claim_class", "headline-eligible", name="sync_simplemlp.json")
        assert any("not declarable" in m for m in validate_deployment_config(refused))
        assert "derived by the target expansion" in str(_cli_refusal(refused, tmp_path))
        bundled = _load("sync_simplemlp.json")
        owned = set(target_bundles()["odin"].values) | {"max_axons", "max_neurons", "has_bias"}
        bundled["platform_constraints"] = {k: v for k, v in bundled["platform_constraints"].items() if k not in owned}
        bundled["platform_constraints"].update(target="odin", target_claim_class="quarantined")
        assert any("not declarable" in m for m in validate_deployment_config(bundled))
        assert _merged_or_refusal(bundled)["target_claim_class"] == "quarantined"
        bundled["platform_constraints"]["target_claim_class"] = "headline-eligible"
        assert "differs from target_claim_class='quarantined'" in _merged_or_refusal(bundled)

    def test_the_applied_schema_is_judged_where_its_relevance_holds(self):
        """Key amendment K1 through the registry: ``weight_bits``' applied schema (INT in [2, 32]) is judged where
        weight quantization is on, and its reason is stated; under float weights the value is not applied."""
        quantized = _declared("platform_constraints", "weight_bits", 1, name="sync_simplemlp.json")
        (message,) = document_declaration_errors(quantized)
        assert message.startswith("weight_bits=1 is below the minimum 2: the signed weight grid")
        assert validate_deployment_config(quantized) == [message]
        float_weights = copy.deepcopy(quantized)
        float_weights["pipeline_mode"] = "vanilla"
        float_weights["deployment_parameters"]["weight_quantization"] = False
        assert document_declaration_errors(float_weights) == []
        assert declaration_judge.applied_errors({"weight_quantization": False}, {"weight_bits": 1}) == []

    def test_an_applied_schema_is_conditional_typed_and_reasoned(self):
        """The registry refuses an applied schema of another type, one judged under an unconditional condition (it
        would be the key's own schema), one without the consumer it names, and one that names no reason (review
        round 1: the condition is the consumer's, ``applied_when``, never the key's relevance)."""
        from_registry = REGISTRY["weight_bits"]
        when, by = from_registry.applied_when, from_registry.applied_by
        assert from_registry.applied_schema.low == 2 and when.op != "always" and from_registry.relevant.op == "always"
        base = dict(flat_key="k", group="hardware", owner="o", type=from_registry.type, category=Category.BASIC,
                    label="K", doc="d", section="platform_constraints")
        good = from_registry.applied_schema
        for broken in (dict(applied_schema=good, applied_by=by),
                       dict(applied_schema=good, applied_when=Relevance.always(), applied_by=by),
                       dict(applied_schema=good, applied_when=when),
                       dict(applied_schema=ValueSchema(kind="float", low=2.0, doc="why"), applied_when=when,
                            applied_by=by),
                       dict(applied_schema=ValueSchema(kind="int", low=2), applied_when=when, applied_by=by),
                       dict(applied_when=when, applied_by=by)):
            with pytest.raises(ValueError, match="applied_schema"):
                ConfigKeySchema(**base, **broken)
        ConfigKeySchema(**base, applied_schema=good, applied_when=when, applied_by=by)

    def test_the_wizard_keys_a_declaration_in_the_other_section(self):
        """G4 interface note 3: the wizard gets keyed, remediable rows for the judge's refusals, a key declared in
        the other section included (``platform_constraints.lr``)."""
        document = _declared("platform_constraints", "lr", -1.0)
        rows = [row for row in resolve_draft(document).errors if row["message"] == _message("lr", -1.0)]
        assert [(row["key"], row["rule_id"]) for row in rows] == [("lr", "field_domain")]

    def test_the_contract_reads_the_pipeline_mode_the_derivation_reads(self, tmp_path):
        """The session's quantization assembly contract read the top-level mode, else the deployment parameters'
        own, while the derivation reads the run's layout; a platform-constraints mode shadowing a misspelt
        deployment-parameters one ran the derivation's vanilla over an explicit weight quantization unrefused. The
        contract reads the layout's mode, so the run refuses the contradiction by name, as the document door does.
        Review round 3 re-pin: the misspelt shadowed copy is itself refused by its schema with its place at both
        doors (``declaration_places.reading_errors``), so the contract is pinned on the platform mode alone."""
        document = _load("sync_simplemlp.json")
        document.pop("pipeline_mode", None)
        document["deployment_parameters"].update(pipeline_mode="Vanilla", weight_quantization=True)
        document["platform_constraints"]["pipeline_mode"] = "vanilla"
        shadowed = ("deployment_parameters.pipeline_mode: pipeline_mode='Vanilla' is not a valid option; choose one "
                    "of ['vanilla', 'phased']")
        assert validate_deployment_config(copy.deepcopy(document)) == [shadowed]
        refusal = _cli_refusal(copy.deepcopy(document), tmp_path)
        assert isinstance(refusal, DeclarationValueError) and str(refusal) == shadowed
        del document["deployment_parameters"]["pipeline_mode"]
        (message,) = validate_deployment_config(document)
        assert message.startswith("quantization contract violation: explicit weight_quantization=true contradicts")
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DerivationRefusal) and str(refusal) == message

    @pytest.mark.parametrize("section", ["platform_constraints", "deployment_parameters"])
    def test_a_section_that_is_no_object_is_refused_by_name_and_never_raises(self, section, tmp_path):
        """A declared section that is no object: the document door and the CLI path refuse it by name with one
        message (the CLI path cast or iterated it), the wizard states it without raising and derives no preview,
        and emit writes it as declared (G4 final review, the emit regression on a malformed draft)."""
        document = _load("sync_simplemlp.json")
        document[section] = "x"
        message = f"{section} must be a dict"
        assert message in validate_deployment_config(document)
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DeclarationValueError) and str(refusal) == message
        resolution = resolve_draft(document)
        assert message in [row["message"] for row in resolution.errors] and resolution.resolved == {}
        assert emit_deployment_config(document)[section] == "x"


    def test_the_domain_is_the_core_semantics_the_run_lays(self, tmp_path):
        """A ``core_semantics`` declared in the platform constraints is the run's (the merge lays both sections
        flat), so the domain rule reads the run's layout: a spiking fixture moved to ``mvm`` that way refuses its
        event-domain declarations by name at both doors and on the CLI path, where both doors admitted it and the
        run raised at the model check."""
        document = _load("sync_simplemlp.json")
        document["deployment_parameters"].pop("core_semantics", None)
        document["platform_constraints"]["core_semantics"] = "mvm"
        stated = validate_deployment_config(document)
        assert stated and all("value-domain (core_semantics='mvm') deployment" in m for m in stated)
        assert document_declaration_errors(document) == stated
        refusal = _cli_refusal(document, tmp_path)
        assert isinstance(refusal, DeclarationValueError) and _parts([str(refusal)]) == _parts(stated)

    @pytest.mark.parametrize("key,value", [("core_semantics", 7), ("core_semantics", "bogus"),
                                           ("tile_grid_rows", "x"), ("tile_grid_cols", [1])])
    def test_the_document_door_states_what_its_rules_cannot_read(self, key, value):
        """S0-B interface note (carried since amendment B): the document door raised ValueError where a document
        rule could not read the value (an unknown core_semantics in the domain rules, a tile grid the floorplan
        pairing cannot cast); it states the value schema's refusal instead, once, as the run does."""
        document = _declared(REGISTRY[key].section, key, value, name="sync_simplemlp.json")
        message = _message(key, value)
        assert validate_deployment_config(document) == document_declaration_errors(document) == [message]

    def test_the_run_s_state_reads_a_null_as_unset(self):
        """The state the legal-value set and the applied schemas are judged under is the run's: the schema defaults
        beneath the top-level declarations and the layout, a null declaration unset (``run_layers`` lays nothing
        for it), so a null reads as the default, never as None."""
        from_judge = declaration_judge._sections_view({"spiking_family": None, "lr": None}, {"simulation_steps": 8},
                                                      {"pipeline_mode": None})
        layout, own, state, refused = from_judge
        assert layout == {"simulation_steps": 8} and refused == frozenset()
        assert own == {"spiking_family": None, "lr": None, "simulation_steps": 8}
        defaults = {key: entry.default for key, entry in REGISTRY.items() if entry.has_default()}
        assert state["spiking_family"] == defaults["spiking_family"] and state["lr"] == defaults["lr"]
        assert state["simulation_steps"] == 8 and "pipeline_mode" not in state

    def test_the_applied_schema_reads_the_top_level_pipeline_mode(self, tmp_path):
        """The run's state holds the top-level declarations: under a top-level ``pipeline_mode: vanilla`` the weight
        grid is not applied, so ``weight_bits: 1`` passes the document door and the CLI path (the session lays the
        mode beneath the deployment parameters), as the run computes it float."""
        document = _load("sync_simplemlp.json")
        document["pipeline_mode"] = "vanilla"
        document["deployment_parameters"].pop("weight_quantization", None)
        document["platform_constraints"]["weight_bits"] = 1
        assert document_declaration_errors(document) == validate_deployment_config(document) == []
        assert _through_the_run(document, tmp_path) == []

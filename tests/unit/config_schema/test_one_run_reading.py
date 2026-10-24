"""Every reader of a run reads one layout of a document (the S0-B and G4 joint closure, review round 1).

A run lays a document once (``declaration_places.run_layers``): the top-level seed and pipeline mode beneath the
deployment parameters, the deployment parameters, the platform default layer, the platform constraints, a null
declaration unset. The merge, the derivation, the session's data provider (seed, preprocessing, batch size), its
hardware-search branch and pipeline mode, and the quantization assembly contract all read that layout. So a
declaration has one reading, or the one judge refuses it by name at every door: one no reader reads
(``placement_errors``), and a key declared in two places the run reads whose readings differ
(``reading_errors``). The judge's state holds only the declarations their schemas admit, a legal set that reads a
refused one is undetermined, the document door states the run's own refusals (the builder registration, the
workload reading, the run door's consumer readings), and a top-level key the session reads with no value of its own
is refused absent or null at the document door and on the CLI path alike.
"""

import copy
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
import torch

from mimarsinan.common.pretrained.weight_sets import PretrainedWeightSet
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.workload_profile import ModelWorkloadProfile
from mimarsinan.config_schema.defaults import get_default_platform_constraints
from mimarsinan.config_schema.registry import REGISTRY, Relevance, consumer_reading, declaration_judge
from mimarsinan.config_schema.registry.declaration_judge import (
    applied_errors, declared_sections_errors, document_declaration_errors, legality_rows, sections_run_layout,
)
from mimarsinan.config_schema.registry.types import Category, ConfigKeySchema, FieldType
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError, entry_value_errors
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.runtime import DerivationRefusal
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.mapping.platform.target_bundles import Faithfulness, TargetBundle, target_bundles
from mimarsinan.mapping.platform.target_expansion import target_assignments
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.pipelines.deployment_specs import get_pipeline_step_specs
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.pipelining.determinism import isolated_rng_stream
from mimarsinan.pipelining.session import PipelineSession, parse_deployment_config

DP, PC = "deployment_parameters", "platform_constraints"
_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _ROOT / "tests" / "fixtures" / "deployment_configs"
SPIKING, LIF = "sync_simplemlp.json", "lif_mmixcore.json"
SEARCH_SPACE = {"num_core_types": 2, "core_type_counts": [200, 200], "core_axons_bounds": [64, 1024],
                "core_neurons_bounds": [64, 1024]}


def _load(name):
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _golden():
    spec = importlib.util.spec_from_file_location(
        "_golden_for_one_reading", _ROOT / "tests" / "unit" / "config_schema" / "test_golden_resolution_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GOLDEN = _golden()


class _Provider:
    """The golden snapshot's provider stub over the run's layout (dataset I/O replaced by facts)."""

    def __init__(self, document):
        dp = document.get(DP) if isinstance(document.get(DP), dict) else {}
        pc = document.get(PC) if isinstance(document.get(PC), dict) else {}
        self._dp = sections_run_layout(dp, pc)

    def create(self):
        return _GOLDEN.stub_provider("MNIST_DataProvider", self._dp)


def _parts(messages):
    return sorted(part for message in messages for part in message.split("; "))


def _session(document, tmp_path):
    """The CLI path (``PipelineSession.from_config``) with the session's global determinism put back."""
    document = {**copy.deepcopy(document), "_working_directory": str(tmp_path)}
    flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
    try:
        with isolated_rng_stream():
            return PipelineSession.from_config(document, data_provider_factory=_Provider(document))
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.benchmark = flags[2]
        torch.set_float32_matmul_precision(flags[3])


def _cli(document, tmp_path):
    """The CLI path's refusal (each ``; `` part), [] when the session and the run door admit; a refusal must be a
    ``ToolchainRefusal``, and any other exception propagates."""
    try:
        session = _session(document, tmp_path)
        build_platform_constraints_resolved(session.pipeline.config)
    except ToolchainRefusal as refusal:
        return str(refusal).split("; ")
    return []


def _merge(document):
    """The run door of the two sections (``merge_pipeline_config``, then the resolver): its refusal's parts."""
    try:
        build_platform_constraints_resolved(merge_pipeline_config(copy.deepcopy(document[DP]),
                                                                  copy.deepcopy(document[PC])))
    except ToolchainRefusal as refusal:
        return str(refusal).split("; ")
    return []


def _wizard(document):
    return [row["message"] for row in resolve_draft(copy.deepcopy(document)).errors]


def _refused_everywhere(document, message, tmp_path, *, merge=True):
    """``message`` is stated by the document door, the wizard, the CLI path and (``merge``) the run door."""
    assert message in validate_deployment_config(copy.deepcopy(document))
    assert message in _wizard(document)
    assert message in _parts([*_cli(document, tmp_path)])
    if merge:
        assert message in _parts(_merge(document))


def _admitted_everywhere(document, tmp_path):
    assert validate_deployment_config(copy.deepcopy(document)) == []
    assert _wizard(document) == []
    assert _merge(document) == []
    assert _cli(document, tmp_path) == []


def _hardware(dp_mode, pc_mode):
    document = _load(LIF)
    document[DP]["arch_search"] = {"pop_size": 8, "generations": 2}
    document[DP].pop("hw_config_mode", None)
    for section, mode in ((DP, dp_mode), (PC, pc_mode)):
        if mode is not None:
            document[section]["hw_config_mode"] = mode
    document[PC]["search_space"] = dict(SEARCH_SPACE)
    return document


def _twice(key, first, second, places=(DP, PC)):
    """The judge's message for a key declared in two places the run reads, with differing readings."""
    where = [f"at the top level as {v!r}" if p == "top" else f"in {p} as {v!r}"
             for p, v in zip(places, (first, second))]
    return f"{key} is declared {' and '.join(where)}, and a run reads one {key}: declare it once"


class TestEveryReaderReadsTheLayout:
    """The blocker of review round 1: the session read some declarations from the section itself or the top
    level, while the derivation and the flat config read the run's layout."""

    def test_a_platform_hardware_mode_runs_the_hardware_search_with_its_bounds(self, tmp_path):
        """``platform_constraints.hw_config_mode: search`` is the run's mode, so the session merges the search
        space into ``arch_search``: the hardware search runs with its declared bounds."""
        document = _hardware(None, "search")
        _admitted_everywhere(document, tmp_path)
        session = _session(document, tmp_path / "s")
        assert derive_search_mode(session.pipeline.config) == "hardware"
        assert set(SEARCH_SPACE) <= set(session.parsed.deployment_parameters["arch_search"])
        assert "search_space" not in session.parsed.platform_constraints

    @pytest.mark.parametrize("holder", ["search_space", "arch_search"])
    def test_the_search_merge_reads_the_section_the_layout_reads(self, holder, tmp_path):
        """A search space or an ``arch_search`` declared in the other section is the one the layout reads: the
        hardware search merges the one into the other wherever they are declared."""
        document = _hardware("search", None)
        if holder == "search_space":
            document[DP]["search_space"] = document[PC].pop("search_space")
        else:
            document[PC]["arch_search"] = document[DP].pop("arch_search")
        session = _session(document, tmp_path)
        assert set(SEARCH_SPACE) <= set(session.pipeline.config["arch_search"])
        assert {"pop_size", "generations"} <= set(session.pipeline.config["arch_search"])

    @pytest.mark.parametrize("dp_mode,pc_mode", [("fixed", "search"), ("search", "fixed")])
    def test_a_hardware_mode_under_another_copy_refuses_at_every_door(self, dp_mode, pc_mode, tmp_path):
        _refused_everywhere(_hardware(dp_mode, pc_mode), _twice("hw_config_mode", dp_mode, pc_mode), tmp_path)

    @pytest.mark.parametrize("key,first,second", [
        ("batch_size", 32, 64), ("batch_size", 64, 32),
        ("preprocessing", {"resize_to": 32, "interpolation": "bilinear"}, {"resize_to": 28}),
    ], ids=["batch_32_64", "batch_64_32", "preprocessing"])
    def test_a_provider_declaration_and_another_copy_refuse_at_every_door(self, key, first, second, tmp_path):
        document = _load(SPIKING)
        document[DP][key], document[PC][key] = first, second
        _refused_everywhere(document, _twice(key, first, second), tmp_path)

    @pytest.mark.parametrize("key,value", [("batch_size", 64), ("preprocessing", {"resize_to": 28})])
    def test_a_platform_copy_alone_reaches_the_data_provider(self, key, value, tmp_path):
        document = _load(SPIKING)
        document[DP].pop(key, None)
        document[PC][key] = value
        parsed = parse_deployment_config({**document, "_working_directory": str(tmp_path)})
        assert getattr(parsed.data_provider_factory, f"_{key}") == value
        assert merge_pipeline_config(copy.deepcopy(document[DP]), copy.deepcopy(document[PC]))[key] == value

    def test_a_deployment_parameters_seed_alone_seeds_the_provider_and_the_plan(self, tmp_path):
        document = _load(SPIKING)
        document.pop("seed")
        document[DP]["seed"] = 9
        parsed = parse_deployment_config({**document, "_working_directory": str(tmp_path)})
        assert parsed.data_provider_factory._seed == 9
        assert _session(document, tmp_path / "s").pipeline.plan.seed == 9

    def test_a_platform_copy_of_a_laid_top_level_key_alone_is_the_run_s(self, tmp_path):
        """The session lays the top-level seed beneath the deployment parameters only where neither section
        declares one, so a platform-constraints seed alone seeds the provider and the plan at every door."""
        document = _load(SPIKING)
        document.pop("seed")
        document[PC]["seed"] = 9
        _admitted_everywhere(document, tmp_path)
        parsed = parse_deployment_config({**document, "_working_directory": str(tmp_path)})
        assert parsed.data_provider_factory._seed == 9
        assert _session(document, tmp_path / "s").pipeline.plan.seed == 9

    def test_the_presence_rules_read_the_layout(self, tmp_path):
        """The model and search keys a mode needs are present where the run reads them: a platform-constraints
        ``model_type`` builds the model the run reads, so no door states it missing."""
        document = _load(SPIKING)
        document[PC]["model_type"] = document[DP].pop("model_type")
        _admitted_everywhere(document, tmp_path)

    @pytest.mark.parametrize("key,top,section", [("seed", 3, 9), ("pipeline_mode", "vanilla", "phased")])
    def test_a_top_level_key_and_a_section_copy_refuse_at_every_door(self, key, top, section, tmp_path):
        """The merge-only run door holds no top level: its sections alone are one reading."""
        document = _load(SPIKING)
        document[key], document[DP][key] = top, section
        if key == "pipeline_mode":
            document[DP].pop("weight_quantization")
        _refused_everywhere(document, _twice(key, top, section, ("top", DP)), tmp_path, merge=False)

    def test_equal_copies_are_one_reading(self, tmp_path):
        document = _load(SPIKING)
        document[PC]["batch_size"] = document[DP]["batch_size"]
        document[DP]["seed"] = document["seed"]
        _admitted_everywhere(document, tmp_path)

    def test_copies_its_consumer_reads_alike_are_one_reading(self, tmp_path):
        """Two copies are compared as their consumer reads them, and a refusal names each reading: a spelling the
        consumer normalizes declares the canonical value (``nevresim_connectivity_mode``: surrounding whitespace)."""
        document = _load(SPIKING)
        document[DP]["nevresim_connectivity_mode"] = " compile_time"
        document[PC]["nevresim_connectivity_mode"] = "compile_time"
        _admitted_everywhere(document, tmp_path)
        document[PC]["nevresim_connectivity_mode"] = "runtime"
        _refused_everywhere(document, _twice("nevresim_connectivity_mode", "compile_time", "runtime"), tmp_path / "r")

    def test_a_null_lays_nothing(self, tmp_path):
        """A null declaration is unset, so it never replaces an earlier section's declaration in the layout."""
        document = _load(SPIKING)
        document[DP]["batch_size"], document[PC]["batch_size"] = 32, None
        _admitted_everywhere(document, tmp_path)
        parsed = parse_deployment_config({**document, "_working_directory": str(tmp_path)})
        assert parsed.data_provider_factory._batch_size == 32
        assert merge_pipeline_config(copy.deepcopy(document[DP]), copy.deepcopy(document[PC]))["batch_size"] == 32
        assert declaration_judge.run_layers({"a": 1}, {"a": None}, defaults=False) == {"a": 1}

    def test_the_contract_reads_the_layout(self, tmp_path):
        """Review round 1, minor: the quantization assembly contract reads each key where the run lays it."""
        misplaced = _load(SPIKING)
        misplaced[DP].pop("weight_quantization")
        misplaced[PC].update(weight_quantization=False, weight_bits=1)
        (message,) = validate_deployment_config(copy.deepcopy(misplaced))
        assert message.startswith("quantization contract violation: weight quantization is bits-driven")
        with pytest.raises(DerivationRefusal, match="bits-driven"):
            _session(misplaced, tmp_path)
        disagreeing = _load(SPIKING)
        disagreeing[PC]["weight_quantization"] = False
        _refused_everywhere(disagreeing, _twice("weight_quantization", True, False), tmp_path)
        # The wizard applies the contract to what the judge admits, as the document door states it.
        assert not any(m.startswith("quantization contract") for m in _wizard(disagreeing))

    def test_the_wizard_reads_the_core_semantics_the_layout_reads(self, tmp_path):
        """The wizard's dormant-domain split reads the run's layout: a value-domain document whose core semantics
        the platform constraints declare keeps its value-domain keys, as both doors and the CLI path read it."""
        document = _load("mvm_aq_lenet5.json")
        document[PC]["core_semantics"] = document[DP].pop("core_semantics")
        _admitted_everywhere(document, tmp_path)
        assert resolve_draft(copy.deepcopy(document)).dormant == []
        assert "activation_bits" in resolve_draft(copy.deepcopy(document)).resolved

    def test_a_null_top_level_pipeline_mode_runs_phased(self, tmp_path, monkeypatch):
        """Review round 1, minor (mutant R40): a null top-level mode is the session's default, never ``"None"``."""
        document = _load(SPIKING)
        document["pipeline_mode"] = None
        session = _session(document, tmp_path)
        assert session.parsed.pipeline_mode == "phased" and session.pipeline.config["pipeline_mode"] == "phased"
        monkeypatch.chdir(tmp_path)
        assert parse_deployment_config(copy.deepcopy(document)).working_directory.endswith("_phased_deployment_run")
        assert resolve_draft(copy.deepcopy(document)).resolved["pipeline_mode"] == "phased"


#: A registry key declared where no reader reads it: the path, the declaration, the document's fixture.
_UNREAD = {
    "top_weight_quantization": ("weight_quantization", lambda d: d.update(weight_quantization=False)),
    "top_weight_bits": ("weight_bits", lambda d: d.update(weight_bits=1)),
    "nested_weight_quantization": ("platform_constraints.platform_constraints.weight_quantization",
                                   lambda d: d[PC].update(platform_constraints={"weight_quantization": "false"})),
    "nested_weight_bits": ("platform_constraints.platform_constraints.weight_bits",
                           lambda d: d[PC].update(platform_constraints={"weight_bits": 1})),
    "top_level_object": ("extra.layer.weight_bits", lambda d: d.update(extra={"layer": {"weight_bits": 1}})),
    "deep_weight_bits": ("deployment_parameters.extra.layer.weight_bits",
                         lambda d: d[DP].update(extra={"layer": {"weight_bits": 1}})),
    "listed_weight_bits": ("deployment_parameters.extra[0].weight_bits",
                           lambda d: d[DP].update(extra=[{"weight_bits": 1}])),
    "under_the_platform_defaults": ("deployment_parameters.allow_coalescing",
                                    lambda d: d[DP].update(allow_coalescing=False)),
    "session_only_top_key": ("deployment_parameters.target_metric_override",
                             lambda d: d[DP].update(target_metric_override=0.5)),
}


class TestADeclarationNoReaderReads:
    """Review round 1, major: every registry key found where no reader reads it refuses by name at every door
    (the top level for a section key, an object the parser does not read at any depth, a deployment-parameters
    key the platform default layer replaces, a top-level key the session reads at the top level only)."""

    @pytest.mark.parametrize("case", sorted(_UNREAD))
    def test_it_refuses_by_name_at_every_door(self, case, tmp_path):
        path, place = _UNREAD[case]
        document = _load(SPIKING)
        place(document)
        messages = [m for m in validate_deployment_config(copy.deepcopy(document)) if m.startswith(f"{path}: ")]
        assert len(messages) == 1, messages
        _refused_everywhere(document, messages[0], tmp_path, merge=path.startswith((DP, PC)))

    def test_every_section_key_at_the_top_level_and_every_platform_default_in_the_deployment_parameters(self):
        """Generic over the registry: no key is left out of the rule."""
        for key, entry in REGISTRY.items():
            if entry.section != "top" and entry.category is not Category.RUNTIME:
                refusals = declaration_judge.placement_errors({key: 1})
                assert any(m.startswith(f"{key}: {key} is a ") for m in refusals), key
        for key in get_default_platform_constraints():
            assert declared_sections_errors({key: 1}, {})[0].startswith(f"{DP}.{key}: {key} is a "), key

    def test_the_nested_bias_object_s_own_bias_is_read(self, tmp_path):
        document = _load(SPIKING)
        document[PC]["platform_constraints"] = {"has_bias": True}
        _admitted_everywhere(document, tmp_path)

    def test_a_bias_object_a_platform_one_replaces_is_unread(self):
        document = _load(SPIKING)
        document[DP]["platform_constraints"] = {"has_bias": True}
        document[PC]["platform_constraints"] = {"has_bias": True}
        assert any(m.startswith("deployment_parameters.platform_constraints.has_bias: ")
                   for m in validate_deployment_config(document))

    def test_no_shipped_document_declares_one_but_the_one_already_refused(self):
        found = set()
        for path in sorted({*_ROOT.glob("templates/**/*.json"), *_ROOT.glob("examples/**/*.json"),
                            *_ROOT.glob("tests/fixtures/**/*.json")}):
            document = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(document, dict) and isinstance(document.get(DP), dict)
                    and declaration_judge.placement_errors(document)):
                found.add(str(path.relative_to(_ROOT)))
                assert validate_deployment_config(copy.deepcopy(document)), path
        assert found == {"examples/mnist_core_coalescing.json"}


class TestTheJudgedStateHoldsAdmittedDeclarations:
    """Review round 1, major: a declaration its schema refuses is never the state another key's legal set is
    computed from, so it gives the schema's message once and never raises."""

    @pytest.mark.parametrize("section,key,value", [
        (DP, "ttfs_cycle_schedule", []), (DP, "ttfs_cycle_schedule", {}),
        (PC, "ttfs_cycle_schedule", []), (PC, "ttfs_cycle_schedule", {}),
        (DP, "spiking_family", {}), (DP, "spiking_variant", []),
    ], ids=lambda x: repr(x))
    def test_a_malformed_declaration_is_the_schema_s_one_message(self, section, key, value, tmp_path):
        document = _load(SPIKING)
        document[section][key] = value
        message = "; ".join(entry_value_errors(REGISTRY[key], value))
        doc, wizard = validate_deployment_config(copy.deepcopy(document)), _wizard(document)
        assert doc.count(message) == 1 and not any("is not legal here" in m for m in doc), doc
        # The wizard's keyed retired row supersedes the key's other rows (a retired key in the deployment parameters).
        assert wizard.count(message) <= 1 and not any("is not legal here" in m for m in wizard), wizard
        assert _cli(document, tmp_path) == [message]
        assert _merge(document) == [message]

    def test_a_legal_set_that_reads_a_refused_declaration_judges_nothing(self):
        state = {"spiking_family": "lif", "spiking_variant": "streamed"}
        illegal = {"firing_mode": "TTFS"}
        assert legality_rows(state, illegal)
        assert legality_rows(state, illegal, refused={"spiking_family"}) == []
        reads = consumer_reading.ReadRecorder({"a": 1})
        assert reads.get("a") == 1 and "b" not in reads and reads.read == {"a", "b"}

    def test_an_applied_condition_that_reads_a_refused_declaration_judges_nothing(self):
        assert applied_errors({}, {"weight_bits": 1})
        assert applied_errors({}, {"weight_bits": 1}, refused={"weight_quantization"}) == []


#: Every ``src/`` site that reads a run's ``weight_bits`` from its config or resolved platform, and whether it
#: applies the value only under weight quantization (the weight grid) or reads it in every run (review round 1:
#: the K2 claim that weight_bits is inert under float weights was false). A new site fails until it is classified.
WEIGHT_BITS_READERS = {
    "pipelining/pipeline_steps/quantization/weight_quantization_step.py": "grid",
    "pipelining/pipeline_steps/quantization/quantization_verification_step.py": "grid",
    "pipelining/pipeline_steps/mapping/core_quantization_verification_step.py": "grid",
    "advisories/rules_graph_scale.py": "grid",
    "pipelining/core/platform_constraints_resolver.py": "every run",
    "pipelining/pipeline_steps/mapping/soft_core_mapping_step.py": "every run",
    "pipelining/pipeline_steps/mapping/soft_core_structured_pruning.py": "every run",
    "pipelining/pipeline_steps/mapping/hard_core_mapping_step.py": "every run",
    "pipelining/pipeline_steps/mapping/deployment_record_emission.py": "every run",
    "pipelining/pipeline_steps/verification/deployment_record_assembly.py": "every run",
    "pipelining/pipeline_steps/verification/fidelity_emission.py": "every run",
    "deployment_record/objectives/context.py": "every run",
    "deployment_record/introspection/scaling_ladder.py": "every run",
    "deployment_record/cost/absolute/energy.py": "every run",
    "deployment_record/cost/absolute/metrics.py": "every run",
    "tuning/tuners/ttfs_cycle_adaptation_tuner.py": "every run",
    "tuning/tuners/pruning/pruning_tuner_masks.py": "every run",
    "search/problems/joint/model_build.py": "every run",
    "search/problems/joint/validate.py": "every run",
    "mapping/platform/target_bundles.py": "every run",
    # CL-SG: the per-axon register bound (a sign and a (bits - 1)-bit magnitude) on every per-axon run.
    "pipelining/core/simulation_factory.py": "every run",
    # Wave 2 WI-5g: the width of the layers no layer-to-type rule selects, read only on a chip with typed widths.
    "mapping/support/core_types.py": "grid",
}
_READ = re.compile(r"\.get\(\s*[\"']weight_bits[\"']|\[\s*[\"']weight_bits[\"']\s*\]|for key in \(\s*\"weight_bits\"")


class TestWeightBitsIsReadInEveryRun:
    """Review round 1, major: ``weight_bits``' consumers read it in every run (payload sizes, pricing, the record,
    the mapping's register bound), so its relevance is ``always``; only the weight grid applies it under weight
    quantization alone (``applied_when``), and a float bundle's label names the grid, never the key."""

    def test_every_reader_is_classified(self):
        src = _ROOT / "src" / "mimarsinan"
        readers = {str(path.relative_to(src)) for path in src.rglob("*.py")
                   if "config_schema/registry" not in str(path) and _READ.search(path.read_text(encoding="utf-8"))}
        assert readers == set(WEIGHT_BITS_READERS)

    def test_the_relevance_is_its_consumers_and_the_grid_is_conditional(self):
        entry = REGISTRY["weight_bits"]
        assert entry.relevant == Relevance.always()
        assert entry.applied_by == "weight grid" and entry.applied_schema.low == 2
        assert not entry.applied_when.evaluate({"pipeline_mode": "vanilla"})
        assert not entry.applied_when.evaluate({"weight_quantization": False})
        assert entry.applied_when.evaluate({})

    def test_the_float_truenorth_cell_names_the_grid_its_readers_skip(self):
        path = _ROOT / "templates" / "tier_0" / "t0_62_lifs_simplemlp_fp_s8_truenorth.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        flat = merge_pipeline_config(copy.deepcopy(document[DP]), copy.deepcopy(document[PC]))
        label = flat["target_faithfulness_class"]
        assert label.endswith("; weight grid not applied") and "weight_bits not applied" not in label
        assert build_platform_constraints_resolved(flat)["weight_bits"] == 1  # every-run readers read 1
        steps = {name for name, _ in get_pipeline_step_specs({**flat, "input_shape": (1, 28, 28),
                                                              "num_classes": 10, "device": "cpu"})}
        assert {"Weight Quantization", "Quantization Verification"}.isdisjoint(steps)  # the grid readers

    def test_a_bundle_s_own_keys_switch_another_s_relevance(self):
        """Review round 1, minor (mutant R43): the "not applied" relevance reads the bundle's own values and the
        overrides beneath the declarations."""
        bundle = TargetBundle("pruned_chip", "recom_64x64_256", "quarantined", Faithfulness("exact"),
                              json.dumps({"pruning": True, "pruning_fraction": 0.5}))
        written = target_assignments({}, {"target": "pruned_chip"}, write_derived=True, bundles={"pruned_chip": bundle})
        assert written[PC]["target_faithfulness_class"] == "exact"
        overridden = {"target": "pruned_chip", "target_overrides": {"pruning": {"value": False, "note": "dense"}}}
        written = target_assignments({}, overridden, write_derived=True, bundles={"pruned_chip": bundle})
        assert written[PC]["target_faithfulness_class"] == "approximate: pruning declared; pruning_fraction not applied"


#: A value of a key no value schema judges whole, and the consumer that refuses it at the run.
_CONSUMER_REFUSED = [
    (DP, "calibration_set_policy", "x"), (DP, "calibration_set_policy", 7),
    (DP, "calibration_set_policy", [{"bogus": 1}]), (PC, "calibration_set_policy", {"bogus": 1}),
    (PC, "core_value_granularity", "x"), (PC, "core_value_granularity", 7),
    (PC, "core_value_granularity", {"bogus": "per_core"}), (DP, "core_value_granularity", [{"bogus": 1}]),
    (PC, "platform_physics_overrides", "x"), (PC, "platform_physics_overrides", {"bogus": 1}),
    (DP, "platform_physics_overrides", 7), (PC, "platform_physics_profile", "x"),
]


class TestTheDocumentDoorStatesTheRunsRefusal:
    """Review round 1, major: the document door and the CLI path give one verdict, one message, on S0-B's broad
    probes: a consumer's refusal of a key no schema judges, the builder registration's legal sets, a key the
    registration owns."""

    @pytest.mark.parametrize("section,key,value", _CONSUMER_REFUSED, ids=lambda x: repr(x))
    def test_a_consumer_s_refusal_is_named_at_every_door(self, section, key, value, tmp_path):
        document = _load(SPIKING)
        document[section][key] = value
        (message,) = validate_deployment_config(copy.deepcopy(document))
        assert message.startswith(f"{key}={value!r}: refused by its consumer: ")
        assert _parts([message]) == _parts(_cli(document, tmp_path)) == _parts(_merge(document))

    @pytest.mark.parametrize("section", [DP, PC])
    def test_the_builder_registration_judges_the_pretrained_regime_at_every_door(self, section, tmp_path):
        document = _load(SPIKING)
        document[section]["preload_weights"] = True
        message = ("preload_weights=True is not legal here: the current config admits {False}. Remove "
                   "preload_weights to accept the derived value.")
        assert validate_deployment_config(copy.deepcopy(document)) == [message]
        assert _cli(document, tmp_path) == _merge(document) == [message]

    @pytest.mark.parametrize("value", [7.5, "false", True, {}], ids=repr)
    def test_a_registration_owned_key_declared_otherwise_refuses_at_every_door(self, value, tmp_path):
        document = _load(SPIKING)
        document[PC]["pretrained_weight_sets"] = value
        message = ("pretrained_weight_sets is not declarable in a config document — the derivation "
                   "(workload_profile/weight_preloading) owns it. Remove the key.")
        _refused_everywhere(document, message, tmp_path)

    def test_a_floorplan_too_small_for_the_cores_is_named_at_every_door(self, tmp_path):
        document = _load(SPIKING)
        document[PC].update(cores_per_tile=4, tile_grid_rows=2, tile_grid_cols=3)
        (message,) = validate_deployment_config(copy.deepcopy(document))
        assert message.startswith("cores_per_tile=4, tile_grid_rows=2, tile_grid_cols=3: refused by its consumer")
        assert _parts(_cli(document, tmp_path)) == _parts(_merge(document)) == _parts([message])


class TestARequiredKeyIsMissingAbsentOrNull:
    """Review round 1, major: a top-level key the session reads with no value of its own is missing when it is
    absent or null, at the document door, the wizard and the CLI path, before the session reads any."""

    @pytest.mark.parametrize("key", ["experiment_name", "generated_files_path", "data_provider_name"])
    def test_a_null_one_refuses_by_name(self, key, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        document = _load(SPIKING)
        document[key] = None
        message = f"Missing top-level key: {key}"
        assert message in validate_deployment_config(copy.deepcopy(document))
        assert message in _wizard(document)
        with pytest.raises(DeclarationValueError, match=message):
            parse_deployment_config(document)
        with pytest.raises(DeclarationValueError, match=message):  # whichever factory a caller hands in
            parse_deployment_config(document, data_provider_factory=_Provider(document))
        assert list(tmp_path.iterdir()) == []

    def test_an_absent_deployment_parameters_refuses_by_name(self, tmp_path):
        document = _load(SPIKING)
        del document[DP]
        message = "Missing top-level key: deployment_parameters"
        assert message in validate_deployment_config(copy.deepcopy(document)) and message in _wizard(document)
        assert not any("model_type is missing" in m for m in _wizard(document))
        with pytest.raises(DeclarationValueError, match=message):
            parse_deployment_config({**document, "_working_directory": str(tmp_path)})


class TestASectionThatIsNoObject:
    """Review round 1, minor: the wizard keys a falsy non-object section as both doors state it, and emit keeps
    it as declared."""

    @pytest.mark.parametrize("section", [DP, PC])
    @pytest.mark.parametrize("value", [None, []], ids=repr)
    def test_it_is_the_shape_rule_s_at_the_doors_and_kept_by_emit(self, section, value, tmp_path):
        document = _load(SPIKING)
        document[section] = value
        message = f"{section} must be a dict"
        assert message in validate_deployment_config(copy.deepcopy(document))
        assert message in _wizard(document) and not any("is missing" in m for m in _wizard(document))
        assert emit_deployment_config(copy.deepcopy(document))[section] == value
        with pytest.raises(DeclarationValueError, match=message):
            parse_deployment_config({**document, "_working_directory": str(tmp_path)})


class TestANonDeclarableKeyInEitherSection:
    """Review round 1, minor: the not-declarable rule reads both sections, where the run's layout reads the key."""

    @pytest.mark.parametrize("section", [DP, PC])
    def test_it_refuses_by_name_at_the_document_door(self, section, tmp_path):
        """The document door refuses it in either section; the CLI path runs one equal to the registration it
        bridges (the stated boundary of an equal non-declarable key: the document door only)."""
        document = _load(SPIKING)
        document[section]["pretrained_weight_sets"] = []
        assert any(m.startswith("pretrained_weight_sets is not declarable")
                   for m in validate_deployment_config(document))
        assert _cli(document, tmp_path) == []


class TestTheDatasetFactsOnlyTheRunHolds:
    """A registered weight set the provider's input shape rules out is refused by name on the CLI path, before the
    plan reads the regime (the document door holds no provider facts: a stated boundary)."""

    def test_a_weight_set_the_provider_shape_rules_out_refuses_by_name(self, tmp_path, monkeypatch):
        weights = PretrainedWeightSet(id="rgb_only", label="RGB", task="image classification", dataset="ImageNet-1K",
                                      input_shape=(3, 224, 224), num_classes=10, source="torchvision")
        monkeypatch.setattr(ModelRegistry, "get_workload_profile",
                            classmethod(lambda cls, model_id: ModelWorkloadProfile(pretrained_weight_sets=(weights,))))
        document = _load(SPIKING)
        document[DP]["preload_weights"] = True
        assert validate_deployment_config(copy.deepcopy(document)) == []
        with pytest.raises(DerivationRefusal, match=r"preload_weights=True is not legal here"):
            _session(document, tmp_path)


def test_a_laid_top_level_key_is_a_top_level_key():
    """``laid_beneath`` names the section a top-level key is laid beneath, and only a top-level key's."""
    base = dict(flat_key="k", group="run", owner="o", type=FieldType.INT, category=Category.BASIC, label="K", doc="d")
    with pytest.raises(ValueError, match="laid_beneath"):
        ConfigKeySchema(**base, laid_beneath=DP)
    with pytest.raises(ValueError, match="laid_beneath"):
        ConfigKeySchema(**base, section="top", laid_beneath="top")
    ConfigKeySchema(**base, section="top", laid_beneath=DP)
    assert {key for key, entry in REGISTRY.items() if entry.laid_beneath} == {"seed", "pipeline_mode"}


def test_every_target_cell_reads_one_layout():
    """The five target cells and the ODIN cell lay no declaration where no reader reads it."""
    for path in sorted((_ROOT / "templates" / "tier_0").glob("t0_6*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document_declaration_errors(document) == [], path
    assert "truenorth" in target_bundles()

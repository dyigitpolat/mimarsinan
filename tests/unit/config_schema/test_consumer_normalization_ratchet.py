"""The consumer-normalization ratchet: every string normalization call in ``src/mimarsinan`` is classified.

Both doors read a declaration as its consumer reads it (registry ``consumer_reading``), so a consumer that
starts normalizing a declared value is a new reading both doors must take, or both refuse a spelling the
consumer runs. The audit behind the registry readings classified every ``lower``, ``upper``, ``casefold``,
``strip``, ``lstrip`` and ``rstrip`` call by its file and enclosing function or class, and counted each site's
normalizations by method name (S0 amendment B: a ``str.<method>`` reference passed as a callable, as in
``map(str.lower, names)``, counts as one). This test re-runs that scan and fails on a site the audit has not
classified (a new normalizing scope), on a classified site whose count of any method moved (a normalization
added to, removed from or swapped in a classified scope, a second call on one line included), and on a
classified site that is gone. Each change is re-audited and re-pinned by hand.
"""

import ast
import collections
import importlib
import inspect
import pathlib
import textwrap

import mimarsinan
from mimarsinan.config_schema.registry import REGISTRY

_NORMALIZATIONS = frozenset({"lower", "upper", "casefold", "strip", "lstrip", "rstrip"})
_SRC = pathlib.Path(mimarsinan.__file__).parent

#: The three classes of a site that is no registry reading (the audit's sections).
NESTED = "a nested JSON field of a declared key, judged at neither door: its consumer exposes no reading"
JUDGED = ("a nested JSON field judged as written at both doors: its legal set is the spellings every consumer "
          "reads alike (owner decision 8's audit)")
DERIVED = "a derivation-owned key a document cannot declare"
OTHER = "no registry value"

#: (file under src/mimarsinan, enclosing function or class scope): (the declared key whose consumer reading
#: the site is (registry ``consumer_reading``), else the class of a site that reads no declared value; the
#: scope's normalizations by method name).
CLASSIFIED = {
    # Declared keys whose consumer normalizes them: the registry reading references these functions.
    ("tuning/orchestration/temporal_allocation.py", "resolve_s_allocation_mode"): ("s_allocation", {"lower": 1}),
    ("chip_simulation/nevresim/connectivity.py", "resolve_nevresim_connectivity_mode"):
        ("nevresim_connectivity_mode", {"strip": 1}),
    # training_recipe / tuning_recipe optimizer and scheduler, preprocessing.normalize; preprocessing.interpolation
    # is judged at both doors by the spellings its two consumers read alike (S0 amendment B).
    ("model_training/training_recipe.py", "build_optimizer"): (NESTED, {"lower": 1}),
    ("model_training/training_recipe.py", "build_scheduler"): (NESTED, {"lower": 1}),
    ("data_handling/preprocessing.py", "_resolve_normalize"): (NESTED, {"lower": 1}),
    ("data_handling/ffcv/spec_builder.py", "_PILResize.__init__"): (JUDGED, {"lower": 1}),
    # spiking_mode: retired as a document key, written canonical by fold_spiking_axes.
    ("mapping/pruning/liveness_semantics.py", "bias_can_activate"): (DERIVED, {"lower": 1}),
    ("mapping/pruning/liveness_transfer/constant_policy.py", "domain_admits_nonzero_constants"):
        (DERIVED, {"lower": 1}),
    # Simulator backend names, the optimization driver, registered weight-set ids.
    ("chip_simulation/behavior_config.py", "NeuralBehaviorConfig.require_backend"): (OTHER, {"lower": 1}),
    ("chip_simulation/firing_strategy.py", "FiringStrategy.capabilities"): (OTHER, {"lower": 1}),
    ("chip_simulation/spiking_semantics.py", "backend_capabilities"): (OTHER, {"lower": 1}),
    ("tuning/orchestration/optimization_driver.py", "resolve_optimization_driver"): (OTHER, {"lower": 1}),
    ("tuning/orchestration/ttfs_adaptation_plan.py", "TtfsAdaptationPlan.resolve"): (OTHER, {"lower": 1}),
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weight_set"): (OTHER, {"lower": 1}),
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weights"): (OTHER, {"upper": 1}),
    # Wave 0 (frozen patch): torch operator names, never a declared value.
    ("torch_mapping/converter_handlers/symbolic/batch_domain/kernels.py", "_pointwise_name"):
        (OTHER, {"rstrip": 1}),
    ("torch_mapping/converter_handlers/symbolic/entered_modes.py", "_native_call"): (OTHER, {"strip": 1}),
    # Environment variables.
    ("common/env.py", "gui_no_browser"): (OTHER, {"lower": 1, "strip": 1}),
    ("common/env.py", "gui_resource_render_override"): (OTHER, {"lower": 1, "strip": 1}),
    ("common/env.py", "imagenet_root"): (OTHER, {"strip": 1}),
    ("common/env.py", "loihi_wave_workers"): (OTHER, {"strip": 1}),
    ("common/env.py", "nevresim_root"): (OTHER, {"strip": 1}),
    ("common/env.py", "simulation_step_timeout_override"): (OTHER, {"strip": 1}),
    # IR op types and activation-type strings.
    ("mapping/pruning/boundary_policy.py", "_computeop_relays_deadness"): (OTHER, {"lower": 1}),
    ("mapping/pruning/certificate/dyadic_grid.py", "_op_preserves_dyadic_grid"): (OTHER, {"lower": 1}),
    ("mapping/pruning/certificate/zero_preserving.py", "is_zero_preserving_host_op"): (OTHER, {"lower": 1}),
    ("mapping/pruning/liveness_transfer/transfer_registry.py", "derive_liveness_transfer"): (OTHER, {"lower": 1}),
    ("models/spiking/ttfs_activation.py", "ttfs_activation_from_type"): (OTHER, {"strip": 2}),
    ("pipelining/pipeline_steps/adaptation/activation_adaptation_step.py", "ActivationAdaptationStep.process"):
        (OTHER, {"strip": 1}),
    # Parameter names, dataset split names, provider class names, preset registration, dataset labels.
    ("model_training/training_recipe.py", "_is_no_decay"): (OTHER, {"lower": 1}),
    ("model_training/training_recipe.py", "build_param_groups"): (OTHER, {"lower": 1}),
    ("data_handling/ffcv/pipeline_spec.py", "normalize_split_name"): (OTHER, {"lower": 1, "strip": 1}),
    ("data_handling/ffcv/spec_builder.py", "_provider_id"): (OTHER, {"lower": 1}),
    ("data_handling/preprocessing.py", "register_normalization_preset"): (OTHER, {"lower": 1}),
    ("chip_simulation/pareto.py", "_norm_dataset"): (OTHER, {"lower": 1, "strip": 1}),
    # Process output, logs, files, HTTP and GUI strings, notes, coverage rows and axes, display labels.
    ("chip_simulation/coverage_ci.py", "assert_axes_screening_sound"): (OTHER, {"strip": 1}),
    ("chip_simulation/coverage_reporting.py", "FlagMetadata.is_unowned"): (OTHER, {"strip": 1}),
    ("chip_simulation/coverage_rows.py", "classify_validity_tier"): (OTHER, {"upper": 1}),
    ("chip_simulation/coverage_rows.py", "flag_owner_of"): (OTHER, {"strip": 1}),
    ("chip_simulation/coverage_rows.py", "mine_flagged_ops"): (OTHER, {"upper": 1}),
    ("chip_simulation/coverage_rows.py", "parse_ledger_timestamp"): (OTHER, {"strip": 1}),
    ("chip_simulation/cross_sim_parity.py", "assert_cross_sim_screen_sound"): (OTHER, {"strip": 2}),
    ("chip_simulation/hypervolume_axes.py", "HypervolumeAxis.__post_init__"): (OTHER, {"strip": 1}),
    ("chip_simulation/hypervolume_axis_encoder.py", "_truthy"): (OTHER, {"lower": 1, "strip": 1}),
    ("chip_simulation/hypervolume_axis_encoder.py", "normalize_dataset"): (OTHER, {"lower": 1, "strip": 1}),
    ("chip_simulation/nevresim/execute_nevresim.py", "_collect_worker_outputs"): (OTHER, {"strip": 1}),
    ("chip_simulation/nevresim/execute_nevresim.py", "_parse_stdout_tokens"): (OTHER, {"strip": 1}),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_membrane_records"): (OTHER, {"strip": 1}),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_records"): (OTHER, {"strip": 1}),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_trains"): (OTHER, {"strip": 1}),
    ("chip_simulation/nevresim/instrumented_builds.py", "emit_and_compile_flagged"): (OTHER, {"lower": 1}),
    ("chip_simulation/pareto.py", "load_deep_cnn_rows"): (OTHER, {"strip": 1}),
    ("chip_simulation/ttfs/ttfs_recorder.py", "format_first_ttfs_diff"): (OTHER, {"strip": 1}),
    ("common/dependency_manifest.py", "_canonical"): (OTHER, {"lower": 1}),
    ("common/dependency_manifest.py", "_entries"): (OTHER, {"strip": 1}),
    ("common/dependency_manifest.py", "declared_direct_reference"): (OTHER, {"strip": 1}),
    ("common/dependency_manifest.py", "declared_requirement"): (OTHER, {"strip": 2}),
    ("common/dependency_manifest.py", "declared_specifier"): (OTHER, {"strip": 1}),
    ("common/reporter.py", "DefaultReporter.console_log"): (OTHER, {"strip": 1}),
    ("config_schema/display_view_meta.py", "_snake_to_label"): (OTHER, {"strip": 1}),
    ("config_schema/registry/value_schema.py", "_bound_text"): (OTHER, {"upper": 1}),
    ("config_schema/registry/value_schema.py", "describe"): (OTHER, {"upper": 1}),
    ("deployment_record/cost/model.py", "_require_no_double_count_note"): (OTHER, {"lower": 1}),
    ("deployment_record/platform_physics/host_calibration.py", "read_rapl_package_j"): (OTHER, {"strip": 1}),
    ("deployment_record/platform_physics/profile.py", "PhysicsConstantValue.__post_init__"): (OTHER, {"strip": 1}),
    ("gui/runs.py", "classify_artifact"): (OTHER, {"lower": 1}),
    ("gui/runs.py", "list_dir_artifacts"): (OTHER, {"lower": 1}),
    ("gui/runtime/active_run_tailers.py", "JsonlTailer._tick"): (OTHER, {"strip": 1}),
    ("gui/runtime/persistence/load.py", "load_console_logs"): (OTHER, {"strip": 1}),
    ("gui/runtime/persistence/load.py", "load_events"): (OTHER, {"strip": 1}),
    ("gui/runtime/persistence/load.py", "load_live_metrics"): (OTHER, {"strip": 1}),
    ("gui/runtime/process_spawn.py", "start_console_reader._drain"): (OTHER, {"rstrip": 2}),
    ("gui/server/routes_resources.py", "conditional_bytes_response"): (OTHER, {"strip": 1}),
    ("gui/snapshot/console_events.py", "_coerce"): (OTHER, {"rstrip": 1, "strip": 2}),
    ("gui/snapshot/console_events.py", "_parse_line"): (OTHER, {"strip": 1}),
    ("gui/templates.py", "name_and_deployment_from_post_body"): (OTHER, {"strip": 1}),
    ("gui/templates.py", "save_template"): (OTHER, {"strip": 1}),
    ("gui/viewmodel/step_metrics_vm.py", "metric_category"): (OTHER, {"lower": 1}),
    # G7: a refusal's message text, its first line kept as the census verdict's reason.
    ("mapping/capability/member.py", "_refused"): (OTHER, {"strip": 1}),
    ("mapping/verification/suggester/hw_config_suggester_scheduled.py", "suggest_hardware_config_scheduled"):
        (OTHER, {"rstrip": 1}),
    ("pipelining/core/simulation_factory.py", "run_hcm_mapping_metric"): (OTHER, {"lower": 1}),
    ("pipelining/core/simulation_factory.py", "run_hcm_spiking_test"): (OTHER, {"lower": 1}),
    ("pipelining/core/simulation_factory.py", "run_scm_identity_metric"): (OTHER, {"lower": 1}),
    ("search/optimizers/compilagent/compilagent_optimizer.py", "CompilagentOptimizer._build_system_instructions"):
        (OTHER, {"strip": 1}),
    ("search/optimizers/llm/trace_format.py", "parse_json_object"): (OTHER, {"strip": 1}),
    ("search/optimizers/llm/trace_format.py", "split_prompt_for_trace"): (OTHER, {"strip": 4}),
    ("spiking/theta_cotrain.py", "emit_theta_install_witness"): (OTHER, {"lower": 1}),
    ("visualization/graphviz/common.py", "try_render_dot"): (OTHER, {"lower": 1, "strip": 1}),
    ("visualization/search_viz/html/sections_table.py", "render_table_section"): (OTHER, {"lower": 1}),
}


def _normalization_sites():
    """{(file, enclosing scope): {method: count}} under src/mimarsinan, as the audit scan finds them: every call
    of a normalization method, and every ``str.<method>`` reference passed uncalled (a callable argument)."""
    sites = collections.defaultdict(collections.Counter)
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}

        def visit(node, scope, rel=rel, called=called):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, scope + (child.name,))
                    continue
                if isinstance(child, ast.Attribute) and child.attr in _NORMALIZATIONS and (
                        id(child) in called
                        or (isinstance(child.value, ast.Name) and child.value.id == "str")):
                    sites[(rel, ".".join(scope) or "<module>")][child.attr] += 1
                visit(child, scope)

        visit(tree, ())
    return {site: dict(counts) for site, counts in sites.items()}


def test_every_normalization_site_is_classified():
    sites = _normalization_sites()
    new = sorted(set(sites) - set(CLASSIFIED))
    assert not new, (
        f"new string normalizations {new}: classify each in CLASSIFIED; one that reads a declared registry "
        "value is a consumer reading both doors must take (registry consumer_reading), else both doors refuse "
        "a spelling its consumer runs")
    moved = sorted((site, CLASSIFIED[site][1], counts) for site, counts in sites.items()
                   if site in CLASSIFIED and CLASSIFIED[site][1] != counts)
    assert not moved, (
        f"normalizations added to, removed from or swapped in classified scopes (site, pinned, found) {moved}: "
        "re-audit each against the registry readings, then re-pin its counts in CLASSIFIED")
    gone = sorted(set(CLASSIFIED) - set(sites))
    assert not gone, f"classified sites no longer in src/mimarsinan {gone}: remove them from CLASSIFIED"


def test_the_sites_that_read_a_declared_key_are_the_registry_readings():
    classes = {NESTED, JUDGED, DERIVED, OTHER}
    labels = [label for label, _ in CLASSIFIED.values()]
    assert all(label in REGISTRY or label in classes for label in labels)
    read_keys = {label for label in labels if label in REGISTRY}
    assert read_keys == {key for key, entry in REGISTRY.items() if entry.consumer_reading is not None}


def test_each_registry_reading_reaches_its_consumer_site_by_reference():
    """The reading is the consumer's function, or calls it by the name its module imports: never a copy."""
    for (rel, scope), (key, _) in CLASSIFIED.items():
        if key not in REGISTRY:
            continue
        module = importlib.import_module("mimarsinan." + rel[:-len(".py")].replace("/", "."))
        site = getattr(module, scope)
        reading = REGISTRY[key].consumer_reading
        assert reading is site or reading.__globals__.get(scope) is site, (key, scope)
        source = textwrap.dedent(inspect.getsource(reading))
        names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
        assert reading is site or scope in names, (key, scope)

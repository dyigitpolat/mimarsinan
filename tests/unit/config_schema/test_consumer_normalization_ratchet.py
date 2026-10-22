"""The consumer-normalization ratchet: every string normalization in ``src/mimarsinan`` is a classified site.

Both doors read a declaration as its consumer reads it (registry ``consumer_reading``), so a consumer that
starts normalizing a declared value is a new reading both doors must take, or both refuse a spelling the
consumer runs. The audit behind the registry readings classified every ``lower``, ``upper``, ``casefold``,
``strip``, ``lstrip`` and ``rstrip`` call by its file and enclosing function or class; this test re-runs that
scan and fails on a site the audit has not classified (a new normalization) and on a classified site that is
gone (the list only shrinks with the code).
"""

import ast
import importlib
import inspect
import pathlib
import textwrap

import mimarsinan
from mimarsinan.config_schema.registry import REGISTRY

_NORMALIZATIONS = frozenset({"lower", "upper", "casefold", "strip", "lstrip", "rstrip"})
_SRC = pathlib.Path(mimarsinan.__file__).parent

#: The three classes of a site that is no registry reading (the audit's sections).
NESTED = "a nested JSON field of a declared key, judged at neither door (owner decision 8's audit)"
DERIVED = "a derivation-owned key a document cannot declare"
OTHER = "no registry value"

#: (file under src/mimarsinan, enclosing function or class scope): the declared key whose consumer reading the
#: site is (registry ``consumer_reading``), else the class of a site that reads no declared value.
CLASSIFIED = {
    # Declared keys whose consumer normalizes them: the registry reading references these functions.
    ("tuning/orchestration/temporal_allocation.py", "resolve_s_allocation_mode"): "s_allocation",
    ("chip_simulation/nevresim/connectivity.py", "resolve_nevresim_connectivity_mode"):
        "nevresim_connectivity_mode",
    # training_recipe / tuning_recipe optimizer and scheduler, preprocessing.normalize and .interpolation.
    ("model_training/training_recipe.py", "build_optimizer"): NESTED,
    ("model_training/training_recipe.py", "build_scheduler"): NESTED,
    ("data_handling/preprocessing.py", "_resolve_normalize"): NESTED,
    ("data_handling/ffcv/spec_builder.py", "_PILResize.__init__"): NESTED,
    # spiking_mode: retired as a document key, written canonical by fold_spiking_axes.
    ("mapping/pruning/liveness_semantics.py", "bias_can_activate"): DERIVED,
    ("mapping/pruning/liveness_transfer/constant_policy.py", "domain_admits_nonzero_constants"): DERIVED,
    # Simulator backend names, the optimization driver, registered weight-set ids.
    ("chip_simulation/behavior_config.py", "NeuralBehaviorConfig.require_backend"): OTHER,
    ("chip_simulation/firing_strategy.py", "FiringStrategy.capabilities"): OTHER,
    ("chip_simulation/spiking_semantics.py", "backend_capabilities"): OTHER,
    ("tuning/orchestration/optimization_driver.py", "resolve_optimization_driver"): OTHER,
    ("tuning/orchestration/ttfs_adaptation_plan.py", "TtfsAdaptationPlan.resolve"): OTHER,
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weight_set"): OTHER,
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weights"): OTHER,
    # Wave 0 (frozen patch): torch operator names, never a declared value.
    ("torch_mapping/converter_handlers/symbolic/batch_domain/kernels.py", "_pointwise_name"): OTHER,
    ("torch_mapping/converter_handlers/symbolic/entered_modes.py", "_native_call"): OTHER,
    # Environment variables.
    ("common/env.py", "gui_no_browser"): OTHER,
    ("common/env.py", "gui_resource_render_override"): OTHER,
    ("common/env.py", "imagenet_root"): OTHER,
    ("common/env.py", "loihi_wave_workers"): OTHER,
    ("common/env.py", "nevresim_root"): OTHER,
    ("common/env.py", "simulation_step_timeout_override"): OTHER,
    # IR op types and activation-type strings.
    ("mapping/pruning/boundary_policy.py", "_computeop_relays_deadness"): OTHER,
    ("mapping/pruning/certificate/dyadic_grid.py", "_op_preserves_dyadic_grid"): OTHER,
    ("mapping/pruning/certificate/zero_preserving.py", "is_zero_preserving_host_op"): OTHER,
    ("mapping/pruning/liveness_transfer/transfer_registry.py", "derive_liveness_transfer"): OTHER,
    ("models/spiking/ttfs_activation.py", "ttfs_activation_from_type"): OTHER,
    ("pipelining/pipeline_steps/adaptation/activation_adaptation_step.py", "ActivationAdaptationStep.process"):
        OTHER,
    # Parameter names, dataset split names, provider class names, preset registration, dataset labels.
    ("model_training/training_recipe.py", "_is_no_decay"): OTHER,
    ("model_training/training_recipe.py", "build_param_groups"): OTHER,
    ("data_handling/ffcv/pipeline_spec.py", "normalize_split_name"): OTHER,
    ("data_handling/ffcv/spec_builder.py", "_provider_id"): OTHER,
    ("data_handling/preprocessing.py", "register_normalization_preset"): OTHER,
    ("chip_simulation/pareto.py", "_norm_dataset"): OTHER,
    # Process output, logs, files, HTTP and GUI strings, notes, coverage rows and axes, display labels.
    ("chip_simulation/coverage_ci.py", "assert_axes_screening_sound"): OTHER,
    ("chip_simulation/coverage_reporting.py", "FlagMetadata.is_unowned"): OTHER,
    ("chip_simulation/coverage_rows.py", "classify_validity_tier"): OTHER,
    ("chip_simulation/coverage_rows.py", "flag_owner_of"): OTHER,
    ("chip_simulation/coverage_rows.py", "mine_flagged_ops"): OTHER,
    ("chip_simulation/coverage_rows.py", "parse_ledger_timestamp"): OTHER,
    ("chip_simulation/cross_sim_parity.py", "assert_cross_sim_screen_sound"): OTHER,
    ("chip_simulation/hypervolume_axes.py", "HypervolumeAxis.__post_init__"): OTHER,
    ("chip_simulation/hypervolume_axis_encoder.py", "_truthy"): OTHER,
    ("chip_simulation/hypervolume_axis_encoder.py", "normalize_dataset"): OTHER,
    ("chip_simulation/nevresim/execute_nevresim.py", "_collect_worker_outputs"): OTHER,
    ("chip_simulation/nevresim/execute_nevresim.py", "_parse_stdout_tokens"): OTHER,
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_membrane_records"): OTHER,
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_records"): OTHER,
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_trains"): OTHER,
    ("chip_simulation/nevresim/instrumented_builds.py", "emit_and_compile_flagged"): OTHER,
    ("chip_simulation/pareto.py", "load_deep_cnn_rows"): OTHER,
    ("chip_simulation/ttfs/ttfs_recorder.py", "format_first_ttfs_diff"): OTHER,
    ("common/dependency_manifest.py", "_canonical"): OTHER,
    ("common/dependency_manifest.py", "_entries"): OTHER,
    ("common/dependency_manifest.py", "declared_direct_reference"): OTHER,
    ("common/dependency_manifest.py", "declared_requirement"): OTHER,
    ("common/dependency_manifest.py", "declared_specifier"): OTHER,
    ("common/reporter.py", "DefaultReporter.console_log"): OTHER,
    ("config_schema/display_view_meta.py", "_snake_to_label"): OTHER,
    ("config_schema/registry/value_schema.py", "_bound_text"): OTHER,
    ("config_schema/registry/value_schema.py", "describe"): OTHER,
    ("deployment_record/cost/model.py", "_require_no_double_count_note"): OTHER,
    ("deployment_record/platform_physics/host_calibration.py", "read_rapl_package_j"): OTHER,
    ("deployment_record/platform_physics/profile.py", "PhysicsConstantValue.__post_init__"): OTHER,
    ("gui/runs.py", "classify_artifact"): OTHER,
    ("gui/runs.py", "list_dir_artifacts"): OTHER,
    ("gui/runtime/active_run_tailers.py", "JsonlTailer._tick"): OTHER,
    ("gui/runtime/persistence/load.py", "load_console_logs"): OTHER,
    ("gui/runtime/persistence/load.py", "load_events"): OTHER,
    ("gui/runtime/persistence/load.py", "load_live_metrics"): OTHER,
    ("gui/runtime/process_spawn.py", "start_console_reader._drain"): OTHER,
    ("gui/server/routes_resources.py", "conditional_bytes_response"): OTHER,
    ("gui/snapshot/console_events.py", "_coerce"): OTHER,
    ("gui/snapshot/console_events.py", "_parse_line"): OTHER,
    ("gui/templates.py", "name_and_deployment_from_post_body"): OTHER,
    ("gui/templates.py", "save_template"): OTHER,
    ("gui/viewmodel/step_metrics_vm.py", "metric_category"): OTHER,
    ("mapping/verification/suggester/hw_config_suggester_scheduled.py", "suggest_hardware_config_scheduled"):
        OTHER,
    ("pipelining/core/simulation_factory.py", "run_hcm_mapping_metric"): OTHER,
    ("pipelining/core/simulation_factory.py", "run_hcm_spiking_test"): OTHER,
    ("pipelining/core/simulation_factory.py", "run_scm_identity_metric"): OTHER,
    ("search/optimizers/compilagent/compilagent_optimizer.py", "CompilagentOptimizer._build_system_instructions"):
        OTHER,
    ("search/optimizers/llm/trace_format.py", "parse_json_object"): OTHER,
    ("search/optimizers/llm/trace_format.py", "split_prompt_for_trace"): OTHER,
    ("spiking/theta_cotrain.py", "emit_theta_install_witness"): OTHER,
    ("visualization/graphviz/common.py", "try_render_dot"): OTHER,
    ("visualization/search_viz/html/sections_table.py", "render_table_section"): OTHER,
}


def _normalization_sites():
    """(file, enclosing scope) of every normalization call under src/mimarsinan, as the audit scan finds them."""
    sites = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_SRC).as_posix()

        def visit(node, scope, rel=rel):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, scope + (child.name,))
                    continue
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr in _NORMALIZATIONS):
                    sites.add((rel, ".".join(scope) or "<module>"))
                visit(child, scope)

        visit(ast.parse(path.read_text(encoding="utf-8")), ())
    return sites


def test_every_normalization_site_is_classified():
    sites = _normalization_sites()
    new = sorted(sites - set(CLASSIFIED))
    assert not new, (
        f"new string normalizations {new}: classify each in CLASSIFIED; one that reads a declared registry "
        "value is a consumer reading both doors must take (registry consumer_reading), else both doors refuse "
        "a spelling its consumer runs")
    gone = sorted(set(CLASSIFIED) - sites)
    assert not gone, f"classified sites no longer in src/mimarsinan {gone}: remove them from CLASSIFIED"


def test_the_sites_that_read_a_declared_key_are_the_registry_readings():
    classes = {NESTED, DERIVED, OTHER}
    assert all(label in REGISTRY or label in classes for label in CLASSIFIED.values())
    read_keys = {label for label in CLASSIFIED.values() if label in REGISTRY}
    assert read_keys == {key for key, entry in REGISTRY.items() if entry.consumer_reading is not None}


def test_each_registry_reading_reaches_its_consumer_site_by_reference():
    """The reading is the consumer's function, or calls it by the name its module imports: never a copy."""
    for (rel, scope), key in CLASSIFIED.items():
        if key not in REGISTRY:
            continue
        module = importlib.import_module("mimarsinan." + rel[:-len(".py")].replace("/", "."))
        site = getattr(module, scope)
        reading = REGISTRY[key].consumer_reading
        assert reading is site or reading.__globals__.get(scope) is site, (key, scope)
        source = textwrap.dedent(inspect.getsource(reading))
        names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
        assert reading is site or scope in names, (key, scope)

"""The consumer-normalization ratchet: every string normalization call in ``src/mimarsinan`` is classified.

Both doors read a declaration as its consumer reads it (registry ``consumer_reading``), so a consumer that
starts normalizing a declared value is a new reading both doors must take, or both refuse a spelling the
consumer runs. The audit behind the registry readings classified every ``lower``, ``upper``, ``casefold``,
``strip``, ``lstrip`` and ``rstrip`` call by its file and enclosing function or class, and counted the calls
of each site. This test re-runs that scan and fails on any change of the classified calls: a site the audit
has not classified (a new normalizing scope), a classified site whose call count moved (a normalization added
to or removed from a classified scope, a second call on one line included), and a classified site that is
gone. Each change is re-audited and re-pinned by hand.
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
NESTED = "a nested JSON field of a declared key, judged at neither door (owner decision 8's audit)"
DERIVED = "a derivation-owned key a document cannot declare"
OTHER = "no registry value"

#: (file under src/mimarsinan, enclosing function or class scope): (the declared key whose consumer reading
#: the site is (registry ``consumer_reading``), else the class of a site that reads no declared value; the
#: number of normalization calls in the scope).
CLASSIFIED = {
    # Declared keys whose consumer normalizes them: the registry reading references these functions.
    ("tuning/orchestration/temporal_allocation.py", "resolve_s_allocation_mode"): ("s_allocation", 1),
    ("chip_simulation/nevresim/connectivity.py", "resolve_nevresim_connectivity_mode"):
        ("nevresim_connectivity_mode", 1),
    # training_recipe / tuning_recipe optimizer and scheduler, preprocessing.normalize and .interpolation.
    ("model_training/training_recipe.py", "build_optimizer"): (NESTED, 1),
    ("model_training/training_recipe.py", "build_scheduler"): (NESTED, 1),
    ("data_handling/preprocessing.py", "_resolve_normalize"): (NESTED, 1),
    ("data_handling/ffcv/spec_builder.py", "_PILResize.__init__"): (NESTED, 1),
    # spiking_mode: retired as a document key, written canonical by fold_spiking_axes.
    ("mapping/pruning/liveness_semantics.py", "bias_can_activate"): (DERIVED, 1),
    ("mapping/pruning/liveness_transfer/constant_policy.py", "domain_admits_nonzero_constants"): (DERIVED, 1),
    # Simulator backend names, the optimization driver, registered weight-set ids.
    ("chip_simulation/behavior_config.py", "NeuralBehaviorConfig.require_backend"): (OTHER, 1),
    ("chip_simulation/firing_strategy.py", "FiringStrategy.capabilities"): (OTHER, 1),
    ("chip_simulation/spiking_semantics.py", "backend_capabilities"): (OTHER, 1),
    ("tuning/orchestration/optimization_driver.py", "resolve_optimization_driver"): (OTHER, 1),
    ("tuning/orchestration/ttfs_adaptation_plan.py", "TtfsAdaptationPlan.resolve"): (OTHER, 1),
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weight_set"): (OTHER, 1),
    ("models/builders/torch/torchvision_builder_utils.py", "torchvision_weights"): (OTHER, 1),
    # Wave 0 (frozen patch): torch operator names, never a declared value.
    ("torch_mapping/converter_handlers/symbolic/batch_domain/kernels.py", "_pointwise_name"): (OTHER, 1),
    ("torch_mapping/converter_handlers/symbolic/entered_modes.py", "_native_call"): (OTHER, 1),
    # Environment variables.
    ("common/env.py", "gui_no_browser"): (OTHER, 2),
    ("common/env.py", "gui_resource_render_override"): (OTHER, 2),
    ("common/env.py", "imagenet_root"): (OTHER, 1),
    ("common/env.py", "loihi_wave_workers"): (OTHER, 1),
    ("common/env.py", "nevresim_root"): (OTHER, 1),
    ("common/env.py", "simulation_step_timeout_override"): (OTHER, 1),
    # IR op types and activation-type strings.
    ("mapping/pruning/boundary_policy.py", "_computeop_relays_deadness"): (OTHER, 1),
    ("mapping/pruning/certificate/dyadic_grid.py", "_op_preserves_dyadic_grid"): (OTHER, 1),
    ("mapping/pruning/certificate/zero_preserving.py", "is_zero_preserving_host_op"): (OTHER, 1),
    ("mapping/pruning/liveness_transfer/transfer_registry.py", "derive_liveness_transfer"): (OTHER, 1),
    ("models/spiking/ttfs_activation.py", "ttfs_activation_from_type"): (OTHER, 2),
    ("pipelining/pipeline_steps/adaptation/activation_adaptation_step.py", "ActivationAdaptationStep.process"):
        (OTHER, 1),
    # Parameter names, dataset split names, provider class names, preset registration, dataset labels.
    ("model_training/training_recipe.py", "_is_no_decay"): (OTHER, 1),
    ("model_training/training_recipe.py", "build_param_groups"): (OTHER, 1),
    ("data_handling/ffcv/pipeline_spec.py", "normalize_split_name"): (OTHER, 2),
    ("data_handling/ffcv/spec_builder.py", "_provider_id"): (OTHER, 1),
    ("data_handling/preprocessing.py", "register_normalization_preset"): (OTHER, 1),
    ("chip_simulation/pareto.py", "_norm_dataset"): (OTHER, 2),
    # Process output, logs, files, HTTP and GUI strings, notes, coverage rows and axes, display labels.
    ("chip_simulation/coverage_ci.py", "assert_axes_screening_sound"): (OTHER, 1),
    ("chip_simulation/coverage_reporting.py", "FlagMetadata.is_unowned"): (OTHER, 1),
    ("chip_simulation/coverage_rows.py", "classify_validity_tier"): (OTHER, 1),
    ("chip_simulation/coverage_rows.py", "flag_owner_of"): (OTHER, 1),
    ("chip_simulation/coverage_rows.py", "mine_flagged_ops"): (OTHER, 1),
    ("chip_simulation/coverage_rows.py", "parse_ledger_timestamp"): (OTHER, 1),
    ("chip_simulation/cross_sim_parity.py", "assert_cross_sim_screen_sound"): (OTHER, 2),
    ("chip_simulation/hypervolume_axes.py", "HypervolumeAxis.__post_init__"): (OTHER, 1),
    ("chip_simulation/hypervolume_axis_encoder.py", "_truthy"): (OTHER, 2),
    ("chip_simulation/hypervolume_axis_encoder.py", "normalize_dataset"): (OTHER, 2),
    ("chip_simulation/nevresim/execute_nevresim.py", "_collect_worker_outputs"): (OTHER, 1),
    ("chip_simulation/nevresim/execute_nevresim.py", "_parse_stdout_tokens"): (OTHER, 1),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_membrane_records"): (OTHER, 1),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_records"): (OTHER, 1),
    ("chip_simulation/nevresim/execute_nevresim.py", "parse_spike_trains"): (OTHER, 1),
    ("chip_simulation/nevresim/instrumented_builds.py", "emit_and_compile_flagged"): (OTHER, 1),
    ("chip_simulation/pareto.py", "load_deep_cnn_rows"): (OTHER, 1),
    ("chip_simulation/ttfs/ttfs_recorder.py", "format_first_ttfs_diff"): (OTHER, 1),
    ("common/dependency_manifest.py", "_canonical"): (OTHER, 1),
    ("common/dependency_manifest.py", "_entries"): (OTHER, 1),
    ("common/dependency_manifest.py", "declared_direct_reference"): (OTHER, 1),
    ("common/dependency_manifest.py", "declared_requirement"): (OTHER, 2),
    ("common/dependency_manifest.py", "declared_specifier"): (OTHER, 1),
    ("common/reporter.py", "DefaultReporter.console_log"): (OTHER, 1),
    ("config_schema/display_view_meta.py", "_snake_to_label"): (OTHER, 1),
    ("config_schema/registry/value_schema.py", "_bound_text"): (OTHER, 1),
    ("config_schema/registry/value_schema.py", "describe"): (OTHER, 1),
    ("deployment_record/cost/model.py", "_require_no_double_count_note"): (OTHER, 1),
    ("deployment_record/platform_physics/host_calibration.py", "read_rapl_package_j"): (OTHER, 1),
    ("deployment_record/platform_physics/profile.py", "PhysicsConstantValue.__post_init__"): (OTHER, 1),
    ("gui/runs.py", "classify_artifact"): (OTHER, 1),
    ("gui/runs.py", "list_dir_artifacts"): (OTHER, 1),
    ("gui/runtime/active_run_tailers.py", "JsonlTailer._tick"): (OTHER, 1),
    ("gui/runtime/persistence/load.py", "load_console_logs"): (OTHER, 1),
    ("gui/runtime/persistence/load.py", "load_events"): (OTHER, 1),
    ("gui/runtime/persistence/load.py", "load_live_metrics"): (OTHER, 1),
    ("gui/runtime/process_spawn.py", "start_console_reader._drain"): (OTHER, 2),
    ("gui/server/routes_resources.py", "conditional_bytes_response"): (OTHER, 1),
    ("gui/snapshot/console_events.py", "_coerce"): (OTHER, 3),
    ("gui/snapshot/console_events.py", "_parse_line"): (OTHER, 1),
    ("gui/templates.py", "name_and_deployment_from_post_body"): (OTHER, 1),
    ("gui/templates.py", "save_template"): (OTHER, 1),
    ("gui/viewmodel/step_metrics_vm.py", "metric_category"): (OTHER, 1),
    ("mapping/verification/suggester/hw_config_suggester_scheduled.py", "suggest_hardware_config_scheduled"):
        (OTHER, 1),
    ("pipelining/core/simulation_factory.py", "run_hcm_mapping_metric"): (OTHER, 1),
    ("pipelining/core/simulation_factory.py", "run_hcm_spiking_test"): (OTHER, 1),
    ("pipelining/core/simulation_factory.py", "run_scm_identity_metric"): (OTHER, 1),
    ("search/optimizers/compilagent/compilagent_optimizer.py", "CompilagentOptimizer._build_system_instructions"):
        (OTHER, 1),
    ("search/optimizers/llm/trace_format.py", "parse_json_object"): (OTHER, 1),
    ("search/optimizers/llm/trace_format.py", "split_prompt_for_trace"): (OTHER, 4),
    ("spiking/theta_cotrain.py", "emit_theta_install_witness"): (OTHER, 1),
    ("visualization/graphviz/common.py", "try_render_dot"): (OTHER, 2),
    ("visualization/search_viz/html/sections_table.py", "render_table_section"): (OTHER, 1),
}


def _normalization_sites():
    """{(file, enclosing scope): normalization calls} under src/mimarsinan, as the audit scan finds them."""
    sites = collections.Counter()
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
                    sites[(rel, ".".join(scope) or "<module>")] += 1
                visit(child, scope)

        visit(ast.parse(path.read_text(encoding="utf-8")), ())
    return sites


def test_every_normalization_site_is_classified():
    sites = _normalization_sites()
    new = sorted(set(sites) - set(CLASSIFIED))
    assert not new, (
        f"new string normalizations {new}: classify each in CLASSIFIED; one that reads a declared registry "
        "value is a consumer reading both doors must take (registry consumer_reading), else both doors refuse "
        "a spelling its consumer runs")
    moved = sorted((site, CLASSIFIED[site][1], count) for site, count in sites.items()
                   if site in CLASSIFIED and CLASSIFIED[site][1] != count)
    assert not moved, (
        f"normalization calls added to or removed from classified scopes (site, pinned, found) {moved}: "
        "re-audit each call against the registry readings, then re-pin its count in CLASSIFIED")
    gone = sorted(set(CLASSIFIED) - set(sites))
    assert not gone, f"classified sites no longer in src/mimarsinan {gone}: remove them from CLASSIFIED"


def test_the_sites_that_read_a_declared_key_are_the_registry_readings():
    classes = {NESTED, DERIVED, OTHER}
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

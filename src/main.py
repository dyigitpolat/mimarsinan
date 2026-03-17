from init import *

from mimarsinan.common.reporter import DefaultReporter
from mimarsinan.pipelining.pipelines.deployment_pipeline import (
    DeploymentPipeline,
)
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
import mimarsinan.data_handling.data_providers

import sys
import json
import os
import threading


def _parse_deployment_config(deployment_config):
    """Parse deployment config dict into args for pipeline creation. Used by main() and run_pipeline_from_config."""
    data_provider_name = deployment_config['data_provider_name']
    seed = deployment_config.get("seed", 0)
    data_provider_factory = BasicDataProviderFactory(data_provider_name, "./datasets", seed=seed)

    deployment_name = deployment_config['experiment_name']
    deployment_parameters = dict(deployment_config['deployment_parameters'])

    platform_constraints_raw = deployment_config["platform_constraints"]
    if isinstance(platform_constraints_raw, dict) and "mode" in platform_constraints_raw:
        mode = platform_constraints_raw.get("mode", "user")
        if mode == "user":
            platform_constraints = platform_constraints_raw.get(
                "user",
                {k: v for k, v in platform_constraints_raw.items() if k != "mode"},
            )
        elif mode == "auto":
            auto = platform_constraints_raw.get("auto", {}) or {}
            fixed = auto.get("fixed", {}) or {}
            search_space = auto.get("search_space", {}) or {}
            arch_cfg = deployment_parameters.setdefault("arch_search", {})
            for k, v in search_space.items():
                arch_cfg.setdefault(k, v)
            platform_constraints = fixed
        else:
            raise ValueError(f"Invalid platform_constraints.mode: {mode}")
    else:
        platform_constraints = platform_constraints_raw

    pipeline_mode = deployment_config.get("pipeline_mode", "phased")
    working_directory = (
        deployment_config['generated_files_path'] + "/"
        + deployment_name + "_" + pipeline_mode + "_deployment_run"
    )
    os.makedirs(working_directory + "/_RUN_CONFIG", exist_ok=True)
    with open(working_directory + "/_RUN_CONFIG/config.json", 'w') as f:
        json.dump(deployment_config, f, indent=4)

    return {
        "pipeline_mode": pipeline_mode,
        "data_provider_factory": data_provider_factory,
        "deployment_name": deployment_name,
        "platform_constraints": platform_constraints,
        "deployment_parameters": deployment_parameters,
        "working_directory": working_directory,
        "start_step": deployment_config.get("start_step"),
        "stop_step": deployment_config.get("stop_step"),
        "target_metric_override": deployment_config.get("target_metric_override"),
    }


def run_pipeline_from_config(deployment_config, collector, gui_port=8501):
    """Create pipeline from config dict, attach collector and hooks, run pipeline in a background thread.

    Used by the GUI when user clicks RUN in the wizard. Returns immediately; pipeline runs in thread.
    """
    parsed = _parse_deployment_config(deployment_config)
    pipeline_mode = parsed["pipeline_mode"]
    deployment_parameters = parsed["deployment_parameters"]
    DeploymentPipeline.apply_preset(pipeline_mode, deployment_parameters)

    reporter = DefaultReporter()
    pipeline = DeploymentPipeline(
        data_provider_factory=parsed["data_provider_factory"],
        deployment_parameters=deployment_parameters,
        platform_constraints=parsed["platform_constraints"],
        reporter=reporter,
        working_directory=parsed["working_directory"],
    )

    resolved_start_step = None
    if parsed["start_step"] is not None:
        try:
            resolved_start_step = pipeline.get_resolved_start_step(parsed["start_step"])
        except Exception:
            resolved_start_step = parsed["start_step"]

    from mimarsinan.gui import GUIHandle
    from mimarsinan.gui import _make_json_safe
    from mimarsinan.gui.composite_reporter import CompositeReporter

    gui = GUIHandle(pipeline, collector)
    pipeline.reporter = CompositeReporter([reporter, gui.reporter])
    pipeline.register_pre_step_hook(gui.on_step_start)
    pipeline.register_post_step_hook(gui.on_step_end)

    safe_config = _make_json_safe(pipeline.config)
    collector.set_pipeline_info([name for name, _ in pipeline.steps], safe_config)

    if parsed["target_metric_override"] is not None:
        pipeline.set_target_metric(parsed["target_metric_override"])

    def _run():
        try:
            if resolved_start_step is None:
                pipeline.run(stop_step=parsed["stop_step"])
            else:
                pipeline.run_from(step_name=resolved_start_step, stop_step=parsed["stop_step"])
        finally:
            try:
                reporter.finish()
            except Exception:
                pass
            finally:
                collector.set_pipeline_thread(None)

    thread = threading.Thread(target=_run, daemon=False, name="pipeline-run")
    collector.set_pipeline_thread(thread)
    thread.start()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <deployment_config_json>")
        exit(1)

    deployment_config_path = sys.argv[1]
    with open(deployment_config_path, 'r') as f:
        deployment_config = json.load(f)

    parsed = _parse_deployment_config(deployment_config)
    run_pipeline(
        pipeline_mode=parsed["pipeline_mode"],
        data_provider_factory=parsed["data_provider_factory"],
        deployment_name=parsed["deployment_name"],
        platform_constraints=parsed["platform_constraints"],
        deployment_parameters=parsed["deployment_parameters"],
        working_directory=parsed["working_directory"],
        start_step=parsed["start_step"],
        stop_step=parsed["stop_step"],
        target_metric_override=parsed["target_metric_override"],
    )

def run_pipeline(
    pipeline_mode,
    data_provider_factory, 
    deployment_name, 
    platform_constraints, 
    deployment_parameters, 
    working_directory,
    start_step = None,
    stop_step = None,
    target_metric_override = None,
    gui_port = 8501):

    # Merge pipeline_mode preset into a copy of deployment_parameters
    # so that explicit user values always win over preset defaults.
    merged_params = dict(deployment_parameters)
    DeploymentPipeline.apply_preset(pipeline_mode, merged_params)

    reporter = DefaultReporter()

    pipeline = DeploymentPipeline(
        data_provider_factory=data_provider_factory,
        deployment_parameters=merged_params,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory,
    )

    # Resolve actual start step (may be earlier if dependencies are missing)
    resolved_start_step = None
    if start_step is not None:
        try:
            resolved_start_step = pipeline.get_resolved_start_step(start_step)
        except Exception:
            resolved_start_step = start_step

    # Start the browser-based monitoring GUI
    gui_started = False
    try:
        from mimarsinan.gui import start_gui
        from mimarsinan.gui.composite_reporter import CompositeReporter

        gui = start_gui(pipeline, port=gui_port, start_step=resolved_start_step)
        pipeline.reporter = CompositeReporter([reporter, gui.reporter])
        pipeline.register_pre_step_hook(gui.on_step_start)
        pipeline.register_post_step_hook(gui.on_step_end)
        gui_started = True
    except Exception as e:
        print(f"[GUI] Failed to start monitoring GUI (non-fatal): {e}")

    if target_metric_override is not None:
        pipeline.set_target_metric(target_metric_override)
    
    if start_step is None:
        pipeline.run(stop_step=stop_step)
    else:
        pipeline.run_from(step_name=start_step, stop_step=stop_step)

    # Finish reporter (e.g. flush / no-op)
    try:
        reporter.finish()
    except Exception:
        pass  # Non-fatal

    # Keep the process (and GUI server) alive until user confirms exit
    if gui_started:
        print("\n────────────────────────────────────────")
        print("Pipeline complete. GUI server still running.")
        print("Press Enter to exit and close the GUI...")
        print("────────────────────────────────────────")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass  # Graceful exit on Ctrl+C or EOF

if __name__ == "__main__":
    init()
    main()

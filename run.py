import sys
sys.path.append('./src')

# `--debug` must take effect before any CUDA context is created. Strip the
# flag from argv here (before importing mimarsinan) and set the env vars.
_DEBUG_FLAG = "--debug"
DEBUG_ENABLED = _DEBUG_FLAG in sys.argv
if DEBUG_ENABLED:
    sys.argv = [a for a in sys.argv if a != _DEBUG_FLAG]
    from mimarsinan.common.diagnostics import enable_cuda_debug
    enable_cuda_debug()

from src.init import init
from src.main import main, run_pipeline_from_config


def _run_headless(config_path: str) -> None:
    """Run a pipeline headlessly with file-based monitoring (no GUI server).

    Used by ProcessManager to spawn isolated pipeline processes.
    Writes run_info.json, steps.json, and live_metrics.jsonl to _GUI_STATE/.
    Force-exits the process on completion or SIGTERM to avoid lingering threads.
    """
    import json
    import os
    import signal
    from src.main import _parse_deployment_config

    with open(config_path, 'r') as f:
        deployment_config = json.load(f)

    if DEBUG_ENABLED:
        deployment_config.setdefault("deployment_parameters", {})["cuda_debug"] = True

    parsed = _parse_deployment_config(deployment_config)
    working_dir = parsed["working_directory"]

    from mimarsinan.pipelining.pipelines.deployment_pipeline import DeploymentPipeline
    from mimarsinan.common.reporter import DefaultReporter
    from mimarsinan.gui.data_collector import DataCollector
    from mimarsinan.gui import GUIHandle, _make_json_safe, _backfill_skipped_steps
    from mimarsinan.gui.composite_reporter import CompositeReporter
    from mimarsinan.gui.persistence import save_run_info, update_run_status

    def _sigterm_handler(_signum, _frame):
        try:
            update_run_status(working_dir, "stopped")
        except Exception:
            pass
        os._exit(143)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    deployment_parameters = parsed["deployment_parameters"]
    DeploymentPipeline.apply_preset(parsed["pipeline_mode"], deployment_parameters)

    reporter = DefaultReporter()
    pipeline = DeploymentPipeline(
        data_provider_factory=parsed["data_provider_factory"],
        deployment_parameters=deployment_parameters,
        platform_constraints=parsed["platform_constraints"],
        reporter=reporter,
        working_directory=working_dir,
    )

    resolved_start_step = None
    if parsed["start_step"] is not None:
        try:
            resolved_start_step = pipeline.get_resolved_start_step(parsed["start_step"])
        except Exception:
            resolved_start_step = parsed["start_step"]

    collector = DataCollector()
    gui = GUIHandle(pipeline, collector, persist_metrics=True, capture_stdio=False)
    collector._metric_callback = gui.on_metric
    pipeline.reporter = CompositeReporter([reporter, gui.reporter])
    pipeline.register_pre_step_hook(gui.on_step_start)
    pipeline.register_post_step_hook(gui.on_step_end)

    step_names = [name for name, _ in pipeline.steps]
    safe_config = _make_json_safe(pipeline.config)
    collector.set_pipeline_info(step_names, safe_config)

    save_run_info(working_dir, os.getpid(), step_names, {
        "experiment_name": parsed["deployment_name"],
        "pipeline_mode": parsed["pipeline_mode"],
    })

    if resolved_start_step is not None:
        _backfill_skipped_steps(pipeline, collector, step_names, resolved_start_step)

    if parsed["target_metric_override"] is not None:
        pipeline.set_target_metric(parsed["target_metric_override"])

    exit_code = 0
    try:
        if resolved_start_step is None:
            pipeline.run(stop_step=parsed["stop_step"])
        else:
            pipeline.run_from(step_name=resolved_start_step, stop_step=parsed["stop_step"])
        update_run_status(working_dir, "completed")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        update_run_status(working_dir, "failed", error=str(e))
        exit_code = 1
    finally:
        try:
            reporter.finish()
        except Exception:
            pass
        try:
            gui.restore_streams()
        except Exception:
            pass
        os._exit(exit_code)


if __name__ == "__main__":
    init()
    if len(sys.argv) >= 2 and sys.argv[1] == "--ui":
        from mimarsinan.gui.data_collector import DataCollector
        from mimarsinan.gui.server import start_server

        collector = DataCollector()
        start_server(collector, run_config_fn=run_pipeline_from_config)
        try:
            input("Press Enter to exit...\n")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            if not collector.join_pipeline_thread(timeout=60.0):
                print("Pipeline still running; exiting after timeout.")
    elif len(sys.argv) >= 3 and sys.argv[1] == "--headless":
        _run_headless(sys.argv[2])
    else:
        main()

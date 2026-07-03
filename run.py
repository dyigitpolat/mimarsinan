import os
import sys

sys.path.append('./src')
sys.path.append('./spikingjelly')

# cuBLAS float32 matmuls must use one reduction order across launches: the
# spiking forward's ceil(S*(1-V/θ)) flips on near-boundary neurons otherwise
# (~3 pp accuracy drift). Must be set before any CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# --debug must take effect before any CUDA context; strip it before imports.
_DEBUG_FLAG = "--debug"
DEBUG_ENABLED = _DEBUG_FLAG in sys.argv
if DEBUG_ENABLED:
    sys.argv = [a for a in sys.argv if a != _DEBUG_FLAG]
    from mimarsinan.common.diagnostics import enable_cuda_debug
    enable_cuda_debug()

from src.init import init
from src.main import main, run_pipeline_from_config


def _run_headless(config_path: str) -> None:
    """Run a config headlessly with file-based monitoring, then hard-exit."""
    import json
    import signal

    from mimarsinan.common.best_effort import best_effort
    from mimarsinan.gui import GUIHandle, backfill_skipped_steps, to_json_safe
    from mimarsinan.gui.resources import ResourceStore
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.persistence import save_run_info, update_run_status
    from mimarsinan.model_training.weight_loading import UnsupportedPreloadError
    from mimarsinan.pipelining.session import PipelineSession

    with open(config_path, 'r') as f:
        deployment_config = json.load(f)
    if DEBUG_ENABLED:
        deployment_config.setdefault("deployment_parameters", {})["cuda_debug"] = True

    session = PipelineSession.from_config(deployment_config)
    working_dir = session.parsed.working_directory

    def _sigterm_handler(_signum, _frame):
        with best_effort("record stopped status"):
            update_run_status(working_dir, "stopped")
        os._exit(143)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    collector = DataCollector()
    collector.set_resource_store(ResourceStore())
    gui = GUIHandle(session.pipeline, collector, persist_metrics=True, capture_stdio=False)
    collector.set_metric_callback(gui.on_metric)
    session.attach_gui(gui)

    step_names = [name for name, _ in session.pipeline.steps]
    collector.set_pipeline_info(step_names, to_json_safe(session.pipeline.config))
    save_run_info(working_dir, os.getpid(), step_names, {
        "experiment_name": session.parsed.deployment_name,
        "pipeline_mode": session.parsed.pipeline_mode,
    })

    start_step = session.resolved_start_step()
    if start_step is not None:
        backfill_skipped_steps(session.pipeline, collector, step_names, start_step)

    exit_code = 0
    try:
        session.run()
        update_run_status(working_dir, "completed")
    except UnsupportedPreloadError as e:
        # An ill-posed pretrained arm is a CLEAN campaign skip, not a failure.
        sys.stderr.write(f"[run] UNSUPPORTED preload, skipping cleanly: {e}\n")
        sys.stderr.flush()
        update_run_status(working_dir, "skipped", error=f"UNSUPPORTED_PRELOAD: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        update_run_status(working_dir, "failed", error=str(e))
        exit_code = 1
    finally:
        session.finish()
        with best_effort("drain and shut down GUI snapshots"):
            # Snapshot rendering can exceed a minute on the largest pipelines;
            # a short timeout would silently truncate monitor-UI resources.
            if not gui.wait_snapshots_idle(timeout=600.0):
                sys.stderr.write(
                    "[run] WARNING: snapshot executor did not drain within "
                    "600 s; some monitor-UI resources may be missing.\n"
                )
                sys.stderr.flush()
            gui.shutdown()
        with best_effort("restore stdio streams"):
            gui.restore_streams()
        os._exit(exit_code)


if __name__ == "__main__":
    init()
    if len(sys.argv) >= 2 and sys.argv[1] == "--ui":
        from mimarsinan.gui.runtime.collector import DataCollector
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

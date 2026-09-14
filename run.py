import os
import sys

sys.path.append('./src')

# cuBLAS float32 matmuls must use one reduction order across launches: the
# spiking forward's ceil(S*(1-V/θ)) flips on near-boundary neurons otherwise
# (~3 pp accuracy drift). Must be set before any CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Cycle-accurate spiking evals alternate large/small allocations (adaptive
# chunking): the default caching allocator fragments (measured 42 GiB reserved
# -but-unallocated while a 9 GiB ask failed). Must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --debug must take effect before any CUDA context; strip it before imports.
_DEBUG_FLAG = "--debug"
DEBUG_ENABLED = _DEBUG_FLAG in sys.argv
if DEBUG_ENABLED:
    sys.argv = [a for a in sys.argv if a != _DEBUG_FLAG]
    from mimarsinan.common.diagnostics import enable_cuda_debug
    enable_cuda_debug()

from mimarsinan.common.lifecycle.exit_contract import exit_process, install_exit_contract
from src.init import init
from src.main import main, run_pipeline_from_config

# A headless run renders nothing, so this drains source writes -- seconds, not
# the minutes rendering used to cost -- but the cap stays generous because a
# short budget would silently truncate monitor-UI resources rather than delay
# them. Children are already reaped by the time this runs, so the wait costs
# latency, never a leak.
SNAPSHOT_DRAIN_BUDGET_S = 600.0
SNAPSHOT_HEARTBEAT_S = 30.0


def _drain_gui(gui) -> None:
    """Persist outstanding monitor snapshots, then release the GUI's stdio."""
    import time

    from mimarsinan.common.best_effort import best_effort

    with best_effort("drain and shut down GUI snapshots"):
        deadline = time.monotonic() + SNAPSHOT_DRAIN_BUDGET_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                sys.stderr.write(
                    f"[run] WARNING: snapshot executor did not drain within "
                    f"{SNAPSHOT_DRAIN_BUDGET_S:.0f} s; some monitor-UI resources "
                    "may be missing.\n"
                )
                sys.stderr.flush()
                break
            if gui.wait_snapshots_idle(timeout=min(SNAPSHOT_HEARTBEAT_S, remaining)):
                break
            waited = SNAPSHOT_DRAIN_BUDGET_S - remaining
            sys.stderr.write(
                f"[run] still persisting monitor snapshots ({waited:.0f} s of "
                f"{SNAPSHOT_DRAIN_BUDGET_S:.0f} s)\n"
            )
            sys.stderr.flush()
        gui.shutdown()
    with best_effort("restore stdio streams"):
        gui.restore_streams()


def _run_headless(config_path: str) -> None:
    """Run a config headlessly with file-based monitoring, then hard-exit."""
    import json

    from mimarsinan.gui import GUIHandle, backfill_skipped_steps, to_json_safe
    from mimarsinan.gui.resources import ResourceRenderPolicy, ResourceStore
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

    def _record_stopped(_signum: int) -> None:
        update_run_status(working_dir, "stopped")

    install_exit_contract(on_terminate=_record_stopped)

    collector = DataCollector()
    collector.set_resource_store(ResourceStore())
    # Nobody is watching a headless run, so it persists resource SOURCES and
    # renders nothing: the render backlog is what used to keep the process --
    # and, under a scheduler, its whole node -- alive for minutes past its last
    # step. Whoever opens the run later renders from the source data.
    gui = GUIHandle(
        session.pipeline, collector,
        persist_metrics=True, capture_stdio=False,
        render_policy=ResourceRenderPolicy.DEFERRED,
    )
    collector.set_metric_callback(gui.on_metric)
    collector.set_event_callback(gui.on_event)
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
        # Reaping first is load-bearing: until it runs, leftover forkserver and
        # dataloader processes hold duplicates of our stdout/stderr, so pipe-EOF
        # waiters (run_tier, sbatch) hang and any SIGKILL during the drain
        # orphans the whole cohort.
        exit_process(exit_code, teardown=lambda: _drain_gui(gui))


def _run_ui() -> None:
    """Serve the wizard GUI until the operator ends the session."""
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.server import start_server

    collector = DataCollector()
    try:
        start_server(collector, run_config_fn=run_pipeline_from_config)
        try:
            input("Press Enter to exit...\n")
        except (KeyboardInterrupt, EOFError):
            pass
    finally:
        if not collector.join_pipeline_thread(timeout=60.0):
            print("Pipeline still running; exiting after timeout.")


if __name__ == "__main__":
    # Installed before anything can spawn a child, so no termination path -- and
    # no failure above -- can reach an exit that skips the reap.
    install_exit_contract()
    code = 0
    try:
        init()
        if len(sys.argv) >= 2 and sys.argv[1] == "--ui":
            _run_ui()
        elif len(sys.argv) >= 3 and sys.argv[1] == "--headless":
            _run_headless(sys.argv[2])
        else:
            main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    exit_process(code)

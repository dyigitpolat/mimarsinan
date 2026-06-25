import os
import sys
sys.path.append('./src')
# Spikingjelly is consumed as a vendored source tree.
sys.path.append('./spikingjelly')
# Lava is installed as ``lava-nc`` from PyPI in the env310 venv; no
# sys.path injection. (The repo includes a ``lava/`` submodule for
# reference / patches, but the runtime resolves ``import lava.*`` from
# site-packages.)

# cuBLAS sgemm picks a parallel-reduction order per launch based on workspace
# availability; without this env var the float32 matmul in
# ``SpikingUnifiedCoreFlow._forward_ttfs_quantized`` accumulates in slightly
# different orders across calls, flipping ``ceil(S*(1-V/θ))`` on near-boundary
# neurons and producing ~3 pp accuracy drift. Setting ``:4096:8`` pins
# cuBLAS to a stable reduction path with no measurable wall-time cost on
# modern GPUs. Must be set before any CUDA context (hence before
# ``enable_cuda_debug`` below and before any ``torch`` import).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline
    from mimarsinan.common.reporter import DefaultReporter
    from mimarsinan.gui import GUIHandle, backfill_skipped_steps, to_json_safe
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.composite_reporter import CompositeReporter
    from mimarsinan.gui.runtime.persistence import save_run_info, update_run_status

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

    from mimarsinan.gui.resources import ResourceStore

    collector = DataCollector()
    collector.set_resource_store(ResourceStore())
    gui = GUIHandle(pipeline, collector, persist_metrics=True, capture_stdio=False)
    collector._metric_callback = gui.on_metric
    pipeline.reporter = CompositeReporter([reporter, gui.reporter])
    pipeline.register_pre_step_hook(gui.on_step_start)
    pipeline.register_post_step_hook(gui.on_step_end)

    step_names = [name for name, _ in pipeline.steps]
    safe_config = to_json_safe(pipeline.config)
    collector.set_pipeline_info(step_names, safe_config)

    save_run_info(working_dir, os.getpid(), step_names, {
        "experiment_name": parsed["deployment_name"],
        "pipeline_mode": parsed["pipeline_mode"],
    })

    if resolved_start_step is not None:
        backfill_skipped_steps(pipeline, collector, step_names, resolved_start_step)

    if parsed["target_metric_override"] is not None:
        pipeline.set_target_metric(parsed["target_metric_override"])

    from mimarsinan.model_training.weight_loading import UnsupportedPreloadError

    exit_code = 0
    try:
        if resolved_start_step is None:
            pipeline.run(stop_step=parsed["stop_step"])
        else:
            pipeline.run_from(step_name=resolved_start_step, stop_step=parsed["stop_step"])
        update_run_status(working_dir, "completed")
    except UnsupportedPreloadError as e:
        # Ill-posed pretrained arm (no pretrained source for this builder): a CLEAN
        # skip, not a failure — record UNSUPPORTED and exit 0 so the campaign ledgers
        # it as a skip instead of an opaque rc=1.
        sys.stderr.write(f"[run] UNSUPPORTED preload, skipping cleanly: {e}\n")
        sys.stderr.flush()
        update_run_status(working_dir, "skipped", error=f"UNSUPPORTED_PRELOAD: {e}")
        exit_code = 0
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
            # Drain the snapshot executor so all pending step_completed
            # broadcasts and steps.json persistence flush before the
            # hard _exit below (which skips atexit handlers).
            #
            # Budget rationale: a single soft-core IR heatmap render is
            # ~30 ms of matplotlib + PNG encoding; a typical model has
            # ~600 cores × 2 heatmaps (post + pre-pruning) plus per-bank
            # heatmaps and per-hard-core heatmaps. Even with the IR-graph
            # de-duplication on the Hardware tab (see
            # ``snapshot_ir_graph(source_step_name=...)``) this can still
            # take well over a minute on the largest pipelines. A short
            # timeout here silently truncates the on-disk resource folder
            # and shows missing-image icons in the monitor UI for
            # historical runs, so we err on the side of waiting.
            drained = gui.wait_snapshots_idle(timeout=600.0)
            if not drained:
                import sys as _sys
                _sys.stderr.write(
                    "[run] WARNING: snapshot executor did not drain within "
                    "600 s; some monitor-UI resources may be missing. "
                    "Consider profiling resource persistence (see "
                    "GUIHandle._persist_resources).\n"
                )
                _sys.stderr.flush()
            gui.shutdown()
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

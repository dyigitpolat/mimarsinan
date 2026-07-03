"""CLI and GUI frontends over the PipelineSession composition root."""

from init import init

import json
import sys
import threading
from pathlib import Path


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


_load_project_dotenv()

from mimarsinan.pipelining.session import PipelineSession


def run_pipeline_from_config(deployment_config, collector, gui_port=8501):
    """Run a config in a background thread with the wizard GUI attached."""
    from mimarsinan.gui import GUIHandle, to_json_safe

    session = PipelineSession.from_config(deployment_config)
    gui = GUIHandle(session.pipeline, collector)
    session.attach_gui(gui)

    collector.set_pipeline_info(
        [name for name, _ in session.pipeline.steps],
        to_json_safe(session.pipeline.config),
    )

    def _run():
        try:
            session.run()
        finally:
            session.finish()
            collector.set_pipeline_thread(None)

    thread = threading.Thread(target=_run, daemon=False, name="pipeline-run")
    collector.set_pipeline_thread(thread)
    thread.start()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <deployment_config_json>")
        raise SystemExit(1)

    with open(sys.argv[1], "r") as f:
        deployment_config = json.load(f)

    session = PipelineSession.from_config(deployment_config)

    gui = None
    try:
        from mimarsinan.gui import start_gui

        gui = start_gui(
            session.pipeline, port=8501, start_step=session.resolved_start_step()
        )
        session.attach_gui(gui)
    except Exception as e:
        print(f"[GUI] Failed to start monitoring GUI (non-fatal): {e}")

    session.run()
    session.finish()

    if gui is not None:
        print("\n────────────────────────────────────────")
        print("Pipeline complete. GUI server still running.")
        print("Press Enter to exit and close the GUI...")
        print("────────────────────────────────────────")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    init()
    main()

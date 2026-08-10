"""A crashing step must leave an honest persisted failure state, not 'running'."""

import json
from pathlib import Path

import pytest

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep


class _FakePipeline:
    def __init__(self, working_dir):
        self.working_directory = str(working_dir)

    def get_target_metric(self):
        return 0.5


def _handle(tmp_path):
    collector = DataCollector()
    handle = GUIHandle(
        _FakePipeline(tmp_path), collector,
        persist_metrics=True, capture_stdio=False,
    )
    return handle, collector


def _persisted_steps(tmp_path):
    with open(Path(tmp_path) / "_GUI_STATE" / "steps.json", encoding="utf-8") as f:
        return json.load(f)["steps"]


class TestOnStepFailed:
    def test_persists_failed_status_and_error(self, tmp_path):
        handle, collector = _handle(tmp_path)
        handle.on_step_start("Train", object())
        handle.on_step_failed("Train", object(), RuntimeError("boom"))

        entry = _persisted_steps(tmp_path)["Train"]
        assert entry["status"] == "failed"
        assert entry["error"] == "boom"
        assert entry["end_time"] is not None
        assert entry["start_time"] is not None

    def test_collector_records_error_and_overview_shows_failed(self, tmp_path):
        handle, collector = _handle(tmp_path)
        collector.set_pipeline_info(["Train"], {})
        handle.on_step_start("Train", object())
        handle.on_step_failed("Train", object(), RuntimeError("boom"))

        detail = collector.get_step_detail("Train")
        assert detail is not None
        assert detail["status"] == "failed"
        assert detail["error"] == "boom"

        overview_step = collector.get_pipeline_overview()["steps"][0]
        assert overview_step["status"] == "failed"
        assert overview_step["error"] == "boom"


# ---------------------------------------------------------------------------
# Headless crash-run fixture: real engine + real GUIHandle on a tmp dir.
# ---------------------------------------------------------------------------

class _OkStep(PipelineStep):
    def __init__(self, pipeline):
        super().__init__(requires=[], promises=["data"], updates=[], clears=[], pipeline=pipeline)

    def process(self):
        self.add_entry("data", 1)

    def validate(self):
        return 1.0


class _CrashStep(PipelineStep):
    def __init__(self, pipeline):
        super().__init__(requires=["data"], promises=[], updates=[], clears=[], pipeline=pipeline)

    def process(self):
        _ = self.get_entry("data")
        raise RuntimeError("mid-step crash")

    def validate(self):
        return 1.0


class TestHeadlessCrashRunFixture:
    def test_crash_persists_the_failure_state_machine(self, tmp_path):
        working_dir = str(tmp_path / "run")
        pipeline = Pipeline(working_dir)
        ok = _OkStep(pipeline)
        crash = _CrashStep(pipeline)
        pipeline.add_pipeline_step("Build", ok)
        pipeline.add_pipeline_step("Train", crash)

        collector = DataCollector()
        collector.set_pipeline_info(["Build", "Train"], {})
        gui = GUIHandle(pipeline, collector, persist_metrics=True, capture_stdio=False)
        # Mirror PipelineSession.attach_gui's hook wiring.
        pipeline.register_pre_step_hook(gui.on_step_start)
        pipeline.register_post_step_hook(gui.on_step_end)
        pipeline.register_step_failed_hook(gui.on_step_failed)

        with pytest.raises(RuntimeError, match="mid-step crash"):
            pipeline.run()

        steps = _persisted_steps(working_dir)
        assert steps["Build"]["status"] == "completed"
        assert steps["Train"]["status"] == "failed"
        assert "mid-step crash" in steps["Train"]["error"]
        assert steps["Train"]["end_time"] is not None

        overview = {s["name"]: s for s in collector.get_pipeline_overview()["steps"]}
        assert overview["Build"]["status"] == "completed"
        assert overview["Train"]["status"] == "failed"

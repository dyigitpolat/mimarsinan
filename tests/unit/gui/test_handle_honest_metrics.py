"""GUIHandle persists honest step semantics: metric_kind + verdict, never fabricated."""

import json
from pathlib import Path

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.runtime.collector import DataCollector


class _FakeStep:
    """A verdict-only gate: pipeline metric is the carried previous value."""

    def pipeline_metric_kind(self):
        return "carried"

    def step_verdict(self):
        return {"status": "pass", "rule": "spike parity", "detail": {"samples": 3}}


class _FakeMeasuredStep:
    def pipeline_metric_kind(self):
        return "measured"

    def step_verdict(self):
        return None


class _FakePipeline:
    def __init__(self, working_dir):
        self.working_directory = str(working_dir)

    def get_target_metric(self):
        return 0.91


def _handle(tmp_path):
    collector = DataCollector()
    pipeline = _FakePipeline(tmp_path)
    handle = GUIHandle(pipeline, collector, persist_metrics=True, capture_stdio=False)
    collector.set_event_callback(handle.on_event)
    return handle, collector


def _persisted_steps(tmp_path):
    with open(Path(tmp_path) / "_GUI_STATE" / "steps.json", encoding="utf-8") as f:
        return json.load(f)["steps"]


def test_gate_step_records_carried_kind_and_verdict(tmp_path):
    handle, collector = _handle(tmp_path)
    handle.on_step_start("Loihi Simulation", _FakeStep())
    handle.on_step_end("Loihi Simulation", _FakeStep())

    entry = _persisted_steps(tmp_path)["Loihi Simulation"]
    assert entry["metric_kind"] == "carried"
    assert entry["verdict"]["status"] == "pass"
    assert entry["target_metric"] == 0.91  # still recorded, but labeled carried

    detail = collector.get_step_detail("Loihi Simulation")
    assert detail["metric_kind"] == "carried"
    assert detail["verdict"]["rule"] == "spike parity"

    overview_step = collector.get_pipeline_overview()["steps"][0]
    assert overview_step["metric_kind"] == "carried"
    assert overview_step["verdict"]["status"] == "pass"


def test_measured_step_records_measured_kind(tmp_path):
    handle, collector = _handle(tmp_path)
    handle.on_step_start("Pretraining", _FakeMeasuredStep())
    handle.on_step_end("Pretraining", _FakeMeasuredStep())
    entry = _persisted_steps(tmp_path)["Pretraining"]
    assert entry["metric_kind"] == "measured"
    assert "verdict" not in entry


def test_events_persist_via_the_collector_callback(tmp_path):
    handle, collector = _handle(tmp_path)
    collector.step_started("WQ")
    collector.record_event("mbh_endpoint", {"reached": True})
    events_file = Path(tmp_path) / "_GUI_STATE" / "events.jsonl"
    assert events_file.exists()
    record = json.loads(events_file.read_text().strip())
    assert record["kind"] == "mbh_endpoint"
    assert record["step"] == "WQ"


def test_old_readers_ignore_the_additive_fields(tmp_path):
    """No-regression: the steps.json shape stays a superset of the old one."""
    handle, _ = _handle(tmp_path)
    handle.on_step_start("Loihi Simulation", _FakeStep())
    handle.on_step_end("Loihi Simulation", _FakeStep())
    entry = _persisted_steps(tmp_path)["Loihi Simulation"]
    for legacy_key in ("start_time", "end_time", "target_metric", "metrics",
                       "snapshot", "snapshot_key_kinds", "status"):
        assert legacy_key in entry

"""Structured event channel: record, broadcast, persist, load, tail, backfill."""

import json
import time
from pathlib import Path

from mimarsinan.common.reporter import DefaultReporter, emit_reporter_event
from mimarsinan.gui.runtime.events import PipelineEvent
from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.runtime.active_run_tailers import events_tailer
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.composite_reporter import CompositeReporter
from mimarsinan.gui.runtime.persistence import append_event, load_events
from mimarsinan.gui.snapshot.console_events import parse_console_events


class TestCollectorEvents:
    def test_record_event_round_trip_with_step_attribution(self):
        collector = DataCollector()
        collector.step_started("LIF Adaptation")
        collector.record_event("mbh_gate", {"action": "accept", "rung": 3})
        events = collector.get_events()
        assert len(events) == 1
        assert events[0]["step"] == "LIF Adaptation"
        assert events[0]["kind"] == "mbh_gate"
        assert events[0]["payload"]["action"] == "accept"
        assert events[0]["seq"] == 1

    def test_since_seq_and_step_filters(self):
        collector = DataCollector()
        collector.step_started("A")
        collector.record_event("profile", {"n": 1})
        collector.step_completed("A")
        collector.step_started("B")
        collector.record_event("profile", {"n": 2})
        assert [e["payload"]["n"] for e in collector.get_events(since_seq=1)] == [2]
        assert [e["payload"]["n"] for e in collector.get_events(step_name="B")] == [2]

    def test_event_callback_receives_the_event(self):
        collector = DataCollector()
        seen = []
        collector.set_event_callback(seen.append)
        collector.record_event("lr_refusal", {"tuner": "X"})
        assert len(seen) == 1
        assert isinstance(seen[0], PipelineEvent)
        assert seen[0].kind == "lr_refusal"


class TestReporterFanout:
    def test_gui_reporter_forwards_to_collector(self):
        collector = DataCollector()
        GUIReporter(collector).event("parity", {"agreement": 0.99})
        assert collector.get_events()[0]["kind"] == "parity"

    def test_composite_forwards_and_tolerates_legacy_children(self):
        collector = DataCollector()

        class _Legacy:
            prefix = ""

            def report(self, *a, **k): ...
            def console_log(self, *a, **k): ...

        composite = CompositeReporter([_Legacy(), GUIReporter(collector)])
        composite.event("mbh_hop", {"action": "reaffine"})
        assert collector.get_events()[0]["kind"] == "mbh_hop"

    def test_emit_helper_tolerates_pre_event_reporters(self):
        class _Legacy:
            pass

        emit_reporter_event(_Legacy(), "profile", {})  # must not raise
        collector = DataCollector()
        emit_reporter_event(GUIReporter(collector), "profile", {"wall_s": 1.0})
        assert collector.get_events()[0]["payload"]["wall_s"] == 1.0

    def test_default_reporter_event_is_a_noop(self):
        DefaultReporter().event("profile", {})


class TestPersistence:
    def test_append_and_load_round_trip(self, tmp_path):
        record = {"seq": 1, "step": "WQ", "kind": "mbh_endpoint",
                  "payload": {"reached": True}, "timestamp": time.time()}
        append_event(str(tmp_path), record)
        append_event(str(tmp_path), {**record, "seq": 2})
        loaded = load_events(str(tmp_path))
        assert [e["seq"] for e in loaded] == [1, 2]
        assert load_events(str(tmp_path), since_seq=1)[0]["seq"] == 2
        assert load_events(str(tmp_path), step_name="nope") == []

    def test_events_tailer_emits_typed_frames(self, tmp_path):
        path = Path(tmp_path) / "events.jsonl"
        frames = []
        tailer = events_tailer(path, frames.append)
        try:
            tailer.start()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"seq": 1, "kind": "profile", "payload": {}}) + "\n")
            deadline = time.time() + 3.0
            while not frames and time.time() < deadline:
                time.sleep(0.02)
        finally:
            tailer.stop()
        assert frames and frames[0]["type"] == "event"
        assert frames[0]["kind"] == "profile"


class TestConsoleBackfill:
    _LINES = [
        {"line": "[MBH-GATE] tuner=LIFAdaptationTuner entry best_full_acc=0.912345", "ts": 1.0},
        {"line": "[MBH-GATE] tuner=LIFAdaptationTuner accept rung=0 rate=0.500000 "
                 "full_acc=0.930000 best_full_acc=0.930000", "ts": 2.0},
        {"line": "[MBH-GATE] tuner=LIFAdaptationTuner reject rung=1 attempt=0 rate=1.000000 "
                 "full_acc=0.800000 best_full_acc=0.930000 retry_rate=0.750000", "ts": 3.0},
        {"line": "[MBH-GATE] constructive_stall committed=0.750000 best_full_acc=0.930000", "ts": 4.0},
        {"line": "[MBH-ENDPOINT] tuner=WeightQuantizationTuner target=0.930000 entry=0.910000 "
                 "exit=0.935000 budget=600 steps_used=420 engaged=True reached=True "
                 "rolled_back=False target_floor=0.980000 floor_lifted=True "
                 "entry_gap_armed=False", "ts": 5.0},
        {"line": "[PROFILE] step='Pretraining' wall=  12.30s metric=0.9500 Δ=+0.9500 (prev=0.0000)", "ts": 6.0},
        {"line": "[LR-REFUSE] every probed LR is destructive", "ts": 7.0},
        {"line": "plain console noise", "ts": 8.0},
    ]

    def test_golden_parse(self):
        events = parse_console_events(self._LINES)
        kinds = [e["kind"] for e in events]
        assert kinds == ["mbh_gate", "mbh_gate", "mbh_gate", "mbh_gate",
                         "mbh_endpoint", "profile", "lr_refusal"]
        entry, accept, reject, stall = events[:4]
        assert entry["payload"]["action"] == "entry"
        assert accept["payload"] == {
            "action": "accept", "tuner": "LIFAdaptationTuner", "rung": 0,
            "rate": 0.5, "full_acc": 0.93, "best_full_acc": 0.93,
        }
        assert reject["payload"]["retry_rate"] == 0.75
        assert stall["payload"]["action"] == "stall"
        endpoint = events[4]
        assert endpoint["payload"]["reached"] is True
        assert endpoint["payload"]["floor_lifted"] is True
        profile = events[5]
        assert profile["step"] == "Pretraining"
        assert profile["payload"]["wall_s"] == 12.3
        assert all(e["backfilled"] for e in events)

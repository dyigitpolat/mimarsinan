"""View-models: overview honesty, annotation lanes, staircase invariant, gantt, A6."""

import pytest

from mimarsinan.gui.viewmodel import (
    StaircaseMonotonicityError,
    annotations_for_step,
    build_a6_gauges,
    build_gantt,
    build_overview_chart,
    build_staircase,
    categories_for,
    decorate,
    highwater,
    step_bar_badge,
)


class TestOverviewChart:
    def test_measured_steps_plot_and_carried_steps_never_do(self):
        steps = [
            {"name": "Pretraining", "status": "completed", "target_metric": 0.95,
             "metric_kind": "measured"},
            {"name": "Model Building", "status": "completed", "target_metric": 0.95,
             "metric_kind": "carried"},
            {"name": "LIF Adaptation", "status": "completed", "target_metric": 0.93,
             "metric_kind": "measured"},
        ]
        chart = build_overview_chart(steps)
        assert [p["step"] for p in chart["points"]] == ["Pretraining", "LIF Adaptation"]
        # The carried step is NEVER a data point, but it is never invisible
        # either: it renders as a labeled neutral event line.
        carried = [m for m in chart["markers"] if m["status"] == "carried"]
        assert [m["step"] for m in carried] == ["Model Building"]
        assert "carried" in carried[0]["label"]

    def test_every_completed_step_is_visible_as_point_or_marker(self):
        steps = [
            {"name": "Model Configuration", "status": "completed",
             "target_metric": 0.0, "metric_kind": "carried"},
            {"name": "Pretraining", "status": "completed", "target_metric": 0.95,
             "metric_kind": "measured"},
            {"name": "Core Quantization Verification", "status": "completed",
             "target_metric": 0.95, "metric_kind": "carried",
             "verdict": {"status": "pass", "rule": "chip-quantized"}},
            {"name": "Loihi Simulation", "status": "failed", "error": "boom"},
        ]
        chart = build_overview_chart(steps)
        visible = {p["step"] for p in chart["points"]}
        visible |= {m["step"] for m in chart["markers"]}
        assert visible == {s["name"] for s in steps}

    def test_carried_step_with_verdict_keeps_the_verdict_marker_only(self):
        steps = [
            {"name": "Core Quantization Verification", "status": "completed",
             "target_metric": 0.95, "metric_kind": "carried",
             "verdict": {"status": "pass", "rule": "chip-quantized"}},
        ]
        chart = build_overview_chart(steps)
        assert len(chart["markers"]) == 1
        assert chart["markers"][0]["status"] == "pass"

    def test_verdict_steps_render_as_markers_with_glyphs(self):
        steps = [
            {"name": "Core Quantization Verification", "status": "completed",
             "target_metric": 0.93, "metric_kind": "carried",
             "verdict": {"status": "pass", "rule": "chip-quantized", "detail": {"cores": 4}}},
            {"name": "Loihi Simulation", "status": "failed", "error": "spike mismatch"},
        ]
        chart = build_overview_chart(steps)
        assert chart["points"] == []
        pass_marker, fail_marker = chart["markers"]
        assert pass_marker["glyph"] == "✓" and pass_marker["status"] == "pass"
        assert fail_marker["glyph"] == "✖" and fail_marker["status"] == "fail"
        assert "spike mismatch" in fail_marker["label"]

    def test_running_steps_contribute_nothing(self):
        chart = build_overview_chart([
            {"name": "Pretraining", "status": "running", "target_metric": 0.5,
             "metric_kind": "measured"},
        ])
        assert chart == {"points": [], "markers": []}

    def test_legacy_steps_without_kind_stay_plotted(self):
        chart = build_overview_chart([
            {"name": "Old Step", "status": "completed", "target_metric": 0.9},
        ])
        assert chart["points"] == [{"step": "Old Step", "value": 0.9}]

    def test_step_bar_badge(self):
        assert step_bar_badge({"verdict": {"status": "pass"}})["text"] == "PASS"
        assert step_bar_badge({"verdict": {"status": "fail"}})["text"] == "FAIL"
        assert step_bar_badge({"metric_kind": "carried"})["text"] == ""
        assert step_bar_badge({"target_metric": 0.9123})["text"] == "0.912"


class TestMetricCategories:
    def test_rule_table(self):
        cats = categories_for([
            "Training loss", "Validation accuracy", "Adaptation target",
            "lr", "LIF Adaptation tuning rate", "search_event", "misc",
        ])
        assert cats["Training loss"] == "Loss"
        assert cats["Validation accuracy"] == "Accuracy"
        assert cats["Adaptation target"] == "Accuracy"
        assert cats["lr"] == "Learning Rate"
        assert cats["LIF Adaptation tuning rate"] == "Adaptation"
        assert cats["misc"] == "Other"
        assert "search_event" not in cats


class TestAnnotations:
    def test_gate_events_annotate_adaptation_charts_with_elapsed_x(self):
        events = [
            {"step": "LIF Adaptation", "kind": "mbh_gate", "timestamp": 110.0,
             "payload": {"action": "accept", "rung": 2, "rate": 0.5}},
            {"step": "LIF Adaptation", "kind": "lr_refusal", "timestamp": 120.0,
             "payload": {"tuner": "X"}},
            {"step": "Other Step", "kind": "mbh_gate", "timestamp": 130.0,
             "payload": {"action": "reject"}},
        ]
        annotations = annotations_for_step(events, "LIF Adaptation", step_start=100.0)
        assert len(annotations) == 2
        accept, refusal = annotations
        assert accept["x"] == 10.0
        assert "rung 2" in accept["label"]
        assert accept["tone"] == "good"
        assert "Adaptation" in accept["categories"]
        assert refusal["tone"] == "bad"

    def test_decorate_attaches_display(self):
        record = decorate({"kind": "mbh_endpoint", "seq": 1,
                           "payload": {"reached": True, "steps_used": 420}})
        assert "reached" in record["display"]["label"]
        assert record["seq"] == 1

    def test_category_less_events_still_reach_the_step_timeline(self):
        """Events without chart categories (e.g. profile) are timeline
        entries — the annotation LANES filter by category client-side."""
        events = [
            {"step": "Pretraining", "kind": "profile", "timestamp": 105.0,
             "payload": {"step": "Pretraining", "wall_s": 5.0}},
        ]
        annotations = annotations_for_step(events, "Pretraining", step_start=100.0)
        assert len(annotations) == 1
        assert annotations[0]["categories"] == []
        assert annotations[0]["x"] == 5.0

    def test_annotations_carry_the_event_payload_for_step_insights(self):
        events = [
            {"step": "LIF Adaptation", "kind": "mbh_gate", "timestamp": 110.0,
             "payload": {"action": "accept", "rung": 2, "rate": 0.5,
                         "full_acc": 0.93, "best_full_acc": 0.93}},
        ]
        annotations = annotations_for_step(events, "LIF Adaptation", step_start=100.0)
        assert annotations[0]["payload"]["full_acc"] == 0.93
        assert annotations[0]["payload"]["rung"] == 2


class TestStaircase:
    _EVENTS = [
        {"kind": "mbh_gate", "step": "LIF Adaptation",
         "payload": {"tuner": "LIFAdaptationTuner", "action": "entry", "best_full_acc": 0.90}},
        {"kind": "mbh_gate", "step": "LIF Adaptation",
         "payload": {"tuner": "LIFAdaptationTuner", "action": "accept", "rung": 0,
                     "full_acc": 0.93, "best_full_acc": 0.93}},
        {"kind": "mbh_gate", "step": "LIF Adaptation",
         "payload": {"tuner": "LIFAdaptationTuner", "action": "reject", "rung": 1,
                     "attempt": 0, "full_acc": 0.80, "best_full_acc": 0.93}},
        {"kind": "mbh_gate", "step": "LIF Adaptation",
         "payload": {"tuner": "LIFAdaptationTuner", "action": "stall",
                     "committed": 0.75, "best_full_acc": 0.93}},
    ]

    def test_staircase_shape(self):
        lanes = build_staircase(self._EVENTS)["tuners"]
        assert len(lanes) == 1
        lane = lanes[0]
        assert lane["stalled"] is True
        assert [p["accepted"] for p in lane["probes"]] == [True, False]
        assert [p["best"] for p in lane["staircase"]] == [0.90, 0.93, 0.93]

    def test_monotonicity_property_raises_on_a_falling_ratchet(self):
        broken = list(self._EVENTS[:2]) + [
            {"kind": "mbh_gate", "step": "LIF Adaptation",
             "payload": {"tuner": "LIFAdaptationTuner", "action": "accept", "rung": 1,
                         "full_acc": 0.85, "best_full_acc": 0.85}},
        ]
        with pytest.raises(StaircaseMonotonicityError):
            build_staircase(broken)

    def test_highwater(self):
        assert highwater(self._EVENTS) == 0.93


class TestGantt:
    def test_rows_budget_and_simulator_split(self):
        steps = [
            {"name": "Pretraining", "status": "completed", "start_time": 100.0,
             "end_time": 160.0, "duration": 60.0, "semantic_group": "pretraining"},
            {"name": "SANA-FE Simulation", "status": "completed", "start_time": 160.0,
             "end_time": 190.0, "duration": 30.0, "semantic_group": "simulation"},
        ]
        events = [
            {"kind": "mbh_endpoint", "step": "Weight Quantization",
             "payload": {"tuner": "WQ", "steps_used": 420, "budget_steps": 600,
                         "engaged": True, "reached": True}},
        ]
        gantt = build_gantt(steps, events, config={"endpoint_floor_steps": 17200})
        assert gantt["total_wall_s"] == 90.0
        assert gantt["artifact_wall_s"] == 60.0
        assert gantt["rows"][1]["simulator"] is True
        assert gantt["rows"][0]["offset_s"] == 0.0
        budget = gantt["endpoint_budget"]
        assert budget["consumed_steps"] == 420
        assert budget["budget_steps"] == 17200
        assert budget["stages"][0]["reached"] is True


class TestA6Gauges:
    def test_latest_card_per_gauge_context(self):
        events = [
            {"kind": "mbh_a6", "step": "Activation Quantization",
             "payload": {"gauge": "value", "context": "AQ", "verdict": "FAIL",
                         "levels": 1.4}},
            {"kind": "mbh_a6", "step": "Activation Quantization",
             "payload": {"gauge": "value", "context": "AQ", "verdict": "PASS",
                         "levels": 2.3}},
            {"kind": "mbh_a6", "step": "LIF Adaptation",
             "payload": {"gauge": "temporal", "context": "LIF", "verdict": "PASS",
                         "window": 4}},
        ]
        cards = build_a6_gauges(events)["cards"]
        assert len(cards) == 2
        assert cards[0]["gauge"] == "value"
        assert cards[0]["verdict"] == "PASS"        # latest wins
        assert cards[0]["detail"]["levels"] == 2.3
        assert cards[1]["gauge"] == "temporal"

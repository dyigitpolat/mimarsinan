"""Unit tests for mimarsinan.gui.data_collector."""

import pytest

from mimarsinan.gui.data_collector import DataCollector


class TestStepStartedPurgesOldMetrics:
    """step_started purges old metrics for the step."""

    def test_step_started_purges_old_metrics(self):
        collector = DataCollector()
        collector.set_pipeline_info(["Step A", "Step B"], {})

        collector.step_started("Step A")
        collector.record_metric("accuracy", 0.5)
        collector.record_metric("loss", 0.1)

        metrics_before = collector.get_step_metrics("Step A")
        assert len(metrics_before) == 2

        collector.step_started("Step A")

        metrics_after = collector.get_step_metrics("Step A")
        assert len(metrics_after) == 0
        assert metrics_after == []


class TestStepStartedResetsStepRecordFields:
    """step_started resets end_time, target_metric, snapshot to None."""

    def test_step_started_resets_step_record_fields(self):
        collector = DataCollector()
        collector.set_pipeline_info(["Step A"], {})

        collector.step_started("Step A")
        collector.step_completed(
            "Step A",
            target_metric=0.99,
            snapshot={"layers": 3},
            snapshot_key_kinds={"layers": "new"},
        )

        detail_before = collector.get_step_detail("Step A")
        assert detail_before is not None
        assert detail_before["end_time"] is not None
        assert detail_before["target_metric"] == 0.99
        assert detail_before["snapshot"] == {"layers": 3}
        assert detail_before["snapshot_key_kinds"] == {"layers": "new"}

        collector.step_started("Step A")

        detail_after = collector.get_step_detail("Step A")
        assert detail_after is not None
        assert detail_after["end_time"] is None
        assert detail_after["target_metric"] is None
        assert detail_after["snapshot"] is None
        assert detail_after["snapshot_key_kinds"] is None


class TestMetricCallback:
    """_metric_callback is invoked when record_metric is called."""

    def test_metric_callback_receives_correct_args(self):
        collector = DataCollector()
        collector.set_pipeline_info(["Step A"], {})
        collector.step_started("Step A")

        received: list[tuple] = []

        def capture(step_name: str, metric_name: str, value, seq: int, timestamp: float):
            received.append((step_name, metric_name, value, seq, timestamp))

        collector._metric_callback = capture

        collector.record_metric("accuracy", 0.87, step=10)

        assert len(received) == 1
        step_name, metric_name, value, seq, timestamp = received[0]
        assert step_name == "Step A"
        assert metric_name == "accuracy"
        assert value == 0.87
        assert seq >= 1
        assert timestamp is not None and timestamp > 0


class TestBasicMetricAccumulation:
    """Basic metric accumulation and get_step_metrics."""

    def test_metric_accumulation_and_get_step_metrics(self):
        collector = DataCollector()
        collector.set_pipeline_info(["Step A"], {})
        collector.step_started("Step A")

        collector.record_metric("accuracy", 0.9)
        collector.record_metric("loss", 0.05)
        collector.record_metric("accuracy", 0.92)

        metrics = collector.get_step_metrics("Step A")
        assert len(metrics) == 3
        assert metrics[0]["name"] == "accuracy"
        assert metrics[0]["value"] == 0.9
        assert metrics[1]["name"] == "loss"
        assert metrics[1]["value"] == 0.05
        assert metrics[2]["name"] == "accuracy"
        assert metrics[2]["value"] == 0.92
        assert all("seq" in m and "timestamp" in m for m in metrics)


class TestStepCompletedSetsTargetMetricAndSnapshot:
    """step_completed sets target_metric and snapshot correctly."""

    def test_step_completed_sets_target_metric_and_snapshot(self):
        collector = DataCollector()
        collector.set_pipeline_info(["Step A"], {})
        collector.step_started("Step A")

        collector.step_completed(
            "Step A",
            target_metric=0.98,
            snapshot={"layers": 4, "params": 1000},
            snapshot_key_kinds={"layers": "new", "params": "edited"},
        )

        detail = collector.get_step_detail("Step A")
        assert detail is not None
        assert detail["target_metric"] == 0.98
        assert detail["snapshot"] == {"layers": 4, "params": 1000}
        assert detail["snapshot_key_kinds"] == {"layers": "new", "params": "edited"}
        assert detail["status"] == "completed"
        assert detail["end_time"] is not None

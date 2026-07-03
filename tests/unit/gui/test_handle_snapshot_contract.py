"""GUI step status must reflect pipeline success even when snapshot build fails."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.persistence.resource_paths import resource_disk_path


@pytest.fixture
def handle(monkeypatch):
    collector = DataCollector()
    pipeline = SimpleNamespace(
        get_target_metric=MagicMock(return_value=None),
        working_directory=None,
    )
    monkeypatch.setattr(
        "mimarsinan.gui.handle.build_step_snapshot",
        MagicMock(side_effect=RuntimeError("snapshot boom")),
    )
    return GUIHandle(pipeline, collector, capture_stdio=False), collector


def test_on_step_end_marks_completed_when_snapshot_raises(handle):
    gui, collector = handle
    step = SimpleNamespace()
    collector.step_started("Hard Core Mapping")
    gui.on_step_end("Hard Core Mapping", step)
    detail = collector.get_step_detail("Hard Core Mapping")
    assert detail is not None
    assert detail.get("status") == "completed"
    assert detail.get("snapshot", {}).get("snapshot_error") == "snapshot boom"


class TestPersistResourcesSkipsFailingProducers:
    """A resource producer raising must not crash the snapshot worker, must
    not write anything to disk for that resource, and must not prevent
    sibling resources in the same step from being persisted."""

    def test_failing_producer_is_skipped_others_still_persisted(self, tmp_path, monkeypatch):
        working_dir = str(tmp_path)
        good_payload = {"ok": True}

        def boom_producer():
            raise RuntimeError("resource producer boom")

        def good_producer():
            return good_payload

        descriptors = [
            ResourceDescriptor(kind="connectivity", rid="bad", producer=boom_producer, media_type="application/json"),
            ResourceDescriptor(kind="connectivity", rid="good", producer=good_producer, media_type="application/json"),
        ]
        monkeypatch.setattr(
            "mimarsinan.gui.handle.build_step_snapshot",
            MagicMock(return_value=({"step_name": "S"}, {}, descriptors)),
        )

        collector = DataCollector()
        pipeline = SimpleNamespace(
            get_target_metric=MagicMock(return_value=None),
            working_directory=working_dir,
        )
        gui = GUIHandle(pipeline, collector, capture_stdio=False)
        collector.step_started("S")

        gui.on_step_end("S", SimpleNamespace())  # must not raise
        assert gui.wait_snapshots_idle(timeout=2.0)

        bad_path = resource_disk_path(working_dir, "S", "connectivity", "bad", "application/json")
        good_path = resource_disk_path(working_dir, "S", "connectivity", "good", "application/json")
        assert not bad_path.exists()
        assert good_path.exists()

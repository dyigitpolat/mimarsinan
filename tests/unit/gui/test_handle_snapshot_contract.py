"""GUI step status must reflect pipeline success even when snapshot build fails."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.runtime.collector import DataCollector


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

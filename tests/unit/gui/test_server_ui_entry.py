"""Tests for GUI server welcome URL and browser-open scheduling."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mimarsinan.gui.server.app import gui_entry_url, schedule_open_browser


def test_gui_entry_url() -> None:
    assert gui_entry_url(8501) == "http://127.0.0.1:8501/"
    assert gui_entry_url(9000) == "http://127.0.0.1:9000/"


def test_schedule_open_browser_respects_no_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMARSINAN_GUI_NO_BROWSER", "1")
    timer = MagicMock()
    with patch("mimarsinan.gui.server.app.threading.Timer", return_value=timer) as timer_cls:
        schedule_open_browser("http://127.0.0.1:8501/")
    timer_cls.assert_not_called()
    timer.start.assert_not_called()


def test_schedule_open_browser_starts_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIMARSINAN_GUI_NO_BROWSER", raising=False)
    timer = MagicMock()
    with patch("mimarsinan.gui.server.app.threading.Timer", return_value=timer) as timer_cls:
        schedule_open_browser("http://127.0.0.1:8501/")
    timer_cls.assert_called_once()
    assert timer_cls.call_args[0][0] > 0
    timer.start.assert_called_once()

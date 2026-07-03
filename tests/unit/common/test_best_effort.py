"""best_effort is the single sanctioned log-and-degrade seam."""

import logging

import pytest

from mimarsinan.common.best_effort import best_effort


class TestBestEffort:
    def test_body_runs_normally(self):
        ran = []
        with best_effort("noop"):
            ran.append(1)
        assert ran == [1]

    def test_swallows_and_logs_exceptions(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="mimarsinan.best_effort"):
            with best_effort("render heatmap"):
                raise ValueError("boom")
        assert any("render heatmap" in r.message for r in caplog.records)
        assert any(r.exc_info for r in caplog.records)

    def test_never_swallows_exit_signals(self):
        with pytest.raises(KeyboardInterrupt):
            with best_effort("interruptible"):
                raise KeyboardInterrupt
        with pytest.raises(SystemExit):
            with best_effort("exiting"):
                raise SystemExit(1)

    def test_custom_logger_is_used(self, caplog):
        logger = logging.getLogger("mimarsinan.gui")
        with caplog.at_level(logging.DEBUG, logger="mimarsinan.gui"):
            with best_effort("snapshot", logger=logger):
                raise RuntimeError("panel failed")
        assert any(r.name == "mimarsinan.gui" for r in caplog.records)

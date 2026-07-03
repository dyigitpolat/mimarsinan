"""Error-contract tests: pipelining side-work degrades via best_effort, never crashes."""

import logging

import pytest
import torch.nn as nn

from mimarsinan.pipelining.core.engine import pipeline_resource_debug
from mimarsinan.pipelining.core.engine.pipeline_helpers import (
    run_optional_viz,
    safe_warmup_forward,
)

BEST_EFFORT_LOGGER = "mimarsinan.best_effort"


class _BoomModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("forward boom")


def _raise(exc):
    raise exc


def test_run_optional_viz_suppresses_and_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger=BEST_EFFORT_LOGGER):
        run_optional_viz("StepX", lambda: _raise(RuntimeError("viz boom")))
    assert any("StepX" in r.getMessage() for r in caplog.records)


def test_run_optional_viz_reraises_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        run_optional_viz("StepX", lambda: _raise(KeyboardInterrupt()))


def test_run_optional_viz_runs_fn():
    calls = []
    run_optional_viz("StepX", lambda: calls.append(1))
    assert calls == [1]


def test_safe_warmup_forward_suppresses_and_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger=BEST_EFFORT_LOGGER):
        safe_warmup_forward(_BoomModel(), (3,), "cpu")
    assert any("warmup" in r.getMessage() for r in caplog.records)


def test_resource_snapshot_never_crashes(monkeypatch):
    monkeypatch.setattr(pipeline_resource_debug, "_RESOURCE_DEBUG", True)
    monkeypatch.setattr(
        pipeline_resource_debug.os, "getpid", lambda: _raise(RuntimeError("pid boom"))
    )
    pipeline_resource_debug.log_resource_snapshot("test-tag")

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


def test_safe_warmup_forward_follows_model_device():
    """The dummy runs on the MODEL's parameter device: a builder may leave a
    freshly-built model on CPU regardless of the config device, and a
    device-mismatched warmup silently leaves Lazy modules unmaterialized (the
    t0_20 Model-Building gate crash)."""
    from torch.nn.parameter import UninitializedParameter

    model = nn.Sequential(nn.Linear(3, 4), nn.LazyBatchNorm1d())
    assert any(
        isinstance(p, UninitializedParameter) for p in model.parameters()
    )
    # Config device says "cuda" but the model (and this CPU-only test host)
    # live on cpu; the warmup must still materialize the lazy modules.
    safe_warmup_forward(model, (3,), "cuda")
    assert not any(
        isinstance(p, UninitializedParameter) for p in model.parameters()
    )


def test_safe_warmup_forward_paramless_uses_passed_device():
    calls = []

    class _Probe(nn.Module):
        def forward(self, x):
            calls.append(str(x.device))
            return x

    safe_warmup_forward(_Probe(), (3,), "cpu")
    assert calls == ["cpu"]


def test_resource_snapshot_reads_env_at_call_time(monkeypatch, capsys):
    monkeypatch.delenv("MIMARSINAN_RESOURCE_DEBUG", raising=False)
    pipeline_resource_debug.log_resource_snapshot("disabled-tag")
    assert "disabled-tag" not in capsys.readouterr().err
    monkeypatch.setenv("MIMARSINAN_RESOURCE_DEBUG", "1")
    pipeline_resource_debug.log_resource_snapshot("enabled-tag")
    assert "enabled-tag" in capsys.readouterr().err


def test_resource_snapshot_never_crashes(monkeypatch):
    monkeypatch.setenv("MIMARSINAN_RESOURCE_DEBUG", "1")
    monkeypatch.setattr(
        pipeline_resource_debug.os, "getpid", lambda: _raise(RuntimeError("pid boom"))
    )
    pipeline_resource_debug.log_resource_snapshot("test-tag")

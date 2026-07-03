"""Error-contract tests for the compilagent backend helpers."""

import logging
import time
from types import SimpleNamespace

import pytest

from mimarsinan.search.optimizers.compilagent.backend.backend_layout import (
    collect_layout_payload,
)
from mimarsinan.search.optimizers.compilagent.backend.backend_tools import fire_pass
from mimarsinan.search.problems.joint.problem import json_key

LAYOUT_LOGGER = "mimarsinan.search.optimizers.compilagent.backend.backend_layout"
BEST_EFFORT_LOGGER = "mimarsinan.best_effort"


class _RebuildFailProblem:
    search_mode = "joint"
    _hw_only_cache = None

    def __init__(self, cached_config=None, hw_objectives=None):
        self._validation_cache = {}
        if cached_config is not None:
            self._validation_cache[json_key(cached_config)] = SimpleNamespace(
                model=None, total_params=1.0, hw_objectives=dict(hw_objectives or {}),
            )

    def _build_model(self, mc, pcfg):
        raise RuntimeError("rebuild failed")


def _config():
    return {"model_config": {"w": 8}, "platform_constraints": {"cores": []}}


class TestCollectLayoutPayloadErrorContract:
    def test_rebuild_failure_with_cache_serves_degraded_payload_and_warns(self, caplog):
        config = _config()
        problem = _RebuildFailProblem(
            cached_config=config, hw_objectives={"fragmentation_pct": 12.5},
        )
        with caplog.at_level(logging.WARNING, logger=LAYOUT_LOGGER):
            payload = collect_layout_payload(problem, config)
        assert payload["softcore_count"] == 0
        assert payload["per_softcore"] == []
        assert payload["hw_objectives"] == {"fragmentation_pct": 12.5}
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_rebuild_failure_without_cache_propagates(self):
        problem = _RebuildFailProblem()
        with pytest.raises(RuntimeError, match="rebuild failed"):
            collect_layout_payload(problem, _config())


class TestFirePassErrorContract:
    def test_callback_failure_is_swallowed_and_logged(self, caplog):
        def bad_callback(event):
            raise RuntimeError("host gone")

        with caplog.at_level(logging.DEBUG, logger=BEST_EFFORT_LOGGER):
            fire_pass(bad_callback, "validate", "validate_detailed", time.perf_counter())
        assert any("pass event" in r.getMessage() for r in caplog.records)

    def test_callback_receives_pass_event(self):
        events = []
        fire_pass(events.append, "validate", "validate_detailed", time.perf_counter())
        assert len(events) == 1
        assert events[0].stage == "validate"
        assert events[0].name == "validate_detailed"

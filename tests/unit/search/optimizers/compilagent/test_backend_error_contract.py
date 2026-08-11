"""Error-contract tests for the compilagent backend helpers.

Driven against the REAL problem: the layout payload is a projection of
``JointArchHwProblem.candidate_layout``, so its error contract IS that method's
— a candidate-scoped failure crosses typed and the backend renders it as a
failed compile, while problem-level breakage propagates untyped and aborts.
"""

import logging
import time
from pathlib import Path

import pytest
from compilagent import Plan

from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.backend.backend_layout import (
    collect_layout_payload,
)
from mimarsinan.search.optimizers.compilagent.backend.backend_tools import fire_pass
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)
from mimarsinan.search.problem import CandidateInfeasibleError
from mimarsinan.search.problems.joint import JointArchHwProblem

from .real_problem import make_problem, make_workload, pipeline_config

BEST_EFFORT_LOGGER = "mimarsinan.best_effort"


class _BrokenFixtureProblem(JointArchHwProblem):
    """A hardware search whose candidate-INDEPENDENT model fixture is broken."""

    def _ensure_hw_only_cache(self):
        raise RuntimeError("hw-only fixture broken")


def _like(problem, cls):
    from dataclasses import fields

    return cls(**{f.name: getattr(problem, f.name) for f in fields(problem) if f.init})


def _configuration(width: int = 16):
    cfg = pipeline_config(width=width)
    return {
        "model_config": dict(cfg["model_config"]),
        "platform_constraints": {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        },
    }


class TestCollectLayoutPayloadErrorContract:
    def test_a_candidate_the_model_cannot_map_raises_typed(self):
        problem = make_problem("hardware", cfg=pipeline_config(width=200))
        configuration = _configuration(width=200)
        configuration["platform_constraints"]["cores"][0]["max_axons"] = 64
        with pytest.raises(CandidateInfeasibleError, match="fan-in"):
            collect_layout_payload(problem, configuration)

    def test_a_candidate_platform_that_does_not_resolve_raises_typed(self):
        problem = make_problem("hardware")
        configuration = _configuration()
        configuration["platform_constraints"]["cores"][0]["count"] = 0
        with pytest.raises(ValueError, match="does not resolve into a chip"):
            collect_layout_payload(problem, configuration)

    def test_problem_level_fixture_failure_propagates(self):
        problem = _like(make_problem("hardware"), _BrokenFixtureProblem)
        with pytest.raises(RuntimeError, match="hw-only fixture broken"):
            collect_layout_payload(problem, _configuration())

    def test_the_backend_renders_a_typed_failure_as_a_failed_compile(self, tmp_path):
        # The whole point of the typed boundary: one unmappable candidate costs
        # a compile result, never the session.
        problem = make_problem("hardware", cfg=pipeline_config(width=600))
        workload_id = "real_layout_error_contract"
        register_problem(workload_id, problem)
        try:
            backend = MimarsinanLayoutBackend()
            result = backend.compile(
                make_workload(workload_id), Plan(), artifact_dir=Path(tmp_path),
            )
        finally:
            unregister_problem(workload_id)
        assert result.ok is False
        assert result.metadata["failure_phase"] == "hw_conversion"
        assert "fan-in" in (result.diagnostics or "")

    def test_the_backend_lets_problem_level_breakage_abort(self, tmp_path):
        problem = _like(make_problem("hardware"), _BrokenFixtureProblem)
        workload_id = "real_layout_broken_fixture"
        register_problem(workload_id, problem)
        try:
            backend = MimarsinanLayoutBackend()
            with pytest.raises(RuntimeError, match="hw-only fixture broken"):
                backend.compile(
                    make_workload(workload_id), Plan(), artifact_dir=Path(tmp_path),
                )
        finally:
            unregister_problem(workload_id)


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

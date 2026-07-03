"""Error-contract tests for search optimizers: explicit penalties, best-effort telemetry."""

import logging

import numpy as np
import pytest

from mimarsinan.search.optimizers.agent_evolve import AgentEvolveOptimizer
from mimarsinan.search.optimizers.agent_evolve_prompts import parse_candidates
from mimarsinan.search.optimizers.llm.trace import emit_search_event, parse_json_object
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.results import ObjectiveSpec

NSGA2_LOGGER = "mimarsinan.search.optimizers.nsga2_optimizer"
BATCH_EVAL_LOGGER = "mimarsinan.search.optimizers.agent_evolve.batch_eval"
BEST_EFFORT_LOGGER = "mimarsinan.best_effort"


class _EncodedToyProblem:
    objectives = (ObjectiveSpec("score", "min"),)
    n_var = 2
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    def __init__(self, fail_above=None):
        self.fail_above = fail_above

    def decode(self, x):
        return {"x0": float(x[0]), "x1": float(x[1])}

    def validate(self, configuration):
        return True

    def constraint_violation(self, configuration):
        return 0.0

    def evaluate(self, configuration):
        if self.fail_above is not None and configuration["x0"] > self.fail_above:
            raise RuntimeError("evaluation blew up")
        return {"score": configuration["x0"] + configuration["x1"]}


class TestNSGA2ErrorContract:
    def test_failed_candidate_gets_penalty_and_warning(self, caplog):
        optimizer = NSGA2Optimizer(pop_size=8, generations=2, seed=0, verbose=False)
        with caplog.at_level(logging.WARNING, logger=NSGA2_LOGGER):
            result = optimizer.optimize(_EncodedToyProblem(fail_above=0.5))
        assert result.best is not None
        penalized = [
            c for c in result.all_candidates
            if c.objectives.get("score") == optimizer.invalid_penalty
        ]
        assert penalized
        assert any(
            r.levelno == logging.WARNING and "evaluation blew up" in r.getMessage()
            for r in caplog.records
        )

    def test_reporter_failure_degrades_without_crashing(self, caplog):
        def bad_reporter(*args, **kwargs):
            raise RuntimeError("reporter down")

        optimizer = NSGA2Optimizer(pop_size=4, generations=2, seed=0, verbose=False)
        with caplog.at_level(logging.DEBUG, logger=NSGA2_LOGGER):
            result = optimizer.optimize(_EncodedToyProblem(), reporter=bad_reporter)
        assert result.best is not None
        assert any("best-effort" in r.getMessage() for r in caplog.records)


class _ExplodingProblem:
    objectives = (ObjectiveSpec("score", "min"),)

    def validate(self, configuration):
        return True

    def evaluate(self, configuration):
        raise RuntimeError("evaluate blew up")


class TestBatchEvalErrorContract:
    def test_evaluation_exception_records_penalty_and_warns(self, caplog):
        optimizer = AgentEvolveOptimizer(verbose=False)
        specs = [ObjectiveSpec("score", "min")]
        with caplog.at_level(logging.WARNING, logger=BATCH_EVAL_LOGGER):
            valid, failed = optimizer._evaluate_batch(
                _ExplodingProblem(), [{"x": 1}], specs,
            )
        assert valid == []
        assert len(failed) == 1
        assert not failed[0].is_valid
        assert "evaluate blew up" in failed[0].error_message
        assert failed[0].objectives == {"score": optimizer.invalid_penalty}
        assert any(
            r.levelno == logging.WARNING and "evaluate blew up" in r.getMessage()
            for r in caplog.records
        )

    def test_report_generation_metrics_swallows_reporter_failure(self):
        optimizer = AgentEvolveOptimizer(verbose=False)

        def bad_reporter(*args, **kwargs):
            raise RuntimeError("reporter down")

        optimizer._report_generation_metrics(bad_reporter, 1, 2, 3)


class TestTraceErrorContract:
    def test_emit_search_event_swallows_reporter_failure(self, caplog):
        def bad_reporter(*args, **kwargs):
            raise RuntimeError("reporter down")

        with caplog.at_level(logging.DEBUG, logger=BEST_EFFORT_LOGGER):
            emit_search_event(bad_reporter, {"type": "x"})
        assert any("search_event" in r.getMessage() for r in caplog.records)

    def test_emit_search_event_delivers_events(self):
        calls = []
        emit_search_event(lambda name, payload: calls.append((name, payload)), {"type": "x"})
        assert len(calls) == 1
        assert calls[0][0] == "search_event"

    def test_emit_llm_trace_swallows_reporter_failure(self):
        from types import SimpleNamespace

        optimizer = AgentEvolveOptimizer(verbose=False)
        optimizer._trace_reporter = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        optimizer._emit_llm_trace(
            "initial_candidates",
            "prompt text",
            {"reasoning": str},
            SimpleNamespace(reasoning="r", candidates=[]),
        )

    def test_parse_json_object_valid(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_parse_json_object_embedded_in_prose(self):
        assert parse_json_object('sure! {"a": 1} hope that helps') == {"a": 1}

    def test_parse_json_object_garbage_degrades_to_empty(self):
        assert parse_json_object("not json at all") == {}


class TestParseCandidatesErrorContract:
    def test_malformed_json_string_is_skipped_with_warning(self):
        logs = []
        parsed = parse_candidates(['{"a": 1}', "not json"], 2, logs.append)
        assert parsed == [{"a": 1}]
        assert any("Could not parse" in m for m in logs)

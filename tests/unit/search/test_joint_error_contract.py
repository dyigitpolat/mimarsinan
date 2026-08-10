"""Error-contract tests for the joint search problem mixins.

The typed taxonomy: candidate-scoped failures in the validate path degrade to
explicit invalid results with a warning log; candidate-scoped failures in the
evaluate path cross the problem boundary as ``CandidateInfeasibleError`` (the
optimizer converts them to penalties); problem-level (candidate-independent)
failures propagate untyped and abort.
"""

import logging

import pytest

from mimarsinan.search.problem import CandidateInfeasibleError, ValidationResult
from mimarsinan.search.problems.joint.evaluate import JointEvaluateMixin
from mimarsinan.search.problems.joint.types import ValidationEntry
from mimarsinan.search.problems.joint.validate import JointValidateMixin
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME, ObjectiveSpec

VALIDATE_LOGGER = "mimarsinan.search.problems.joint.validate"
EVALUATE_LOGGER = "mimarsinan.search.problems.joint.evaluate"


class _ValidateHarness(JointValidateMixin):
    def __init__(self, search_mode="joint", validate_fn=None, constraint_fn=None):
        self._validation_cache = {}
        self._validation_errors = {}
        self._cache = {}
        self.validate_fn = validate_fn
        self.constraint_fn = constraint_fn
        self.search_mode = search_mode
        self.accuracy_seed = 0
        self.input_shape = (1, 4, 4)
        self.objectives = [
            ObjectiveSpec(ACCURACY_OBJECTIVE_NAME, "max"),
            ObjectiveSpec("total_params", "min"),
        ]

    def _ensure_hw_only_cache(self):
        raise RuntimeError("hw-only fixture broken")

    def _build_raw_model(self, mc, pcfg):
        raise ValueError("candidate arch invalid")

    def _ensure_mapper_repr(self, model):
        raise AssertionError("should not be reached")

    def _collect_softcores(self, model, pcfg):
        raise AssertionError("should not be reached")

    def _compute_hw_objectives(self, softcores, pcfg, total_params, host_segments):
        raise AssertionError("should not be reached")


def _config():
    return {"model_config": {}, "platform_constraints": {}}


class TestValidateErrorContract:
    def test_hw_only_fixture_failure_propagates(self):
        harness = _ValidateHarness(search_mode="hardware")
        with pytest.raises(RuntimeError, match="hw-only fixture broken"):
            harness.validate_detailed(_config())

    def test_candidate_model_build_failure_is_explicit_invalid(self, caplog):
        harness = _ValidateHarness(search_mode="joint")
        with caplog.at_level(logging.WARNING, logger=VALIDATE_LOGGER):
            vr = harness.validate_detailed(_config())
        assert not vr.is_valid
        assert vr.failure_phase == "model_build"
        assert "candidate arch invalid" in vr.error_message
        assert any(
            r.levelno == logging.WARNING and "candidate arch invalid" in r.getMessage()
            for r in caplog.records
        )

    def test_structural_validate_fn_exception_is_explicit_invalid(self, caplog):
        def bad_validate_fn(mc, pcfg, input_shape):
            raise TypeError("validate_fn blew up")

        harness = _ValidateHarness(search_mode="joint", validate_fn=bad_validate_fn)
        with caplog.at_level(logging.WARNING, logger=VALIDATE_LOGGER):
            vr = harness.validate_detailed(_config())
        assert not vr.is_valid
        assert vr.failure_phase == "structural"
        assert "validate_fn blew up" in vr.error_message
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_constraint_fn_failure_returns_large_violation_and_warns(self, caplog):
        def bad_constraint_fn(mc, pcfg, input_shape):
            raise RuntimeError("constraint blew up")

        harness = _ValidateHarness(search_mode="joint", constraint_fn=bad_constraint_fn)
        with caplog.at_level(logging.WARNING, logger=VALIDATE_LOGGER):
            cv = harness.constraint_violation(_config())
        assert cv == 1e6
        assert any(
            r.levelno == logging.WARNING and "constraint blew up" in r.getMessage()
            for r in caplog.records
        )

    def test_constraint_violation_propagates_problem_level_failures(self):
        harness = _ValidateHarness(search_mode="hardware")
        with pytest.raises(RuntimeError, match="hw-only fixture broken"):
            harness.constraint_violation(_config())


class _EvaluateHarness(JointEvaluateMixin):
    accuracy_seed = 0
    search_mode = "joint"

    def __init__(self, inner_error=None):
        self._cache = {}
        self._validation_cache = {}
        self._inner_error = inner_error
        self.objectives = [
            ObjectiveSpec(ACCURACY_OBJECTIVE_NAME, "max"),
            ObjectiveSpec("total_params", "min"),
        ]

    def validate_detailed(self, configuration):
        return ValidationResult(is_valid=True)

    def _penalty_objectives(self):
        return {s.name: (0.0 if s.goal == "max" else 1e18) for s in self.objectives}

    def _evaluate_accuracy(self, model):
        raise RuntimeError("training exploded")

    def _evaluate_inner(self, mc, pcfg):
        raise self._inner_error


class TestEvaluateErrorContract:
    def test_cached_accuracy_failure_records_penalty_and_warns(self, caplog):
        harness = _EvaluateHarness()
        vc = ValidationEntry(
            model=object(), total_params=1.0, hw_objectives={"total_params": 5.0},
        )
        with caplog.at_level(logging.WARNING, logger=EVALUATE_LOGGER):
            obj = harness._evaluate_from_cache(vc, _config())
        assert obj[ACCURACY_OBJECTIVE_NAME] == 0.0
        assert obj["total_params"] == 5.0
        assert vc.model is None
        assert any(
            r.levelno == logging.WARNING and "training exploded" in r.getMessage()
            for r in caplog.records
        )

    def test_evaluate_propagates_candidate_infeasibility_typed(self):
        # The problem boundary raises typed; converting to penalties is the
        # optimizer's job, so nothing may be swallowed here.
        harness = _EvaluateHarness(
            inner_error=CandidateInfeasibleError("candidate collapsed"),
        )
        with pytest.raises(CandidateInfeasibleError, match="candidate collapsed"):
            harness.evaluate(_config())

    def test_evaluate_propagates_problem_level_inner_failure(self):
        harness = _EvaluateHarness(inner_error=RuntimeError("inner exploded"))
        with pytest.raises(RuntimeError, match="inner exploded"):
            harness.evaluate(_config())


class _InnerHarness(JointEvaluateMixin):
    """Exercises the REAL ``_evaluate_inner`` classification of raise-sites."""

    accuracy_seed = 0
    search_mode = "joint"

    def __init__(self, build_error=None, mapper_error=None):
        self._cache = {}
        self._validation_cache = {}
        self._build_error = build_error
        self._mapper_error = mapper_error
        self.objectives = [ObjectiveSpec("total_params", "min")]

    def validate_detailed(self, configuration):
        return ValidationResult(is_valid=True)

    def _penalty_objectives(self):
        return {s.name: (0.0 if s.goal == "max" else 1e18) for s in self.objectives}

    def _ensure_hw_only_cache(self):
        raise RuntimeError("hw-only fixture broken")

    def _build_raw_model(self, mc, pcfg):
        if self._build_error is not None:
            raise self._build_error
        return object(), 1.0

    def _ensure_mapper_repr(self, model):
        if self._mapper_error is not None:
            raise self._mapper_error
        return model

    def _collect_softcores(self, model, pcfg):
        raise AssertionError("should not be reached")


class TestEvaluateInnerRaiseSiteClassification:
    def test_candidate_model_build_failure_raises_typed(self):
        harness = _InnerHarness(build_error=ValueError("candidate arch invalid"))
        with pytest.raises(CandidateInfeasibleError, match="candidate arch invalid") as ei:
            harness._evaluate_inner({}, {})
        assert isinstance(ei.value.__cause__, ValueError)

    def test_candidate_mapping_collapse_raises_typed(self):
        harness = _InnerHarness(mapper_error=RuntimeError("conversion collapsed"))
        with pytest.raises(CandidateInfeasibleError, match="conversion collapsed") as ei:
            harness._evaluate_inner({}, {})
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_hw_only_fixture_failure_propagates_untyped(self):
        harness = _InnerHarness()
        harness.search_mode = "hardware"
        with pytest.raises(RuntimeError, match="hw-only fixture broken"):
            harness._evaluate_inner({}, {})

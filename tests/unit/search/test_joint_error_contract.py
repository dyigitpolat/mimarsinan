"""Error-contract tests for the joint search problem mixins.

The typed taxonomy: candidate-scoped failures in the validate path degrade to
explicit invalid results with a warning log; candidate-scoped failures in the
evaluate path cross the problem boundary as ``CandidateInfeasibleError`` (the
optimizer converts them to penalties); problem-level (candidate-independent)
failures propagate untyped and abort.
"""

import logging

import pytest

from mimarsinan.deployment_record.objectives import OBJECTIVES, CandidateStaticView
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.search.problem import CandidateInfeasibleError, ValidationResult
from mimarsinan.search.problems.joint.evaluate import JointEvaluateMixin
from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin
from mimarsinan.search.problems.joint.types import ValidationEntry
from mimarsinan.search.problems.joint.validate import JointValidateMixin
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME, ObjectiveSpec

VALIDATE_LOGGER = "mimarsinan.search.problems.joint.validate"
EVALUATE_LOGGER = "mimarsinan.search.problems.joint.evaluate"

ACTIVE_NAMES = (ACCURACY_OBJECTIVE_NAME, "total_params")


class _Harness(JointValidateMixin, JointLayoutMixin, JointEvaluateMixin):
    """The joint mixins on a bare host: only the host members, nothing simulated."""

    search_mode = "joint"
    accuracy_seed = 0
    # [H4] the per-problem fragment-need memo the real problem carries; a
    # fresh dict per harness class keeps active-set differences separate.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._fragment_needs_cache = {}
    _fragment_needs_cache: dict = {}
    input_shape = (1, 4, 4)
    validate_fn = None
    constraint_fn = None
    active_objective_names = ACTIVE_NAMES
    encoding_placement = "subsume"
    pruning_fraction = 0.0
    onchip_min_fraction = 0.0

    def __init__(self, active_names=ACTIVE_NAMES):
        self._cache = {}
        self._validation_cache = {}
        self._validation_errors = {}
        self._constraint_census = {}
        self._hw_only_cache = None
        self.active_objective_names = tuple(active_names)

    @property
    def _searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def active_specs(self):
        return OBJECTIVES.resolve_active(self.search_mode, self.active_objective_names)

    @property
    def objectives(self):
        return [ObjectiveSpec(s.name, s.goal) for s in self.active_specs]

    def _resolved_configuration(self, configuration):
        """The identity resolution: these harnesses declare platforms directly."""
        return configuration


class _ValidateHarness(_Harness):
    def __init__(self, search_mode="joint", validate_fn=None, constraint_fn=None):
        super().__init__()
        self.validate_fn = validate_fn
        self.constraint_fn = constraint_fn
        self.search_mode = search_mode

    def _ensure_hw_only_cache(self, _placement):
        raise RuntimeError("hw-only fixture broken")

    def _build_raw_model(self, mc, pcfg, _placement):
        raise ValueError("candidate arch invalid")

    def _ensure_mapper_repr(self, model, _placement):
        raise AssertionError("should not be reached")

    def _collect_softcores(self, model, pcfg, *, collect_census=False):
        raise AssertionError("should not be reached")

    def _pack_candidate(self, softcores, pcfg):
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


class _EvaluateHarness(_Harness):
    def __init__(self, inner_error=None):
        super().__init__()
        self._inner_error = inner_error

    def validate_detailed(self, configuration):
        return ValidationResult(is_valid=True)

    def _evaluate_accuracy(self, model):
        raise RuntimeError("training exploded")

    def _evaluate_inner(self, mc, pcfg, _placement):
        raise self._inner_error


def _entry(total_params=5.0):
    return ValidationEntry(
        model=object(),
        view=CandidateStaticView(
            layout=None,
            chip_param_capacity=None,
            total_params=total_params,
            host_side_segment_count=None,
        ),
    )


class TestEvaluateErrorContract:
    def test_accuracy_failure_records_penalty_and_warns(self, caplog):
        harness = _EvaluateHarness()
        entry = _entry()
        with caplog.at_level(logging.WARNING, logger=EVALUATE_LOGGER):
            obj = harness._objectives_from_entry(entry)
        assert obj[ACCURACY_OBJECTIVE_NAME] == 0.0
        assert obj["total_params"] == 5.0
        assert entry.model is None
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


class _InnerHarness(_Harness):
    """Exercises the REAL ``_evaluate_inner`` classification of raise-sites."""

    def __init__(self, build_error=None, mapper_error=None):
        super().__init__(active_names=("total_params",))
        self._build_error = build_error
        self._mapper_error = mapper_error

    def _ensure_hw_only_cache(self, _placement):
        raise RuntimeError("hw-only fixture broken")

    def _build_raw_model(self, mc, pcfg, _placement):
        if self._build_error is not None:
            raise self._build_error
        return object(), 1.0

    def _ensure_mapper_repr(self, model, _placement):
        if self._mapper_error is not None:
            raise self._mapper_error
        return model

    def _collect_softcores(self, model, pcfg, *, collect_census=False):
        raise AssertionError("should not be reached")


class _LayoutInnerHarness(_InnerHarness):
    """Same, with a layout-bearing objective active so the mapping path runs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_objective_names = ("total_params", "param_utilization_pct")


class _UnpackableHarness(_LayoutInnerHarness):
    """A candidate whose facts resolve but whose softcores do not fit the chip."""

    def _collect_softcores(self, model, pcfg, *, collect_census=False):
        return [
            LayoutSoftCoreSpec(
                input_count=64, output_count=64, residency_class_id=0, name="sc",
            )
        ], 0, None


TOO_SMALL_CHIP = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 1}]}


class TestEvaluateInnerRaiseSiteClassification:
    def test_candidate_model_build_failure_raises_typed(self):
        harness = _InnerHarness(build_error=ValueError("candidate arch invalid"))
        with pytest.raises(CandidateInfeasibleError, match="candidate arch invalid") as ei:
            harness._evaluate_inner({}, {}, harness.encoding_placement)
        assert isinstance(ei.value.__cause__, ValueError)

    def test_candidate_mapping_collapse_raises_typed(self):
        harness = _LayoutInnerHarness(
            mapper_error=RuntimeError("conversion collapsed"),
        )
        with pytest.raises(CandidateInfeasibleError, match="conversion collapsed") as ei:
            harness._evaluate_inner({}, {}, harness.encoding_placement)
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_hw_only_fixture_failure_propagates_untyped(self):
        harness = _InnerHarness()
        harness.search_mode = "hardware"
        with pytest.raises(RuntimeError, match="hw-only fixture broken"):
            harness._evaluate_inner({}, {}, harness.encoding_placement)

    def test_a_layoutless_objective_set_never_touches_the_mapping(self):
        # ``_collect_softcores`` asserts it is unreachable: with no layout-bearing
        # axis active, the candidate view is built without the mapping at all.
        harness = _InnerHarness()
        assert harness._evaluate_inner({}, {}, harness.encoding_placement) == {"total_params": 1.0}

    def test_a_candidate_that_does_not_fit_is_SCORED_not_raised(self, caplog):
        # The one penalized phase: a chip that cannot hold the candidate is a
        # ranked, dominated row — raising here would let a whole population of
        # ambitious-but-oversized candidates abort the search instead.
        harness = _UnpackableHarness()
        with caplog.at_level(logging.WARNING, logger=EVALUATE_LOGGER):
            objectives = harness._evaluate_inner({}, TOO_SMALL_CHIP, harness.encoding_placement)
        assert objectives == harness._penalty_objectives()
        assert any(
            r.levelno == logging.WARNING and "returning full penalty" in r.getMessage()
            for r in caplog.records
        )

    def test_the_packing_penalty_reports_the_census_that_explains_it(self):
        harness = _UnpackableHarness()
        _entry, failure = harness._resolve_entry({}, TOO_SMALL_CHIP, harness.encoding_placement)
        assert failure is not None
        assert failure.phase == "hw_packing"
        assert "softcores=1" in failure.message
        assert "total_hw_capacity=64" in failure.message

"""A candidate below the on-chip floor is INFEASIBLE to the optimizer, not a crash."""

import numpy as np
import pytest

from mimarsinan.search.constraints import ONCHIP_FLOOR_CONSTRAINT
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.problems.joint.problem import JointArchHwProblem
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)

from conftest import TinyPerceptronFlow

_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
    "weight_bits": 8,
    "encoding_layer_placement": "subsume",
}


def _resolver(overlay):
    return build_platform_constraints_resolved({**_PLATFORM, **dict(overlay)})


def _builder_factory(device, input_shape, num_classes, config):
    class _Builder:
        def build(self, model_config):
            return TinyPerceptronFlow(input_shape=input_shape, num_classes=num_classes)

    return _Builder()


def _problem(*, floor, axes=None, **over):
    kwargs = dict(
        data_provider_factory=None,
        device="cpu",
        input_shape=(1, 8, 8),
        num_classes=4,
        target_tq=4,
        lr=1e-3,
        search_mode="hardware",
        platform_resolver=_resolver,
        builder_factory=_builder_factory,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        fixed_model_config={},
        onchip_min_fraction=floor,
        option_axes=build_option_axes(axes),
    )
    kwargs.update(over)
    return JointArchHwProblem(**kwargs)


class TestTheFloorIsDeclaredNotAssumed:
    def test_a_zero_floor_declares_no_constraint(self):
        problem = _problem(floor=0.0)
        configuration = problem.decode(problem.xl)
        assert problem.onchip_constraint(configuration) is None

    def test_an_unreachable_floor_reports_a_violation(self):
        """A floor of 100% cannot be met by a model with any host op at all."""
        problem = _problem(floor=1.0)
        configuration = problem.decode(problem.xl)
        report = problem.onchip_constraint(configuration)
        assert report is not None
        assert report.constraint == ONCHIP_FLOOR_CONSTRAINT
        assert report.violation > 0

    def test_the_report_names_what_it_measured(self):
        problem = _problem(floor=1.0)
        report = problem.onchip_constraint(problem.decode(problem.xl))
        assert "on chip" in report.detail or "on-chip" in report.detail


class TestItReachesTheOptimizersConstraintChannel:
    def test_a_violating_candidate_reports_a_positive_constraint_violation(self):
        problem = _problem(floor=1.0)
        cv = problem.constraint_violation(problem.decode(problem.xl))
        assert cv > 0.0

    def test_a_satisfying_candidate_reports_zero(self):
        problem = _problem(floor=0.0)
        cv = problem.constraint_violation(problem.decode(problem.xl))
        assert cv == 0.0

    def test_the_violation_is_never_a_crash(self):
        """The point of the whole exercise: the floor is a feasibility boundary the
        optimizer can see, not an exception a run discovers after picking a winner."""
        problem = _problem(floor=1.0)
        configuration = problem.decode(problem.xl)
        assert isinstance(problem.constraint_violation(configuration), float)


class TestTheCensusIsReportable:
    def test_violations_accumulate_by_constraint_name(self):
        problem = _problem(floor=1.0)
        problem.constraint_violation(problem.decode(problem.xl))
        census = problem.constraint_census()
        assert census[ONCHIP_FLOOR_CONSTRAINT] >= 1

    def test_a_clean_search_reports_an_empty_census(self):
        problem = _problem(floor=0.0)
        problem.constraint_violation(problem.decode(problem.xl))
        assert problem.constraint_census() == {}


class TestPlacementMovesTheFraction:
    def test_offload_puts_more_on_chip_than_subsume(self):
        """The measured axis: placement decides which side of the boundary the
        encoder lands on, so the constraint must see the candidate's own choice."""
        problem = _problem(floor=0.0, axes=["encoding_layer_placement"])
        fractions = {}
        for coord, name in ((0.0, "subsume"), (1.0, "offload")):
            x = np.concatenate([problem.xl[:-1], np.array([coord], dtype=float)])
            configuration = problem.decode(x)
            assert configuration["deployment_options"][
                "encoding_layer_placement"] == name
            fractions[name] = problem.onchip_fraction(configuration)
        assert fractions["offload"] > fractions["subsume"]


class TestTheEvaluationPathUsesTheCandidatesPlacement:
    """Not just the constraint: the packing an evaluation SCORES must be the
    candidate's own, or two placements score identically and the axis is inert."""

    def _configurations(self, problem):
        out = {}
        for coord, name in ((0.0, "subsume"), (1.0, "offload")):
            x = np.concatenate([problem.xl[:-1], np.array([coord], dtype=float)])
            out[name] = problem.decode(x)
        return out

    def test_the_two_placements_lay_out_differently(self):
        problem = _problem(floor=0.0, axes=["encoding_layer_placement"])
        configurations = self._configurations(problem)
        counts = {
            name: len(problem.candidate_layout(cfg).softcores)
            for name, cfg in configurations.items()
        }
        assert counts["offload"] != counts["subsume"], counts

    def test_the_two_placements_score_differently(self):
        problem = _problem(floor=0.0, axes=["encoding_layer_placement"])
        configurations = self._configurations(problem)
        scores = {n: problem.evaluate(c) for n, c in configurations.items()}
        assert scores["offload"] != scores["subsume"], scores

    def test_validation_packs_the_candidates_own_placement(self):
        """validate_detailed caches the entry an evaluation then scores off, so it
        must resolve under the candidate's placement too."""
        problem = _problem(floor=0.0, axes=["encoding_layer_placement"])
        configurations = self._configurations(problem)
        for cfg in configurations.values():
            assert problem.validate_detailed(cfg).is_valid
        entries = list(problem._validation_cache.values())
        assert len(entries) == 2, "two placements are two candidates"
        # The CHIP is identical (only the placement differs), so the hard-core
        # count cannot separate them — the softcore census is what placement moves.
        assert (
            entries[0].view.layout.total_softcores
            != entries[1].view.layout.total_softcores
        )

"""A searched option must reach the deployment it describes — and move the census."""

import numpy as np
import pytest

from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.problems.joint.problem import JointArchHwProblem
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)

_BASE = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
    "weight_bits": 8,
    "max_schedule_passes": 8,
    "encoding_layer_placement": "subsume",
    "pruning_fraction": 0.0,
}


def _resolver(declared=None):
    declared = {**_BASE, **(declared or {})}

    def resolve(overlay):
        return build_platform_constraints_resolved({**declared, **dict(overlay)})

    return resolve


def _problem(axes, *, search_mode="hardware", **over):
    kwargs = dict(
        data_provider_factory=None,
        device="cpu",
        input_shape=(1, 8, 8),
        num_classes=4,
        target_tq=4,
        lr=1e-3,
        search_mode=search_mode,
        platform_resolver=_resolver(),
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        option_axes=build_option_axes(axes),
    )
    kwargs.update(over)
    return JointArchHwProblem(**kwargs)


class TestTheEncodingGrows:
    def test_each_axis_adds_exactly_one_dimension(self):
        bare = _problem(None)
        with_axes = _problem(
            ["encoding_layer_placement", {"weight_bits": {"bounds": [2, 8]}}]
            if False else
            {"encoding_layer_placement": None, "weight_bits": {"bounds": [2, 8]}}
        )
        assert with_axes.n_var == bare.n_var + 2

    def test_the_option_dims_follow_the_hardware_block(self):
        """Appending keeps every existing dimension at its index, so a resumed or
        re-decoded population still means what it meant."""
        problem = _problem(["encoding_layer_placement"])
        assert problem.xl[: -1].tolist() == _problem(None).xl.tolist()
        assert problem.xu[: -1].tolist() == _problem(None).xu.tolist()

    def test_a_choice_axis_spans_its_index_range_in_the_box(self):
        problem = _problem(["encoding_layer_placement"])
        assert problem.xl[-1] == 0.0
        assert problem.xu[-1] == 1.0

    def test_a_numeric_axis_spans_its_declared_bounds(self):
        problem = _problem({"weight_bits": {"bounds": [2, 8]}})
        assert (problem.xl[-1], problem.xu[-1]) == (2.0, 8.0)

    def test_no_axes_leaves_the_encoding_byte_identical(self):
        problem = _problem(None)
        assert problem.n_var == 3
        assert problem.xl.tolist() == [64.0, 64.0, 8.0]


class TestDecode:
    def _decode(self, problem, option_coords):
        x = np.concatenate([problem.xl[: problem.n_var - len(option_coords)],
                            np.array(option_coords, dtype=float)])
        return problem.decode(x)

    def test_a_searched_option_lands_in_the_candidates_deployment_options(self):
        problem = _problem(["encoding_layer_placement"])
        assert self._decode(problem, [0.0])["deployment_options"] == {
            "encoding_layer_placement": "subsume"
        }
        assert self._decode(problem, [1.0])["deployment_options"] == {
            "encoding_layer_placement": "offload"
        }

    def test_a_platform_option_reaches_the_resolved_chip(self):
        """The searched value must be the chip's, not the declaration's — otherwise
        the axis moves nothing and the search is flat along it."""
        problem = _problem({"weight_bits": {"bounds": [2, 8]}})
        decoded = self._decode(problem, [4.0])
        assert decoded["platform_constraints"]["weight_bits"] == 4
        assert decoded["deployment_options"]["weight_bits"] == 4

    def test_an_unsearched_option_keeps_the_declared_value(self):
        problem = _problem(["encoding_layer_placement"])
        decoded = self._decode(problem, [1.0])
        assert decoded["platform_constraints"]["max_schedule_passes"] == 8

    def test_without_axes_the_candidate_declares_no_options(self):
        problem = _problem(None)
        assert problem.decode(problem.xl)["deployment_options"] == {}

    def test_decode_is_still_the_resolvers_own_answer(self):
        """The law: a searched chip is a DEPLOYABLE chip — decode must equal the
        deployment resolver over base + the searched overlay, key for key."""
        problem = _problem({"weight_bits": {"bounds": [2, 8]}})
        decoded = self._decode(problem, [4.0])
        expected = build_platform_constraints_resolved({
            **_BASE,
            "cores": decoded["platform_constraints"]["cores"],
            "target_tq": 4,
            "weight_bits": 4,
        })
        assert decoded["platform_constraints"] == expected


class TestTheOptionsReachTheDeployment:
    def test_the_candidate_placement_follows_the_searched_axis(self):
        """The layout hook must build each candidate under ITS OWN placement, not
        the problem's declaration — placement moves ~75% of parameters across the
        NeuralOps/ComputeOps boundary, so a fixed one makes the axis inert."""
        problem = _problem(["encoding_layer_placement"], encoding_placement="subsume")
        subsumed = self._placement(problem, 0.0)
        offloaded = self._placement(problem, 1.0)
        assert subsumed == "subsume"
        assert offloaded == "offload"

    def _placement(self, problem, coord):
        x = np.concatenate([problem.xl[:-1], np.array([coord], dtype=float)])
        decoded = problem.decode(x)
        return problem.candidate_encoding_placement(decoded)

    def test_an_unsearched_placement_is_the_problems_declaration(self):
        problem = _problem(None, encoding_placement="offload")
        decoded = problem.decode(problem.xl)
        assert problem.candidate_encoding_placement(decoded) == "offload"

    def test_pruning_cannot_be_declared_as_an_axis(self):
        """[P3] Superseded by design: pruning's accuracy impact is unmodeled
        at candidate time, so the axis is refused at declaration — the
        declared run value drives the candidate's pruned-shape twins instead
        (test_candidate_pruned_shapes)."""
        with pytest.raises(ValueError, match="pruning_fraction"):
            _problem({"pruning_fraction": {"bounds": [0.0, 0.5]}})

"""W5.1 — the searched chip IS the deployed chip, by construction.

Every candidate platform crossing the problem surface is produced by the SAME
``build_platform_constraints_resolved`` the deployment path runs, overlaid with
nothing but the searched decision variables (core dimensions and ``target_tq``).
No hand-carried key list stands between a candidate and its deployed twin, so a
resolver key added tomorrow reaches the search for free.
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.platform.core_residency import RESIDENCY_KEY
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    build_fixed_platform_constraints,
    make_platform_resolver,
)
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.search_space_description import CORE_DIM_GRANULARITY

HW_OBJECTIVES = [
    "total_param_capacity",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "axon_wastage_pct",
    "fragmentation_pct",
]

# The run's timestep budget, deliberately different from the one the declared
# platform carries: the search's ``target_tq`` is a decision the candidate
# overlay must state, not a value the declared platform happens to agree on.
SEARCH_TARGET_TQ = 4
DECLARED_TARGET_TQ = 32


def _pipeline_config():
    """A platform whose every resolver-visible knob sits OFF its default.

    A decode that reconstructs the platform from defaults (or from a carried
    subset of keys) cannot reproduce this config, so the golden below is a real
    round-trip and not an accident of shared defaults.
    """
    return {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": DECLARED_TARGET_TQ,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "schedule_policy": "bank_clustered",
        "max_schedule_passes": 128,
        RESIDENCY_KEY: {"threshold": "per_neuron"},
        # truenorth wires 1 core/tile; loihi (the default) wires 4.
        "sanafe_arch_preset": "truenorth",
        "platform_constraints": {"has_bias": False},
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
    }


def _fixed_model_config():
    return {"mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU"}


def _hw_problem(cfg=None, platform_resolver="from_config"):
    cfg = cfg if cfg is not None else _pipeline_config()
    if platform_resolver == "from_config":
        platform_resolver = make_platform_resolver(cfg)
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        target_tq=SEARCH_TARGET_TQ,
        lr=cfg["lr"],
        search_mode="hardware",
        builder_factory=SimpleMLPBuilder,
        arch_options=(),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=_fixed_model_config(),
        platform_resolver=platform_resolver,
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


def _mid_x(problem):
    return (np.asarray(problem.xl) + np.asarray(problem.xu)) / 2.0


def _snap(value, lo, hi):
    """The encoding contract: clip into the bound, then snap to the core grid."""
    clipped = int(max(lo, min(hi, int(round(float(value))))))
    snapped = int(round(clipped / CORE_DIM_GRANULARITY)) * CORE_DIM_GRANULARITY
    return max(CORE_DIM_GRANULARITY, snapped)


def _decision_cores(problem, x):
    """The decision variables alone — reconstructed WITHOUT calling decode."""
    cores = []
    for i in range(int(problem.num_core_types)):
        base = 3 * i
        cores.append({
            "max_axons": _snap(x[base], *problem.core_axons_bounds),
            "max_neurons": _snap(x[base + 1], *problem.core_neurons_bounds),
            "count": int(max(
                problem.core_count_bounds[0],
                min(problem.core_count_bounds[1], int(round(float(x[base + 2])))),
            )),
        })
    return cores


class TestDecodeIsTheDeploymentResolution:
    def test_decode_equals_the_resolver_over_base_plus_overlay(self):
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        x = _mid_x(problem)

        decoded = problem.decode(x)["platform_constraints"]
        expected = build_platform_constraints_resolved(
            {**cfg, "cores": _decision_cores(problem, x),
             "target_tq": SEARCH_TARGET_TQ},
        )

        assert decoded == expected, (
            "the decoded candidate platform must BE the deployment resolution of "
            "the declared platform overlaid with the decision variables"
        )

    def test_every_resolver_key_survives_the_decode(self):
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        decoded = problem.decode(_mid_x(problem))["platform_constraints"]

        assert decoded["schedule_policy"] == "bank_clustered"
        assert decoded["max_schedule_passes"] == 128
        assert decoded[RESIDENCY_KEY] == {"threshold": "per_neuron"}
        assert decoded["allow_scheduling"] is True
        assert decoded["allow_coalescing"] is False
        assert decoded["weight_bits"] == 4, "the declared width, never a hardcoded 8"
        assert decoded["target_tq"] == SEARCH_TARGET_TQ, (
            "the run's timestep budget is a searched declaration; the platform's "
            "own value does not survive it"
        )
        assert "allow_weight_reuse" not in decoded

    def test_candidate_cores_inherit_the_declared_bias_capability(self):
        problem = _hw_problem()
        decoded = problem.decode(_mid_x(problem))["platform_constraints"]
        assert decoded["cores"], "a candidate declares at least one core type"
        for core in decoded["cores"]:
            assert core["has_bias"] is False, (
                "a param-encoded-bias platform must stay param-encoded under search"
            )

    def test_bias_capability_declared_per_core_also_reaches_the_candidate(self):
        # A suggested platform states has_bias on the core rows themselves; the
        # candidate core types replace those rows and must still inherit it.
        cfg = _pipeline_config()
        cfg.pop("platform_constraints")
        cfg["cores"] = [
            {"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": False},
        ]
        problem = _hw_problem(cfg)
        decoded = problem.decode(_mid_x(problem))["platform_constraints"]
        for core in decoded["cores"]:
            assert core["has_bias"] is False

    def test_the_floorplan_follows_the_candidate_core_count(self):
        # The base declares 64 cores; a candidate declaring fewer is a smaller
        # chip, and its NoC floorplan must be the one its OWN deployment gets.
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        base = build_fixed_platform_constraints(cfg)

        x = _mid_x(problem)
        x[2] = float(problem.core_count_bounds[0])  # smallest core count
        decoded = problem.decode(x)["platform_constraints"]

        count = decoded["cores"][0]["count"]
        assert count == problem.core_count_bounds[0]
        assert decoded["cores_per_tile_resolved"] == 1, "truenorth wires 1 core/tile"
        tiles = (
            decoded["tile_grid_rows_resolved"] * decoded["tile_grid_cols_resolved"]
        )
        assert tiles * decoded["cores_per_tile_resolved"] >= count
        assert tiles < (
            base["tile_grid_rows_resolved"] * base["tile_grid_cols_resolved"]
        ), "a smaller chip must not carry the base platform's floorplan"


class TestTheGoldenIsSensitive:
    """Mutation guard: the golden must fail when the overlay loses a key."""

    def test_dropping_target_tq_from_the_overlay_breaks_the_equality(self):
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        x = _mid_x(problem)
        decoded = problem.decode(x)["platform_constraints"]

        without_tq = {k: v for k, v in cfg.items() if k != "target_tq"}
        mutated = build_platform_constraints_resolved(
            {**without_tq, "cores": _decision_cores(problem, x)},
        )
        assert mutated != decoded

    def test_dropping_the_declared_schedule_policy_breaks_the_equality(self):
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        x = _mid_x(problem)
        decoded = problem.decode(x)["platform_constraints"]

        without_policy = {k: v for k, v in cfg.items() if k != "schedule_policy"}
        mutated = build_platform_constraints_resolved(
            {**without_policy, "cores": _decision_cores(problem, x),
             "target_tq": SEARCH_TARGET_TQ},
        )
        assert mutated != decoded


class TestTheCoreGridIsHonoured:
    """Core dimensions land ON the declared grid — never between its lines."""

    def test_an_off_grid_decision_variable_snaps_onto_the_grid(self):
        # The optimizers propose continuous vectors; a 99-axon core is not a
        # chip anyone can build, and an unsnapped dimension would make the
        # searched chip differ from the one the levers describe.
        problem = _hw_problem()
        decoded = problem.decode(np.array([99.0, 101.0, 16.0]))
        core = decoded["platform_constraints"]["cores"][0]
        assert core["max_axons"] == 96, "99 snaps DOWN to the nearest grid line"
        assert core["max_neurons"] == 104, "101 snaps UP to the nearest grid line"
        assert core["count"] == 16, "core count is a count, not a grid dimension"
        for dim in ("max_axons", "max_neurons"):
            assert core[dim] % CORE_DIM_GRANULARITY == 0

    def test_a_dimension_below_one_grid_line_snaps_up_to_one(self):
        # Rounding toward zero would decode a 0-axon core: not a small chip,
        # no chip at all.
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        problem.core_axons_bounds = (1, 256)
        problem.core_neurons_bounds = (1, 256)
        decoded = problem.decode(np.array([1.0, 3.0, 16.0]))
        core = decoded["platform_constraints"]["cores"][0]
        assert core["max_axons"] == CORE_DIM_GRANULARITY
        assert core["max_neurons"] == CORE_DIM_GRANULARITY


class TestCandidatePlatformResolutionIsTheOneSeam:
    def test_resolution_is_a_fixpoint(self):
        # Re-resolving an already resolved candidate changes nothing, so the
        # problem boundary may normalize every incoming candidate unconditionally.
        problem = _hw_problem()
        once = problem.resolve_candidate_platform(
            {"cores": [{"max_axons": 128, "max_neurons": 128, "count": 16}]}
        )
        assert problem.resolve_candidate_platform(once) == once

    def test_a_candidate_declared_outside_decode_is_resolved_at_the_boundary(self):
        # LLM optimizers hand the problem a raw platform dict instead of an
        # encoded vector; it must reach validation as the SAME resolved chip.
        problem = _hw_problem()
        raw = {
            "model_config": _fixed_model_config(),
            "platform_constraints": {
                "cores": [{"max_axons": 128, "max_neurons": 128, "count": 16}],
            },
        }
        resolved = {
            "model_config": _fixed_model_config(),
            "platform_constraints": problem.resolve_candidate_platform(
                raw["platform_constraints"]
            ),
        }
        assert problem.evaluate(raw) == problem.evaluate(resolved)
        assert len(problem._cache) == 1, (
            "the raw and resolved declarations are one candidate, one cache row"
        )

    def test_the_resolved_base_is_the_deployment_resolution(self):
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        assert problem.fixed_platform_constraints == build_fixed_platform_constraints(cfg)

    def test_the_constraint_check_sees_the_resolved_chip(self):
        # ``constraint_fn`` is the caller's own feasibility rule (a builder's
        # ``validate_config``). Handing it the raw declaration would ask it
        # about a chip nobody deploys — it could not even see the platform's
        # bias capability, which decides how a layer's fan-in is counted.
        # (``validate_detailed`` re-resolves idempotently, so passing it the
        # resolved candidate is the same call; the ``constraint_fn`` argument
        # is the observable half of that contract.)
        seen = []
        cfg = _pipeline_config()
        problem = _hw_problem(cfg)
        problem.constraint_fn = lambda mc, pcfg, shape: seen.append(pcfg) or 0.0

        raw = {
            "model_config": _fixed_model_config(),
            "platform_constraints": {
                "cores": [{"max_axons": 128, "max_neurons": 128, "count": 16}],
            },
        }
        assert problem.constraint_violation(raw) == 0.0
        assert seen == [problem.resolve_candidate_platform(raw["platform_constraints"])]
        assert seen[0]["cores"][0]["has_bias"] is False, (
            "the declared bias capability must reach the constraint check"
        )
        assert seen[0]["schedule_policy"] == "bank_clustered"

    def test_a_constraint_violation_is_reported_on_the_resolved_chip(self):
        # The mirror: a rule that rejects the RESOLVED chip must be able to.
        problem = _hw_problem()
        problem.constraint_fn = lambda mc, pcfg, shape: (
            5.0 if pcfg["cores"][0].get("has_bias") is False else 0.0
        )
        violation = problem.constraint_violation({
            "model_config": _fixed_model_config(),
            "platform_constraints": {
                "cores": [{"max_axons": 128, "max_neurons": 128, "count": 16}],
            },
        })
        assert violation == 5.0


class TestAnUnresolvableDeclarationIsScopedCorrectly:
    """A candidate's bad declaration is scored; the RUN's bad declaration aborts."""

    def test_a_candidate_platform_that_cannot_resolve_is_invalid_not_fatal(self):
        problem = _hw_problem()
        broken = {
            "model_config": _fixed_model_config(),
            # A tile grid too small for the declared capacity: the resolver's own
            # capacity invariant, tripped by the candidate rather than the run.
            "platform_constraints": {
                "cores": [{"max_axons": 64, "max_neurons": 64, "count": 40}],
                "cores_per_tile": 4, "tile_grid_rows": 2, "tile_grid_cols": 2,
            },
        }
        result = problem.validate_detailed(broken)
        assert not result.is_valid
        assert result.failure_phase == "structural"
        assert "does not resolve into a chip" in result.error_message
        assert problem.evaluate(broken) == problem._penalty_objectives()
        assert problem.constraint_violation(broken) > 0

    def test_a_declared_platform_that_cannot_resolve_aborts_the_run(self):
        cfg = _pipeline_config()
        cfg.update({"cores_per_tile": 4, "tile_grid_rows": 2, "tile_grid_cols": 2})
        problem = _hw_problem(cfg)
        with pytest.raises(ValueError, match="capacity"):
            problem.evaluate({
                "model_config": _fixed_model_config(), "platform_constraints": {},
            })


class TestBrokenProblemAborts:
    def test_decode_without_a_platform_resolver_aborts(self):
        problem = _hw_problem(platform_resolver=None)
        with pytest.raises(ValueError, match="platform_resolver"):
            problem.decode(np.array([128.0, 128.0, 16.0]))

    def test_evaluate_without_a_platform_resolver_aborts(self):
        problem = _hw_problem(platform_resolver=None)
        with pytest.raises(ValueError, match="platform_resolver"):
            problem.evaluate({"model_config": {}, "platform_constraints": {}})

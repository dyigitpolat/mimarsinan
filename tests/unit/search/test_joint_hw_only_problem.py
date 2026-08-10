"""Hardware-only ``JointArchHwProblem`` over the resolved platform base.

The searched chip and the deployed chip must be the same chip: candidates
decode from ``fixed_platform_constraints`` (the resolver output) overlaid with
only the decision variables, and a problem constructed without the resolved
base aborts the search instead of degenerating into all-penalty rows.
"""

import math

import numpy as np
import pytest
import torch

from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    build_fixed_platform_constraints,
)
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.problems.joint import JointArchHwProblem

HW_OBJECTIVES = [
    "total_param_capacity",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "axon_wastage_pct",
    "fragmentation_pct",
]


def _tiny_pipeline_config():
    return {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
    }


def _fixed_model_config():
    return {"mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU"}


def _make_hw_problem(fixed_platform_constraints="from_config"):
    cfg = _tiny_pipeline_config()
    if fixed_platform_constraints == "from_config":
        fixed_platform_constraints = build_fixed_platform_constraints(cfg)
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        target_tq=cfg["target_tq"],
        lr=cfg["lr"],
        search_mode="hardware",
        builder_factory=SimpleMLPBuilder,
        arch_options=(),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=_fixed_model_config(),
        fixed_platform_constraints=fixed_platform_constraints,
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


def _mid_x(problem):
    return (np.asarray(problem.xl) + np.asarray(problem.xu)) / 2.0


class TestHwOnlyDecodeCarriesResolvedBase:
    def test_decoded_platform_is_base_plus_decision_variables(self):
        problem = _make_hw_problem()
        decoded = problem.decode(_mid_x(problem))
        pcfg = decoded["platform_constraints"]

        # Decision variables come from x.
        assert len(pcfg["cores"]) == 1
        core = pcfg["cores"][0]
        assert 64 <= core["max_axons"] <= 256
        assert 64 <= core["max_neurons"] <= 256
        assert 8 <= core["count"] <= 64
        assert pcfg["target_tq"] == 4

        # Everything else is the resolved base, not synthesized defaults.
        assert pcfg["weight_bits"] == 4, "must carry config weight_bits, not a hardcoded 8"
        assert pcfg["allow_scheduling"] is True
        assert pcfg["allow_coalescing"] is False
        assert "schedule_policy" in pcfg
        assert "max_schedule_passes" in pcfg
        # W1.1: weight reuse is always on — the retired knob never resolves.
        assert "allow_weight_reuse" not in pcfg

    def test_decoded_cores_inherit_base_bias_capability(self):
        problem = _make_hw_problem()
        decoded = problem.decode(_mid_x(problem))
        for core in decoded["platform_constraints"]["cores"]:
            assert core["has_bias"] is True

    def test_fixed_model_config_is_carried(self):
        problem = _make_hw_problem()
        decoded = problem.decode(_mid_x(problem))
        assert decoded["model_config"] == _fixed_model_config()


class TestHwOnlyValidateAndEvaluate:
    def test_validate_and_evaluate_produce_finite_objectives(self):
        problem = _make_hw_problem()
        decoded = problem.decode(_mid_x(problem))

        vr = problem.validate_detailed(decoded)
        assert vr.is_valid, f"{vr.failure_phase}: {vr.error_message}"

        obj = problem.evaluate(decoded)
        assert set(obj) == set(HW_OBJECTIVES)
        for name, value in obj.items():
            assert math.isfinite(value), f"{name} not finite: {value}"
            assert abs(value) < 1e17, f"{name} looks like a penalty: {value}"

    def test_nsga2_hardware_search_completes_with_candidates(self):
        problem = _make_hw_problem()
        optimizer = NSGA2Optimizer(pop_size=4, generations=2, seed=0, verbose=False)
        result = optimizer.optimize(problem)
        assert result.best.configuration, "search must produce a best candidate"
        assert result.best.configuration["platform_constraints"]["cores"]
        for name, value in result.best.objectives.items():
            assert abs(value) < 1e17, f"{name} looks like a penalty: {value}"


class TestBrokenProblemAborts:
    def test_missing_base_aborts_nsga2_run(self):
        # A problem-level defect must abort the search loudly, never degrade
        # into penalty rows and a misleading "no candidates" result.
        problem = _make_hw_problem(fixed_platform_constraints=None)
        optimizer = NSGA2Optimizer(pop_size=4, generations=2, seed=0, verbose=False)
        with pytest.raises(ValueError, match="fixed_platform_constraints"):
            optimizer.optimize(problem)

    def test_decode_requires_resolved_base(self):
        problem = _make_hw_problem(fixed_platform_constraints=None)
        with pytest.raises(ValueError, match="fixed_platform_constraints"):
            problem.decode(np.array([128.0, 128.0, 16.0]))

    def test_hw_only_cache_requires_cores_in_base(self):
        problem = _make_hw_problem(fixed_platform_constraints={"target_tq": 4})
        with pytest.raises(ValueError, match="cores"):
            problem._ensure_hw_only_cache()

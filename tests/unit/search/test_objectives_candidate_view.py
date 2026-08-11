"""The candidate static view reproduces the layout hook's objective values exactly."""

import torch

from mimarsinan.deployment_record.objectives import (
    OBJECTIVES,
    CandidateStaticView,
    chip_param_capacity,
)
from mimarsinan.mapping.platform.coalescing import normalize_coalescing_config
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    build_fixed_platform_constraints,
)
from mimarsinan.search.problems.joint import JointArchHwProblem

HW_OBJECTIVES = [
    "total_param_capacity",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "axon_wastage_pct",
    "fragmentation_pct",
]


def _pipeline_config():
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


def _hw_problem():
    cfg = _pipeline_config()
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
        fixed_model_config={"mlp_width_1": 16, "mlp_width_2": 16,
                            "base_activation": "ReLU"},
        fixed_platform_constraints=build_fixed_platform_constraints(cfg),
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


class TestCandidateViewMatchesTheLayoutHook:
    def test_the_registry_reproduces_every_hook_objective_value(self):
        problem = _hw_problem()
        cache = problem._ensure_hw_only_cache()
        pcfg = dict(problem.fixed_platform_constraints or {})
        normalize_coalescing_config(pcfg)

        hook_values, error = problem._compute_hw_objectives(
            cache.softcores, pcfg, cache.total_params, cache.host_side_segment_count
        )
        assert error is None and hook_values is not None

        stats, stats_error = compute_mapping_stats(
            softcores=cache.softcores,
            core_types=problem._make_core_types(pcfg),
            **ChipCapabilities.from_platform_constraints(pcfg).permission_kwargs(),
        )
        assert stats_error is None or stats.feasible

        view = CandidateStaticView(
            layout=stats,
            chip_param_capacity=chip_param_capacity(pcfg["cores"]),
            total_params=cache.total_params,
            host_side_segment_count=cache.host_side_segment_count,
        )
        assert OBJECTIVES.extract(view) == hook_values

    def test_the_hook_capacity_helper_is_the_registry_helper(self):
        pcfg = dict(build_fixed_platform_constraints(_pipeline_config()))
        assert JointArchHwProblem._compute_chip_capacity(pcfg) == chip_param_capacity(
            pcfg["cores"]
        )

"""The candidate view the search builds IS the registry's view of the candidate."""

import torch

from mimarsinan.deployment_record.objectives import (
    OBJECTIVES,
    CandidateStaticView,
    chip_param_capacity,
    declared_core_capacity,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    build_fixed_platform_constraints,
    make_platform_resolver,
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
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=HW_OBJECTIVES,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


class TestCandidateViewMatchesTheLayoutHook:
    def test_the_hook_view_reproduces_the_registry_read_of_the_packing(self):
        problem = _hw_problem()
        pcfg = dict(problem.fixed_platform_constraints or {})
        cache = problem._ensure_hw_only_cache(problem.encoding_placement)
        softcores, host_segments = problem._collect_softcores(cache.model, pcfg)

        packed, error = problem._pack_candidate(softcores, pcfg)
        assert error is None and packed.feasible
        view = problem._static_view(
            packed, pcfg, cache.total_params, host_segments
        )

        # The SAME forwarding the hook uses: a layout answer computed with the
        # three permission bits alone is a different layout, so "byte parity with
        # the hook" would be asserted against a config the hook never runs.
        stats, stats_error = compute_mapping_stats(
            softcores=softcores,
            core_types=problem._make_core_types(pcfg),
            **ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs(),
        )
        assert stats_error is None or stats.feasible

        expected = CandidateStaticView(
            layout=stats,
            chip_param_capacity=chip_param_capacity(pcfg["cores"]),
            total_params=cache.total_params,
            host_side_segment_count=host_segments,
        )
        assert OBJECTIVES.extract(view) == OBJECTIVES.extract(expected)

    def test_the_hook_capacity_is_the_registry_capacity(self):
        pcfg = dict(build_fixed_platform_constraints(_pipeline_config()))
        assert declared_core_capacity(pcfg) == chip_param_capacity(pcfg["cores"])

    def test_a_candidate_view_carries_no_record_only_axis(self):
        # The candidate view answers only what a candidate holds; the sealed
        # record's axes stay unavailable instead of reading as zero.
        problem = _hw_problem()
        pcfg = dict(problem.fixed_platform_constraints or {})
        cache = problem._ensure_hw_only_cache(problem.encoding_placement)
        softcores, host_segments = problem._collect_softcores(cache.model, pcfg)
        packed, _error = problem._pack_candidate(softcores, pcfg)
        view = problem._static_view(
            packed, pcfg, cache.total_params, host_segments
        )
        assert "mj_per_sample" not in OBJECTIVES.extract(view)
        assert "estimated_accuracy" not in OBJECTIVES.extract(view)

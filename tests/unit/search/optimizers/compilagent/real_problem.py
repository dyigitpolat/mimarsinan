"""The REAL ``JointArchHwProblem`` the compilagent backend tests drive.

The backend is an ADAPTER over the problem's own surface — its platform
resolver, its model fixture, its layout hook, the registry read of its candidate
view. A hand-written double of that surface goes stale in silence: it did, and
the backend broke against every real problem (``HwOnlyCache.softcores``,
``ValidationEntry.hw_objectives``, ``_compute_hw_objectives`` — all gone) while
its tests stayed green on the double. So these fixtures are the real problem,
kept CHEAP rather than fake: a 3-softcore MLP over an 8x8 input, CPU-only, no
data provider (nothing in this suite trains).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from compilagent import ToleranceConfig, WorkloadKind, WorkloadSpec

from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
)
from mimarsinan.search.problems.joint import JointArchHwProblem

BACKEND_ID = "mimarsinan_layout"

HW_OBJECTIVES = (
    "total_param_capacity",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "axon_wastage_pct",
    "fragmentation_pct",
)
JOINT_OBJECTIVES = ("estimated_accuracy", "total_params", "param_utilization_pct")
ARCH_OPTIONS = (("mlp_width_1", (16, 32)), ("mlp_width_2", (16, 32)))


def pipeline_config(width: int = 16, has_bias: Optional[bool] = None) -> Dict[str, Any]:
    """A declared deployment: 8x8 inputs, 4 classes, one 256x256 core type."""
    cfg: Dict[str, Any] = {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "model_config": {
            "mlp_width_1": width,
            "mlp_width_2": width,
            "base_activation": "ReLU",
        },
    }
    if has_bias is not None:
        cfg["platform_constraints"] = {"has_bias": bool(has_bias)}
    return cfg


def make_problem(
    search_mode: str = "hardware",
    cfg: Optional[Dict[str, Any]] = None,
    objectives: Optional[Sequence[str]] = None,
) -> JointArchHwProblem:
    """The problem this backend adapts, built exactly as the pipeline step builds it."""
    cfg = pipeline_config() if cfg is None else cfg
    searches_model = search_mode in ("model", "joint")
    default_objectives = JOINT_OBJECTIVES if searches_model else HW_OBJECTIVES
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=int(cfg["num_classes"]),
        target_tq=int(cfg["target_tq"]),
        lr=float(cfg["lr"]),
        search_mode=search_mode,
        builder_factory=SimpleMLPBuilder,
        arch_options=ARCH_OPTIONS if searches_model else (),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=None if searches_model else dict(cfg["model_config"]),
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=tuple(
            objectives if objectives is not None else default_objectives
        ),
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


def make_workload(workload_id: str) -> WorkloadSpec:
    return WorkloadSpec(
        id=workload_id,
        title="tiny mlp",
        description="the real joint problem, kept small",
        kind=WorkloadKind.FULL_MODEL,
        backend_id=BACKEND_ID,
        tolerance=ToleranceConfig(atol=1.0, rtol=1.0),
    )

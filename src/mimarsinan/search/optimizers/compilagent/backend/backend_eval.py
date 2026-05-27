"""Objective evaluation and timing helpers for MimarsinanLayoutBackend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

from compilagent import (
    CompileResult,
    CorrectnessResult,
    Objective,
    Plan,
    TimingResult,
    ToleranceConfig,
    WorkloadSpec,
)

from mimarsinan.search.results import ObjectiveSpec

from ..plan_codec import CodecDefaults, decode_plan
from ..workload import lookup_problem


_PCT_OBJECTIVES = frozenset(
    {
        "param_utilization_pct",
        "neuron_wastage_pct",
        "axon_wastage_pct",
        "fragmentation_pct",
    }
)
_PARAM_OBJECTIVES = frozenset({"total_params", "total_param_capacity"})
_BARRIER_OBJECTIVES = frozenset({"total_sync_barriers"})
_ACC_OBJECTIVES = frozenset({"estimated_accuracy"})


def unit_for(name: str) -> str:
    if name in _PCT_OBJECTIVES:
        return "%"
    if name in _PARAM_OBJECTIVES:
        return "params"
    if name in _BARRIER_OBJECTIVES:
        return "barriers"
    if name in _ACC_OBJECTIVES:
        return ""
    return ""


def time_workload(
    backend: Any,
    workload: WorkloadSpec,
    plan: Plan,
    *,
    warmup: int,
    repetitions: int,
    max_seconds: Optional[float] = None,
) -> TimingResult:
    problem = lookup_problem(workload.id)
    description = backend._description_for(problem)
    defaults = CodecDefaults.from_description(
        description,
        fixed_model_config=getattr(problem, "fixed_model_config", None),
        fixed_platform_constraints=getattr(problem, "fixed_platform_constraints", None),
    )
    configuration = decode_plan(plan, defaults)
    evaluator = getattr(problem, "evaluate", None)
    if evaluator is None:
        return TimingResult(
            timings_ms=(),
            median_ms=None,
            p20_ms=None,
            p80_ms=None,
            diagnostics="problem has no evaluate method",
        )

    objectives = evaluator(configuration)
    return TimingResult(
        timings_ms=(),
        median_ms=None,
        p20_ms=None,
        p80_ms=None,
        profile_metrics={"objectives": dict(objectives or {})},
    )


def validate_correctness(
    workload: WorkloadSpec,
    baseline: CompileResult,
    candidate: CompileResult,
    tolerance: ToleranceConfig,
) -> CorrectnessResult:
    return CorrectnessResult(
        ok=True,
        diagnostics=(
            "layout-mapping integration: numerical correctness owned by "
            "mimarsinan's downstream pipeline; not enforced here"
        ),
    )


def objectives_for_candidate(
    workload: WorkloadSpec,
    plan: Plan,
    compile_result: CompileResult,
    timing_result: Optional[TimingResult],
) -> Mapping[str, Objective]:
    if not compile_result.ok or timing_result is None:
        return {}
    problem = lookup_problem(workload.id)
    objective_specs: Sequence[ObjectiveSpec] = list(problem.objectives)
    objective_values: Dict[str, float] = dict(
        (timing_result.profile_metrics or {}).get("objectives", {})
    )
    result: Dict[str, Objective] = {}
    for spec in objective_specs:
        if spec.name not in objective_values:
            continue
        result[spec.name] = Objective(
            name=spec.name,
            value=float(objective_values[spec.name]),
            goal=spec.goal,
            unit=unit_for(spec.name),
        )
    return result

"""Compilagent Backend adapter wrapping a live JointArchHwProblem."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from compilagent import (
    Analysis,
    BackendBase,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    Intervention,
    Objective,
    Plan,
    SearchSpace,
    TimingResult,
    ToleranceConfig,
    ValidationResult,
    WorkloadSpec,
)

from .backend_eval import (
    objectives_for_candidate as _objectives_for_candidate,
    time_workload as _time_workload,
    validate_correctness as _validate_correctness,
)
from .backend_layout import collect_layout_payload
from .backend_tools import (
    description_for,
    fire_pass,
    list_introspection_tools,
    platform_to_json_safe,
)
from ..lever_factory import levers_from_description
from ..plan_codec import CodecDefaults, PlanCodecError, decode_plan
from ..workload import lookup_problem
from .backend_validate import validate_intervention as _validate_intervention

logger = logging.getLogger(__name__)


class MimarsinanLayoutBackend(BackendBase):
    """Compilagent backend bridging mimarsinan's JointArchHwProblem."""

    id: str = "mimarsinan_layout"
    artifact_stages: Tuple[str, ...] = ("config", "softcores", "layout_stats")

    def __init__(self) -> None:
        self._candidate_payloads: Dict[str, Dict[str, Any]] = {}

    def device_capability(self) -> DeviceCapability:
        return DeviceCapability(
            arch="snn-crossbar",
            capability_int=None,
            name="Mimarsinan layout-mapping (heterogeneous neuromorphic crossbars)",
            memory_total_bytes=None,
            memory_peak_bandwidth_gbps=None,
            extra={"vendor": "mimarsinan", "kind": "neuromorphic"},
        )

    def infer_workload_family(self, workload: WorkloadSpec) -> Optional[str]:
        return "snn-mapping"

    def analyze(
        self,
        workload: WorkloadSpec,
        *,
        baseline_artifacts: Sequence[Path],
    ) -> Analysis:
        problem = lookup_problem(workload.id)
        description = self._description_for(problem)
        defaults = CodecDefaults.from_description(
            description,
            fixed_model_config=getattr(problem, "fixed_model_config", None),
            fixed_platform_constraints=getattr(
                problem, "fixed_platform_constraints", None
            ),
        )
        summary: Dict[str, Any] = {
            "kind": workload.kind.value,
            "search_mode": getattr(problem, "search_mode", "joint"),
            "objective_catalog": [
                {"name": s.name, "goal": s.goal} for s in problem.objectives
            ],
        }
        extra: Dict[str, Any] = {
            "defaults": {
                "model_config": dict(defaults.model_config),
                "platform_constraints": platform_to_json_safe(
                    defaults.platform_constraints
                ),
            },
        }
        try:
            baseline_payload = collect_layout_payload(
                problem,
                {
                    "model_config": dict(defaults.model_config),
                    "platform_constraints": platform_to_json_safe(
                        defaults.platform_constraints
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Baseline layout payload collection failed for workload %s; "
                "recording baseline_error in analysis",
                workload.id, exc_info=True,
            )
            extra["baseline_error"] = repr(exc)
        else:
            summary["softcore_count_baseline"] = baseline_payload["softcore_count"]
            summary["layer_count"] = len(baseline_payload["per_layer"])
            extra["baseline"] = baseline_payload
        return Analysis(summary=summary, extra=extra)

    def derive_search_space(
        self,
        workload: WorkloadSpec,
        analysis: Analysis,
    ) -> SearchSpace:
        problem = lookup_problem(workload.id)
        description = self._description_for(problem)
        levers = levers_from_description(
            description, workload_id=workload.id, backend_id=self.id,
        )
        return SearchSpace(
            workload_id=workload.id, backend_id=self.id, levers=tuple(levers),
        )

    def validate_intervention(self, intervention: Intervention) -> ValidationResult:
        return _validate_intervention(intervention)

    def compile(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        artifact_dir: Path,
        pass_callback: Any = None,
    ) -> CompileResult:
        problem = lookup_problem(workload.id)
        description = self._description_for(problem)
        defaults = CodecDefaults.from_description(
            description,
            fixed_model_config=getattr(problem, "fixed_model_config", None),
            fixed_platform_constraints=getattr(
                problem, "fixed_platform_constraints", None
            ),
        )

        try:
            configuration = decode_plan(plan, defaults)
        except PlanCodecError as exc:
            return CompileResult(
                ok=False,
                diagnostics=f"plan decode failed: {exc}",
                metadata={"failure_phase": "plan_decode"},
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)

        compile_started = time.perf_counter()
        if pass_callback is not None:
            fire_pass(pass_callback, "decode", "decode_plan", compile_started)

        validate = getattr(problem, "validate_detailed", None)
        if validate is None:
            return CompileResult(
                ok=False,
                diagnostics="problem has no validate_detailed method",
                metadata={"failure_phase": "internal"},
            )

        validate_started = time.perf_counter()
        vr = validate(configuration)
        if pass_callback is not None:
            fire_pass(
                pass_callback, "validate", "validate_detailed", validate_started,
            )

        if not vr.is_valid:
            elapsed = (time.perf_counter() - compile_started) * 1000.0
            return CompileResult(
                ok=False,
                elapsed_ms=elapsed,
                diagnostics=vr.error_message or "validation failed",
                metadata={
                    "failure_phase": vr.failure_phase or "validation",
                    "config": configuration,
                },
            )

        try:
            payload = collect_layout_payload(problem, configuration)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Layout payload collection failed for workload %s config %.500s; "
                "returning failed CompileResult",
                workload.id, configuration, exc_info=True,
            )
            elapsed = (time.perf_counter() - compile_started) * 1000.0
            return CompileResult(
                ok=False,
                elapsed_ms=elapsed,
                diagnostics=f"layout payload collection failed: {exc!r}",
                metadata={
                    "failure_phase": "layout_collection",
                    "config": configuration,
                },
            )

        config_path = artifact_dir / "config.json"
        config_path.write_text(json.dumps(configuration, indent=2, default=str))
        softcores_path = artifact_dir / "softcores.json"
        softcores_path.write_text(json.dumps(payload["per_softcore"], indent=2, default=str))
        layout_stats_path = artifact_dir / "layout_stats.json"
        layout_stats_path.write_text(json.dumps(payload["layout_stats"], indent=2, default=str))

        elapsed = (time.perf_counter() - compile_started) * 1000.0
        candidate_id = artifact_dir.name
        cached_payload = {
            "config": configuration,
            "softcores": payload["per_softcore"],
            "per_layer": payload["per_layer"],
            "layout_stats": payload["layout_stats"],
            "hw_objectives": payload["hw_objectives"],
            "objective_catalog": [
                {"name": s.name, "goal": s.goal} for s in problem.objectives
            ],
        }
        self._candidate_payloads[candidate_id] = cached_payload
        return CompileResult(
            ok=True,
            elapsed_ms=elapsed,
            artifacts=(config_path, softcores_path, layout_stats_path),
            metadata=cached_payload,
        )

    def get_candidate_payload(self, candidate_id: str) -> Dict[str, Any]:
        return self._candidate_payloads[candidate_id]

    def known_candidate_ids(self) -> Tuple[str, ...]:
        return tuple(self._candidate_payloads.keys())

    def time_workload(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        *,
        warmup: int,
        repetitions: int,
        max_seconds: Optional[float] = None,
    ) -> TimingResult:
        return _time_workload(
            self, workload, plan,
            warmup=warmup, repetitions=repetitions, max_seconds=max_seconds,
        )

    def validate_correctness(
        self,
        workload: WorkloadSpec,
        baseline: CompileResult,
        candidate: CompileResult,
        tolerance: ToleranceConfig,
    ) -> CorrectnessResult:
        return _validate_correctness(workload, baseline, candidate, tolerance)

    def objectives_for_candidate(
        self,
        workload: WorkloadSpec,
        plan: Plan,
        compile_result: CompileResult,
        timing_result: Optional[TimingResult],
    ) -> Mapping[str, Objective]:
        return _objectives_for_candidate(
            workload, plan, compile_result, timing_result,
        )

    def list_introspection_tools(self) -> Sequence[Any]:
        return list_introspection_tools(self)

    def _description_for(self, problem: Any):
        return description_for(problem)


__all__ = ["MimarsinanLayoutBackend"]

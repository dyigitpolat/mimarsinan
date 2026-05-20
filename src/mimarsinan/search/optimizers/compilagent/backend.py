"""Compilagent Backend adapter wrapping a live JointArchHwProblem."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compilagent import (
    Analysis,
    BackendBase,
    CompileResult,
    CorrectnessResult,
    DeviceCapability,
    Intervention,
    Objective,
    PassEvent,
    Plan,
    SearchSpace,
    TimingResult,
    ToleranceConfig,
    ValidationResult,
    WorkloadSpec,
)

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.layout_verification_stats import compute_mapping_stats
from mimarsinan.search.results import ObjectiveSpec, resolve_active_objectives
from mimarsinan.search.search_space_description import SearchSpaceDescription

from .lever_factory import levers_from_description
from .plan_codec import (
    ARCH_KIND,
    HW_CORE_KIND,
    HW_DIM_NAMES,
    CodecDefaults,
    PlanCodecError,
    decode_plan,
)
from .workload import lookup_problem


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
        """Inspect the baseline configuration and surface layout internals."""

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
                "platform_constraints": _platform_to_json_safe(
                    defaults.platform_constraints
                ),
            },
        }
        try:
            baseline_payload = self._collect_layout_payload(
                problem,
                {
                    "model_config": dict(defaults.model_config),
                    "platform_constraints": _platform_to_json_safe(
                        defaults.platform_constraints
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
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
        target = intervention.target
        if target.kind not in (ARCH_KIND, HW_CORE_KIND):
            return ValidationResult(
                ok=False,
                errors=(
                    f"unknown target.kind {target.kind!r}; expected "
                    f"{ARCH_KIND!r} or {HW_CORE_KIND!r}",
                ),
            )
        if target.kind == ARCH_KIND:
            if not target.selector:
                return ValidationResult(
                    ok=False,
                    errors=("arch intervention requires a non-empty selector",),
                )
            return ValidationResult(ok=True)
        parts = target.selector.split(".")
        if len(parts) != 2 or parts[1] not in HW_DIM_NAMES:
            return ValidationResult(
                ok=False,
                errors=(
                    f"hw.core selector must be `<core_index>.{{{'|'.join(HW_DIM_NAMES)}}}`, "
                    f"got {target.selector!r}",
                ),
            )
        try:
            int(parts[0])
            value = int(intervention.payload)
        except (TypeError, ValueError):
            return ValidationResult(
                ok=False,
                errors=(
                    f"hw.core core index and payload must be integers, "
                    f"got {parts[0]!r}/{intervention.payload!r}",
                ),
            )
        if value <= 0:
            return ValidationResult(
                ok=False,
                errors=(
                    f"hw.core payload for {target.selector} must be a "
                    f"positive integer, got {value}",
                ),
            )
        _HARD_MAX = 65536
        if value > _HARD_MAX:
            return ValidationResult(
                ok=False,
                errors=(
                    f"hw.core payload for {target.selector} must be "
                    f"<= {_HARD_MAX}, got {value}",
                ),
            )
        # max_axons/max_neurons should be multiples of 8 for the layout packer.
        if parts[1] in ("max_axons", "max_neurons") and value % 8 != 0:
            return ValidationResult(
                ok=False,
                errors=(
                    f"hw.core payload for {target.selector} should be a "
                    f"multiple of 8 (the layout packer pads otherwise); "
                    f"got {value}. Snap to {(value // 8) * 8} or "
                    f"{((value + 7) // 8) * 8}.",
                ),
            )
        return ValidationResult(ok=True)

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
            self._fire_pass(pass_callback, "decode", "decode_plan", compile_started)

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
            self._fire_pass(
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
            payload = self._collect_layout_payload(problem, configuration)
        except Exception as exc:  # noqa: BLE001
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
        """Return the cached compile payload for ``candidate_id``."""

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
        problem = lookup_problem(workload.id)
        description = self._description_for(problem)
        defaults = CodecDefaults.from_description(
            description,
            fixed_model_config=getattr(problem, "fixed_model_config", None),
            fixed_platform_constraints=getattr(
                problem, "fixed_platform_constraints", None
            ),
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
        self,
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
        self,
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
                unit=_unit_for(spec.name),
            )
        return result

    def list_introspection_tools(self) -> Sequence[Any]:
        from .tools import build_introspection_tools

        return build_introspection_tools(self)

    def _description_for(self, problem: Any) -> SearchSpaceDescription:
        """Reconstruct the SearchSpaceDescription from the live problem."""

        return SearchSpaceDescription(
            search_mode=str(getattr(problem, "search_mode", "joint")),
            arch_options=tuple(
                (str(k), tuple(v)) for k, v in getattr(problem, "arch_options", ())
            ),
            num_core_types=int(getattr(problem, "num_core_types", 1)),
            core_axons_bounds=tuple(getattr(problem, "core_axons_bounds", (64, 1024))),
            core_neurons_bounds=tuple(
                getattr(problem, "core_neurons_bounds", (64, 1024))
            ),
            core_count_bounds=tuple(getattr(problem, "core_count_bounds", (50, 500))),
            target_tq=int(getattr(problem, "target_tq", 32)),
            weight_bits=int(8),
        )

    def _collect_layout_payload(
        self,
        problem: Any,
        configuration: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pull per-softcore, per-layer, and layout_stats data via the problem."""

        from mimarsinan.search.problems.joint_arch_hw_problem import _json_key

        pcfg = configuration.get("platform_constraints", {})
        cores_cfg = pcfg.get("cores", [])
        from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

        core_types = [
            LayoutHardCoreType(
                max_axons=int(c["max_axons"]),
                max_neurons=int(c["max_neurons"]),
                count=int(c["count"]),
            )
            for c in cores_cfg
        ]

        cache = getattr(problem, "_hw_only_cache", None)
        softcores: List[LayoutSoftCoreSpec]
        host_segments: int
        total_params: float
        if cache is not None and getattr(problem, "search_mode", "joint") == "hardware":
            softcores = list(cache.softcores)
            host_segments = int(cache.host_side_segment_count)
            total_params = float(cache.total_params)
        else:
            mc = configuration.get("model_config", {})
            try:
                model, total_params = problem._build_model(mc, pcfg)
            except Exception:
                key = _json_key(configuration)
                vc = getattr(problem, "_validation_cache", {}).get(key)
                if vc is None:
                    raise
                hw_obj = vc.hw_objectives
                return {
                    "softcore_count": 0,
                    "per_softcore": [],
                    "per_layer": [],
                    "layout_stats": {},
                    "hw_objectives": dict(hw_obj),
                }
            softcores, host_segments = problem._collect_softcores(model, pcfg)

        stats, _err = compute_mapping_stats(
            softcores=softcores,
            core_types=core_types,
            allow_scheduling=bool(pcfg.get("allow_scheduling", False)),
            allow_neuron_splitting=bool(pcfg.get("allow_neuron_splitting", False)),
            allow_coalescing=bool(pcfg.get("allow_coalescing", False)),
        )
        per_softcore = [_softcore_to_dict(sc, idx) for idx, sc in enumerate(softcores)]
        per_layer = _aggregate_per_layer(softcores)
        hw_objectives, _ = problem._compute_hw_objectives(
            softcores, pcfg, total_params, host_segments,
        )

        return {
            "softcore_count": len(softcores),
            "per_softcore": per_softcore,
            "per_layer": per_layer,
            "layout_stats": stats.to_dict() if stats else {},
            "hw_objectives": dict(hw_objectives or {}),
        }

    @staticmethod
    def _fire_pass(
        callback: Any, stage: str, name: str, started_at: float,
    ) -> None:
        try:
            callback(
                PassEvent(
                    stage=stage,
                    name=name,
                    duration_ms=(time.perf_counter() - started_at) * 1000.0,
                )
            )
        except Exception:
            pass


def _platform_to_json_safe(pcfg: Mapping[str, Any]) -> Dict[str, Any]:
    from mimarsinan.gui.json_util import to_json_safe

    return to_json_safe(dict(pcfg))


def _softcore_to_dict(sc: LayoutSoftCoreSpec, index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "name": sc.name,
        "input_count": int(sc.input_count),
        "output_count": int(sc.output_count),
        "area": int(sc.area),
        "threshold_group_id": int(sc.threshold_group_id),
        "latency_tag": (None if sc.latency_tag is None else int(sc.latency_tag)),
        "segment_id": (None if sc.segment_id is None else int(sc.segment_id)),
    }


def _aggregate_per_layer(
    softcores: Sequence[LayoutSoftCoreSpec],
) -> List[Dict[str, Any]]:
    """Roll per-softcore facts up to per-layer rows for the agent."""

    by_layer: Dict[str, Dict[str, Any]] = {}
    for sc in softcores:
        key = _layer_key(sc)
        row = by_layer.setdefault(
            key,
            {
                "layer": key,
                "softcore_count": 0,
                "total_area": 0,
                "max_input_count": 0,
                "max_output_count": 0,
                "threshold_groups": set(),
                "latency_tags": set(),
                "segments": set(),
            },
        )
        row["softcore_count"] += 1
        row["total_area"] += int(sc.area)
        row["max_input_count"] = max(row["max_input_count"], int(sc.input_count))
        row["max_output_count"] = max(row["max_output_count"], int(sc.output_count))
        row["threshold_groups"].add(int(sc.threshold_group_id))
        if sc.latency_tag is not None:
            row["latency_tags"].add(int(sc.latency_tag))
        if sc.segment_id is not None:
            row["segments"].add(int(sc.segment_id))

    rows: List[Dict[str, Any]] = []
    for row in by_layer.values():
        rows.append(
            {
                "layer": row["layer"],
                "softcore_count": row["softcore_count"],
                "total_area": row["total_area"],
                "max_input_count": row["max_input_count"],
                "max_output_count": row["max_output_count"],
                "threshold_group_count": len(row["threshold_groups"]),
                "latency_tag_count": len(row["latency_tags"]),
                "segment_count": len(row["segments"]),
            }
        )
    rows.sort(key=lambda r: r["layer"])
    return rows


def _layer_key(sc: LayoutSoftCoreSpec) -> str:
    name = sc.name or f"unnamed_tg{int(sc.threshold_group_id)}"
    for sep in ("_tile_", "_psum_pos_", "_psum_neg_", "_psum_accum_", "_pos", "_col"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name


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


def _unit_for(name: str) -> str:
    if name in _PCT_OBJECTIVES:
        return "%"
    if name in _PARAM_OBJECTIVES:
        return "params"
    if name in _BARRIER_OBJECTIVES:
        return "barriers"
    if name in _ACC_OBJECTIVES:
        return ""
    return ""


__all__ = ["MimarsinanLayoutBackend"]

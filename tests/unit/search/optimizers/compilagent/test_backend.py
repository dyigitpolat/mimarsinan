"""Tests for ``MimarsinanLayoutBackend``.

These tests use a synthetic ``JointArchHwProblem``-like fake so they
remain fast (no torch model build, no data loading) while still
exercising the full delegation path through the real
``compute_mapping_stats`` / softcore helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import pytest
from compilagent import (
    Intervention,
    Plan,
    Target,
    ToleranceConfig,
    WorkloadKind,
    WorkloadSpec,
)

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.backend.backend_eval import unit_for as _unit_for
from mimarsinan.search.optimizers.compilagent.backend.backend_layout import (
    aggregate_per_layer as _aggregate_per_layer,
    layer_key as _layer_key,
    softcore_to_dict as _softcore_to_dict,
)
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)
from mimarsinan.search.problem import ValidationResult
from mimarsinan.search.results import ObjectiveSpec


# ---------------------------------------------------------------------- fakes


@dataclass
class _HwOnlyCache:
    softcores: List[LayoutSoftCoreSpec]
    total_params: float
    host_side_segment_count: int


@dataclass
class _FakeProblem:
    """Mimics the public surface of ``JointArchHwProblem`` the backend reads.

    HW-only mode keeps the backend on the cheap path (no model build) so
    the tests stay deterministic and torch-free.
    """

    softcores: List[LayoutSoftCoreSpec]
    fixed_platform_constraints: Dict[str, Any] = field(default_factory=dict)
    fixed_model_config: Dict[str, Any] = field(default_factory=dict)
    arch_options: tuple = ()
    num_core_types: int = 1
    core_axons_bounds: tuple = (64, 1024)
    core_neurons_bounds: tuple = (64, 1024)
    core_count_bounds: tuple = (50, 500)
    target_tq: int = 32
    search_mode: str = "hardware"
    _last_validate_config: Dict[str, Any] = field(default_factory=dict)
    _validation_cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def _hw_only_cache(self):
        return _HwOnlyCache(
            softcores=list(self.softcores),
            total_params=128.0,
            host_side_segment_count=1,
        )

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return [
            ObjectiveSpec("total_param_capacity", "min"),
            ObjectiveSpec("param_utilization_pct", "max"),
            ObjectiveSpec("fragmentation_pct", "min"),
        ]

    def validate_detailed(self, configuration: Dict[str, Any]) -> ValidationResult:
        self._last_validate_config = configuration
        cores = configuration.get("platform_constraints", {}).get("cores", [])
        if not cores or any(c.get("max_axons", 0) <= 0 for c in cores):
            return ValidationResult(
                is_valid=False,
                error_message="cores must have positive max_axons",
                failure_phase="structural",
            )
        # Specific failure trigger for tests: ``count == 9999`` is the
        # sentinel for "passes basic sanity but fails problem-side
        # validation" (e.g. simulates a packing failure that surfaces
        # only after the model is built).
        if any(c.get("count") == 9999 for c in cores):
            return ValidationResult(
                is_valid=False,
                error_message="simulated packing failure",
                failure_phase="hw_packing",
            )
        return ValidationResult(is_valid=True)

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        # Trivial: return a deterministic dict per config.
        cores = configuration["platform_constraints"]["cores"]
        capacity = sum(c["max_axons"] * c["max_neurons"] * c["count"] for c in cores)
        return {
            "total_param_capacity": float(capacity),
            "param_utilization_pct": 75.0,
            "fragmentation_pct": 12.5,
        }

    def _compute_hw_objectives(
        self, softcores, pcfg, total_params, host_side_segment_count,
    ):
        # Backend only calls this from `_collect_layout_payload`; mirror
        # `evaluate()` so tests can compare them directly.
        cores = pcfg.get("cores", [])
        capacity = sum(c["max_axons"] * c["max_neurons"] * c["count"] for c in cores)
        return (
            {
                "total_param_capacity": float(capacity),
                "param_utilization_pct": 75.0,
                "fragmentation_pct": 12.5,
            },
            None,
        )


def _make_softcores() -> List[LayoutSoftCoreSpec]:
    return [
        LayoutSoftCoreSpec(
            input_count=64, output_count=32, threshold_group_id=0,
            latency_tag=0, segment_id=0, name="conv1_pos0_0",
        ),
        LayoutSoftCoreSpec(
            input_count=64, output_count=32, threshold_group_id=0,
            latency_tag=0, segment_id=0, name="conv1_pos1_0",
        ),
        LayoutSoftCoreSpec(
            input_count=128, output_count=64, threshold_group_id=1,
            latency_tag=1, segment_id=0, name="fc1_tile_0_64",
        ),
    ]


def _make_problem() -> _FakeProblem:
    return _FakeProblem(
        softcores=_make_softcores(),
        fixed_platform_constraints={
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 100}],
            "target_tq": 32,
            "weight_bits": 8,
        },
    )


def _make_workload(workload_id: str) -> WorkloadSpec:
    return WorkloadSpec(
        id=workload_id,
        title="fake",
        description="fake",
        kind=WorkloadKind.FULL_MODEL,
        backend_id="mimarsinan_layout",
        tolerance=ToleranceConfig(atol=1.0, rtol=1.0),
    )


@pytest.fixture
def registered_problem():
    workload_id = "fake_layout_test"
    problem = _make_problem()
    register_problem(workload_id, problem)
    try:
        yield workload_id, problem
    finally:
        unregister_problem(workload_id)


# ------------------------------------------------------------------- tests


class TestStaticHelpers:
    def test_softcore_to_dict_round_trip(self):
        sc = _make_softcores()[0]
        d = _softcore_to_dict(sc, 7)
        assert d["index"] == 7
        assert d["name"] == "conv1_pos0_0"
        assert d["input_count"] == 64
        assert d["output_count"] == 32
        assert d["area"] == 64 * 32

    def test_layer_key_strips_pos_suffix(self):
        sc = _make_softcores()[0]
        assert _layer_key(sc) == "conv1"

    def test_layer_key_strips_tile_suffix(self):
        sc = _make_softcores()[2]
        assert _layer_key(sc) == "fc1"

    def test_aggregate_per_layer_collapses_tiles(self):
        rows = _aggregate_per_layer(_make_softcores())
        layer_names = sorted(r["layer"] for r in rows)
        assert layer_names == ["conv1", "fc1"]
        conv = next(r for r in rows if r["layer"] == "conv1")
        assert conv["softcore_count"] == 2
        assert conv["total_area"] == 64 * 32 * 2
        assert conv["threshold_group_count"] == 1
        assert conv["latency_tag_count"] == 1
        assert conv["segment_count"] == 1
        fc = next(r for r in rows if r["layer"] == "fc1")
        assert fc["softcore_count"] == 1
        assert fc["max_input_count"] == 128

    def test_unit_for_known_objectives(self):
        assert _unit_for("fragmentation_pct") == "%"
        assert _unit_for("total_params") == "params"
        assert _unit_for("total_sync_barriers") == "barriers"
        assert _unit_for("estimated_accuracy") == ""
        assert _unit_for("unknown") == ""


class TestDeviceCapabilityAndAnalyse:
    def test_device_capability_arch(self):
        backend = MimarsinanLayoutBackend()
        cap = backend.device_capability()
        assert cap.arch == "snn-crossbar"
        assert cap.extra["vendor"] == "mimarsinan"

    def test_analyze_includes_per_layer_and_layout_stats(self, registered_problem):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        analysis = backend.analyze(workload, baseline_artifacts=())
        assert analysis.summary["kind"] == "full_model"
        assert analysis.summary["search_mode"] == "hardware"
        assert "layer_count" in analysis.summary
        assert analysis.summary["layer_count"] == 2  # conv1 + fc1
        baseline = analysis.extra.get("baseline")
        assert baseline is not None
        assert baseline["softcore_count"] == 3
        assert {row["layer"] for row in baseline["per_layer"]} == {"conv1", "fc1"}
        assert baseline["layout_stats"]
        assert "total_param_capacity" in baseline["hw_objectives"]


class TestSearchSpace:
    def test_derive_search_space_returns_levers(self, registered_problem):
        workload_id, problem = registered_problem
        # Add a single arch option so we get an arch lever as well.
        problem.arch_options = (("activation", ("ReLU", "GELU")),)
        problem.search_mode = "joint"
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        analysis = backend.analyze(workload, baseline_artifacts=())
        space = backend.derive_search_space(workload, analysis)
        assert space.workload_id == workload_id
        assert space.backend_id == "mimarsinan_layout"
        kinds = {lv.target_kind for lv in space.levers}
        assert kinds == {"arch", "hw.core"}


class TestValidateIntervention:
    def test_unknown_kind_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(target=Target(kind="bogus", selector="x"), payload=1)
        assert backend.validate_intervention(iv).ok is False

    def test_arch_without_selector_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(target=Target(kind="arch", selector=""), payload=1)
        assert backend.validate_intervention(iv).ok is False

    def test_hw_core_with_bad_selector_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(
            target=Target(kind="hw.core", selector="0.bogus"), payload=128
        )
        assert backend.validate_intervention(iv).ok is False

    def test_valid_arch_intervention_is_accepted(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(target=Target(kind="arch", selector="activation"), payload="ReLU")
        assert backend.validate_intervention(iv).ok is True

    def test_valid_hw_core_intervention_is_accepted(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(
            target=Target(kind="hw.core", selector="0.max_axons"), payload=256
        )
        assert backend.validate_intervention(iv).ok is True

    def test_zero_or_negative_axon_count_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        for bad_value in (0, -8):
            iv = Intervention(
                target=Target(kind="hw.core", selector="0.max_axons"),
                payload=bad_value,
            )
            result = backend.validate_intervention(iv)
            assert result.ok is False
            assert "positive integer" in result.errors[0]

    def test_obviously_huge_payload_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(
            target=Target(kind="hw.core", selector="0.count"),
            payload=10**8,
        )
        result = backend.validate_intervention(iv)
        assert result.ok is False

    def test_non_8_multiple_axons_is_rejected_with_snap_hint(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(
            target=Target(kind="hw.core", selector="0.max_neurons"),
            payload=250,
        )
        result = backend.validate_intervention(iv)
        assert result.ok is False
        assert "multiple of 8" in result.errors[0]
        assert "248" in result.errors[0] and "256" in result.errors[0]

    def test_count_does_not_require_multiple_of_8(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(
            target=Target(kind="hw.core", selector="0.count"),
            payload=13,
        )
        assert backend.validate_intervention(iv).ok is True


class TestCompile:
    def test_empty_plan_compiles_against_defaults(self, registered_problem, tmp_path):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        result = backend.compile(workload, Plan(), artifact_dir=tmp_path)
        assert result.ok, result.diagnostics
        assert result.artifacts and len(result.artifacts) == 3
        # Each artifact should be a real, non-empty JSON file
        for path in result.artifacts:
            data = json.loads(path.read_text())
            assert data is not None
        # Metadata exposes the rich layout payload + hw objectives
        assert "softcores" in result.metadata
        assert "per_layer" in result.metadata
        assert "layout_stats" in result.metadata
        assert "hw_objectives" in result.metadata

    def test_invalid_plan_returns_failed_compile(self, registered_problem, tmp_path):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        # Force the validator to fail by setting max_axons to 0.
        plan = Plan(
            interventions=(
                Intervention(
                    target=Target(kind="hw.core", selector="0.max_axons"),
                    payload=0,
                ),
            )
        )
        result = backend.compile(workload, plan, artifact_dir=tmp_path)
        assert result.ok is False
        assert result.metadata["failure_phase"] == "structural"
        assert "positive max_axons" in (result.diagnostics or "")


class TestTimeWorkload:
    def test_time_workload_leaves_single_axis_empty_and_exposes_full_objectives(
        self, registered_problem, tmp_path,
    ):
        """Multi-objective backends deliberately leave ``median_ms`` as
        ``None`` so compilagent's single-axis leaderboard becomes
        informationless and the agent is forced to use ``pareto_front`` /
        ``metric_summary`` / ``query_top_candidates`` instead. The full
        objective tuple lives in ``profile_metrics['objectives']`` and
        on the ``Backend.objectives_for_candidate`` hook."""

        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        timing = backend.time_workload(
            workload, Plan(), warmup=0, repetitions=1, max_seconds=1.0,
        )
        assert timing.median_ms is None
        assert "primary_objective" not in timing.profile_metrics
        full = timing.profile_metrics["objectives"]
        assert set(full) == {
            "total_param_capacity", "param_utilization_pct", "fragmentation_pct",
        }


class TestObjectivesForCandidate:
    def test_returns_objective_objects_with_goals(self, registered_problem, tmp_path):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        compile_outcome = backend.compile(workload, Plan(), artifact_dir=tmp_path)
        timing = backend.time_workload(
            workload, Plan(), warmup=0, repetitions=1, max_seconds=1.0,
        )
        objectives = backend.objectives_for_candidate(
            workload, Plan(), compile_outcome, timing,
        )
        assert set(objectives) == {
            "total_param_capacity", "param_utilization_pct", "fragmentation_pct",
        }
        assert objectives["param_utilization_pct"].goal == "max"
        assert objectives["fragmentation_pct"].goal == "min"
        assert objectives["fragmentation_pct"].unit == "%"

    def test_empty_dict_when_compile_failed(self, registered_problem):
        from compilagent import CompileResult

        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        bad = CompileResult(ok=False, diagnostics="x")
        objectives = backend.objectives_for_candidate(
            workload, Plan(), bad, timing_result=None,
        )
        assert objectives == {}

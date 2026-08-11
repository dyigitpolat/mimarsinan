"""Tests for ``MimarsinanLayoutBackend``.

The backend is driven against the REAL ``JointArchHwProblem`` (see
``real_problem.py``): it reads that problem's resolver, model fixture, layout
hook and objective registry, so a stand-in for those members is a stand-in for
the very thing under test. Only two doubles remain, and both SUBCLASS the real
problem to override exactly one method — the raise-site whose rendering is being
pinned.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from compilagent import (
    Intervention,
    Plan,
    Target,
)

from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.backend.backend_eval import unit_for as _unit_for
from mimarsinan.search.optimizers.compilagent.backend.backend_layout import (
    aggregate_per_layer as _aggregate_per_layer,
    collect_layout_payload,
    layer_key as _layer_key,
    softcore_to_dict as _softcore_to_dict,
)
from mimarsinan.search.optimizers.compilagent.plan_codec import CodecDefaults, decode_plan
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)

from .real_problem import (
    HW_OBJECTIVES,
    make_problem,
    make_workload as _make_workload,
    pipeline_config,
)


def _make_problem():
    """The default registered problem: a hardware search over the tiny MLP."""
    return make_problem("hardware")


def _make_softcores() -> List[LayoutSoftCoreSpec]:
    """Synthetic softcores for the PURE name/aggregation helpers below."""
    return [
        LayoutSoftCoreSpec(
            input_count=64, output_count=32, residency_class_id=0,
            latency_tag=0, segment_id=0, name="conv1_pos0_0",
        ),
        LayoutSoftCoreSpec(
            input_count=64, output_count=32, residency_class_id=0,
            latency_tag=0, segment_id=0, name="conv1_pos1_0",
        ),
        LayoutSoftCoreSpec(
            input_count=128, output_count=64, residency_class_id=1,
            latency_tag=1, segment_id=0, name="fc1_tile_0_64",
        ),
    ]


def _plan(*interventions) -> Plan:
    return Plan(interventions=tuple(interventions))


def _hw_core(selector: str, payload: Any) -> Intervention:
    return Intervention(target=Target(kind="hw.core", selector=selector), payload=payload)


def _baseline_configuration(problem) -> Dict[str, Any]:
    """The configuration an empty plan decodes to — the backend's own path."""
    backend = MimarsinanLayoutBackend()
    description = backend._description_for(problem)
    return decode_plan(
        Plan(),
        CodecDefaults.from_description(
            description,
            fixed_model_config=problem.fixed_model_config,
            fixed_platform_constraints=problem.fixed_platform_constraints,
        ),
    )


@pytest.fixture
def registered_problem():
    workload_id = "real_layout_test"
    problem = _make_problem()
    register_problem(workload_id, problem)
    try:
        yield workload_id, problem
    finally:
        unregister_problem(workload_id)


def _registered(workload_id: str, problem):
    register_problem(workload_id, problem)
    return problem


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
        assert conv["residency_class_count"] == 1
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


class TestLayoutPayloadIsTheProblemsOwnCandidateLayout:
    """The payload is the problem's ``candidate_layout``, rendered — nothing else."""

    def test_payload_softcores_are_the_candidates_own_layout(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)

        layout = problem.candidate_layout(configuration)
        assert payload["softcore_count"] == len(layout.softcores)
        assert payload["softcore_count"] > 0, "the fixture must map something"
        assert payload["per_softcore"] == [
            _softcore_to_dict(sc, idx) for idx, sc in enumerate(layout.softcores)
        ]
        assert payload["per_layer"] == _aggregate_per_layer(layout.softcores)
        assert payload["layout_stats"] == layout.stats.to_dict()

    def test_payload_objectives_are_the_registry_read_of_the_candidate_view(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)

        layout = problem.candidate_layout(configuration)
        assert payload["hw_objectives"] == OBJECTIVES.extract(layout.view)
        assert set(payload["hw_objectives"]) >= set(HW_OBJECTIVES), (
            "every active hardware axis must be readable off the payload"
        )
        assert "estimated_accuracy" not in payload["hw_objectives"], (
            "a static view carries no training proxy"
        )

    def test_payload_axes_agree_with_what_the_search_scores(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)
        scored = problem.evaluate(configuration)
        for name, value in scored.items():
            assert payload["hw_objectives"][name] == value, (
                f"{name}: the agent is shown a different number than the search "
                f"optimizes"
            )

    def test_the_payload_is_computed_on_the_RESOLVED_candidate_chip(self):
        # A plan decodes into core dimensions only; the chip a deployment builds
        # from that declaration also carries the platform's bias capability,
        # which re-tiles the model. Laying the payload out on the raw dict would
        # show the agent a chip nobody deploys.
        cfg = pipeline_config(has_bias=False)
        problem = make_problem("hardware", cfg=cfg)
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)

        raw_pcfg = configuration["platform_constraints"]
        assert all("has_bias" not in core for core in raw_pcfg["cores"]), (
            "the fixture must decode a declaration that still needs resolving"
        )
        model, _ = problem._build_model(configuration["model_config"], raw_pcfg)
        raw_softcores, _ = problem._collect_softcores(model, raw_pcfg)
        raw_rows = [_softcore_to_dict(sc, i) for i, sc in enumerate(raw_softcores)]
        assert payload["per_softcore"] != raw_rows, (
            "the payload must describe the resolved chip, not the declaration"
        )
        assert payload["per_softcore"] == [
            _softcore_to_dict(sc, i)
            for i, sc in enumerate(problem.candidate_layout(configuration).softcores)
        ]

    def test_a_candidate_that_cannot_be_laid_out_raises(self):
        problem = make_problem("hardware", cfg=pipeline_config(width=200))
        configuration = _baseline_configuration(problem)
        configuration["platform_constraints"]["cores"] = [
            {"max_axons": 64, "max_neurons": 256, "count": 64},
        ]
        with pytest.raises(Exception, match="fan-in"):
            collect_layout_payload(problem, configuration)


class TestDeviceCapabilityAndAnalyse:
    def test_device_capability_arch(self):
        backend = MimarsinanLayoutBackend()
        cap = backend.device_capability()
        assert cap.arch == "snn-crossbar"
        assert cap.extra["vendor"] == "mimarsinan"

    def test_analyze_includes_per_layer_and_layout_stats(self, registered_problem):
        workload_id, problem = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        analysis = backend.analyze(workload, baseline_artifacts=())
        assert analysis.summary["kind"] == "full_model"
        assert analysis.summary["search_mode"] == "hardware"
        assert "baseline_error" not in analysis.extra, analysis.extra.get(
            "baseline_error"
        )
        baseline = analysis.extra["baseline"]
        expected = problem.candidate_layout(_baseline_configuration(problem))
        assert baseline["softcore_count"] == len(expected.softcores)
        assert analysis.summary["softcore_count_baseline"] == len(expected.softcores)
        assert analysis.summary["layer_count"] == len(baseline["per_layer"]) > 0
        assert baseline["layout_stats"]["feasible"] is True
        assert baseline["hw_objectives"] == OBJECTIVES.extract(expected.view)

    def test_analyze_records_a_baseline_error_instead_of_aborting(self):
        # A workload whose baseline cannot be laid out still analyses: the
        # failure is REPORTED, never swallowed into a silently empty baseline.
        # A 600-wide layer overflows the 512-axon cores the codec defaults to.
        problem = make_problem("hardware", cfg=pipeline_config(width=600))
        workload_id = "real_layout_unmappable_baseline"
        _registered(workload_id, problem)
        try:
            backend = MimarsinanLayoutBackend()
            analysis = backend.analyze(_make_workload(workload_id), baseline_artifacts=())
        finally:
            unregister_problem(workload_id)
        assert "baseline" not in analysis.extra
        assert "fan-in" in analysis.extra["baseline_error"]


class TestSearchSpace:
    def test_derive_search_space_returns_levers(self):
        workload_id = "real_layout_joint_space"
        _registered(workload_id, make_problem("joint"))
        try:
            workload = _make_workload(workload_id)
            backend = MimarsinanLayoutBackend()
            analysis = backend.analyze(workload, baseline_artifacts=())
            space = backend.derive_search_space(workload, analysis)
        finally:
            unregister_problem(workload_id)
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
        iv = _hw_core("0.bogus", 128)
        assert backend.validate_intervention(iv).ok is False

    def test_valid_arch_intervention_is_accepted(self):
        backend = MimarsinanLayoutBackend()
        iv = Intervention(target=Target(kind="arch", selector="activation"), payload="ReLU")
        assert backend.validate_intervention(iv).ok is True

    def test_valid_hw_core_intervention_is_accepted(self):
        backend = MimarsinanLayoutBackend()
        assert backend.validate_intervention(_hw_core("0.max_axons", 256)).ok is True

    def test_zero_or_negative_axon_count_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        for bad_value in (0, -8):
            result = backend.validate_intervention(_hw_core("0.max_axons", bad_value))
            assert result.ok is False
            assert "positive integer" in result.errors[0]

    def test_obviously_huge_payload_is_rejected(self):
        backend = MimarsinanLayoutBackend()
        assert backend.validate_intervention(_hw_core("0.count", 10**8)).ok is False

    def test_non_8_multiple_axons_is_rejected_with_snap_hint(self):
        backend = MimarsinanLayoutBackend()
        result = backend.validate_intervention(_hw_core("0.max_neurons", 250))
        assert result.ok is False
        assert "multiple of 8" in result.errors[0]
        assert "248" in result.errors[0] and "256" in result.errors[0]

    def test_count_does_not_require_multiple_of_8(self):
        backend = MimarsinanLayoutBackend()
        assert backend.validate_intervention(_hw_core("0.count", 13)).ok is True


class TestCompile:
    def test_empty_plan_compiles_against_defaults(self, registered_problem, tmp_path):
        workload_id, problem = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        result = backend.compile(workload, Plan(), artifact_dir=tmp_path)
        assert result.ok, result.diagnostics
        assert result.artifacts and len(result.artifacts) == 3
        for path in result.artifacts:
            data = json.loads(path.read_text())
            assert data is not None
        expected = problem.candidate_layout(_baseline_configuration(problem))
        assert len(result.metadata["softcores"]) == len(expected.softcores)
        assert result.metadata["layout_stats"] == expected.stats.to_dict()
        assert result.metadata["hw_objectives"] == OBJECTIVES.extract(expected.view)
        assert {row["name"] for row in result.metadata["objective_catalog"]} == set(
            HW_OBJECTIVES
        )

    def test_an_unresolvable_chip_is_a_structural_failed_compile(
        self, registered_problem, tmp_path,
    ):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        # Zero cores is not a chip: the deployment resolver refuses it, and the
        # refusal is the CANDIDATE's — a failed compile, never a dead session.
        result = backend.compile(
            workload, _plan(_hw_core("0.count", 0)), artifact_dir=tmp_path,
        )
        assert result.ok is False
        assert result.metadata["failure_phase"] == "structural"
        assert "does not resolve into a chip" in (result.diagnostics or "")

    def test_a_chip_the_model_cannot_map_is_a_conversion_failed_compile(
        self, registered_problem, tmp_path,
    ):
        workload_id, _ = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        result = backend.compile(
            workload, _plan(_hw_core("0.max_axons", 8)), artifact_dir=tmp_path,
        )
        assert result.ok is False
        assert result.metadata["failure_phase"] == "hw_conversion"
        assert "fan-in" in (result.diagnostics or "")

    def test_candidate_infeasible_validate_becomes_failed_compile(self, tmp_path):
        from mimarsinan.search.problem import CandidateInfeasibleError
        from mimarsinan.search.problems.joint import JointArchHwProblem

        class _TypedFailProblem(JointArchHwProblem):
            def validate_detailed(self, configuration):
                raise CandidateInfeasibleError("candidate collapsed at build")

        workload_id = "real_layout_typed_fail"
        problem = _make_problem()
        _registered(workload_id, _TypedFailProblem(**_ctor_kwargs(problem)))
        try:
            workload = _make_workload(workload_id)
            backend = MimarsinanLayoutBackend()
            result = backend.compile(workload, Plan(), artifact_dir=tmp_path)
        finally:
            unregister_problem(workload_id)
        assert result.ok is False
        assert result.metadata["failure_phase"] == "candidate_infeasible"
        assert "candidate collapsed at build" in (result.diagnostics or "")

    def test_candidate_infeasible_evaluate_becomes_diagnostic_timing(self, tmp_path):
        from mimarsinan.search.problem import CandidateInfeasibleError
        from mimarsinan.search.problems.joint import JointArchHwProblem

        class _TypedEvalFailProblem(JointArchHwProblem):
            def evaluate(self, configuration):
                raise CandidateInfeasibleError("candidate collapsed at evaluate")

        workload_id = "real_layout_typed_eval_fail"
        problem = _make_problem()
        _registered(workload_id, _TypedEvalFailProblem(**_ctor_kwargs(problem)))
        try:
            workload = _make_workload(workload_id)
            backend = MimarsinanLayoutBackend()
            timing = backend.time_workload(
                workload, Plan(), warmup=0, repetitions=1, max_seconds=1.0,
            )
        finally:
            unregister_problem(workload_id)
        assert "candidate infeasible" in (timing.diagnostics or "")
        assert "objectives" not in (timing.profile_metrics or {})

    def test_problem_level_validate_failure_propagates(self, tmp_path):
        from mimarsinan.search.problems.joint import JointArchHwProblem

        class _BrokenProblem(JointArchHwProblem):
            def validate_detailed(self, configuration):
                raise RuntimeError("problem fixture broken")

        workload_id = "real_layout_broken"
        problem = _make_problem()
        _registered(workload_id, _BrokenProblem(**_ctor_kwargs(problem)))
        try:
            workload = _make_workload(workload_id)
            backend = MimarsinanLayoutBackend()
            with pytest.raises(RuntimeError, match="problem fixture broken"):
                backend.compile(workload, Plan(), artifact_dir=tmp_path)
        finally:
            unregister_problem(workload_id)


def _ctor_kwargs(problem) -> Dict[str, Any]:
    """The real problem's constructor arguments, for the two raise-site doubles."""
    from dataclasses import fields

    return {
        f.name: getattr(problem, f.name) for f in fields(problem) if f.init
    }


class TestTimeWorkload:
    def test_time_workload_leaves_single_axis_empty_and_exposes_full_objectives(
        self, registered_problem,
    ):
        """Multi-objective backends deliberately leave ``median_ms`` as
        ``None`` so compilagent's single-axis leaderboard becomes
        informationless and the agent is forced to use ``pareto_front`` /
        ``metric_summary`` / ``query_top_candidates`` instead. The full
        objective tuple lives in ``profile_metrics['objectives']`` and
        on the ``Backend.objectives_for_candidate`` hook."""

        workload_id, problem = registered_problem
        workload = _make_workload(workload_id)
        backend = MimarsinanLayoutBackend()
        timing = backend.time_workload(
            workload, Plan(), warmup=0, repetitions=1, max_seconds=1.0,
        )
        assert timing.median_ms is None
        assert "primary_objective" not in timing.profile_metrics
        full = timing.profile_metrics["objectives"]
        assert set(full) == set(HW_OBJECTIVES)
        assert full == problem.evaluate(_baseline_configuration(problem))


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
        assert set(objectives) == set(HW_OBJECTIVES)
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

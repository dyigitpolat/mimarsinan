"""Tests for ``MimarsinanLayoutBackend``.

Two contracts meet in this file, and both are pinned here.

The backend is driven against the REAL ``JointArchHwProblem`` (see
``real_problem.py``): it reads that problem's resolver, model fixture, layout
seam and objective registry, so a stand-in for those members is a stand-in for
the very thing under test. Only the raise-site doubles remain, and each
SUBCLASSES the real problem to override exactly one method.

What that problem's layout ANSWERS is then served through the introspection
registry — a declared, versioned surface — so the assertions below are about
served envelopes (``payload``/``payload_version``) and the flat keys that
project them, never about a hand-rolled rendering. The registry's own row
semantics (bank sharing degree, rollup identity, rename-invariance, version
refusals) are pinned against synthetic specs in
``tests/unit/deployment_record/test_introspection_registry.py``; what belongs
here is that the SERVED payload is the problem's own candidate layout.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from compilagent import (
    Intervention,
    Plan,
    Target,
)

import mimarsinan.deployment_record.introspection.views as _introspection_views
from mimarsinan.deployment_record.introspection import (
    INTROSPECTION_FORMAT_VERSION,
    INTROSPECTION_REGISTRY,
    CANDIDATE_LAYOUT,
)
from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.backend.backend_eval import unit_for as _unit_for
from mimarsinan.search.optimizers.compilagent.backend.backend_layout import (
    collect_layout_payload,
    write_layout_artifacts as _write_layout_artifacts,
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

#: The payloads a search candidate — as opposed to a sealed record — can answer.
CANDIDATE_PAYLOADS = {
    spec.name
    for spec in INTROSPECTION_REGISTRY.all()
    if CANDIDATE_LAYOUT in spec.builders
}


def _make_problem():
    """The default registered problem: a hardware search over the tiny MLP."""
    return make_problem("hardware")


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


def _shapes(softcores):
    """(name, in, out, area, perceptron) per spec — what a served row must carry."""
    return [
        (
            sc.name, int(sc.input_count), int(sc.output_count), int(sc.area),
            sc.perceptron_index,
        )
        for sc in softcores
    ]


def _row_shapes(rows):
    return [
        (
            row["name"], row["input_count"], row["output_count"], row["area"],
            row["perceptron_index"],
        )
        for row in rows
    ]


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
    def test_unit_for_known_objectives(self):
        assert _unit_for("fragmentation_pct") == "%"
        assert _unit_for("total_params") == "params"
        assert _unit_for("total_sync_barriers") == "barriers"
        assert _unit_for("estimated_accuracy") == ""
        assert _unit_for("unknown") == ""


class TestLayoutPayloadIsTheProblemsOwnCandidateLayout:
    """The payload is the problem's ``candidate_layout``, served — nothing else."""

    def test_payload_softcores_are_the_candidates_own_layout(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)

        layout = problem.candidate_layout(configuration)
        assert payload["softcore_count"] == len(layout.softcores)
        assert payload["softcore_count"] > 0, "the fixture must map something"
        assert _row_shapes(payload["per_softcore"]) == _shapes(layout.softcores)
        assert payload["layout_stats"] == layout.stats.to_dict()
        assert sum(row["softcore_count"] for row in payload["per_layer"]) == len(
            layout.softcores
        )

    def test_the_backend_serves_the_problems_packing_instead_of_re_running_it(
        self, monkeypatch,
    ):
        """A recomputed layout is a SECOND answer, free to drift from the scored one.

        The problem already packed this candidate; the channel takes that census
        (``CandidateLayoutView.packed``). Reverting the backend to a view that
        lays the candidate out itself trips this refusal — the problem's own
        packing goes through ``layout_hook``'s import, not this one.
        """
        problem = _make_problem()
        configuration = _baseline_configuration(problem)

        def _refuse(*args, **kwargs):
            raise AssertionError(
                "the layout backend must not compute a layout of its own"
            )

        monkeypatch.setattr(
            _introspection_views, "compute_mapping_stats", _refuse,
        )
        payload = collect_layout_payload(problem, configuration)
        assert payload["softcore_count"] > 0
        assert payload["layout_stats"]["feasible"] is True

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
        model, _ = problem._build_model(configuration["model_config"], raw_pcfg, problem.encoding_placement)
        raw_softcores, _ = problem._collect_softcores(model, raw_pcfg)
        assert _row_shapes(payload["per_softcore"]) != _shapes(raw_softcores), (
            "the payload must describe the resolved chip, not the declaration"
        )
        assert _row_shapes(payload["per_softcore"]) == _shapes(
            problem.candidate_layout(configuration).softcores
        )

    def test_a_candidate_that_cannot_be_laid_out_raises(self):
        problem = make_problem("hardware", cfg=pipeline_config(width=200))
        configuration = _baseline_configuration(problem)
        configuration["platform_constraints"]["cores"] = [
            {"max_axons": 64, "max_neurons": 256, "count": 64},
        ]
        with pytest.raises(Exception, match="fan-in"):
            collect_layout_payload(problem, configuration)


class TestBothCacheStatesOfTheRealProblem:
    """The two ways a real problem holds its model, driven end to end.

    A hardware-only search reuses one candidate-INDEPENDENT model (its cache is
    WARM after the first candidate); a model-bearing search builds a new model
    per candidate and never fills that cache at all. The backend used to read
    the cache's softcores in the first case and rebuild the model in the second
    — both of which went stale in silence when the problem surface changed.
    Neither path may touch a private member now: the seam is the same public
    call, so the two states cannot diverge.
    """

    def _payload_matches_the_layout(self, problem, configuration):
        payload = collect_layout_payload(problem, configuration)
        layout = problem.candidate_layout(configuration)
        assert set(payload["introspection"]) == CANDIDATE_PAYLOADS
        assert _row_shapes(payload["per_softcore"]) == _shapes(layout.softcores)
        assert payload["layout_stats"] == layout.stats.to_dict()
        assert payload["hw_objectives"] == OBJECTIVES.extract(layout.view)
        return payload

    def test_a_hardware_search_serves_off_a_warm_model_fixture(self):
        problem = make_problem("hardware")
        configuration = _baseline_configuration(problem)
        assert not problem._hw_only_cache, "the fixture starts cold"
        # The public seam warms it, exactly as the first scored candidate would.
        problem.candidate_layout(configuration)
        assert problem._hw_only_cache, "a hardware search caches its model per placement"

        payload = self._payload_matches_the_layout(problem, configuration)
        assert payload["softcore_count"] > 0

    def test_a_joint_search_serves_with_no_model_fixture_at_all(self):
        problem = make_problem("joint")
        configuration = _baseline_configuration(problem)
        payload = self._payload_matches_the_layout(problem, configuration)
        assert not problem._hw_only_cache, (
            "a model-bearing search builds per candidate; nothing is cached"
        )
        assert payload["softcore_count"] > 0

    def test_a_model_search_serves_with_no_model_fixture_at_all(self):
        problem = make_problem("model")
        configuration = _baseline_configuration(problem)
        self._payload_matches_the_layout(problem, configuration)
        assert not problem._hw_only_cache


class TestServedPayload:
    """The agent surface is the introspection registry's, versioned and identified."""

    def test_every_candidate_answerable_payload_is_served(self):
        problem = _make_problem()
        served = collect_layout_payload(
            problem, _baseline_configuration(problem),
        )["introspection"]
        assert set(served) == CANDIDATE_PAYLOADS
        assert set(served) == {
            "softcores", "layer_rollup", "bank_composition", "schedule",
            "capabilities", "layout_stats",
        }
        assert all(env["payload"] == name for name, env in served.items())
        assert all(env["payload_version"] >= 1 for env in served.values())

    def test_softcore_rows_carry_the_layer_identity_the_mapper_assigned(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        rows = collect_layout_payload(
            problem, configuration,
        )["introspection"]["softcores"]["softcores"]
        layout = problem.candidate_layout(configuration)
        assert [row["perceptron_index"] for row in rows] == [
            sc.perceptron_index for sc in layout.softcores
        ]
        assert all(row["perceptron_index"] is not None for row in rows), (
            "the fixture's cores all come from a source perceptron"
        )
        assert [row["bank_id"] for row in rows] == [
            sc.bank_id for sc in layout.softcores
        ]

    def test_the_rollup_keys_on_identity_not_on_the_name(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)
        rows = payload["introspection"]["layer_rollup"]["layers"]
        layout = problem.candidate_layout(configuration)
        assert [row["perceptron_index"] for row in rows] == sorted(
            {sc.perceptron_index for sc in layout.softcores}
        )
        assert [row["layer"] for row in rows] == [
            f"perceptron_{row['perceptron_index']}" for row in rows
        ]
        assert sum(row["total_area"] for row in rows) == sum(
            int(sc.area) for sc in layout.softcores
        )

    def test_the_bank_payload_states_who_owns_their_weights(self):
        # This MLP fixture shares no weight bank, and the payload says so rather
        # than serving an empty table that reads like "not measured". The
        # sharing-degree rows are pinned in the registry's own suite.
        problem = _make_problem()
        payload = collect_layout_payload(problem, _baseline_configuration(problem))
        banks = payload["introspection"]["bank_composition"]
        assert banks["banks"] == []
        assert banks["unbanked_softcore_count"] == payload["softcore_count"]

    def test_the_schedule_names_the_policy_the_candidate_chip_declares(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)
        schedule = payload["introspection"]["schedule"]
        layout = problem.candidate_layout(configuration)
        capabilities = ChipCapabilities.from_platform_constraints(layout.platform)
        assert schedule["schedule_policy"] == capabilities.schedule_policy == "pool"
        assert schedule["max_schedule_passes"] == capabilities.max_schedule_passes
        assert schedule["sync_count"] == int(layout.stats.schedule_sync_count)
        assert schedule["pass_count"] == int(layout.stats.schedule_pass_count)

    def test_the_capability_bits_are_the_candidate_chips_whole_declaration(self):
        problem = _make_problem()
        configuration = _baseline_configuration(problem)
        payload = collect_layout_payload(problem, configuration)
        bits = payload["introspection"]["capabilities"]["bits"]
        layout = problem.candidate_layout(configuration)
        assert bits == ChipCapabilities.from_platform_constraints(
            layout.platform
        ).capability_bits()
        assert [name for name, value in bits.items() if value is None] == [], (
            "a served null must mean 'the platform declares none', never 'unread'"
        )
        assert bits["allow_scheduling"] is True, "the fixture declares it"

    def test_the_legacy_flat_keys_project_the_same_rows(self):
        problem = _make_problem()
        payload = collect_layout_payload(problem, _baseline_configuration(problem))
        assert payload["softcore_count"] == len(payload["per_softcore"]) > 0
        assert payload["per_softcore"] == (
            payload["introspection"]["softcores"]["softcores"]
        )
        assert payload["per_layer"] == (
            payload["introspection"]["layer_rollup"]["layers"]
        )
        assert payload["layout_stats"] == (
            payload["introspection"]["layout_stats"]["stats"]
        )


class TestThePersistedArtifactCarriesTheChannelVersion:
    """A stored file outlives the writer, so it states its envelope convention.

    Each payload inside carries its own ``payload_version``; the file states the
    CHANNEL's ``INTROSPECTION_FORMAT_VERSION`` — otherwise reshaping the served
    map itself would be an undetectable break for anything reading the artifact.
    """

    def test_introspection_json_is_the_versioned_artifact(self, tmp_path):
        problem = _make_problem()
        payload = collect_layout_payload(problem, _baseline_configuration(problem))
        written = _write_layout_artifacts(tmp_path, {"cfg": 1}, payload)
        assert [p.name for p in written] == [
            "config.json", "softcores.json", "layout_stats.json",
            "introspection.json",
        ]
        stored = json.loads((tmp_path / "introspection.json").read_text())
        assert stored["introspection_format_version"] == INTROSPECTION_FORMAT_VERSION
        assert stored["payloads"] == payload["introspection"]
        assert all(
            envelope["payload_version"] >= 1
            for envelope in stored["payloads"].values()
        )

    def test_the_other_artifacts_stay_the_bare_bodies(self, tmp_path):
        problem = _make_problem()
        payload = collect_layout_payload(problem, _baseline_configuration(problem))
        _write_layout_artifacts(tmp_path, {"cfg": 1}, payload)
        assert json.loads((tmp_path / "config.json").read_text()) == {"cfg": 1}
        assert json.loads(
            (tmp_path / "softcores.json").read_text()
        ) == payload["per_softcore"]


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
        assert analysis.summary["introspection_payloads"] == sorted(CANDIDATE_PAYLOADS)
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
        assert result.artifacts and len(result.artifacts) == 4
        for path in result.artifacts:
            data = json.loads(path.read_text())
            assert data is not None
        expected = problem.candidate_layout(_baseline_configuration(problem))
        assert len(result.metadata["softcores"]) == len(expected.softcores)
        assert result.metadata["layout_stats"] == expected.stats.to_dict()
        assert result.metadata["hw_objectives"] == OBJECTIVES.extract(expected.view)
        assert set(result.metadata["introspection"]) == CANDIDATE_PAYLOADS
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

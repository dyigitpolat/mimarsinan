"""W5.1 — one evaluation contract for every search mode.

A candidate is evaluated exactly once, in one way: its static facts become a
``CandidateStaticView`` and the ACTIVE registry objectives are read off that
view. Nothing assembles objectives per mode any more; accuracy attaches where —
and only where — the registry says the mode can carry it.
"""

import math

import numpy as np
import pytest
import torch

from mimarsinan.deployment_record.objectives import (
    OBJECTIVES,
    CandidateStaticView,
    chip_param_capacity,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
)
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.problem import CandidateInfeasibleError
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.problems.joint import validate as joint_validate

HW_OBJECTIVES = [
    "total_param_capacity",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "axon_wastage_pct",
    "fragmentation_pct",
]
MODEL_OBJECTIVES = ["estimated_accuracy", "total_params"]
JOINT_OBJECTIVES = [
    "estimated_accuracy",
    "total_params",
    "param_utilization_pct",
    "neuron_wastage_pct",
    "fragmentation_pct",
]
STUB_ACCURACY = 0.75

ARCH_OPTIONS = (("mlp_width_1", [16, 32]), ("mlp_width_2", [16, 32]))


def _pipeline_config(width=16):
    return {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "model_config": {
            "mlp_width_1": width, "mlp_width_2": width, "base_activation": "ReLU",
        },
    }


def _problem(search_mode, objective_names, cfg=None):
    cfg = cfg if cfg is not None else _pipeline_config()
    searches_model = search_mode in ("model", "joint")
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        target_tq=cfg["target_tq"],
        lr=cfg["lr"],
        search_mode=search_mode,
        builder_factory=SimpleMLPBuilder,
        arch_options=ARCH_OPTIONS if searches_model else (),
        model_config_assembler=lambda raw: {**raw, "base_activation": "ReLU"},
        fixed_model_config=None if searches_model else dict(cfg["model_config"]),
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=objective_names,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


def _mid_x(problem):
    return (np.asarray(problem.xl) + np.asarray(problem.xu)) / 2.0


def _stub_accuracy(problem):
    """Replace training with a sentinel and count the attachments."""
    calls = []

    def _evaluate(model):
        calls.append(model)
        return STUB_ACCURACY

    problem._evaluate_accuracy = _evaluate
    return calls


def _independent_view(problem, configuration, accuracy=None):
    """The candidate's static facts, built WITHOUT the problem's evaluate path."""
    mc = configuration["model_config"]
    pcfg = configuration["platform_constraints"]
    model, total_params = problem._build_model(mc, pcfg, problem.encoding_placement)
    softcores, host_segments, _census = problem._collect_softcores(model, pcfg)
    stats, error = compute_mapping_stats(
        softcores=softcores,
        core_types=problem._make_core_types(pcfg),
        **ChipCapabilities.from_platform_constraints(pcfg).permission_kwargs(),
    )
    assert stats.feasible, error
    return CandidateStaticView(
        layout=stats,
        chip_param_capacity=chip_param_capacity(pcfg["cores"]),
        total_params=total_params,
        host_side_segment_count=host_segments,
        estimated_accuracy=accuracy,
    )


class TestOneEvaluationContract:
    @pytest.mark.parametrize(
        "search_mode,objective_names",
        [
            ("hardware", HW_OBJECTIVES),
            ("model", MODEL_OBJECTIVES),
            ("joint", JOINT_OBJECTIVES),
        ],
    )
    def test_evaluate_returns_exactly_the_active_registry_axes(
        self, search_mode, objective_names,
    ):
        problem = _problem(search_mode, objective_names)
        _stub_accuracy(problem)
        objectives = problem.evaluate(problem.decode(_mid_x(problem)))

        assert set(objectives) == set(objective_names)
        for name, value in objectives.items():
            assert math.isfinite(value), f"{name} not finite: {value}"
            assert abs(value) < 1e17, f"{name} looks like a penalty: {value}"

    def test_the_values_are_the_registry_read_of_the_candidate_view(self):
        problem = _problem("hardware", HW_OBJECTIVES)
        configuration = problem.decode(_mid_x(problem))

        view = _independent_view(problem, configuration)
        expected = {
            spec.key: spec.value(view)
            for spec in OBJECTIVES.resolve_active("hardware", HW_OBJECTIVES)
        }
        assert problem.evaluate(configuration) == expected

    def test_the_validation_cache_is_an_optimization_not_a_dependency(
        self, monkeypatch,
    ):
        # The cache is bounded, so a validated candidate whose entry has been
        # evicted must still be scored — by resolving it again, to the SAME
        # numbers. (This is what keeps ``_evaluate_inner`` a live path rather
        # than a comforting dead branch.)
        problem = _problem("hardware", HW_OBJECTIVES)
        configuration = problem.decode(_mid_x(problem))
        expected = problem.evaluate(configuration)

        evicted = _problem("hardware", HW_OBJECTIVES)
        monkeypatch.setattr(joint_validate, "VALIDATION_CACHE_MAX_SIZE", 0)
        objectives = evicted.evaluate(configuration)

        assert evicted._validation_cache == {}, "the fixture must evict the entry"
        assert objectives == expected

    def test_joint_values_are_the_registry_read_including_accuracy(self):
        problem = _problem("joint", JOINT_OBJECTIVES)
        _stub_accuracy(problem)
        configuration = problem.decode(_mid_x(problem))

        view = _independent_view(problem, configuration, accuracy=STUB_ACCURACY)
        expected = {
            spec.key: spec.value(view)
            for spec in OBJECTIVES.resolve_active("joint", JOINT_OBJECTIVES)
        }
        assert problem.evaluate(configuration) == expected


class TestAccuracyAttachesWhereTheModeTrains:
    def test_hardware_search_never_trains(self):
        problem = _problem("hardware", HW_OBJECTIVES)
        calls = _stub_accuracy(problem)
        objectives = problem.evaluate(problem.decode(_mid_x(problem)))
        assert calls == [], "a hardware search carries no accuracy axis to fill"
        assert "estimated_accuracy" not in objectives

    @pytest.mark.parametrize(
        "search_mode,objective_names",
        [("model", MODEL_OBJECTIVES), ("joint", JOINT_OBJECTIVES)],
    )
    def test_model_bearing_searches_attach_the_estimate_once(
        self, search_mode, objective_names,
    ):
        problem = _problem(search_mode, objective_names)
        calls = _stub_accuracy(problem)
        objectives = problem.evaluate(problem.decode(_mid_x(problem)))
        assert objectives["estimated_accuracy"] == STUB_ACCURACY
        assert len(calls) == 1, "the estimate is produced once per candidate"


class TestUnavailableObjectivesFailLoudThroughTheProblem:
    def test_accuracy_requested_for_a_hardware_search_aborts(self):
        problem = _problem("hardware", ["estimated_accuracy"])
        with pytest.raises(ValueError, match="estimated_accuracy"):
            _ = problem.objectives

    def test_an_unavailable_objective_aborts_the_evaluation(self):
        problem = _problem("hardware", HW_OBJECTIVES + ["estimated_accuracy"])
        configuration = {
            "model_config": _pipeline_config()["model_config"],
            "platform_constraints": {
                "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
            },
        }
        with pytest.raises(ValueError, match="estimated_accuracy"):
            problem.evaluate(configuration)

    def test_a_view_that_cannot_answer_an_active_axis_aborts(self):
        # The contract produces EVERY active axis or fails; a short objective
        # vector would silently re-rank the whole population.
        problem = _problem("joint", JOINT_OBJECTIVES)
        layoutless = CandidateStaticView(
            layout=None,
            chip_param_capacity=None,
            total_params=1.0,
            host_side_segment_count=None,
            estimated_accuracy=STUB_ACCURACY,
        )
        with pytest.raises(ValueError, match="param_utilization_pct"):
            problem._objectives_from_view(layoutless)

    def test_an_unknown_objective_name_aborts(self):
        problem = _problem("joint", ["no_such_axis"])
        with pytest.raises(ValueError, match="no_such_axis"):
            _ = problem.objectives

    def test_a_record_only_objective_is_not_searchable(self):
        # ``mj_per_sample`` needs a sealed record; a candidate view cannot carry
        # it, and the registry refuses instead of quietly dropping the axis.
        problem = _problem("joint", ["mj_per_sample"])
        with pytest.raises(ValueError, match="mj_per_sample"):
            _ = problem.objectives


class TestCandidateChipIsTheChipTheModelIsMappedOnto:
    """The layout is re-derived for the candidate's own core geometry."""

    def test_a_candidate_too_narrow_for_the_model_is_infeasible(self):
        # The fixed model has a 200-wide fan-in: it maps onto the declared
        # 256-axon platform but NOT onto a 64-axon candidate. Reusing the
        # base platform's tiling would report that candidate as deployable.
        cfg = _pipeline_config(width=200)
        problem = _problem("hardware", HW_OBJECTIVES, cfg=cfg)

        wide = problem.decode(np.array([256.0, 256.0, 64.0]))
        assert problem.validate_detailed(wide).is_valid

        narrow = problem.decode(np.array([64.0, 256.0, 64.0]))
        result = problem.validate_detailed(narrow)
        assert not result.is_valid, (
            "a 64-axon chip cannot map a 200-wide fan-in; the search must not "
            "score it off the declared platform's tiling"
        )
        assert result.failure_phase == "hw_conversion"
        assert problem.evaluate(narrow) == problem._penalty_objectives()

    def test_a_candidate_with_narrow_neurons_is_re_tiled_not_rejected(self):
        # The mirror failure: a 64-NEURON candidate cannot hold the declared
        # platform's 200-neuron softcores, but it maps the model perfectly well
        # once the layout is re-derived for it. Reusing the declared platform's
        # tiling would throw this deployable chip away.
        cfg = _pipeline_config(width=200)
        problem = _problem("hardware", HW_OBJECTIVES, cfg=cfg)

        candidate = problem.decode(np.array([256.0, 64.0, 64.0]))
        pcfg = candidate["platform_constraints"]
        assert (pcfg["cores"][0]["max_axons"], pcfg["cores"][0]["max_neurons"]) == (256, 64)

        assert problem.validate_detailed(candidate).is_valid, (
            "a narrower chip re-tiles the model instead of failing to hold the "
            "declared platform's softcores"
        )
        base_cores = problem.fixed_platform_constraints["cores"]
        model, _params = problem._build_model(
            cfg["model_config"], {**pcfg, "cores": base_cores},
            problem.encoding_placement,
        )
        base_softcores, _, _ = problem._collect_softcores(
            model, {**pcfg, "cores": base_cores},
        )
        candidate_softcores, _, _ = problem._collect_softcores(model, pcfg)
        assert len(candidate_softcores) > len(base_softcores), (
            "the candidate's own tiling is what the search must score"
        )

    def test_unmappable_candidates_do_not_sink_the_search(self):
        # The typed taxonomy end to end: a population containing chips the model
        # cannot map completes and returns a REAL chip, never a penalty row and
        # never an aborted run.
        cfg = _pipeline_config(width=200)
        problem = _problem("hardware", HW_OBJECTIVES, cfg=cfg)
        optimizer = NSGA2Optimizer(pop_size=6, generations=2, seed=0, verbose=False)

        result = optimizer.optimize(problem)
        assert result.best.configuration["platform_constraints"]["cores"]
        for name, value in result.best.objectives.items():
            assert abs(value) < 1e17, f"{name} looks like a penalty: {value}"

    def test_the_fixed_model_is_built_once_across_candidates(self):
        problem = _problem("hardware", HW_OBJECTIVES)
        builds = []
        original = problem._build_raw_model

        def _counting(mc, pcfg, _placement):
            builds.append(pcfg)
            return original(mc, pcfg, _placement)

        problem._build_raw_model = _counting
        for count in (16.0, 32.0, 64.0):
            problem.evaluate(problem.decode(np.array([256.0, 256.0, count])))
        assert len(builds) == 1, (
            "the model does not depend on the candidate platform in a hardware "
            "search; it is trained/built once"
        )

    def test_the_model_side_repr_is_converted_once_across_candidates(self):
        # The candidate-dependent half of the mapping is the LAYOUT, which every
        # candidate re-derives. The model-side representation is not: converting
        # a torch model into mapper form depends on the model alone, so paying
        # for it per candidate is pure waste in a hardware search.
        problem = _problem("hardware", HW_OBJECTIVES)
        conversions = []
        original = problem._convert_to_mapper_repr

        def _counting(model, _placement):
            conversions.append(model)
            return original(model, _placement)

        problem._convert_to_mapper_repr = _counting
        layouts = []
        for count in (16.0, 32.0, 64.0):
            configuration = problem.decode(np.array([256.0, 256.0, count]))
            problem.evaluate(configuration)
            layouts.append(problem.candidate_layout(configuration))

        assert len(conversions) == 1, (
            "the model-side representation does not depend on the candidate "
            "chip; a hardware search converts it once per run"
        )
        assert all(lay.softcores for lay in layouts), (
            "every candidate still derives its own layout off that one repr"
        )


class TestTheCandidateLayoutSeam:
    """``candidate_layout`` — the candidate's chip, softcores and view in one call."""

    def test_the_layout_is_the_problems_own_mapping_of_the_candidate_chip(self):
        problem = _problem("hardware", HW_OBJECTIVES)
        configuration = problem.decode(_mid_x(problem))
        layout = problem.candidate_layout(configuration)

        pcfg = configuration["platform_constraints"]
        model, total_params = problem._build_model(
            configuration["model_config"], pcfg, problem.encoding_placement,
        )
        expected_softcores, expected_host, _ = problem._collect_softcores(model, pcfg)

        assert layout.platform == pcfg
        assert [sc.name for sc in layout.softcores] == [
            sc.name for sc in expected_softcores
        ]
        assert [sc.input_count for sc in layout.softcores] == [
            sc.input_count for sc in expected_softcores
        ]
        assert layout.host_side_segment_count == expected_host
        assert layout.stats.feasible
        assert layout.view.total_params == total_params
        assert layout.view.layout is layout.stats

    def test_the_view_answers_exactly_what_the_evaluation_reads(self):
        problem = _problem("hardware", HW_OBJECTIVES)
        configuration = problem.decode(_mid_x(problem))
        layout = problem.candidate_layout(configuration)
        assert problem._objectives_from_view(layout.view) == problem.evaluate(
            configuration
        )

    def test_a_raw_declaration_is_laid_out_on_its_RESOLVED_chip(self):
        # An LLM-declared candidate states core dimensions only. Laying it out
        # on that raw dict maps the model onto a chip nobody deploys — the bias
        # capability alone re-tiles it.
        cfg = _pipeline_config()
        cfg["platform_constraints"] = {"has_bias": False}
        problem = _problem("hardware", HW_OBJECTIVES, cfg=cfg)

        raw = {
            "model_config": cfg["model_config"],
            "platform_constraints": {
                "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
            },
        }
        layout = problem.candidate_layout(raw)
        assert layout.platform["cores"][0]["has_bias"] is False

        raw_params = resolve_platform_mapping_params(
            raw["platform_constraints"]["cores"],
        )
        resolved_params = resolve_platform_mapping_params(layout.platform["cores"])
        assert raw_params.hardware_bias != resolved_params.hardware_bias, (
            "the fixture must make the raw and resolved chips map differently"
        )
        model, _ = problem._build_model(
            cfg["model_config"], raw["platform_constraints"],
            problem.encoding_placement,
        )
        raw_softcores, _, _ = problem._collect_softcores(
            model, raw["platform_constraints"],
        )
        assert [sc.input_count for sc in layout.softcores] != [
            sc.input_count for sc in raw_softcores
        ], "the layout must be the resolved chip's, not the declaration's"

    def test_a_candidate_that_cannot_be_laid_out_raises_typed(self):
        cfg = _pipeline_config(width=200)
        problem = _problem("hardware", HW_OBJECTIVES, cfg=cfg)
        narrow = problem.decode(np.array([64.0, 256.0, 64.0]))
        with pytest.raises(CandidateInfeasibleError):
            problem.candidate_layout(narrow)

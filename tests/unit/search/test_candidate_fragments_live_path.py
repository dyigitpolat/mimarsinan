"""N0 — the fragments the registry gates on ride the LIVE evaluation path.

The incident: ``resolve_active_specs(physics=declared)`` ADMITS the absolute
axes, but ``layout_hook._static_view`` built the candidate view with neither
``physics`` nor ``quantity_context`` — so every candidate extraction raised
"not available on the candidate_static view". The gate and the extraction
answered from two different sources; these pins keep them one.
"""

import numpy as np
import torch

from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
)
from mimarsinan.search.problems.joint import JointArchHwProblem

PRICED = ["chip_area_mm2", "energy_per_inference_mj"]

#: The candidate model keeps host-side work (flatten/softmax), so the absolute
#: latency/energy headliners honestly require declared host rates — without
#: them the refusal chain names host_macs_per_s (owner decision: host pricing
#: is declared, never defaulted).
_HOST_RATES = {
    "host_macs_per_s": {"nominal": 10.0, "unit": "G/s",
                        "evidence_kind": "estimated", "note": "test host rate"},
    "p_host": {"nominal": 20.0, "unit": "W",
               "evidence_kind": "estimated", "note": "test host power"},
}


def _physics_cfg(**extra):
    return _cfg(
        platform_physics_profile="truenorth",
        platform_physics_overrides=dict(_HOST_RATES),
        activity_factor=0.05,
        **extra,
    )


def _cfg(**extra):
    base = {
        "device": "cpu",
        "input_shape": (1, 8, 8),
        "num_classes": 4,
        "target_tq": 4,
        "weight_bits": 4,
        "lr": 0.001,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "model_config": {
            "mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU",
        },
    }
    base.update(extra)
    return base


def _problem(cfg, names):
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
        model_config_assembler=lambda raw: {**raw, "base_activation": "ReLU"},
        fixed_model_config=dict(cfg["model_config"]),
        platform_resolver=make_platform_resolver(cfg),
        active_objective_names=names,
        num_core_types=1,
        core_axons_bounds=(64, 256),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 64),
        accuracy_seed=0,
    )


def _candidate(problem):
    x = (np.asarray(problem.xl) + np.asarray(problem.xu)) / 2.0
    return problem.decode(x)


class TestPhysicsRidesTheLivePath:
    def test_priced_axes_evaluate_to_finite_positive_numbers(self):
        """The incident shape, end-to-end: admitted axes must EXTRACT."""
        cfg = _physics_cfg()
        problem = _problem(cfg, PRICED + ["param_utilization_pct"])
        out = problem.evaluate(_candidate(problem))
        for key in PRICED:
            assert np.isfinite(out[key]) and out[key] > 0.0, (key, out)

    def test_every_admitted_axis_is_extractable_on_the_live_view(self):
        cfg = _physics_cfg()
        problem = _problem(cfg, PRICED)
        view = problem.candidate_layout(_candidate(problem)).view
        for spec in problem.active_specs:
            assert spec.available(view), (
                f"{spec.key} admitted by the physics gate but not extractable "
                f"on the live candidate view — the two answered differently"
            )

    def test_physics_comes_from_the_candidates_own_platform(self):
        """Per-candidate pcfg carries ``platform_physics_resolved``; the view
        prices with THAT declaration, not a problem-level copy."""
        cfg = _physics_cfg()
        problem = _problem(cfg, PRICED)
        view = problem.candidate_layout(_candidate(problem)).view
        assert view.physics is not None
        assert view.cost_report() is not None


class TestQuantityContextRidesTheLivePath:
    def test_the_view_answers_the_candidates_own_declarations(self):
        """The candidate's OWN decoded chip, not the problem's base."""
        cfg = _physics_cfg()
        problem = _problem(cfg, PRICED + ["param_utilization_pct"])
        cand = _candidate(problem)
        pcfg = cand["platform_constraints"]
        cores = pcfg["cores"]
        q = problem.candidate_layout(cand).view.quantities
        assert q.get("timesteps").value == 32.0  # simulation_steps registry default
        assert q.get("weight_bits").value == 4.0
        assert q.get("cores_physical").value == float(
            sum(ct["count"] for ct in cores)
        )
        assert q.get("neurons_physical").value == float(
            sum(ct["count"] * ct["max_neurons"] for ct in cores)
        )
        assert q.get("axons_physical").value == float(
            sum(ct["count"] * ct["max_axons"] for ct in cores)
        )
        assert q.get("tiles").value > 0.0

    def test_declared_activity_yields_modeled_events(self):
        """An active priced axis pulls the MAC census; events become claimable."""
        cfg = _physics_cfg()
        problem = _problem(cfg, PRICED)
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        assert q.get("synaptic_events").provenance == "modeled"
        assert q.get("onchip_macs").value > 0.0

    def test_proxy_only_search_skips_the_census(self):
        """No active axis needs the context -> no flow walks are spent on it."""
        cfg = _physics_cfg()
        problem = _problem(cfg, ["param_utilization_pct"])
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        assert not q.has("onchip_macs")

    def test_undeclared_activity_keeps_event_claims_absent(self):
        """0 is the registry sentinel for 'undeclared': no assumption, no
        claim — and the run-level gate refuses activity-dependent axes BY
        NAME at resolution, before any candidate is built."""
        import pytest

        cfg = _cfg(
            platform_physics_profile="truenorth",
            platform_physics_overrides=dict(_HOST_RATES),
        )
        problem = _problem(cfg, ["chip_area_mm2"])
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        assert not q.has("synaptic_events")
        with pytest.raises(ValueError, match="energy_per_inference_mj"):
            _problem(cfg, ["energy_per_inference_mj"]).active_specs


class TestNoPhysicsStaysBare:
    def test_a_run_declaring_no_physics_prices_nothing(self):
        problem = _problem(_cfg(), ["param_utilization_pct"])
        view = problem.candidate_layout(_candidate(problem)).view
        assert view.physics is None
        assert view.cost_report() is None


class TestNocRidesTheLivePath:
    def test_noc_total_hops_evaluates_on_the_live_view(self):
        """The wireload model end-to-end: fragments collected, placed on the
        candidate's own resolved floorplan, priced at the declared activity."""
        problem = _problem(_physics_cfg(), ["noc_total_hops"])
        out = problem.evaluate(_candidate(problem))
        assert np.isfinite(out["noc_total_hops"])
        assert out["noc_total_hops"] >= 0.0

    def test_hops_need_no_physics(self):
        """Hops are a count: a profile-less run may still search NoC traffic."""
        problem = _problem(_cfg(activity_factor=0.05), ["noc_total_hops"])
        out = problem.evaluate(_candidate(problem))
        assert np.isfinite(out["noc_total_hops"])

    def test_undeclared_activity_refuses_extraction_by_name(self):
        """No declared switching activity -> no modeled traffic claim; the
        active axis refuses loudly instead of pricing an unstated assumption."""
        import pytest

        problem = _problem(_cfg(), ["noc_total_hops"])
        with pytest.raises(ValueError, match="noc_total_hops"):
            problem.evaluate(_candidate(problem))

    def test_inactive_noc_axis_skips_the_fragment_work(self):
        """No active NoC axis -> no wire-census materialisation, no fragment
        packing; the census gate answers from the registry."""
        problem = _problem(_cfg(activity_factor=0.05), ["param_utilization_pct"])
        layout = problem.candidate_layout(_candidate(problem))
        assert layout.noc is None
        assert not layout.view.quantities.has("noc_total_hops")

    def test_candidate_quantities_carry_the_modeled_noc_census(self):
        problem = _problem(_physics_cfg(), ["noc_total_hops"])
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        for key in ("noc_total_hops", "noc_total_packets",
                    "noc_intra_tile_packets", "noc_inter_tile_packets"):
            assert q.get(key).provenance == "modeled", key

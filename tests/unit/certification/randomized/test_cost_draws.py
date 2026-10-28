"""Code-level draws: exactly the registry's cost_only set, partitioned; program-changing fields drawn one by one."""

from __future__ import annotations

import pytest

import mimarsinan.certification.randomized.cost_draws as cd
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COMPOSITE, COST_ONLY, FUNCTION_BEARING
from mimarsinan.mapping.support.period_census import RESIDENCY_STEP_RULES, STEP_INDEX


class TestPartition:
    def test_the_partition_is_the_registry_cost_only_set(self):
        parts = cd.partition()
        listed = {p for paths in parts.values() for p in paths}
        assert listed == cd.cost_only_set()
        # closure amendment: cores.weight_sign_granularity; N2 amendment: cores.operand.write_source; N2b amendment:
        # cores.write_physics (the last two pricing-only); W02 (pack): position_multiplexing (deferred by name);
        # W08 (lanes): onchip_arithmetic.lane_block and onchip_arithmetic.lanes_per_core (pricing-only); E-6:
        # cores.compute_physics (pricing-only); U-S (the event census): event_census_samples (pricing-only, a
        # measurement of the built program); U-F4: pipelined_multiplexing (deferred by name)
        assert len(cd.cost_only_set()) == 35

    @pytest.mark.parametrize("change", ["add", "drop"])
    def test_registry_drift_breaks_the_partition(self, monkeypatch, change):
        """A cost-only key the runner does not classify (or one it lists that left the set) refuses by name."""
        registry = cd.cost_only_set()
        drifted = registry | {"brand_new_key"} if change == "add" else registry - {"write_ports"}
        monkeypatch.setattr(cd, "cost_only_set", lambda: drifted)
        with pytest.raises(cd.CostDrawError, match="brand_new_key" if change == "add" else "write_ports"):
            cd.partition()

    def test_has_bias_is_function_bearing_and_composites_are_composite(self):
        classes = dict(cd.registry_paths())
        assert classes["cores.has_bias"] == FUNCTION_BEARING
        assert classes["target"] == COMPOSITE and classes["target_overrides"] == COMPOSITE


class TestDraws:
    @pytest.mark.parametrize("seed", range(30))
    def test_a_draw_sets_program_changing_cost_only_fields_only(self, seed):
        drawn = cd.drawn_paths(cd.draw_declaration(seed))
        assert drawn <= set(cd.PROGRAM_CHANGING)
        assert not drawn & set(cd.PRICING_ONLY)
        assert "cores.has_bias" not in drawn
        classes = dict(cd.registry_paths())
        assert {classes[p] for p in drawn} == {COST_ONLY}

    def test_the_cores_fields_are_drawn_one_by_one(self):
        draws = [cd.draw_declaration(seed) for seed in range(200)]
        for field in ("max_axons", "max_neurons", "count"):
            values = {core[field] for d in draws for core in d["cores"]}
            assert len(values) > 10, field
        pairs = {(core["max_axons"], core["max_neurons"]) for d in draws for core in d["cores"]}
        assert len(pairs) > 100

    @pytest.mark.parametrize("seed", range(30))
    def test_every_draw_is_a_legal_declaration(self, seed):
        declaration = cd.platform_constraints(cd.draw_declaration(seed), hardware_bias=seed % 2 == 0)
        assert cd.legal(declaration) == []
        assert all(core["has_bias"] == (seed % 2 == 0) for core in declaration["cores"])

    def test_the_draw_space_is_the_declared_one(self):
        space = cd.DEFAULT_DRAWS
        for seed in range(100):
            d = cd.draw_declaration(seed)
            assert space.core_types[0] <= len(d["cores"]) <= space.core_types[1]
            assert all(space.axons[0] <= c["max_axons"] <= space.axons[1] for c in d["cores"])
            assert all(space.count[0] <= c["count"] <= space.count[1] for c in d["cores"])
            assert space.passes[0] <= d["max_schedule_passes"] <= space.passes[1]

    def test_pricing_only_paths_are_reported_preserved_by_construction(self):
        from mimarsinan.certification.randomized.screen import coverage

        report = coverage([])
        assert set(report["pricing_only"]) == set(cd.PRICING_ONLY)
        assert set(report["pricing_only"].values()) == {cd.PRESERVED_BY_CONSTRUCTION}
        assert set(report["drawn_by_fields"]) == {"residency_plan"}
        # residency_plan is drawn by its fields (CL1); W02's position_multiplexing and U-F4's pipelined form stay
        # deferred by name
        assert set(report["deferred_to_cl1"]) == {"position_multiplexing", "pipelined_multiplexing"}
        assert set(report["deferred_to_cl1"].values()) == {cd.DEFERRED_REASONS["position_multiplexing"],
                                                           cd.DEFERRED_REASONS["pipelined_multiplexing"]}

    def test_the_registry_cost_only_keys_exist(self):
        for path in cd.PROGRAM_CHANGING + cd.PRICING_ONLY + cd.DERIVED_GEOMETRY + cd.DRAWN_BY_FIELDS + cd.DEFERRED:
            assert path.split(".")[0] in REGISTRY, path


class TestResidencyDraws:
    """CL1: G5's residency_plan is drawn, field by field, wherever the registry's relevance holds."""

    SCHEDULED = {"allow_scheduling": True}

    def test_residency_plan_is_drawn_field_by_field(self):
        draws = [cd.draw_declaration(seed, pinned=self.SCHEDULED) for seed in range(200)]
        plans = [d["residency_plan"] for d in draws if "residency_plan" in d]
        assert 60 < len(plans) < 140
        assert {p["step"] for p in plans} == set(RESIDENCY_STEP_RULES)
        for plan in plans:
            assert ("index" in plan) == (plan["step"] == STEP_INDEX), plan
            assert cd.DEFAULT_DRAWS.residency_index[0] <= plan.get("index", 0) <= cd.DEFAULT_DRAWS.residency_index[1]
        paths = set().union(*(cd.drawn_paths(d) for d in draws))
        assert {"residency_plan.step", "residency_plan.index"} <= paths and "residency_plan" not in paths

    def test_residency_plan_is_absent_where_scheduling_is_off(self):
        for seed in range(100):
            drawn = cd.draw_declaration(seed)
            assert "residency_plan" not in drawn or drawn["allow_scheduling"], drawn
            assert "residency_plan" not in cd.draw_declaration(seed, pinned={"allow_scheduling": False})
        assert "residency_plan" not in cd.settled({"allow_scheduling": False, "residency_plan": {"step": "floor"}})

    def test_the_residency_draw_moves_no_other_field_of_a_seed(self):
        """Drawn after every other field, consuming the same values whether present or absent."""
        for seed in range(100):
            with_plan = cd.draw_declaration(seed, pinned=self.SCHEDULED)
            without = cd.draw_declaration(seed, cd.DrawSpace(residency_rate=0.0), pinned=self.SCHEDULED)
            with_plan.pop("residency_plan", None)
            assert with_plan == without, seed

    @pytest.mark.parametrize("seed", range(20))
    def test_a_drawn_residency_plan_is_a_legal_declaration(self, seed):
        declaration = cd.platform_constraints(cd.draw_declaration(seed, pinned=self.SCHEDULED), hardware_bias=False)
        assert cd.legal(declaration) == []


class TestSignDraws:
    """CL-SG: a core type's weight sign granularity is drawn per core type, after every other field, at the draw
    space's rate (0 for the declared strata, so their seeds are unchanged; the sign-split strata draw it)."""

    SPLIT = cd.DrawSpace(sign_split_rate=1.0)

    def test_the_sign_granularity_is_a_program_changing_path(self):
        assert "cores.weight_sign_granularity" in cd.PROGRAM_CHANGING
        assert "cores.weight_sign_granularity" in cd.cost_only_set()

    @pytest.mark.parametrize("seed", range(30))
    def test_the_declared_strata_draw_no_sign_granularity(self, seed):
        assert all("weight_sign_granularity" not in core for core in cd.draw_declaration(seed)["cores"])

    def test_the_sign_split_strata_declare_per_axon_on_every_core_type(self):
        for seed in range(50):
            drawn = cd.draw_declaration(seed, self.SPLIT)
            assert {core["weight_sign_granularity"] for core in drawn["cores"]} == {"per_axon"}
            assert "cores.weight_sign_granularity" in cd.drawn_paths(drawn)

    def test_the_sign_draw_moves_no_other_field_of_a_seed(self):
        for seed in range(100):
            for pinned in ({}, {"allow_scheduling": True}):
                split = cd.draw_declaration(seed, self.SPLIT, pinned=pinned)
                plain = cd.draw_declaration(seed, pinned=pinned)
                for core in split["cores"]:
                    core.pop("weight_sign_granularity")
                assert split == plain, seed

    @pytest.mark.parametrize("seed", range(20))
    def test_a_sign_split_draw_is_a_legal_declaration(self, seed):
        declaration = cd.platform_constraints(cd.draw_declaration(seed, self.SPLIT), hardware_bias=seed % 2 == 0)
        assert cd.legal(declaration) == []


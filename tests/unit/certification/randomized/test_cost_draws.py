"""Code-level draws: exactly the registry's cost_only set, partitioned; program-changing fields drawn one by one."""

from __future__ import annotations

import pytest

import mimarsinan.certification.randomized.cost_draws as cd
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COMPOSITE, COST_ONLY, FUNCTION_BEARING


class TestPartition:
    def test_the_partition_is_the_registry_cost_only_set(self):
        parts = cd.partition()
        listed = {p for paths in parts.values() for p in paths}
        assert listed == cd.cost_only_set()
        assert len(cd.cost_only_set()) == 26

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
        assert set(report["deferred_to_cl1"]) == {"residency_plan", "residency_plan.index", "residency_plan.step"}

    def test_the_registry_cost_only_keys_exist(self):
        for path in cd.PROGRAM_CHANGING + cd.PRICING_ONLY + cd.DERIVED_GEOMETRY + cd.DEFERRED:
            assert path.split(".")[0] in REGISTRY, path

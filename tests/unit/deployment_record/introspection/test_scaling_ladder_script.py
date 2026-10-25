"""The ladder runner (``scripts/scaling_ladder.py``): rows written as they complete, and a per-point budget.

Plan §12.6: ladder points above a declared per-point budget are recorded as such,
never dropped. The runner writes each row as it completes, so a cell stopped at its
wall or memory cap keeps every completed point, and a point whose stage (convert,
layout, pack, schedule or price) runs past the declared per-point wall budget leaves
one row per K naming that stage and the elapsed wall.
"""

from __future__ import annotations

import importlib.util
import json
import math
import signal
import time
from pathlib import Path

import pytest

from mimarsinan.mapping.platform.packaging_contract import SPIKING_PACKAGING
from unit.deployment_record.introspection.test_scaling_ladder import F_HAND, _network, _OneConv, _platform

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "scaling_ladder.py"


def _script():
    spec = importlib.util.spec_from_file_location("scaling_ladder_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spin(seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        pass


class TestPointBudget:
    def test_a_stage_over_the_budget_is_named_with_its_elapsed_wall(self):
        script = _script()
        clock = script.StageClock()
        with pytest.raises(script.PointAboveBudget) as over:
            with script.point_budget(0.2, clock):
                clock.enter("schedule")
                _spin(5.0)
        assert over.value.stage == "schedule" and over.value.budget_s == 0.2
        assert 0.2 <= over.value.elapsed_s < 5.0

    def test_the_budget_signal_passes_through_broad_exception_handlers(self):
        """The mapper forward and the search plane wrap errors in ``except Exception``; the signal is not one."""
        script = _script()
        clock = script.StageClock()
        with pytest.raises(script.PointAboveBudget):
            with script.point_budget(0.2, clock):
                clock.enter("schedule")
                try:
                    _spin(5.0)
                except Exception:  # noqa: BLE001 - a toolchain handler that would swallow or re-wrap it
                    pytest.fail("the per-point budget signal was caught as an Exception")

    def test_the_previous_alarm_handler_and_timer_are_restored(self):
        script = _script()
        handler = signal.getsignal(signal.SIGALRM)
        armed = signal.getitimer(signal.ITIMER_REAL)[0] > 0.0
        with script.point_budget(30.0, script.StageClock()):
            pass
        assert signal.getsignal(signal.SIGALRM) is handler
        assert (signal.getitimer(signal.ITIMER_REAL)[0] > 0.0) == armed


class TestCellRows:
    def test_a_point_over_the_budget_is_recorded_by_stage_and_the_others_are_priced(self, monkeypatch):
        script = _script()
        scheduled = script.schedule_program  # [CL5] the cell schedules once, then prices per variant and K

        def slow_below_f(network, census, platform, facts, *, tiles=None, on_stage=None):
            if platform["cores"][0]["count"] < F_HAND and on_stage is not None:
                on_stage("schedule")
                _spin(10.0)
            return scheduled(network, census, platform, facts, tiles=tiles, on_stage=on_stage)

        monkeypatch.setattr(script, "schedule_program", slow_below_f)
        rows = list(script.cell_rows(
            _network(4), "hand", "value", _platform(64),
            lambda budget, packed: _platform(max(1, math.floor(budget * packed))),
            budgets=(1.0, 0.5), ks=(1, 16), point_budget_s=1.0, wire_census=True,
        ))
        assert [(r["budget"], r["inference_batch_K"]) for r in rows] == [(1.0, 1), (1.0, 16), (0.5, 1), (0.5, 16)]
        assert all("x_packed" in r and "above_budget" not in r for r in rows[:2])
        for row in rows[2:]:
            assert (row["cores"], row["packed_tiles"]) == (math.floor(0.5 * F_HAND), F_HAND)
            above = row["above_budget"]
            assert above["stage"] == "schedule" and above["point_budget_s"] == 1.0
            assert 1.0 <= above["elapsed_s"] < 10.0

    def test_a_program_with_no_on_chip_softcore_is_a_refusal_row_per_point(self):
        """The host-only program is refused by name at the layout stage; every (budget, K) keeps its row."""
        script = _script()
        rows = list(script.cell_rows(
            _network(4, _OneConv, "host_only_conv", SPIKING_PACKAGING), "hand", "spiking", _platform(64),
            lambda budget, packed: _platform(max(1, math.floor(budget * packed))),
            budgets=(1.0, 0.5), ks=(1, 16), point_budget_s=None, wire_census=True,
        ))
        assert [(r["budget"], r["inference_batch_K"]) for r in rows] == [(1.0, 1), (1.0, 16), (0.5, 1), (0.5, 16)]
        for row in rows:
            assert row["network"] == "host_only_conv" and "x_packed" not in row
            assert (row["refusal"]["name"], row["refusal"]["stage"]) == ("LadderGeometryError", "layout")
            assert "no on-chip softcore" in row["refusal"]["reason"]


class TestRowsAreWrittenAsTheyComplete:
    def test_a_run_stopped_mid_cell_keeps_its_completed_rows(self, tmp_path, monkeypatch):
        script = _script()

        def stopped_run(*args, **kwargs):
            yield {"network": "resnet20", "point": 1}
            yield {"network": "resnet20", "point": 2}
            raise MemoryError("the cell's memory cap, mid-cell")

        monkeypatch.setattr(script, "run", stopped_run)
        out = tmp_path / "rows.jsonl"
        with pytest.raises(MemoryError):
            script.main(["--out", str(out), "--networks", "resnet20", "--targets", "truenorth"])
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["header"] is True and lines[0]["point_budget_s"] == script.POINT_BUDGET_S
        assert [line.get("point") for line in lines[1:]] == [1, 2]

    def test_the_smoke_subset_is_the_executed_one(self):
        script = _script()
        assert script.SMOKE == {"networks": ("vit_leaf", "resnet20", "vgg16_bn"),
                                "targets": ("truenorth", "generic_22nm_on_imc_256x256"),
                                "budgets": (1.0, 0.5, 0.25), "ks": (1, 64)}
        assert all(script.ROSTER[n] is not None for n in script.SMOKE["networks"])
        assert set(script.SMOKE["targets"]) <= set(script.TARGETS)


class TestDeclaredCapabilities:
    """[W03] ``--declare KEY=VALUE`` takes a JSON value, so the exploration can declare the residency planner's key
    (``residency_plan={"step": "floor"}``) on every target; booleans and integers read as before."""

    def test_a_json_value_a_boolean_and_an_integer(self):
        script = _script()
        assert script.declared_values(['residency_plan={"step": "floor"}', "allow_scheduling=False",
                                       "max_schedule_passes=64"]) == {
            "residency_plan": {"step": "floor"}, "allow_scheduling": False, "max_schedule_passes": 64}

    def test_the_resolved_platform_carries_the_declared_residency_key(self):
        script = _script()
        target = next(iter(script.TARGETS))
        platform = script.resolved_platform(target, 1.0, 64, script.declared_values(['residency_plan={"step": "floor"}']))
        assert platform["residency_plan"] == {"step": "floor"}

    def test_a_value_that_is_no_json_refuses_by_name(self):
        with pytest.raises(ValueError, match="--declare max_schedule_passes"):
            _script().declared_values(["max_schedule_passes=eight"])


class TestBundleTargetsVariantsAndArms:
    """[CL5] Targets are the shipped bundles by name, a variant prices one schedule under named override blocks, and
    the arms schedule U1 and the residency plans side by side."""

    def test_every_shipped_bundle_is_a_target_with_its_geometry_and_keys(self):
        from mimarsinan.mapping.platform.target_bundles import target_bundles

        script = _script()
        assert set(target_bundles()) <= set(script.TARGETS)
        macro = script.target_spec("sram_cim_macro")
        assert macro["packaging"] == "value" and macro["bundle"]["physics"] is True
        assert [(c["max_axons"], c["max_neurons"], c["count"]) for c in macro["cores"]] == [(128, 1024, 1)]
        assert script.target_spec("truenorth")["packaging"] == "spiking"
        puma = script.target_spec("puma_node_like")
        assert puma["bundle"]["physics"] is False and puma["packaging"] == "value" and puma["bundle"]["packaging_note"]

    def test_a_bundle_permission_replaces_the_scripts_capability_and_core_fields_reach_every_core(self):
        script = _script()
        platform = script.resolved_platform("sram_cim_macro", 0.5, 64, None, {"write_granularity": "cell"})
        assert platform["allow_neuron_splitting"] is False and platform["allow_coalescing"] is True
        assert [c["count"] for c in platform["cores"]] == [32]
        assert all(c["write_granularity"] == "cell" for c in platform["cores"])
        assert script.resolved_platform("sram_cim_macro", "declared", None)["cores"][0]["count"] == 1

    def test_a_variant_merges_its_named_blocks_and_a_constant_declared_twice_refuses(self, tmp_path):
        from mimarsinan.deployment_record.platform_physics.resolve import OverrideBlockConflictError

        script = _script()
        store = {"e_dma_per_byte": {"nominal": 1.0, "unit": "pJ", "evidence_kind": "estimated", "note": "test"},
                 "t_program_per_byte": {"nominal": 2.0, "unit": "ns", "evidence_kind": "estimated", "note": "test"}}
        host = {"platform_physics_overrides": {
            "p_host": {"nominal": 3.0, "unit": "W", "evidence_kind": "estimated", "note": "test"}}}
        (tmp_path / "store.json").write_text(json.dumps(store))
        (tmp_path / "host.json").write_text(json.dumps(host))
        variants = script.variant_blocks(["shipped=", f"s=host:{tmp_path}/host.json,backing_store:{tmp_path}/store.json"])
        assert variants["shipped"] is None
        assert sorted(variants["s"]) == ["e_dma_per_byte", "p_host", "t_program_per_byte"]
        assert variants["s"]["e_dma_per_byte"]["note"] == "[backing_store] test"
        with pytest.raises(OverrideBlockConflictError):
            script.variant_blocks([f"x=a:{tmp_path}/store.json,b:{tmp_path}/store.json"])
        platform = script.resolved_platform("sram_cim_macro", "declared", None, overrides=variants["s"])
        assert platform["platform_physics_resolved"] is not None

    def test_one_schedule_per_arm_priced_per_variant_and_k(self):
        script = _script()
        physics = _platform(8)["platform_physics_resolved"]
        calls = []
        schedule = script.schedule_program

        def counted(*args, **kwargs):
            calls.append(args[2].get("residency_plan"))
            return schedule(*args, **kwargs)

        script.schedule_program = counted
        try:
            def platform_at(budget, packed, residency=None, overrides=None):
                p = _platform(max(1, math.floor(budget * packed)))
                if residency is not None:
                    p["residency_plan"] = dict(residency)
                if overrides is not None:
                    p["platform_physics_resolved"] = overrides
                return p

            rows = list(script.cell_rows(_network(4), "hand", "value", _platform(64), platform_at,
                                         budgets=(0.5,), ks=(1, 16), point_budget_s=None, wire_census=True,
                                         variants={"a": None, "b": physics}, arms=("absent", "floor")))
        finally:
            script.schedule_program = schedule
        assert calls == [None, {"step": "floor"}]
        assert [(r["arm"], r["variant"], r["inference_batch_K"]) for r in rows] == [
            (arm, v, k) for arm in ("absent", "floor") for v in ("a", "b") for k in (1, 16)]
        assert all("x_packed" in r for r in rows)


class TestRosterKinds:
    def test_every_member_names_a_builder_kind_the_script_reads(self):
        script = _script()
        for name, entry in script.ROSTER.items():
            if entry is None:
                continue
            key = entry[0]
            kind = key.split(":", 1)[0]
            assert kind in ("torchvision", "factory") or key in __import__(
                "mimarsinan.models.builders", fromlist=["BUILDERS_REGISTRY"]).BUILDERS_REGISTRY, name
            assert len(script.member_shape(name, None)) in (3,)

    def test_the_resolution_replaces_the_side_of_an_image_member_only(self):
        script = _script()
        assert script.member_shape("tv_resnet18", None) == (3, 224, 224)
        assert script.member_shape("tv_resnet18", 112) == (3, 112, 112)
        assert script.member_shape("lenet5", None) == (1, 28, 28)


class TestRefusedDeclarations:
    """[CL5] A budget the toolchain refuses to declare (a floorplan too small for the cores) is a point refused by
    name at the "declare" stage; a named toolchain ValueError that is no ToolchainRefusal is recorded and marked."""

    def test_an_undeclarable_budget_is_refused_by_name_and_the_others_are_priced(self):
        from mimarsinan.config_schema.registry.value_schema import DeclarationValueError

        script = _script()

        def platform_at(budget, packed, residency=None, overrides=None):
            if budget == 1.0:
                raise DeclarationValueError("floorplan capacity cannot hold the declared cores (test)")
            return _platform(max(1, math.floor(budget * packed)))

        rows = list(script.cell_rows(_network(4), "hand", "value", _platform(64), platform_at, budgets=(1.0, 0.5),
                                     ks=(1,), point_budget_s=None, wire_census=True, variants={"a": None},
                                     arms=("absent",)))
        assert rows[0]["refusal"]["name"] == "DeclarationValueError" and rows[0]["refusal"]["stage"] == "declare"
        assert "x_packed" in rows[1] and "refusal" not in rows[1]

    def test_a_named_toolchain_value_error_is_recorded_as_a_marked_refusal(self):
        from mimarsinan.mapping.platform.mapping_structure import WideFanInUnsupportedError

        script = _script()

        def refuse(clock):
            clock.enter("layout")
            raise WideFanInUnsupportedError("fan-in 785 exceeds max_axons 127 (test)")

        result, mark = script.attempt(None, refuse)
        assert result is None and mark["refusal"]["name"] == "WideFanInUnsupportedError"
        assert mark["refusal"]["stage"] == "layout" and mark["refusal"]["toolchain_refusal"] is False
        with pytest.raises(ValueError, match="plain"):
            script.attempt(None, lambda clock: (_ for _ in ()).throw(ValueError("plain")))


class TestCapacityRows:
    """[CL5] A pack the toolchain refuses keeps x_fluid (the layout's logical cells over capacity) beside the
    refusal, so the 224 map keeps the fluid overflow of a member no declared packing admits."""

    def test_a_refused_pack_keeps_the_fluid_columns(self, monkeypatch):
        import mimarsinan.deployment_record.introspection.scaling_ladder_types as types

        script = _script()

        def refuse(census, platform):
            raise types.LadderGeometryError("the program does not pack even on unbounded cores (test)")

        monkeypatch.setattr(script, "packed_tiles", refuse)
        (row,) = script.x_rows(_network(4), "hand", _platform(64), point_budget_s=None, wire_census=True)
        assert row["x_fluid"] > 0 and row["x_packed"] is None and row["packed_tiles"] is None
        assert row["pack_refusal"]["name"] == "LadderGeometryError" and row["pack_refusal"]["stage"] == "pack"


class TestAbsoluteBudgets:
    """[CL5b] A budget c:<count> declares an absolute core count, the same chip for every network."""

    def test_an_absolute_budget_ignores_the_packed_tile_count(self):
        script = _script()
        assert script.resolved_platform("neurram", "c:96", 7051)["cores"][0]["count"] == 96
        assert script.resolved_platform("neurram", "c:96", 13)["cores"][0]["count"] == 96
        assert script.resolved_platform("neurram", 0.5, 64)["cores"][0]["count"] == 32

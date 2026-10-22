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
        priced = script.price_program

        def slow_below_f(network, census, platform, facts, *, tiles=None, on_stage=None):
            if platform["cores"][0]["count"] < F_HAND and on_stage is not None:
                on_stage("schedule")
                _spin(10.0)
            return priced(network, census, platform, facts, tiles=tiles, on_stage=on_stage)

        monkeypatch.setattr(script, "price_program", slow_below_f)
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

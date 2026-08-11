"""``/api/runs/{id}/discovered`` is gone, and the search result still reaches the GUI.

The endpoint read ``final_population.json`` — a LIST — with ``.get("best")``
inside ``best_effort``, so it answered ``{"discovered": false}`` unconditionally
and forever, and no frontend module ever called it. The architecture-search
result reaches the GUI through the pipeline entry (``architecture_search_result``,
step-cache-backed) and the run-artifact inventory instead; this pins both halves
so the dead route cannot come back as "the way the GUI learns what was found".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mimarsinan.gui.runs import list_dir_artifacts
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.server import create_app
from mimarsinan.visualization import write_final_population_json


@pytest.fixture
def client() -> TestClient:
    collector = DataCollector()
    collector.set_pipeline_info(["s1"], {})
    return TestClient(create_app(collector, run_config_fn=None))


class TestTheDeadRouteIsGone:
    def test_no_route_named_discovered_is_registered(self, client):
        paths = {getattr(r, "path", "") for r in client.app.routes}
        assert not any(p.endswith("/discovered") for p in paths)

    def test_requesting_it_is_a_404(self, client):
        assert client.get("/api/runs/some_run/discovered").status_code == 404

    def test_no_frontend_module_calls_it(self):
        static = Path(__file__).resolve().parents[3] / "src/mimarsinan/gui/static"
        callers = [
            path.name for path in static.rglob("*.js")
            if "/discovered" in path.read_text(encoding="utf-8")
        ]
        assert not callers, f"a live caller appeared: {callers}"


class TestThePopulationArtifactSurvivesOnItsOwnMerit:
    """``write_final_population_json`` outlives the endpoint: it is a run-directory
    artifact the generic inventory lists, not a private feed for the dead route."""

    def _result_json(self):
        return {
            "pareto_front": [
                {
                    "configuration": {
                        "model_config": {"mlp_width_1": 32},
                        "platform_constraints": {"target_tq": 4},
                    },
                    "objectives": {"fragmentation_pct": 1.5},
                },
            ],
        }

    def test_it_writes_one_flat_row_per_pareto_member(self, tmp_path):
        out = tmp_path / "final_population.json"
        write_final_population_json(self._result_json(), str(out))
        assert json.loads(out.read_text()) == [
            {"mlp_width_1": 32, "target_tq": 4, "fragmentation_pct": 1.5},
        ]

    def test_the_run_artifact_inventory_lists_it(self, tmp_path):
        write_final_population_json(
            self._result_json(), str(tmp_path / "final_population.json"),
        )
        listed = {row["path"]: row for row in list_dir_artifacts(str(tmp_path))}
        assert "final_population.json" in listed
        assert listed["final_population.json"]["kind"] == "json"

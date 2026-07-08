"""Schema-driven wizard API: /api/config_schema and /api/config/resolve end-to-end."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.server.app import create_app

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(DataCollector()))


def _t0_01() -> dict:
    path = os.path.join(
        _REPO_ROOT, "test_configs", "tier0", "t0_01_lif_mmixcore_wq_s4.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestConfigSchemaEndpoint:
    def test_serves_the_full_registry(self, client):
        payload = client.get("/api/config_schema").json()
        assert set(payload["keys"]) == set(REGISTRY)
        assert len(payload["groups"]) == 9

    def test_serves_sub_schema_surfaces(self, client):
        payload = client.get("/api/config_schema").json()
        assert "optimizer" in payload["recipe_fields"]
        assert payload["preprocessing_fields"]["normalize"]["options"]
        assert payload["nas"]["optimizer_options"]
        assert payload["dynamic_options"]["model_type"] == "/api/model_types"

    def test_every_key_record_is_renderable(self, client):
        payload = client.get("/api/config_schema").json()
        for key, record in payload["keys"].items():
            assert record["label"], key
            assert record["doc"], key
            assert record["relevant"]["op"], key
            if record["type"] == "enum":
                assert record["options"], key


class TestResolveEndpoint:
    def test_tier_config_resolves_with_zero_errors_and_a_step_preview(self, client):
        body = client.post("/api/config/resolve", json=_t0_01()).json()
        assert body["ok"] is True
        assert body["errors"] == []
        assert body["unknown_keys"] == []
        steps = body["pipeline"]["steps"]
        assert len(steps) >= 10
        assert "Pretraining" in steps
        assert len(body["pipeline"]["semantic_groups"]) == len(steps)
        assert body["derived"]["activation_quantization"]["value"] is True
        assert "lif" in body["derived"]["activation_quantization"]["why"]

    def test_contract_violation_comes_back_keyed(self, client):
        body = client.post("/api/config/resolve", json={
            "pipeline_mode": "phased",
            "deployment_parameters": {"weight_quantization": False},
            "platform_constraints": {"weight_bits": 5},
        }).json()
        assert body["ok"] is False
        assert any(
            e["key"] == "weight_quantization" and e["rule_id"] == "quantization_assembly"
            for e in body["errors"]
        )
        assert body["pipeline"]["steps"] == []

    def test_resolve_serves_the_emitted_document(self, client):
        """The review pane shows EXACTLY what Launch submits: the emitted
        document rides the resolve payload (lossless for tier configs)."""
        body = client.post("/api/config/resolve", json=_t0_01()).json()
        assert body["emitted"] == _t0_01()

    def test_unknown_keys_reported(self, client):
        draft = _t0_01()
        draft["deployment_parameters"]["endpoint_floor_wall_s"] = 60
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["unknown_keys"] == ["deployment_parameters.endpoint_floor_wall_s"]


class TestStarterEndpoint:
    def test_starter_is_served_and_resolves_clean(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["ok"] is True
        assert body["errors"] == []
        assert body["unknown_keys"] == []
        assert len(body["pipeline"]["steps"]) >= 10

    def test_starter_names_are_fresh(self, client):
        first = client.get("/api/config/starter").json()
        second = client.get("/api/config/starter").json()
        assert first["experiment_name"]
        # Same-second calls may collide on the timestamp; the counter suffix
        # must still separate them.
        assert first["experiment_name"] != second["experiment_name"]


class TestRunEndpoint:
    def test_validation_failure_is_a_400(self, client):
        res = client.post("/api/run?validate=1", json={
            "deployment_parameters": {"spiking_mode": "not_a_mode"},
        })
        assert res.status_code == 400
        assert res.json()["field_errors"]

    def test_wizard_page_serves_the_module_entrypoint(self, client):
        html = client.get("/wizard").text
        assert "/static/js/wizard/main.js" in html
        assert "cdn.plot.ly" not in html

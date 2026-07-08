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
        assert len(payload["groups"]) == 11

    def test_serves_sub_schema_surfaces(self, client):
        payload = client.get("/api/config_schema").json()
        assert "optimizer" in payload["recipe_fields"]
        assert payload["preprocessing_fields"]["normalize"]["options"]
        assert payload["nas"]["optimizer_options"]
        assert payload["dynamic_options"]["model_type"] == "/api/model_types"

    def test_serves_the_hw_search_space_sub_schema(self, client):
        """Round-3 defect 10: search_space renders as a STRUCTURED editor —
        the field schema is served from the search-space SSOT, never
        hardcoded in JS."""
        payload = client.get("/api/config_schema").json()
        fields = payload["hw_search_space_fields"]
        assert fields["num_core_types"]["type"] == "int"
        for key in ("core_axons_bounds", "core_neurons_bounds",
                    "core_count_bounds"):
            spec = fields[key]
            assert spec["type"] == "int_range", key
            lo, hi = spec["default"]
            assert 0 < lo < hi, key
            assert spec["doc"], key

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


class TestPipelinePreviewHonesty:
    """The step preview IS the assembly: simulator steps appear exactly when
    the ConversionPolicy recipe enables their backend for the mode."""

    def _steps_for(self, client, mode, schedule=None):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["spiking_mode"] = mode
        if schedule:
            draft["deployment_parameters"]["ttfs_cycle_schedule"] = schedule
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["ok"] is True, body["errors"]
        return body["pipeline"]["steps"], body["derived"]

    def test_lif_preview_carries_all_three_simulators(self, client):
        steps, derived = self._steps_for(client, "lif")
        assert "Simulation" in steps
        assert "Loihi Simulation" in steps
        assert "SANA-FE Simulation" in steps
        assert derived["enable_loihi_simulation"]["value"] is True

    def test_ttfs_preview_drops_the_loihi_step(self, client):
        steps, derived = self._steps_for(client, "ttfs")
        assert "Loihi Simulation" not in steps
        assert derived["enable_loihi_simulation"]["value"] is False
        assert derived["enable_loihi_simulation"]["why"]

    def test_synchronized_preview_drops_the_nevresim_step(self, client):
        steps, derived = self._steps_for(client, "ttfs_cycle_based", "synchronized")
        assert "Simulation" not in steps
        assert "SANA-FE Simulation" in steps
        assert derived["enable_nevresim_simulation"]["value"] is False


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


class TestVehiclesAlwaysServed:
    """Round-4 defect 5 (server half): the resolve payload serves the vehicle
    rows UNCONDITIONALLY — unrelated draft errors must never remove the
    vehicle state the toggles render from."""

    _ENABLES = (
        "enable_nevresim_simulation", "enable_loihi_simulation",
        "enable_sanafe_simulation",
    )

    def test_starter_serves_all_vehicle_rows(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        rows = {r["key"]: r for r in body["vehicles"]}
        assert set(rows) == set(self._ENABLES)
        for key in self._ENABLES:
            assert rows[key]["supported"] is True, key
            assert rows[key]["on"] is True, key
            assert rows[key]["declared"] is False, key
            assert rows[key]["why"], key

    def test_vehicles_survive_unrelated_errors(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["s_allocation"] = "explicit"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["ok"] is False and body["errors"]
        rows = {r["key"]: r for r in body["vehicles"]}
        assert set(rows) == set(self._ENABLES)
        assert rows["enable_sanafe_simulation"]["on"] is True

    def test_declared_off_is_reported(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["enable_sanafe_simulation"] = False
        body = client.post("/api/config/resolve", json=draft).json()
        row = {r["key"]: r for r in body["vehicles"]}["enable_sanafe_simulation"]
        assert row["supported"] is True
        assert row["on"] is False
        assert row["declared"] is True

    def test_unsupported_backend_is_reported(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["spiking_mode"] = "ttfs"
        body = client.post("/api/config/resolve", json=draft).json()
        row = {r["key"]: r for r in body["vehicles"]}["enable_loihi_simulation"]
        assert row["supported"] is False
        assert row["on"] is False

    def test_unknown_mode_degrades_without_dropping_rows(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["spiking_mode"] = "not_a_mode"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["ok"] is False
        rows = {r["key"]: r for r in body["vehicles"]}
        assert set(rows) == set(self._ENABLES)
        for row in rows.values():
            assert row["supported"] is None
            assert row["why"]


class TestResolvedValuesServed:
    """Round-4 defect 8 (server half): the payload carries the CONCRETE
    resolved value for every registry key, so the UI renders 'derived: <x>'
    instead of prose."""

    def test_starter_resolved_values(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        resolved = body["resolved"]
        assert resolved["firing_mode"] == "Default"
        # Recipe-folded mode-aware default: concrete, not prose.
        assert isinstance(resolved["wq_endpoint_recovery_steps"], int)
        assert isinstance(resolved["tuning_recipe"], dict)
        assert set(resolved) <= set(REGISTRY)

    def test_erroring_draft_serves_no_hypothetical_values(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["spiking_mode"] = "not_a_mode"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["resolved"] == {}


class TestWeightSourceIsBuilderProvided:
    """Round-5 item 4: ONE pretrained-weight-source concept. Turning on the
    regime resolves the source from the model builder's registration (green
    derived in the wizard); a builder that registers nothing fails LOUD, the
    same way the pipeline's DeploymentPlan does."""

    def test_regime_resolves_the_builder_registered_source(self, client, monkeypatch):
        from mimarsinan.common.workload_profile import ModelWorkloadProfile
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        monkeypatch.setattr(
            ModelRegistry, "get_workload_profile",
            classmethod(lambda cls, model_id: ModelWorkloadProfile(
                pretrained_weight_source="torchvision",
            )),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["resolved"]["weight_source"] == "torchvision"
        assert body["derived"]["weight_source"]["value"] == "torchvision"

    def test_regime_without_a_registration_is_a_keyed_error(self, client, monkeypatch):
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        monkeypatch.setattr(
            ModelRegistry, "get_workload_profile",
            classmethod(lambda cls, model_id: None),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        errors = [e for e in body["errors"] if e["key"] == "weight_source"]
        assert len(errors) == 1
        assert errors[0]["rule_id"] == "weight_source_regime"
        assert "registers no pretrained weight source" in errors[0]["message"]
        # No hypothetical source value is served while the draft errors.
        assert body["resolved"] == {}

    def test_explicit_source_wins_over_the_registration(self, client, monkeypatch):
        from mimarsinan.common.workload_profile import ModelWorkloadProfile
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        monkeypatch.setattr(
            ModelRegistry, "get_workload_profile",
            classmethod(lambda cls, model_id: ModelWorkloadProfile(
                pretrained_weight_source="torchvision",
            )),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        draft["deployment_parameters"]["weight_source"] = "/ckpt/best.pt"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["resolved"]["weight_source"] == "/ckpt/best.pt"

    def test_no_regime_no_source(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["derived"]["weight_source"]["value"] is None
        assert "scratch" in body["derived"]["weight_source"]["why"]


class TestBaselineIsTheDiffBaseline:
    """Round-4 defect 7: 'differs from defaults' means differs from the
    STARTER baseline document (the wizard's data-layer default); framework
    defaults stay workload-neutral. Exception: experiment_name is always a
    user delta (its fresh name has no default)."""

    def _differs(self, body):
        return {r["key"] for r in body["diff_vs_defaults"] if r["differs"]}

    def test_pristine_starter_shows_only_the_experiment_name(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        assert self._differs(body) == {"experiment_name"}

    def test_a_true_user_delta_differs_against_the_baseline_value(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["lr"] = 0.004
        body = client.post("/api/config/resolve", json=draft).json()
        rows = {r["key"]: r for r in body["diff_vs_defaults"]}
        assert rows["lr"]["differs"] is True
        assert rows["lr"]["default"] == 0.003  # the baseline, not the framework 0.001

    def test_experiment_name_derives_from_the_baseline_vehicle(self, client):
        draft = client.get("/api/config/starter").json()
        assert draft["experiment_name"].startswith("lenet5_")

    def test_schema_payload_serves_the_baseline_overlay(self, client):
        keys = client.get("/api/config_schema").json()["keys"]
        assert keys["lr"]["baseline"] == 0.003
        assert keys["training_epochs"]["baseline"] == 2
        assert keys["encoding_layer_placement"]["baseline"] == "subsume"
        assert "baseline" not in keys["experiment_name"]
        # Framework defaults stay untouched next to the overlay.
        assert keys["lr"]["default"] == 0.001


class TestMetadataWorkloadFacts:
    """Round-4 defect 8 (data side): the provider metadata carries the
    registered workload facts so the UI can render concrete derived values
    for the profile-injected keys."""

    def test_metadata_carries_workload_facts(self, client, monkeypatch):
        import mimarsinan.data_handling.data_providers  # noqa: F401 — registers providers
        from mimarsinan.common.workload_profile import DataWorkloadProfile
        from mimarsinan.data_handling.data_provider_factory import (
            BasicDataProviderFactory,
        )

        class _FakeProvider:
            def workload_profile(self):
                return DataWorkloadProfile(
                    input_value_range=(0.0, 2.75), eval_subsample_target=4096,
                )

        monkeypatch.setattr(
            BasicDataProviderFactory, "get_metadata",
            classmethod(lambda cls, name, datasets_path="./datasets", *, preprocessing=None: {
                "id": name, "label": name, "input_shape": [1, 28, 28],
                "num_classes": 10, "supports_preprocessing": True,
            }),
        )
        monkeypatch.setattr(
            BasicDataProviderFactory, "create", lambda self: _FakeProvider(),
        )
        body = client.get(
            "/api/data_providers/MNIST_DataProvider/metadata?resize_to=28"
        ).json()
        assert body["workload_facts"] == {
            "input_data_scale": 2.75, "eval_subsample_target": 4096,
        }

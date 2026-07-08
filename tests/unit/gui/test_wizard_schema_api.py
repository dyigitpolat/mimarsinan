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


class TestRound5WorkbenchLayout:
    """Round-5 items 1 and 5, structural half: the mapping-strategy panel is a
    Model-column sibling and derived chips render ONLY in Review."""

    def _codesign(self, client):
        html = client.get("/wizard").text
        start = html.index('data-section-id="codesign"')
        return html[start:html.index("</section>", start)]

    def test_mapping_strategy_sits_below_model_in_the_left_column(self, client):
        section = self._codesign(client)
        left = section[section.index('class="wb-col"'):]
        left = left[: left.index("</div>\n            <div class=\"wb-col\"")] \
            if '</div>\n            <div class="wb-col"' in left else left
        model = section.index('data-groups="model"')
        strategy = section.index('data-groups="mapping_strategy"')
        hardware = section.index('data-groups="hardware"')
        # Left column: Model then Mapping strategy; Hardware opens the right one.
        assert model < strategy < hardware

    def test_the_mapping_panel_spans_the_full_width_beneath(self, client):
        section = self._codesign(client)
        assert section.index('data-groups="mapping_strategy"') < section.index(
            'id="hwStatsPanel"'
        )
        assert 'class="codesign-mapping"' in section

    def test_only_review_hosts_derived_chips(self, client):
        html = client.get("/wizard").text
        assert html.count('id="derivedChips"') == 1
        # The deployment-semantics derived strip is gone: section pages stay clean.
        assert 'id="deploymentDerived"' not in html
        assert "derived-strip" not in self._codesign(client)


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


from mimarsinan.common.workload_profile import ModelWorkloadProfile
from mimarsinan.common.pretrained import PretrainedWeightSet


def _weight_set(**kw):
    base = dict(
        id="imagenet1k_v1", label="ImageNet-1K · V1", task="image classification",
        dataset="ImageNet-1K", input_shape=(3, 224, 224), num_classes=1000,
        source="torchvision", expected_accuracy=0.76, license="BSD-3-Clause",
        num_parameters=25557032, preprocessing={"resize_to": 256},
        adapts_input_shape=True, adapts_num_classes=True,
    )
    base.update(kw)
    return PretrainedWeightSet(**base)


def _register_sets(monkeypatch, *sets):
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

    monkeypatch.setattr(
        ModelRegistry, "get_workload_profile",
        classmethod(lambda cls, model_id: ModelWorkloadProfile(
            pretrained_weight_sets=tuple(sets),
        )),
    )


class TestPretrainedWeightSetsAreBuilderProvided:
    """Round-7: a builder registers a SET of pretrained weight sets. The switch
    and the selector are driven by the builder registration folded into resolve;
    a builder that registers nothing fails LOUD exactly as the pipeline does."""

    def test_regime_resolves_the_chosen_registered_source(self, client, monkeypatch):
        _register_sets(monkeypatch, _weight_set())
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["resolved"]["weight_source"] == "torchvision"
        assert body["derived"]["weight_source"]["value"] == "torchvision"
        assert body["derived"]["pretrained_weight_set"]["value"] == "imagenet1k_v1"

    def test_regime_without_a_registration_is_a_keyed_error(self, client, monkeypatch):
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        monkeypatch.setattr(
            ModelRegistry, "get_workload_profile",
            classmethod(lambda cls, model_id: ModelWorkloadProfile()),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        errors = [e for e in body["errors"] if e["key"] == "weight_source"]
        assert len(errors) == 1
        assert errors[0]["rule_id"] == "weight_source_regime"
        assert "registers no pretrained weights" in errors[0]["message"]
        assert body["resolved"] == {}

    def test_explicit_source_wins_over_the_registration(self, client, monkeypatch):
        _register_sets(monkeypatch, _weight_set())
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        draft["deployment_parameters"]["weight_source"] = "/ckpt/best.pt"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["resolved"]["weight_source"] == "/ckpt/best.pt"
        assert "explicit" in body["derived"]["weight_source"]["why"]

    def test_no_regime_no_source(self, client):
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["derived"]["weight_source"]["value"] is None
        assert "scratch" in body["derived"]["weight_source"]["why"]


class TestPretrainedPanelBlock:
    """The dedicated panel renders from an ALWAYS-served `pretrained` block —
    like the vehicles block, it survives an erroring draft and carries the
    switch state, the reason, the registered records and the selection."""

    def test_lenet5_switch_is_disabled_with_a_reason(self, client):
        draft = client.get("/api/config/starter").json()  # lenet5 registers nothing
        body = client.post("/api/config/resolve", json=draft).json()
        pretrained = body["pretrained"]
        assert pretrained["available"] is False
        assert pretrained["legal_preload"] == [False]
        assert "lenet5" in pretrained["reason"] and "no pretrained" in pretrained["reason"]
        assert pretrained["sets"] == []

    def test_single_set_locks_the_choice(self, client, monkeypatch):
        _register_sets(monkeypatch, _weight_set())
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        pretrained = body["pretrained"]
        assert pretrained["available"] is True
        assert pretrained["legal_set_ids"] == ["imagenet1k_v1"]
        assert pretrained["selected"] == "imagenet1k_v1"
        assert body["legal_values"]["pretrained_weight_set"] == ["imagenet1k_v1"]
        # The full record is revealed — every declared fact.
        record = pretrained["sets"][0]
        assert record["dataset"] == "ImageNet-1K" and record["num_classes"] == 1000
        assert record["expected_accuracy"] == 0.76 and record["license"]

    def test_multiple_sets_offer_a_selection(self, client, monkeypatch):
        _register_sets(
            monkeypatch, _weight_set(),
            _weight_set(id="imagenet1k_v2", label="V2", source="https://x/y.pt"),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["legal_values"]["pretrained_weight_set"] == [
            "imagenet1k_v1", "imagenet1k_v2",
        ]
        assert body["pretrained"]["selected"] == "imagenet1k_v1"

    def test_a_chosen_set_supplies_its_source(self, client, monkeypatch):
        _register_sets(
            monkeypatch, _weight_set(),
            _weight_set(id="imagenet1k_v2", label="V2", source="https://x/y.pt"),
        )
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        draft["deployment_parameters"]["pretrained_weight_set"] = "imagenet1k_v2"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["errors"] == []
        assert body["resolved"]["weight_source"] == "https://x/y.pt"
        assert body["pretrained"]["selected"] == "imagenet1k_v2"

    def test_switch_locks_on_when_a_source_is_declared(self, client):
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["weight_source"] = "/ckpt/best.pt"
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["legal_values"]["preload_weights"] == [True]

    def test_switch_offers_both_when_a_registration_applies(self, client, monkeypatch):
        _register_sets(monkeypatch, _weight_set())
        draft = client.get("/api/config/starter").json()
        body = client.post("/api/config/resolve", json=draft).json()
        assert body["legal_values"]["preload_weights"] == [False, True]

    def test_an_unregistered_set_id_is_a_keyed_error_not_a_500(self, client, monkeypatch):
        _register_sets(monkeypatch, _weight_set())
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["preload_weights"] = True
        draft["deployment_parameters"]["pretrained_weight_set"] = "imagenet1k_v99"
        resp = client.post("/api/config/resolve", json=draft)
        assert resp.status_code == 200
        body = resp.json()
        rows = [e for e in body["errors"] if e["key"] == "pretrained_weight_set"]
        assert rows and rows[0]["rule_id"] == "legal_value_set"
        assert "imagenet1k_v99" in rows[0]["message"]
        assert rows[0]["remedies"]

    def test_an_inapplicable_set_is_a_keyed_incompatibility_error(self, client, monkeypatch):
        """A set requiring a model_config the draft contradicts is filtered from
        the legal set; pinning it names the mismatch, never loads silently."""
        gelu = _weight_set(id="gelu_only",
                           model_config_requires={"base_activation": "GELU"})
        _register_sets(monkeypatch, gelu)
        draft = client.get("/api/config/starter").json()
        draft["deployment_parameters"]["model_config"] = {"base_activation": "ReLU"}
        draft["deployment_parameters"]["preload_weights"] = True
        draft["deployment_parameters"]["pretrained_weight_set"] = "gelu_only"
        resp = client.post("/api/config/resolve", json=draft)
        assert resp.status_code == 200
        body = resp.json()
        rows = [e for e in body["errors"] if e["key"] == "pretrained_weight_set"]
        assert rows and rows[0]["rule_id"] == "legal_value_set"
        assert body["pretrained"]["available"] is False
        assert "base_activation" in body["pretrained"]["reason"]


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


class TestRound6LegalValueSetsAreServed:
    """The resolve payload exposes the legal set of every legality-bearing key
    for the CURRENT state — the frontend locks/limits from that set alone."""

    def _resolve(self, client, **dp):
        body = _t0_01()
        body["deployment_parameters"].update(dp)
        return client.post("/api/config/resolve", json=body).json()

    def test_legal_values_are_served_for_every_legality_bearing_key(self, client):
        payload = self._resolve(client)
        served = payload["legal_values"]
        expected = {k for k, e in REGISTRY.items() if e.legal_values is not None}
        assert set(served) == expected

    def test_ttfs_locks_firing_and_spike_generation(self, client):
        payload = self._resolve(
            client, spiking_mode="ttfs", firing_mode="TTFS",
            spike_generation_mode="TTFS", weight_quantization=True,
        )
        assert payload["legal_values"]["firing_mode"] == ["TTFS"]
        assert payload["legal_values"]["spike_generation_mode"] == ["TTFS"]

    def test_lif_offers_only_the_legal_subset(self, client):
        payload = self._resolve(client)
        assert payload["legal_values"]["firing_mode"] == ["Default", "Novena"]
        assert payload["legal_values"]["spike_generation_mode"] == [
            "Uniform", "Deterministic", "Stochastic",
        ]
        assert "TTFS" not in payload["legal_values"]["spike_generation_mode"]

    def test_legal_values_survive_an_erroring_draft(self, client):
        """Like the vehicle rows: an unrelated error must never remove the sets
        the widgets render from."""
        body = _t0_01()
        body["deployment_parameters"]["weight_quantization"] = False
        payload = client.post("/api/config/resolve", json=body).json()
        assert not payload["ok"]
        assert payload["legal_values"]["firing_mode"]

    def test_an_unknown_mode_still_serves_full_option_sets(self, client):
        payload = self._resolve(client, spiking_mode="banana")
        assert not payload["ok"]
        assert payload["legal_values"]["firing_mode"] == ["Default", "Novena", "TTFS"]


class TestRound6NoUncaughtDerivationErrors:
    """Every authorable document resolves to a keyed error, never a 500."""

    def test_the_reported_bug_returns_a_keyed_remediable_error(self, client):
        body = _t0_01()
        body["deployment_parameters"]["spiking_mode"] = "lif"
        body["deployment_parameters"]["firing_mode"] = "TTFS"
        response = client.post("/api/config/resolve", json=body)
        assert response.status_code == 200
        payload = response.json()
        row = next(e for e in payload["errors"] if e["key"] == "firing_mode")
        assert row["rule_id"] == "legal_value_set"
        assert any(r["action"] == "clear" for r in row["remedies"])
        assert payload["pipeline"]["steps"] == []
        assert payload["resolved"] == {}

    @pytest.mark.parametrize("spiking_mode", ["lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"])
    @pytest.mark.parametrize("key", ["firing_mode", "spike_generation_mode", "thresholding_mode"])
    def test_the_whole_legality_matrix_returns_200(self, client, spiking_mode, key):
        options = REGISTRY[key].resolved_options() or ()
        legal = set(REGISTRY[key].legal_values({"spiking_mode": spiking_mode}))
        for value in [o for o in options if o not in legal]:
            body = _t0_01()
            body["deployment_parameters"].update(
                {"spiking_mode": spiking_mode, key: value}
            )
            response = client.post("/api/config/resolve", json=body)
            assert response.status_code == 200, (spiking_mode, key, value)
            payload = response.json()
            assert not payload["ok"]
            assert any(e["key"] == key for e in payload["errors"])

    def test_a_plan_resolution_failure_is_a_keyed_error_not_a_500(self, client):
        """DeploymentPlan.resolve raises for a bad optimization driver; the
        pipeline PREVIEW must route it through the error channel."""
        body = _t0_01()
        body["deployment_parameters"]["s_allocation"] = "budget"
        response = client.post("/api/config/resolve", json=body)
        assert response.status_code == 200
        payload = response.json()
        assert not payload["ok"]
        assert any(e["key"] == "s_allocation" for e in payload["errors"])

    def test_an_unknown_spiking_mode_never_500s(self, client):
        body = _t0_01()
        body["deployment_parameters"]["spiking_mode"] = "banana"
        response = client.post("/api/config/resolve", json=body)
        assert response.status_code == 200
        assert not response.json()["ok"]

    def test_a_preload_without_registration_stays_keyed(self, client):
        body = _t0_01()
        body["deployment_parameters"]["preload_weights"] = True
        response = client.post("/api/config/resolve", json=body)
        assert response.status_code == 200


class TestRound6DerivedValuesAreConcrete:
    """Every SSOT-sourced key serves the CONCRETE value the deriver produces —
    the wizard's faded-green in-field text is never prose about the source."""

    def _resolve(self, client, **dp):
        body = _t0_01()
        body["deployment_parameters"].update(dp)
        return client.post("/api/config/resolve", json=body).json()

    def test_every_sourced_key_is_served(self, client):
        payload = self._resolve(client)
        sourced = {k for k, e in REGISTRY.items() if e.provenance is not None}
        assert set(payload["derived_values"]) == sourced

    def test_the_deployment_target_family_serves_real_numbers(self, client):
        values = self._resolve(client)["derived_values"]
        assert values["nf_scm_parity_samples"] == 2
        assert values["nf_scm_parity_atol"] == 1e-6
        assert values["nf_scm_parity_max_mismatch_fraction"] == 0.25
        assert values["nf_scm_parity_min_agreement"] == 0.98
        assert values["scm_torch_sim_parity_samples"] == 256
        assert values["onchip_min_fraction"] == 0.2
        assert values["capacity_gate"] is True

    def test_the_sample_count_follows_the_mode(self, client):
        values = self._resolve(
            client, spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
            firing_mode="TTFS", spike_generation_mode="TTFS",
        )["derived_values"]
        assert values["nf_scm_parity_samples"] == 64

    def test_no_derived_value_is_its_empty_means_prose(self, client):
        values = self._resolve(client)["derived_values"]
        for key, value in values.items():
            prose = REGISTRY[key].empty_means
            if prose is not None:
                assert value != prose, key

    def test_a_recipe_override_beats_the_schema_default(self, client):
        """kd_ce_alpha's schema default is 0.3; the lif recipe folds 0.5 —
        rendering the blue default would have been a lie."""
        values = self._resolve(client)["derived_values"]
        assert values["kd_ce_alpha"] == 0.5
        assert REGISTRY["kd_ce_alpha"].default == 0.3

    def test_an_erroring_draft_serves_no_hypothetical_values(self, client):
        body = _t0_01()
        body["deployment_parameters"]["weight_quantization"] = False
        payload = client.post("/api/config/resolve", json=body).json()
        assert not payload["ok"]
        assert payload["derived_values"] == {}


class TestRound6SemanticsLayout:
    """Round-6 item 1: the Deployment-target panel moves to the LEFT column,
    below Spiking semantics; the vehicles card holds the right column."""

    def _semantics(self, client):
        html = client.get("/wizard").text
        start = html.index('data-section-id="semantics"')
        return html[start:html.index("</section>", start)]

    def test_deployment_target_sits_below_spiking_in_the_left_column(self, client):
        section = self._semantics(client)
        spiking = section.index('data-groups="spiking,conversion"')
        target = section.index('data-groups="deployment_target"')
        vehicles = section.index('id="vehiclesCard"')
        assert spiking < target < vehicles

    def test_the_left_column_opens_before_the_right_one(self, client):
        section = self._semantics(client)
        first_col = section.index('class="wb-col"')
        target = section.index('data-groups="deployment_target"')
        second_col = section.index('class="wb-col"', target)
        assert first_col < target < second_col


class TestRound6RemediesAreServedNotHardcoded:
    """Every rule prescribes its own remedies; the frontend has no rule table."""

    def test_the_quantization_contract_serves_its_remedies(self, client):
        body = _t0_01()
        body["deployment_parameters"]["weight_quantization"] = False
        payload = client.post("/api/config/resolve", json=body).json()
        row = next(e for e in payload["errors"] if e["rule_id"] == "quantization_assembly")
        actions = {(r["action"], r["key"]) for r in row["remedies"]}
        assert ("set", "pipeline_mode") in actions
        assert ("clear", "weight_bits") in actions

    def test_the_frontend_carries_no_rule_specific_remedy_table(self):
        review = os.path.join(
            _REPO_ROOT, "src", "mimarsinan", "gui", "static", "js", "wizard", "review.js"
        )
        with open(review, encoding="utf-8") as f:
            source = f.read()
        assert "RULE_REMEDIES" not in source
        assert "quantization_assembly" not in source
        # Only the two generic actions the server prescribes.
        assert "REMEDY_ACTIONS" in source

    def test_the_frontend_never_renders_empty_means_for_a_sourced_key(self):
        """The in-field text of a derivation-owned key is its VALUE; prose about
        the source lives in the badge/tooltip only (round-6 amendment)."""
        fields = os.path.join(
            _REPO_ROOT, "src", "mimarsinan", "gui", "static", "js", "wizard", "fields.js"
        )
        with open(fields, encoding="utf-8") as f:
            source = f.read()
        body = source[source.index("export function placeholderText("):]
        body = body[: body.index("\n}")]
        # The derivation-owned branch returns before any empty_means fallback.
        assert body.index("isDerivationOwned") < body.index("empty_means")


def _static(*parts) -> str:
    path = os.path.join(_REPO_ROOT, "src", "mimarsinan", "gui", "static", *parts)
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestProvenanceOccupiesNoLayout:
    """Round-6 item 6: the SSOT source is TOOLTIP text, never a chip. The only
    surface that renders it inline is the Review 'Derived values' panel, whose
    entire PURPOSE is provenance (the source is that panel's payload, not chrome)."""

    def test_no_provenance_chip_element_is_created(self):
        fields = _static("js", "wizard", "fields.js")
        assert "field-source" not in fields
        assert "data-source-key" not in fields
        assert "sourceBadgeText" not in fields

    def test_no_provenance_chip_is_styled(self):
        assert ".field-source" not in _static("wizard.css")

    def test_the_source_rides_the_immediate_field_tooltip(self):
        fields = _static("js", "wizard", "fields.js")
        help_text = fields[fields.index("function helpText("):]
        help_text = help_text[: help_text.index("\n}")]
        # The source sits beside the doc/effect/bounds/empty text already served.
        assert "'SSOT source: ' + ks.provenance" in help_text
        # ...and helpText is what the field root hands to the immediate tooltip.
        shell = fields[fields.index("function fieldShell("):]
        shell = shell[: shell.index("\n}")]
        assert "field.title = helpText(ks)" in shell

    def test_every_hover_surface_of_a_field_shows_the_same_help(self):
        """Field root, input and toggle all hand tooltip.js the one helpText —
        the source is stated exactly once, never duplicated per surface."""
        fields = _static("js", "wizard", "fields.js")
        assert "setTip(input, helpText(ks))" in fields
        assert "setTip(row, helpText(ks))" in fields
        empty = fields[fields.index("export function emptyMeansText("):]
        empty = empty[: empty.index("\n}")]
        assert "ks.provenance" not in empty, "the source belongs to helpText, once"

    def test_review_keeps_its_inline_sources(self):
        """The panel whose purpose IS provenance still names each source."""
        assert "derived-chip-source" in _static("js", "wizard", "review.js")


class TestOneDerivedVisualLanguage:
    """Round-6 item 7: derived-ness is ONE predicate and ONE class; every widget
    type reads its green treatment from it, and no widget opts out."""

    WIDGET_RULES = (
        ".field-derived input::placeholder",   # text / number inputs
        ".field-derived .toggle-row",          # toggles
        ".field-derived .slider",              # slider thumb
        ".field-derived .slider-combo-box",    # slider numeric box
        ".seg-btn.is-derived",                 # segmented ghost
        ".field-derived .cores-editor",        # structured editors
        ".field-derived.field-locked .locked-value",  # locked fields
    )

    def test_one_predicate_drives_every_widget(self):
        fields = _static("js", "wizard", "fields.js")
        assert "export function isDerivedNow(ks)" in fields
        # The single stamp, applied to the field root (generic + structured).
        assert "classList.toggle('field-derived', isDerivedNow(ks))" in fields
        # No widget carries a private derived flag any more.
        assert "placeholderIsDerived" not in fields
        assert "prefillIsDerived" not in fields
        assert "'is-derived'" not in fields.replace("btn.classList.toggle('is-derived', ghost)", "")

    def test_every_widget_type_has_a_derived_rule(self):
        css = _static("wizard.css")
        for rule in self.WIDGET_RULES:
            assert rule in css, rule

    def test_the_derived_language_names_no_key(self):
        """Generic off the registry state, never per-key CSS."""
        css = _static("wizard.css")
        block = css[css.index("ONE DERIVED VISUAL LANGUAGE"):css.index("input:focus, select:focus")]
        for key in ("firing_mode", "spike_generation_mode", "nf_scm", "onchip", "capacity_gate"):
            assert key not in block, key

    def test_a_structured_editor_is_stamped_too(self):
        fields = _static("js", "wizard", "fields.js")
        custom = fields[fields.index("const custom = customRenderers.get(ks.key);"):]
        custom = custom[: custom.index("const locked =")]
        assert "markDerived(node, ks)" in custom

    def test_a_derivation_owned_key_never_prefills_its_schema_default(self):
        """kd_ce_alpha's schema default is 0.3 while the lif recipe folds 0.5 —
        a prefilled box would state the wrong number in the normal theme."""
        fields = _static("js", "wizard", "fields.js")
        body = fields[fields.index("function displayValue(ks) {"):]
        body = body[: body.index("\n}")]
        assert body.index("isDerivationOwned(ks)") < body.index("wizardDefault(ks)")
        assert REGISTRY["kd_ce_alpha"].provenance and REGISTRY["kd_ce_alpha"].has_default()

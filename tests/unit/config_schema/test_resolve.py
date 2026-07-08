"""resolve_draft: derived values with WHY, keyed errors, diff-vs-defaults."""

import json
import os

from mimarsinan.config_schema.resolve import resolve_draft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _t0_01() -> dict:
    path = os.path.join(
        _REPO_ROOT, "test_configs", "tier0", "t0_01_lif_mmixcore_wq_s4.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestResolveTierConfig:
    def test_resolves_clean(self):
        res = resolve_draft(_t0_01())
        assert res.ok
        assert res.unknown_keys == []
        assert res.resolved["activation_quantization"] is True
        assert res.resolved["pipeline_mode"] == "phased"

    def test_derived_chips_carry_why(self):
        res = resolve_draft(_t0_01())
        aq = res.derived["activation_quantization"]
        assert aq["value"] is True
        assert "lif" in aq["why"]
        assert aq["derived_from"]
        assert res.derived["pipeline_mode"]["why"].startswith("phased")

    def test_explicit_keys_and_diff(self):
        res = resolve_draft(_t0_01())
        assert "endpoint_floor_steps" in res.explicit_keys
        by_key = {row["key"]: row for row in res.diff_vs_defaults}
        assert by_key["lr"]["differs"] is True          # 0.003 vs default 0.001
        assert by_key["lr"]["default"] == 0.001
        assert by_key["sanafe_arch_preset"]["differs"] is False

    def test_explicit_null_means_unset_not_a_diff(self):
        """Tier configs declare start_step/stop_step/target_metric_override as
        null; null against no-default (or a null default) is 'unset', never a
        differing knob — the review diff must not list it."""
        res = resolve_draft(_t0_01())
        by_key = {row["key"]: row for row in res.diff_vs_defaults}
        assert by_key["start_step"]["differs"] is False
        assert by_key["stop_step"]["differs"] is False
        assert by_key["target_metric_override"]["differs"] is False
        # A REAL value against no default still differs.
        assert by_key["max_axons"]["differs"] is True


class TestResolveErrors:
    def test_contradicting_aq_is_a_keyed_error(self):
        draft = {
            "pipeline_mode": "phased",
            "deployment_parameters": {
                "spiking_mode": "lif",
                "activation_quantization": False,
                "weight_quantization": True,
            },
        }
        res = resolve_draft(draft)
        assert not res.ok
        assert any(
            e["key"] == "activation_quantization" and e["rule_id"] == "derivation"
            for e in res.errors
        )

    def test_bits_driven_wq_contract_is_a_keyed_error(self):
        draft = {
            "pipeline_mode": "phased",
            "deployment_parameters": {"weight_quantization": False},
            "platform_constraints": {"weight_bits": 5},
        }
        res = resolve_draft(draft)
        assert any(
            e["key"] == "weight_quantization" and e["rule_id"] == "quantization_assembly"
            for e in res.errors
        )

    def test_reserved_s_allocation_mode_attaches_to_its_key(self):
        draft = dict(_t0_01())
        draft["deployment_parameters"] = dict(draft["deployment_parameters"])
        draft["deployment_parameters"]["s_allocation"] = "explicit"
        res = resolve_draft(draft)
        assert any(e["key"] == "s_allocation" for e in res.errors)

    def test_unknown_keys_are_surfaced_not_dropped(self):
        draft = _t0_01()
        draft["deployment_parameters"]["endpoint_floor_wall_s"] = 60
        res = resolve_draft(draft)
        assert res.unknown_keys == ["deployment_parameters.endpoint_floor_wall_s"]

    def test_config_mistakes_never_raise(self):
        res = resolve_draft({"deployment_parameters": {"spiking_mode": "rate"}})
        assert not res.ok
        assert any(e["key"] == "spiking_mode" for e in res.errors)


def _minimal_draft(**parts) -> dict:
    draft = {
        "experiment_name": "resolve_test",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": "./generated",
        "start_step": None,
        "deployment_parameters": {},
        "platform_constraints": {},
    }
    draft.update(parts)
    dp = draft["deployment_parameters"]
    dp.setdefault("model_type", "simple_mlp")
    dp.setdefault("model_config", {})
    return draft


class TestPolicyOwnedSimulatorEnables:
    """The sim enables are ConversionPolicy-derived per mode; the resolve
    payload must carry them as derived chips with an honest WHY."""

    def _derived_for(self, mode, schedule=None):
        dp = {"spiking_mode": mode}
        if schedule:
            dp["ttfs_cycle_schedule"] = schedule
        res = resolve_draft(_minimal_draft(deployment_parameters=dp))
        assert res.ok, res.errors
        return res.derived

    def test_lif_runs_all_three_backends(self):
        derived = self._derived_for("lif")
        assert derived["enable_nevresim_simulation"]["value"] is True
        assert derived["enable_loihi_simulation"]["value"] is True
        assert derived["enable_sanafe_simulation"]["value"] is True

    def test_ttfs_disables_the_lif_only_loihi_backend(self):
        derived = self._derived_for("ttfs")
        assert derived["enable_loihi_simulation"]["value"] is False
        assert "LIF" in derived["enable_loihi_simulation"]["why"]
        assert derived["enable_nevresim_simulation"]["value"] is True

    def test_synchronized_disables_nevresim(self):
        derived = self._derived_for("ttfs_cycle_based", "synchronized")
        assert derived["enable_nevresim_simulation"]["value"] is False
        assert "synchronized" in derived["enable_nevresim_simulation"]["why"]
        assert derived["enable_sanafe_simulation"]["value"] is True


class TestVehicleToggleSemantics:
    """Round-3 defect 6: derived vehicle rows carry machine-readable support
    meta; a declared OFF on a supported vehicle resolves honestly (value off,
    zero errors); a declared ON on an unsupported one is a keyed error."""

    def test_derived_rows_carry_support_meta(self):
        res = resolve_draft(_minimal_draft(
            deployment_parameters={"spiking_mode": "ttfs"}))
        assert res.ok, res.errors
        assert res.derived["enable_nevresim_simulation"]["meta"]["supported"] is True
        assert res.derived["enable_loihi_simulation"]["meta"]["supported"] is False

    def test_user_off_on_a_supported_vehicle_resolves_clean(self):
        res = resolve_draft(_minimal_draft(deployment_parameters={
            "spiking_mode": "lif", "enable_sanafe_simulation": False}))
        assert res.ok, res.errors
        row = res.derived["enable_sanafe_simulation"]
        assert row["value"] is False
        assert row["meta"]["supported"] is True
        assert "disabled" in row["why"]

    def test_user_off_removes_the_vehicle_step_from_the_preview(self):
        from mimarsinan.gui.wizard.schema_api import resolve_payload

        on = resolve_payload(_minimal_draft(
            deployment_parameters={"spiking_mode": "lif"}))
        off = resolve_payload(_minimal_draft(deployment_parameters={
            "spiking_mode": "lif", "enable_sanafe_simulation": False}))
        assert on["ok"] and off["ok"]
        assert "SANA-FE Simulation" in on["pipeline"]["steps"]
        assert "SANA-FE Simulation" not in off["pipeline"]["steps"]

    def test_user_on_of_an_unsupported_vehicle_is_a_keyed_error(self):
        res = resolve_draft(_minimal_draft(deployment_parameters={
            "spiking_mode": "ttfs", "enable_loihi_simulation": True}))
        assert not res.ok
        assert any(e["key"] == "enable_loihi_simulation" for e in res.errors)


class TestDerivedCoreMaxima:
    """max_axons/max_neurons derive from the core grid at resolve."""

    _CORES = [
        {"max_axons": 784, "max_neurons": 512, "count": 60},
        {"max_axons": 512, "max_neurons": 256, "count": 60},
    ]

    def test_absent_scalars_are_derived_from_cores(self):
        res = resolve_draft(_minimal_draft(
            platform_constraints={"cores": self._CORES}))
        assert res.ok, res.errors
        assert res.resolved["max_axons"] == 784
        assert res.resolved["max_neurons"] == 512
        assert res.derived["max_axons"]["value"] == 784
        assert res.derived["max_axons"]["why"]

    def test_consistent_explicit_scalars_are_accepted(self):
        res = resolve_draft(_minimal_draft(platform_constraints={
            "cores": self._CORES, "max_axons": 784, "max_neurons": 512,
        }))
        assert res.ok, res.errors

    def test_contradicting_scalar_is_a_keyed_error(self):
        res = resolve_draft(_minimal_draft(platform_constraints={
            "cores": self._CORES, "max_axons": 1024,
        }))
        assert not res.ok
        assert any(e["key"] == "max_axons" for e in res.errors)

    def test_scalar_only_documents_keep_their_scalars(self):
        """The legacy / hardware-search shape declares scalar bounds without
        a core grid; the derivation must not touch them."""
        res = resolve_draft(_minimal_draft(platform_constraints={
            "max_axons": 1024, "max_neurons": 1024,
        }))
        assert res.ok, res.errors
        assert res.resolved["max_axons"] == 1024

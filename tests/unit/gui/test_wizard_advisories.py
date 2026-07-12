"""The wizard resolve payload carries config-time deployment advisories."""

from mimarsinan.gui.wizard.schema_api import resolve_payload


def _minimal_draft(**parts) -> dict:
    draft = {
        "experiment_name": "advisory_wizard_test",
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


class TestWizardAdvisories:
    def test_cascaded_pick_shows_the_unsupported_warning(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={"spiking_mode": "ttfs_cycle_based"}
        ))
        assert payload["ok"], payload["errors"]
        by_id = {row["id"]: row for row in payload["advisories"]}
        assert "ADV-CASC-UNSUPPORTED" in by_id
        row = by_id["ADV-CASC-UNSUPPORTED"]
        assert row["severity"] == "UNSUPPORTED"
        assert "not fully supported" in row["detail"]
        assert isinstance(row["suggested_levers"], list)

    def test_lif_draft_has_no_casc_advisory(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={"spiking_mode": "lif"}
        ))
        assert payload["ok"], payload["errors"]
        ids = {row["id"] for row in payload["advisories"]}
        assert "ADV-CASC-UNSUPPORTED" not in ids

    def test_erroring_draft_renders_no_hypothetical_advisories(self):
        payload = resolve_payload(_minimal_draft(
            deployment_parameters={
                "spiking_mode": "lif",
                "activation_quantization": False,
                "weight_quantization": True,
            }
        ))
        assert not payload["ok"]
        assert payload["advisories"] == []

"""Retired config keys: scoped keyed migration rows, one-click ops, honest preview."""

from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import (
    non_declarable_key_errors,
    validate_deployment_config,
)


def _document(pc=None, **dp) -> dict:
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "retired",
        "generated_files_path": "./generated",
        "start_step": None,
        "platform_constraints": dict(pc or {}),
        "deployment_parameters": {
            "model_type": "lenet5",
            "model_config": {"variant": "lenet5"},
            **dp,
        },
    }


def _apply_ops(doc: dict, ops) -> dict:
    """Apply remedy ops the way the wizard does: each op targets the
    sub-document its ``scope`` names."""
    out = {key: dict(value) if isinstance(value, dict) else value
           for key, value in doc.items()}
    for op in ops:
        body = out[op["scope"]]
        if op["action"] == "set":
            body[op["key"]] = op["value"]
        elif op["action"] == "clear":
            body.pop(op["key"], None)
    return out


class TestDocumentValidation:
    def test_every_retired_key_reports(self):
        errors = validate_deployment_config(_document(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
            lif_execution_discipline="streaming", lif_per_hop_retiming=True,
        ))
        for key in ("spiking_mode", "ttfs_cycle_schedule",
                    "lif_execution_discipline", "lif_per_hop_retiming"):
            assert any("retired" in e and key in e for e in errors), key

    def test_retired_keys_are_not_double_reported_as_non_declarable(self):
        errors = non_declarable_key_errors(_document(
            spiking_mode="lif", lif_per_hop_retiming=True,
        ))
        assert errors == []

    def test_axes_document_is_clean(self):
        assert validate_deployment_config(_document(
            spiking_family="ttfs", spiking_variant="synchronized",
        )) == []


class TestResolveChannel:
    def test_retired_row_carries_the_migration_ops(self):
        res = resolve_draft(_document(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
        ))
        row = next(e for e in res.errors if e["key"] == "spiking_mode")
        assert row["rule_id"] == "retired_key"
        assert row["scope"] == "deployment_parameters"
        (remedy,) = row["remedies"]
        dp = "deployment_parameters"
        assert remedy["ops"] == [
            {"action": "set", "key": "spiking_family", "value": "ttfs", "scope": dp},
            {"action": "set", "key": "spiking_variant", "value": "synchronized",
             "scope": dp},
            {"action": "clear", "key": "spiking_mode", "scope": dp},
            {"action": "clear", "key": "ttfs_cycle_schedule", "scope": dp},
        ]

    def test_unknown_legacy_value_gets_a_clear_only_remedy(self):
        res = resolve_draft(_document(spiking_mode="banana"))
        row = next(e for e in res.errors if e["key"] == "spiking_mode")
        assert row["rule_id"] == "retired_key"
        assert [r["action"] for r in row["remedies"]] == ["clear"]

    def test_preview_still_resolves_through_the_meaning_preserving_bridge(self):
        res = resolve_draft(_document(spiking_mode="lif"))
        assert any(e["rule_id"] == "retired_key" for e in res.errors)
        # old 'lif' resolves to the windowed axes — never silently streamed.
        assert res.resolved["spiking_family"] == "lif"
        assert res.resolved["spiking_variant"] == "synchronized"
        assert res.resolved["spiking_mode"] == "lif"

    def test_one_click_migration_yields_a_clean_equivalent_document(self):
        doc = _document(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
        )
        before = resolve_draft(doc)
        row = next(e for e in before.errors if e["key"] == "spiking_mode")
        (remedy,) = row["remedies"]

        after = resolve_draft(_apply_ops(doc, remedy["ops"]))
        assert after.errors == []
        # identical resolved semantics: the migration preserved meaning.
        for key in ("spiking_family", "spiking_variant",
                    "spiking_mode", "ttfs_cycle_schedule"):
            assert after.resolved[key] == before.resolved[key], key


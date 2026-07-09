"""The SSOT field validator: registry type/bounds/options enforced at BOTH doors.

The registry declares each key's type, numeric bounds, and enum options. Before
this validator only the WIZARD honored them (widgets clamp); the RUNTIME path
(``validate_merged_config``) checked spiking-mode + legality only, so a
hand-authored out-of-bounds / wrong-type / illegal-enum value ran clean. These
tests pin the ONE contract at both doors: the runtime flat-config validator and
the wizard resolve error channel must return the SAME verdict.
"""

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.config_schema.validation import (
    validate_against_registry,
    validate_deployment_config,
    validate_merged_config,
)
from mimarsinan.gui.wizard.starter import load_starter_baseline, starter_draft

ROOT = Path(__file__).resolve().parents[3]
TIER_CONFIG_PATHS = sorted(
    glob.glob(str(ROOT / "test_configs" / "tier*" / "t*.json"))
)


def _document(section: str, key: str, value: Any) -> dict:
    dp = {"model_type": "lenet5", "model_config": {"variant": "lenet5"}}
    pc: dict = {}
    (dp if section == "deployment_parameters" else pc)[key] = value
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "field-validation",
        "generated_files_path": "./generated",
        "start_step": None,
        "platform_constraints": pc,
        "deployment_parameters": dp,
    }


def _flat_with(section: str, key: str, value: Any) -> dict:
    dp = {key: value} if section == "deployment_parameters" else {}
    pc = {key: value} if section == "platform_constraints" else {}
    return build_flat_pipeline_config(dp, pc, pipeline_mode="phased")


@dataclass(frozen=True)
class Case:
    key: str
    value: Any
    section: str
    kind: str


# One representative per violation KIND per SECTION, so both doors are proven to
# cover deployment_parameters AND platform_constraints keys.
CASES = (
    Case("nf_scm_parity_samples", -1, "deployment_parameters", "bounds"),
    Case("onchip_min_fraction", 1.5, "deployment_parameters", "bounds"),
    Case("max_simulation_samples", "lots", "deployment_parameters", "type"),
    Case("nevresim_connectivity_mode", "telepathy", "deployment_parameters", "enum"),
    Case("weight_bits", 99, "platform_constraints", "bounds"),
    Case("target_tq", 0, "platform_constraints", "bounds"),
    Case("simulation_steps", 3.5, "platform_constraints", "type"),
    Case("sanafe_arch_preset", "quantum", "deployment_parameters", "enum"),
    Case("simulation_step_timeout_s", 0.5, "deployment_parameters", "bounds"),
)


class TestRegistryWalkingValidator:
    """The core validator: keyed, remediable rows shaped like legality errors."""

    def test_out_of_bounds_is_a_keyed_remediable_error(self):
        rows = validate_against_registry({"nf_scm_parity_samples": -1})
        assert len(rows) == 1
        row = rows[0]
        assert row["key"] == "nf_scm_parity_samples"
        assert row["rule_id"] == "field_domain"
        assert "nf_scm_parity_samples" in row["message"]
        assert row["remedies"] and any(
            r["action"] == "clear" and r["key"] == "nf_scm_parity_samples"
            for r in row["remedies"]
        )

    def test_wrong_type_is_rejected(self):
        rows = validate_against_registry({"max_simulation_samples": "lots"})
        assert [r["key"] for r in rows] == ["max_simulation_samples"]

    def test_illegal_enum_is_rejected(self):
        rows = validate_against_registry({"nevresim_connectivity_mode": "telepathy"})
        assert [r["key"] for r in rows] == ["nevresim_connectivity_mode"]

    def test_bool_is_not_an_int(self):
        """Python conflates bool and int; the registry declares them distinct."""
        rows = validate_against_registry({"weight_bits": True})
        assert [r["key"] for r in rows] == ["weight_bits"]

    def test_int_is_a_legal_float(self):
        assert validate_against_registry({"onchip_min_fraction": 0}) == []

    def test_absent_and_unknown_and_null_are_never_judged(self):
        assert validate_against_registry({}) == []
        assert validate_against_registry({"not_a_registry_key": -999}) == []
        assert validate_against_registry({"nf_scm_parity_samples": None}) == []

    def test_legality_bearing_keys_are_owned_by_the_legal_value_set_law(self):
        """firing_mode/spike_generation_mode/etc. have STATE-dependent admissible
        sets — the field validator must not double-report them."""
        assert validate_against_registry({"firing_mode": "Banana"}) == []
        assert validate_against_registry({"s_allocation": "explicit"}) == []

    def test_a_valid_value_passes(self):
        assert validate_against_registry({"nf_scm_parity_samples": 64}) == []
        assert validate_against_registry({"nevresim_connectivity_mode": "codegen"}) == []
        assert validate_against_registry({"weight_bits": 8}) == []
        assert validate_against_registry({"simulation_step_timeout_s": 300.0}) == []


class TestBothDoorsRejectInvalidValues:
    """The SAME verdict either way: an out-of-bounds / wrong-type / illegal-enum
    value is rejected at the runtime flat-config door AND the wizard resolve door."""

    @pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.key}-{c.kind}")
    def test_runtime_door_rejects(self, case):
        flat = _flat_with(case.section, case.key, case.value)
        errors = validate_merged_config(flat)
        assert any(case.key in message for message in errors), (case, errors)

    @pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.key}-{c.kind}")
    def test_document_door_rejects(self, case):
        errors = validate_deployment_config(_document(case.section, case.key, case.value))
        assert any(case.key in message for message in errors), (case, errors)

    @pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.key}-{c.kind}")
    def test_wizard_resolve_door_surfaces_a_keyed_error(self, case):
        resolution = resolve_draft(_document(case.section, case.key, case.value))
        rows = [
            e for e in resolution.errors
            if e["key"] == case.key and e["rule_id"] == "field_domain"
        ]
        assert rows, (case, resolution.errors)
        assert rows[0]["remedies"], "a field-domain error must prescribe a remedy"
        assert not resolution.ok
        # No structural DUPLICATE of the same key survives beside the keyed row.
        assert sum(1 for e in resolution.errors if e["key"] == case.key) == 1


SENTINEL_KEYS = (
    "nf_scm_parity_samples",
    "scm_torch_sim_parity_samples",
    "max_simulation_samples",
)


class TestSentinelHonesty:
    """0 is a documented disable sentinel (bounds start at 0): 0 legal, -1 not."""

    @pytest.mark.parametrize("key", SENTINEL_KEYS)
    def test_zero_is_legal_at_both_doors(self, key):
        flat = _flat_with("deployment_parameters", key, 0)
        assert not any(key in m for m in validate_merged_config(flat))
        resolution = resolve_draft(_document("deployment_parameters", key, 0))
        assert not [e for e in resolution.errors if e["key"] == key]

    @pytest.mark.parametrize("key", SENTINEL_KEYS)
    def test_negative_is_rejected_at_both_doors(self, key):
        flat = _flat_with("deployment_parameters", key, -1)
        assert any(key in m for m in validate_merged_config(flat))
        resolution = resolve_draft(_document("deployment_parameters", key, -1))
        assert [
            e for e in resolution.errors
            if e["key"] == key and e["rule_id"] == "field_domain"
        ]


class TestTheHeadlineCase:
    """The reported bug: nf_scm_parity_samples:-1 resolved clean and ran."""

    def test_negative_one_is_now_rejected_by_validate_merged_config(self):
        flat = build_flat_pipeline_config(
            {"nf_scm_parity_samples": -1}, {}, pipeline_mode="phased"
        )
        assert flat["nf_scm_parity_samples"] == -1
        errors = validate_merged_config(flat)
        assert any("nf_scm_parity_samples" in m for m in errors), errors

    def test_negative_one_is_rejected_at_the_wizard_door(self):
        resolution = resolve_draft(
            {"deployment_parameters": {"nf_scm_parity_samples": -1}}
        )
        assert [
            e for e in resolution.errors
            if e["key"] == "nf_scm_parity_samples" and e["rule_id"] == "field_domain"
        ]


class TestEveryTierConfigValidatesCleanThroughBothDoors:
    """All 61 tier configs + the starter carry only in-domain values."""

    @pytest.mark.parametrize(
        "path", TIER_CONFIG_PATHS, ids=lambda p: os.path.basename(p)
    )
    def test_runtime_door_is_clean(self, path):
        document = json.loads(Path(path).read_text())
        flat = build_flat_pipeline_config(
            document["deployment_parameters"],
            document["platform_constraints"],
            pipeline_mode=document.get("pipeline_mode", "phased"),
        )
        assert validate_against_registry(flat) == []
        assert validate_merged_config(flat) == []

    @pytest.mark.parametrize(
        "path", TIER_CONFIG_PATHS, ids=lambda p: os.path.basename(p)
    )
    def test_document_and_wizard_doors_are_clean(self, path):
        document = json.loads(Path(path).read_text())
        assert validate_deployment_config(document) == []
        resolution = resolve_draft(document)
        assert [e for e in resolution.errors if e["rule_id"] == "field_domain"] == []

    def test_the_tier_matrix_is_the_expected_61(self):
        assert len(TIER_CONFIG_PATHS) == 61

    def test_starter_is_clean_through_both_doors(self):
        document = load_starter_baseline()
        flat = build_flat_pipeline_config(
            document["deployment_parameters"],
            document["platform_constraints"],
            pipeline_mode=document.get("pipeline_mode", "phased"),
        )
        assert validate_merged_config(flat) == []
        resolution = resolve_draft(starter_draft())
        assert [e for e in resolution.errors if e["rule_id"] == "field_domain"] == []

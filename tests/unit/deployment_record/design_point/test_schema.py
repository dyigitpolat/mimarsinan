"""The instance document's closed schema: a judged row of a built program and a refused placeholder row validate,
unknown or missing fields refuse, and the ViT-Lite softmax block (operand cores, operand writes per inference, host
softmax) fills the same schema unchanged."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from mimarsinan.deployment_record.design_point.schema import (
    INSTANCE, SCHEMA_ID, SCHEMA_VERSION, instance_json_schema, validate_instance,
)
from mimarsinan.deployment_record.design_point.schema_types import field_paths

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "design_point" / "vit_lite_softmax_block.json"


def _vit():
    return json.loads(FIXTURE.read_text())


def test_the_softmax_block_fills_the_schema_unchanged():
    document = _vit()
    assert validate_instance(document) == []
    rows = {r["row"]: r for r in document["rows"]}
    assert {"operand", "static", "host"} <= {o["role"] for o in rows["sram_macro"]["placement"]}
    assert any(e.get("operand_written_cells_per_inference") for e in rows["sram_macro"]["timeline"])
    assert {c["role"] for c in rows["rram_static_sram_operand"]["chip"]["core_types"]} == {"static", "operand"}
    assert rows["rram_alone"]["k_star"]["no_K_admits"] and rows["rram_alone"]["admission"]["verdict"] == "inadmissible"
    assert rows["sram_macro"]["certificate"]["operand_cells"]["compared"] > 0


def test_unknown_and_missing_fields_refuse():
    document = _vit()
    document["rows"][0]["chip"]["colour"] = "red"
    assert any("unknown field 'colour'" in e for e in validate_instance(document))
    document = _vit()
    del document["rows"][0]["k_star"]
    assert any("a judged row carries 'k_star'" in e for e in validate_instance(document))


def test_a_refused_row_carries_only_its_declaration_and_names_its_refusals():
    document = _vit()
    row = document["rows"][1]
    placeholder = {k: row[k] for k in ("row", "role", "title", "beside_of", "chip")}
    placeholder["admission"] = {"verdict": "refused", "at_K": None, "reasons": [],
                                "refusals": [{"name": "per_core_type_write_technology", "reason": "not built"}]}
    document["rows"][1] = placeholder
    assert validate_instance(document) == []
    placeholder["admission"]["refusals"] = []
    assert any("names its refusals" in e for e in validate_instance(document))


def test_the_energy_split_names_every_part_and_census_executors_are_declared():
    document = _vit()
    del document["rows"][0]["pricing"]["variants"][0]["energy_split_mj"]["host"]
    assert any("energy split lacks ['host']" in e for e in validate_instance(document))
    document = _vit()
    document["rows"][0]["activity"][0]["nonzero"]["nevresim"] = 1
    assert any("undeclared executors ['nevresim']" in e for e in validate_instance(document))


def test_the_json_schema_is_closed_and_names_every_field():
    schema = instance_json_schema()
    assert schema["$id"] == f"{SCHEMA_ID}/v{SCHEMA_VERSION}" and schema["additionalProperties"] is False
    text = json.dumps(schema)
    for path in field_paths(INSTANCE):
        assert f'"{path.rsplit(".", 1)[-1].rstrip("[]{}")}"' in text, path


def test_a_wrong_type_is_named_with_its_path():
    document = copy.deepcopy(_vit())
    document["rows"][2]["lifetime"]["hours"] = [1.0, 2.0]
    assert validate_instance(document) == ["$.rows[2].lifetime.hours: expected 3 items, got 2"]


class TestWearPerCoreType:
    """[fix_b] A row states its wear per declared core type, and the reader derives and checks its verdict: the row's
    hours are the corner-wise minimum, its binding type named, every entry endurance / wear / rate."""

    def test_resistive_static_tiles_with_sram_operand_cores_are_admitted_by_their_own_wear(self):
        row = {r["row"]: r for r in _vit()["rows"]}["rram_static_sram_operand"]
        life = row["lifetime"]
        by_role = {e["role"]: e for e in life["per_core_type"]}
        assert by_role["static"]["write_technology"] == "neurram"
        assert by_role["operand"]["write_technology"] == "sram_cim_macro"
        assert by_role["operand"]["max_writes_per_cell_per_inference"] == 1.0
        assert by_role["static"]["max_writes_per_cell_per_inference"] == 4 / life["at_K"]
        assert life["verdict"] == "admitted" == row["admission"]["verdict"] and life["binding"] == "static (neurram)"

    def test_resistive_alone_is_inadmissible_by_nvm_wear_of_the_operand_cells(self):
        row = {r["row"]: r for r in _vit()["rows"]}["rram_alone"]
        (entry,) = row["lifetime"]["per_core_type"]
        assert entry["write_technology"] == "neurram" and entry["max_writes_per_cell_per_inference"] >= 1.0
        assert entry["verdict"] == "inadmissible" == row["admission"]["verdict"]
        assert row["writes"]["operand_written_cells_per_inference"] > 0
        nvm_high = entry["endurance"]["cycles"][2] / (1.0 * 30.0 * 3600.0)
        assert nvm_high < _vit()["declarations"]["min_lifetime_hours"]  # the operand writes alone, at any K

    def test_a_row_of_two_write_technologies_without_its_core_types_is_named(self):
        document = _vit()
        del document["rows"][1]["lifetime"]["per_core_type"]
        assert any("several write technologies states its lifetime per core type" in e
                   for e in validate_instance(document))

    def test_hours_off_endurance_wear_and_rate_are_named(self):
        document = _vit()
        document["rows"][1]["lifetime"]["per_core_type"][0]["hours"] = [140000.0, 4600000.0, 140000000.0]
        problems = validate_instance(document)
        assert any("are not endurance / wear / rate" in e for e in problems)
        assert any("not the corner-wise minimum" in e for e in problems)

    def test_the_binding_type_and_the_worst_verdict_are_checked(self):
        document = _vit()
        document["rows"][1]["lifetime"]["binding"] = "operand (sram_cim_macro)"
        assert any("is not the binding core type" in e for e in validate_instance(document))


class TestDynamicShares:
    """[U-SH, G3-D1] Every stacked cost-bar term of a variant is priced or refused by name: a null MAC or carry share
    of the dynamic energy (``dynamic_shares_mj``) names its refusal in ``dynamic_shares_refusals``, and a priced one
    names none."""

    REASON = "no compute-energy source is declared: the target declares neither e_synaptic_event_total nor e_mac"

    def test_a_null_share_without_a_name_is_refused(self):
        document = _vit()
        document["rows"][0]["pricing"]["variants"][0]["dynamic_shares_mj"]["carry"] = None
        assert ("$.rows[0].pricing.variants[0]: the dynamic share 'carry' is null and names no refusal"
                in validate_instance(document))

    def test_a_null_share_named_by_its_refusal_validates(self):
        document = _vit()
        variant = document["rows"][0]["pricing"]["variants"][0]
        variant["dynamic_shares_mj"] = {"mac": None, "carry": None}
        variant["dynamic_shares_refusals"] = {"mac": self.REASON, "carry": self.REASON}
        assert validate_instance(document) == []
        del variant["dynamic_shares_refusals"]["mac"]
        assert validate_instance(document) == [
            "$.rows[0].pricing.variants[0]: the dynamic share 'mac' is null and names no refusal"]

    def test_a_priced_share_names_no_refusal(self):
        document = _vit()
        document["rows"][0]["pricing"]["variants"][0]["dynamic_shares_refusals"] = {"mac": self.REASON}
        assert validate_instance(document) == [
            "$.rows[0].pricing.variants[0]: the dynamic share 'mac' is priced and names a refusal"]

    def test_a_refusal_of_no_share_is_named(self):
        document = _vit()
        document["rows"][0]["pricing"]["variants"][0]["dynamic_shares_refusals"] = {"heat": self.REASON}
        assert validate_instance(document) == [
            "$.rows[0].pricing.variants[0]: refusals name no dynamic share: ['heat']"]


def test_a_row_whose_planes_disagree_names_every_difference():
    document = _vit()
    row = document["rows"][0]
    row["planes"] = {"agree": False, "fields": [
        {"field": "neural_passes", "executed": 26, "candidate": 23, "agree": False},
        {"field": "cores", "executed": 881, "candidate": 881, "agree": True}]}
    assert any("disagree on ['neural_passes'], unnamed" in e for e in validate_instance(document))
    row["planes"]["named"] = [{"field": "neural_passes", "reason": "a named plane difference"}]
    assert validate_instance(document) == []


def test_a_backing_store_constant_may_state_how_it_composes_with_the_chip():
    """[consts] A store block's channel time composes with the chip's own port (``combine: slower``); the instance
    carries the block's constants as declared, so the field validates, and only the composition rules exist."""
    document = _vit()
    store = document["rows"][0]["chip"]["backing_store"]["constants"]
    store["t_program_per_byte"] = {"nominal": 0.0133, "unit": "ns", "evidence_kind": "published",
                                   "combine": "slower"}
    assert validate_instance(document) == []
    store["t_program_per_byte"]["combine"] = "faster"
    assert any("'faster' is not one of" in e for e in validate_instance(document))

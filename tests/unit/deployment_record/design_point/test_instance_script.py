"""The instance driver end to end on a small residual network: one judged bundle row and one refused placeholder row,
assembled into a document that validates, with the exploration plane's pricing and the plane comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "design_point_instance.py"


def _driver():
    spec = importlib.util.spec_from_file_location("design_point_instance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SPEC = {
    "network": {"name": "residual_block", "builder": "test factory",
                "roster_entry": ["factory:unit.chip_simulation.value_run.integer_programs:residual_block_net", {}, 10,
                                 [3, 16, 16]],
                "width": None, "input_shape": [3, 16, 16], "classes": 10, "weights": "random",
                "accuracy_absent": "random weights"},
    "budget_fraction": 0.25, "declared_K": 1, "k_grid": [1, 16], "inputs": {"count": 4, "seed": 1, "spread": 3},
    "point_budget_s": 600.0, "memory": None,
    "rows": [
        {"row": "rram", "role": "figure", "title": "Resistive cores", "beside_of": None, "target": "neurram",
         "core_fields": {"write_granularity": "cell"}, "core_fields_source": "test", "arithmetic_class": "test"},
        {"row": "placeholder", "role": "figure", "title": "Two technologies", "beside_of": None,
         "core_types": [{"role": "static", "target": "neurram"}, {"role": "operand", "target": "sram_cim_macro",
                                                                   "operand": {"payload_bits": 8, "write_source": "host"}}],
         "arithmetic_class": "test"},
        # [E-6, targeted pass] the same two technologies through a declaration that states each core type's
        # write_physics, on the row's own network (a small softmax block on the operand core type)
        {"row": "mixed", "role": "figure", "title": "Two technologies, declared", "beside_of": None,
         "core_types": [{"role": "static", "target": "neurram"}, {"role": "operand", "target": "sram_cim_macro",
                                                                   "operand": {"payload_bits": 8, "write_source": "host"}}],
         "declaration": "tests/fixtures/design_point/hybrid_neurram_sram_cim.json", "core_counts": [8, 4],
         "declare": {"allow_scheduling": True, "max_schedule_passes": 4096},
         "network": {"name": "softmax_block", "builder": "test factory", "program": "operand",
                     "roster_entry": ["factory:unit.chip_simulation.value_run.operand_programs:attention_block",
                                      {"depth": 1}, 5, [3, 8, 8]],
                     "width": None, "input_shape": [3, 8, 8], "classes": 5, "weights": "random",
                     "block": "softmax attention block", "accuracy_absent": "random weights"},
         "arithmetic_class": "test"},
    ],
}


@pytest.mark.timeout(240)
def test_the_driver_writes_a_valid_document_with_a_judged_and_a_refused_row(tmp_path):
    driver = _driver()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    parts = tmp_path / "parts"
    for row in ("rram", "placeholder", "mixed"):
        assert driver.main(["--spec", str(spec_path), "--parts", str(parts), "--row", row]) == 0
    out = tmp_path / "doc.json"
    assert driver.main(["--spec", str(spec_path), "--parts", str(parts), "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    rows = {r["row"]: r for r in document["rows"]}
    assert rows["placeholder"]["admission"]["verdict"] == "refused"
    # [integration of CL7 with N2] the registry declares the core-type operand field, so only the second write
    # technology still refuses the row by name (the instance prices one physics per chip)
    assert [r["name"] for r in rows["placeholder"]["admission"]["refusals"]] == ["per_core_type_write_technology"]
    judged = rows["rram"]
    assert judged["certificate"]["exact"] and judged["certificate"]["nonvacuous"]
    assert judged["budget"]["cores"] == max(1, int(0.25 * judged["budget"]["packed_tiles_F"]))
    assert {v["K"] for v in judged["pricing"]["variants"]} == {1, 16}
    assert {f["field"] for f in judged["planes"]["fields"]} >= {"cores", "neural_passes", "written_passes"}
    assert judged["planes"]["agree"], judged["planes"]  # [fix_b] the priced program is the executed one
    assert judged["chip"]["core_types"][0]["write_technology"] == "neurram"


@pytest.mark.timeout(240)
def test_a_row_declared_with_each_core_types_write_physics_is_judged_and_priced_on_the_record_plane(tmp_path):
    """[E-6, targeted pass, review M2] the declaration route: the row is built, certified, judged per core type and
    priced on the record plane at every batch of the grid; the document validates."""
    driver = _driver()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({**SPEC, "rows": [r for r in SPEC["rows"] if r["row"] == "mixed"]}))
    parts = tmp_path / "parts"
    assert driver.main(["--spec", str(spec_path), "--parts", str(parts), "--row", "mixed"]) == 0
    out = tmp_path / "doc.json"
    assert driver.main(["--spec", str(spec_path), "--parts", str(parts), "--out", str(out)]) == 0
    row = json.loads(out.read_text())["rows"][0]
    assert row["admission"]["verdict"] == "admitted", row["admission"]
    assert row["certificate"]["exact"]
    assert {e["write_technology"] for e in row["lifetime"]["per_core_type"]} == {"neurram", "sram_cim_macro"}
    assert [(c["role"], c["write_technology"], c["count"]) for c in row["chip"]["core_types"]] == [
        ("static", "neurram", 8), ("operand", "sram_cim_macro", 4)]
    assert row["pricing"]["plane"].startswith("record")
    assert {v["K"] for v in row["pricing"]["variants"]} == {1, 16}
    assert all(v["energy_split_mj"]["dynamic"] is not None for v in row["pricing"]["variants"])
    assert row["network"]["name"] == "softmax_block" and row["budget"]["packed_tiles_F"] is None

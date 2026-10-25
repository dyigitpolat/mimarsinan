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
    ],
}


@pytest.mark.timeout(240)
def test_the_driver_writes_a_valid_document_with_a_judged_and_a_refused_row(tmp_path):
    driver = _driver()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    parts = tmp_path / "parts"
    for row in ("rram", "placeholder"):
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

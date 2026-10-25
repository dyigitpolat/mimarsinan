"""One judged row from a built program: every judged field, the admission verdict at the declared batch, and the
document validates; a program that writes nothing per inference is admitted."""

from __future__ import annotations

from mimarsinan.deployment_record.design_point.executors import EXECUTORS
from mimarsinan.deployment_record.design_point.rows import judged_row, plane_agreement
from mimarsinan.deployment_record.design_point.schema import JUDGED_FIELDS, SCHEMA_ID, SCHEMA_VERSION, validate_instance

from . import programs as P


def _row(count, k=1):
    program = P.residual_program()
    return judged_row(nf=program.model, ir_graph=program.ir, hybrid=P.built(count, k=k), x_int=P.inputs(),
                      platform=P.platform(count, k=k), physics=P.neurram(), k_grid=(1, 16, 64),
                      weight_bits=P.platform(count, k=k)["weight_bits"], names=P.perceptron_names(program),
                      x_fluid=2.0, x_packed=3.0)


def _document(rows):
    return {"schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "status": "provisional",
            "produced_by": {"script": "test", "command": "test", "worktree": "test", "base": "test",
                            "diff_sha256_16": None, "date": "test"},
            "network": {"name": "residual_block", "builder": "test", "width": None, "input_shape": [3, 16, 16],
                        "classes": 10, "weights": "random", "block": None,
                        "accuracy": {"value": None, "absent": "random weights"}},
            "declarations": {"inference_rate_hz": P.RATE_HZ, "min_lifetime_hours": P.FLOOR_H, "declared_K": 1,
                             "k_grid": [1, 16, 64], "budget_rule": "test", "atol": 0.0,
                             "inputs": {"kind": "integer", "count": 16, "seed": 1},
                             "point_cap": {"wall_s": None, "memory": None}},
            "executors": [dict(e) for e in EXECUTORS], "rows": rows}


def test_a_rewriting_program_on_resistive_endurance_is_judged_with_its_boundary():
    judged = _row(3)
    assert set(JUDGED_FIELDS) - {"budget"} <= set(judged)
    assert judged["certificate"]["exact"] and judged["certificate"]["nonvacuous"]
    assert judged["admission"]["verdict"] in ("inadmissible", "straddles")
    assert judged["k_star"]["checked"] and "min_batch" in judged["k_star"]
    row = {"row": "test", "role": "figure", "title": "test", "beside_of": None, "label": "test",
           "chip": {"target": "neurram", "display_name": "test", "core_types": [], "weight_bits": 8,
                    "faithfulness": {"class": "approximate", "named": []}, "claim_class": None,
                    "arithmetic_class": "test", "backing_store": None},
           "budget": {"packed_tiles_F": 10, "fraction": 0.3, "cores": 3, "rule": "test"}, **judged}
    assert validate_instance(_document([row])) == []


def test_plane_agreement_names_each_field():
    reading = plane_agreement({"passes": 3, "cores": 4}, {"passes": 3, "cores": 5})
    assert not reading["agree"] and [f["agree"] for f in reading["fields"]] == [True, False]

"""A ladder row read as a priced variant: the headlines at the row's batch (writes included) with the steady ones
beside, the five-part energy split that closes on the headline, the refusals by name, and a stopped row's refusal."""

from __future__ import annotations

import copy

import pytest

from mimarsinan.deployment_record.design_point.pricing import ladder_pricing, priced_variant
from mimarsinan.deployment_record.design_point.schema import ENERGY_PARTS

# Test constants. At K the write law's headline is the sum of its parts (nominal: dynamic 0.2, static 0.01, host 1.7,
# writes 0.02); the steady headline without writes is a separate band.
ROW = {"variant": "host+hbm2", "inference_batch_K": 1,
       "headlines": {"energy_per_inference_mj": [1.0, 2.0, 3.0], "e2e_latency_s": None},
       "headline_refusals": {"e2e_latency_s": "t_cycle is undeclared", "energy_per_inference_mj": None},
       "energy": {"host_share": 0.9, "terms": {"compute_only": [0.1, 0.2, 0.3], "boundary": [0.0, 0.0, 0.0],
                                               "host": [0.8, 1.7, 2.6]}},
       "write_law": {"terms": {"energy_write_mj": [0.01, 0.02, 0.03], "energy_dynamic_mj": [0.1, 0.2, 0.3],
                               "energy_static_mj": [0.005, 0.01, 0.02], "energy_host_mj": [0.8, 1.7, 2.6],
                               "energy_per_inference_mj": [0.915, 1.93, 2.95], "e2e_latency_s": None},
                     "refusals": {},
                     "verdicts": [{"name": "lifetime_admissible", "verdict": "admitted"}]},
       "latency_semantics": {"time_per_inference_s": None}}


def test_the_headline_is_the_write_law_at_the_batch_and_the_steady_one_rides_beside():
    priced = priced_variant(ROW, operand_cells=0)
    assert priced["energy_mj"] == [0.915, 1.93, 2.95]
    assert priced["energy_without_writes_mj"] == [1.0, 2.0, 3.0]
    assert priced["host_share"] == pytest.approx(1.7 / 1.93)
    assert priced["refusals"] == {"energy_per_inference_mj": None, "e2e_latency_s": "t_cycle is undeclared"}


def test_the_split_names_five_parts_keeps_static_writes_apart_and_closes_on_the_headline():
    priced = priced_variant(ROW, operand_cells=0)
    assert set(priced["energy_split_mj"]) == set(ENERGY_PARTS)
    assert priced["energy_split_mj"]["static_write"] == [0.01, 0.02, 0.03]
    assert priced["energy_split_mj"]["operand_write"] == [0.0, 0.0, 0.0]
    assert priced["split_closes"] is True and priced["lifetime_verdict"] == "admitted"
    assert priced["dynamic_shares_mj"] == {"mac": [0.1, 0.2, 0.3], "carry": [0.0, 0.0, 0.0]}


def test_an_undeclared_part_reads_zero_and_a_split_that_misses_a_part_does_not_close():
    row = copy.deepcopy(ROW)
    row["write_law"]["terms"]["energy_static_mj"] = None
    row["write_law"]["terms"]["energy_per_inference_mj"] = [0.91, 1.92, 2.93]
    assert priced_variant(row, operand_cells=0)["energy_split_mj"]["static"] == [0.0, 0.0, 0.0]
    assert priced_variant(row, operand_cells=0)["split_closes"] is True
    row["write_law"]["terms"]["energy_per_inference_mj"] = [0.915, 1.93, 2.95]
    assert priced_variant(row, operand_cells=0)["split_closes"] is False


def test_a_refused_write_energy_refuses_the_headline_by_name():
    row = copy.deepcopy(ROW)
    reason = "the write energy is refused: the target declares no commit energy"
    row["write_law"]["terms"].update(energy_write_mj=None, energy_per_inference_mj=None)
    row["write_law"]["refusals"] = {"energy_per_inference_mj": reason, "energy_write_mj": "no commit energy"}
    priced = priced_variant(row, operand_cells=0)
    assert priced["energy_mj"] is None and priced["refusals"]["energy_per_inference_mj"] == reason
    assert priced["energy_split_mj"]["static_write"] is None and priced["split_closes"] is None
    assert priced["energy_without_writes_mj"] == [1.0, 2.0, 3.0]


def test_a_row_without_a_declared_batch_reads_the_steady_headlines():
    row = {k: v for k, v in ROW.items() if k != "write_law"}
    assert priced_variant(row, operand_cells=0)["energy_mj"] == [1.0, 2.0, 3.0]


def test_operand_cells_leave_the_write_parts_unsplit():
    priced = priced_variant(ROW, operand_cells=10)
    assert priced["energy_split_mj"]["static_write"] is None and priced["energy_split_mj"]["operand_write"] is None
    assert priced["split_closes"] is None


def test_a_stopped_row_names_its_refusal():
    stopped = {"variant": "host", "inference_batch_K": 1, "stage": "schedule",
               "above_budget": {"name": "above_budget", "reason": "schedule ran past 600 s"}}
    pricing = ladder_pricing([ROW, stopped], operand_cells=0)
    assert len(pricing["variants"]) == 1
    assert pricing["refusal"] == {"name": "above_budget", "reason": "schedule ran past 600 s", "stage": "schedule"}

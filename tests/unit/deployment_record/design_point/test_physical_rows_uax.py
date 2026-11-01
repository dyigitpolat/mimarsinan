"""[U-AX, SDC-D1] The plane-parity gate compares only planes that read the same rows.

A ladder row on a per-axon pool states the rows its walk planned (``physical_rows``): the deployed rows where the walk
read the network's weights on the build's grid, else two per input, an upper bound named with its cause. The
candidate's facts carry that statement (``candidate_facts``), and ``plane_agreement`` names every field on which an
upper-bound candidate disagrees with the executed program by that cause (the planes read different rows), while a
candidate that read the deployed rows is compared as before, never excused. A row on a per-synapse pool states nothing
and reads as before.
"""

from __future__ import annotations

from mimarsinan.deployment_record.design_point.pricing import candidate_facts
from mimarsinan.deployment_record.design_point.rows import plane_agreement

EXECUTED = {"cores": 4, "neural_passes": 2, "written_passes": 2, "most_written_core_writes_per_period": 1,
            "payload_bytes_per_period": 100, "energy_write_mj": [1.0, 2.0, 3.0]}
DEPLOYED = {"basis": "deployed", "reason": None}
BOUND = {"basis": "upper_bound", "reason": "the walk was given no weight grid (weight_bits)"}


def _row(rows=None, payload=100):
    row = {"cores": 4, "census_passes": 2, "reprogram_passes": 2,
           "k_star": {"max_writes_per_cell_per_inference": 1.0}, "reprogrammed_bytes_per_period": payload,
           "write_law": {"terms": {"energy_write_mj": [1.0, 2.0, 3.0]}}}
    if rows is not None:
        row["physical_rows"] = rows
    return row


def test_the_candidate_facts_carry_the_rows_a_row_states():
    assert candidate_facts(_row(DEPLOYED))["physical_rows"] == DEPLOYED
    assert "physical_rows" not in candidate_facts(_row())


def test_an_upper_bound_candidates_disagreements_are_named_by_their_cause():
    planes = plane_agreement(EXECUTED, candidate_facts(_row(BOUND, payload=120)))
    assert not planes["agree"]
    assert [n["field"] for n in planes["named"]] == ["payload_bytes_per_period"]
    assert "upper bound" in planes["named"][0]["reason"] and BOUND["reason"] in planes["named"][0]["reason"]
    assert plane_agreement(EXECUTED, candidate_facts(_row(BOUND)))["agree"]


def test_a_candidate_that_read_the_deployed_rows_is_compared_and_never_excused():
    planes = plane_agreement(EXECUTED, candidate_facts(_row(DEPLOYED, payload=120)))
    assert not planes["agree"] and "named" not in planes
    assert [f["field"] for f in planes["fields"] if not f["agree"]] == ["payload_bytes_per_period"]


def test_a_row_that_states_no_rows_reads_as_before():
    assert plane_agreement(EXECUTED, candidate_facts(_row())) == {
        "agree": True, "fields": [{"field": k, "executed": v, "candidate": v, "agree": True}
                                  for k, v in EXECUTED.items()]}


def test_a_per_synapse_ladder_row_states_no_rows():
    from unit.deployment_record.introspection.test_scaling_ladder import _row as ladder_row

    assert "physical_rows" not in ladder_row()

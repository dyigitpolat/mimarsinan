"""Unit tests for parity contract classification."""

from __future__ import annotations

from mimarsinan.chip_simulation.parity_contract import (
    ParityContractKind,
    classify_backend_contract,
    classify_nf_scm_contract,
    classify_ttfs_quantized_sync_equivalence,
    parity_contract_metadata,
)


def test_ttfs_quantized_and_sync_are_functional_equivalence():
    assert (
        classify_ttfs_quantized_sync_equivalence(
            spiking_mode="ttfs_quantized",
            schedule="analytical",
        )
        == ParityContractKind.FUNCTIONAL_EQUIVALENCE
    )
    assert (
        classify_ttfs_quantized_sync_equivalence(
            spiking_mode="ttfs_cycle_based",
            schedule="synchronized",
        )
        == ParityContractKind.FUNCTIONAL_EQUIVALENCE
    )


def test_nf_scm_contracts():
    assert classify_nf_scm_contract(spiking_mode="ttfs_quantized") == ParityContractKind.FUNCTIONAL_EQUIVALENCE
    assert classify_nf_scm_contract(spiking_mode="ttfs", training_forward_kind="analytical_staircase") == ParityContractKind.BIT_PARITY
    assert classify_nf_scm_contract(spiking_mode="ttfs_cycle_based", schedule="cascaded") == ParityContractKind.FUNCTIONAL_EQUIVALENCE


def test_parity_contract_metadata_on_row():
    meta = parity_contract_metadata(
        {
            "spiking_mode": "ttfs_cycle_based",
            "schedule": "synchronized",
            "backend": "sanafe",
        }
    )
    assert meta["ttfs_quantized_sync_equivalence"] == "FUNCTIONAL_EQUIVALENCE"
    assert meta["nf_scm_contract"] == "BIT_PARITY"
    assert meta["backend_contract"] == "BIT_PARITY"


# ── full (spiking_mode × schedule) behavior tables ───────────────────────────
# Captured cell-by-cell from the pre-refactor literal ladders; the SSOT reroute
# must not move a single cell.

INAPPLICABLE = ParityContractKind.INAPPLICABLE
BIT_PARITY = ParityContractKind.BIT_PARITY
FUNCTIONAL_EQUIVALENCE = ParityContractKind.FUNCTIONAL_EQUIVALENCE

SCHEDULES = (None, "cascaded", "synchronized", "analytical")


def _uniform(kind):
    return {schedule: kind for schedule in SCHEDULES}


SYNC_EQUIVALENCE_TABLE = {
    "lif": _uniform(INAPPLICABLE),
    "ttfs": _uniform(INAPPLICABLE),
    "ttfs_quantized": _uniform(FUNCTIONAL_EQUIVALENCE),
    "ttfs_cycle_based": {
        None: INAPPLICABLE,
        "cascaded": INAPPLICABLE,
        "synchronized": FUNCTIONAL_EQUIVALENCE,
        "analytical": INAPPLICABLE,
    },
    "rate": _uniform(INAPPLICABLE),
    "": _uniform(INAPPLICABLE),
}

NF_SCM_TABLE = {
    "lif": _uniform(BIT_PARITY),
    "ttfs": _uniform(BIT_PARITY),
    "ttfs_quantized": _uniform(FUNCTIONAL_EQUIVALENCE),
    "ttfs_cycle_based": {
        None: BIT_PARITY,  # unrecorded schedule is NOT defaulted to cascaded here
        "cascaded": FUNCTIONAL_EQUIVALENCE,
        "synchronized": BIT_PARITY,
        "analytical": BIT_PARITY,
    },
    "rate": _uniform(INAPPLICABLE),
    "": _uniform(INAPPLICABLE),
}


def test_sync_equivalence_full_mode_schedule_table():
    for mode, row in SYNC_EQUIVALENCE_TABLE.items():
        for schedule, expected in row.items():
            got = classify_ttfs_quantized_sync_equivalence(
                spiking_mode=mode, schedule=schedule
            )
            assert got == expected, (mode, schedule)


def test_nf_scm_full_mode_schedule_table():
    for mode, row in NF_SCM_TABLE.items():
        for schedule, expected in row.items():
            got = classify_nf_scm_contract(spiking_mode=mode, schedule=schedule)
            assert got == expected, (mode, schedule)


def test_nf_scm_analytical_staircase_forward_upgrades_only_inapplicable_and_bit_parity():
    for mode, row in NF_SCM_TABLE.items():
        for schedule, base in row.items():
            got = classify_nf_scm_contract(
                spiking_mode=mode,
                schedule=schedule,
                training_forward_kind="analytical_staircase",
            )
            expected = base if base == FUNCTIONAL_EQUIVALENCE else BIT_PARITY
            assert got == expected, (mode, schedule)


def test_nf_scm_non_staircase_forward_keeps_base_table():
    for mode, row in NF_SCM_TABLE.items():
        for schedule, base in row.items():
            got = classify_nf_scm_contract(
                spiking_mode=mode,
                schedule=schedule,
                training_forward_kind="segment_spike",
            )
            assert got == base, (mode, schedule)


def test_parity_contract_metadata_schedule_placeholders_match_missing_schedule():
    base = parity_contract_metadata(
        {"spiking_mode": "ttfs_cycle_based", "backend": "sanafe"}
    )
    for sync in ("none", "analytical", "bogus"):
        row = {"spiking_mode": "ttfs_cycle_based", "sync": sync, "backend": "sanafe"}
        assert parity_contract_metadata(row) == base, sync


def test_parity_contract_metadata_explicit_cascaded_row():
    meta = parity_contract_metadata(
        {"spiking_mode": "ttfs_cycle_based", "schedule": "cascaded"}
    )
    assert meta["nf_scm_contract"] == "FUNCTIONAL_EQUIVALENCE"
    assert meta["ttfs_quantized_sync_equivalence"] == "INAPPLICABLE"

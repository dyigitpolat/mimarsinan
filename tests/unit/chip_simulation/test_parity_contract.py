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

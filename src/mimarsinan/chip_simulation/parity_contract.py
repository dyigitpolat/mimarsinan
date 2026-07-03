"""Parity and equivalence contract classification for deployment cells."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

__all__ = [
    "ParityContractKind",
    "classify_nf_scm_contract",
    "classify_backend_contract",
    "classify_ttfs_quantized_sync_equivalence",
    "parity_contract_metadata",
]


class ParityContractKind(str, Enum):
    """How strictly two deployment representations must agree."""

    BIT_PARITY = "BIT_PARITY"
    FUNCTIONAL_EQUIVALENCE = "FUNCTIONAL_EQUIVALENCE"
    INAPPLICABLE = "INAPPLICABLE"


def classify_ttfs_quantized_sync_equivalence(
    *,
    spiking_mode: str,
    schedule: str | None,
) -> ParityContractKind:
    """TTFS quantized and synchronized cycle TTFS share the q(x) wire rule."""
    if spiking_mode == "ttfs_quantized":
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    if spiking_mode == "ttfs_cycle_based" and schedule == "synchronized":
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    return ParityContractKind.INAPPLICABLE


def classify_nf_scm_contract(
    *,
    spiking_mode: str,
    schedule: str | None = None,
    training_forward_kind: str | None = None,
) -> ParityContractKind:
    """NF↔SCM gate contract keyed by deployment mode."""
    if spiking_mode == "ttfs_quantized":
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    if spiking_mode == "ttfs_cycle_based" and schedule == "cascaded":
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    if training_forward_kind == "analytical_staircase":
        return ParityContractKind.BIT_PARITY
    if spiking_mode in {"ttfs", "ttfs_cycle_based"}:
        return ParityContractKind.BIT_PARITY
    if spiking_mode == "lif":
        return ParityContractKind.BIT_PARITY
    return ParityContractKind.INAPPLICABLE


def classify_backend_contract(
    *,
    backend: str,
    spiking_mode: str,
    schedule: str | None = None,
) -> ParityContractKind:
    """Backend parity against identity SCM / HCM reference."""
    policy = policy_for_spiking_mode(spiking_mode, schedule=schedule)
    if not policy.supports_backend(backend):
        return ParityContractKind.INAPPLICABLE
    if backend in {"hcm", "scm", "nevresim", "sanafe", "lava"}:
        return ParityContractKind.BIT_PARITY
    return ParityContractKind.INAPPLICABLE


def parity_contract_metadata(row: Mapping[str, Any]) -> dict[str, str]:
    """Attach parity contract kinds to a ledger/campaign row."""
    spiking_mode = str(row.get("spiking_mode") or row.get("firing") or "")
    schedule = row.get("schedule") or row.get("sync")
    if schedule in {"analytical", "none"}:
        schedule = None
    backend = str(row.get("backend") or "sanafe")
    training_forward_kind = row.get("training_forward_kind")
    return {
        "ttfs_quantized_sync_equivalence": classify_ttfs_quantized_sync_equivalence(
            spiking_mode=spiking_mode,
            schedule=str(schedule) if schedule is not None else None,
        ).value,
        "nf_scm_contract": classify_nf_scm_contract(
            spiking_mode=spiking_mode,
            schedule=str(schedule) if schedule is not None else None,
            training_forward_kind=str(training_forward_kind)
            if training_forward_kind is not None
            else None,
        ).value,
        "backend_contract": classify_backend_contract(
            backend=backend,
            spiking_mode=spiking_mode,
            schedule=str(schedule) if schedule is not None else None,
        ).value,
    }

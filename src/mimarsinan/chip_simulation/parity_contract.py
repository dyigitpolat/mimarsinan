"""Parity and equivalence contract classification for deployment cells."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from mimarsinan.chip_simulation.point_capability import backend_refusal_reason
from mimarsinan.chip_simulation.spiking_point import SpikingPoint, recorded_point
from mimarsinan.chip_simulation.spiking_semantics import (
    LIF_MODES,
    forces_activation_quantization,
    is_analytical_ttfs,
    is_cascaded_ttfs,
    is_explicit_ttfs_cycle_schedule,
    requires_ttfs_firing,
    uses_ttfs_floor_ceil_convention,
)

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


def _is_quantized_analytical_ttfs(spiking_mode: str) -> bool:
    return is_analytical_ttfs(spiking_mode) and forces_activation_quantization(
        spiking_mode
    )


def classify_ttfs_quantized_sync_equivalence(
    *,
    spiking_mode: str,
    schedule: str | None,
) -> ParityContractKind:
    """TTFS quantized and synchronized cycle TTFS share the q(x) wire rule."""
    if uses_ttfs_floor_ceil_convention(spiking_mode, schedule):
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    return ParityContractKind.INAPPLICABLE


def classify_nf_scm_contract(
    *,
    spiking_mode: str,
    schedule: str | None = None,
    training_forward_kind: str | None = None,
) -> ParityContractKind:
    """NF↔SCM gate contract keyed by deployment mode."""
    if _is_quantized_analytical_ttfs(spiking_mode):
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    # An unrecorded schedule is NOT defaulted to cascaded here: it stays bit-parity.
    if is_explicit_ttfs_cycle_schedule(schedule) and is_cascaded_ttfs(
        spiking_mode, schedule
    ):
        return ParityContractKind.FUNCTIONAL_EQUIVALENCE
    if training_forward_kind == "analytical_staircase":
        return ParityContractKind.BIT_PARITY
    if requires_ttfs_firing(spiking_mode):
        return ParityContractKind.BIT_PARITY
    if spiking_mode in LIF_MODES:
        return ParityContractKind.BIT_PARITY
    return ParityContractKind.INAPPLICABLE


def classify_backend_contract(*, backend: str, point: SpikingPoint) -> ParityContractKind:
    """Backend parity against identity SCM / HCM reference, at the run's resolved point (a backend that does not
    execute it is INAPPLICABLE)."""
    if backend_refusal_reason(backend, point) is not None:
        return ParityContractKind.INAPPLICABLE
    if backend in {"hcm", "scm", "nevresim", "sanafe", "lava"}:
        return ParityContractKind.BIT_PARITY
    return ParityContractKind.INAPPLICABLE


def parity_contract_metadata(row: Mapping[str, Any]) -> dict[str, str]:
    """Attach parity contract kinds to a ledger/campaign row. [U-LIF2 r2] The backend contract reads the row's point as
    its run resolves the row's axes (the one resolution: the mode, the schedule, the soma keys, the window firing and the
    execution extensions the row declares; ``firing`` and ``sync`` read as the mode and an explicit schedule). [U-LIF2
    r6, m-1] A row is read as written: its window firing is the row's own, else the historical one
    (``recorded_point``)."""
    spiking_mode = str(row.get("spiking_mode") or row.get("firing") or "")
    schedule = row.get("schedule") or row.get("sync")
    if not is_explicit_ttfs_cycle_schedule(schedule):
        schedule = None
    backend = str(row.get("backend") or "sanafe")
    training_forward_kind = row.get("training_forward_kind")
    axes = {key: value for key, value in row.items() if key != "spiking_mode"}
    axes["ttfs_cycle_schedule"] = schedule
    if spiking_mode:
        axes["spiking_mode"] = spiking_mode
    point = recorded_point(axes)  # [U-LIF2 r6, m-1] at the row's own window firing
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
        "backend_contract": classify_backend_contract(backend=backend, point=point).value,
    }

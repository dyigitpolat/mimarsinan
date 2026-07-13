"""[C2] membrane-decode helpers shared by the flat and hybrid nevresim probes."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.models.spiking.hybrid.membrane_readout import membrane_readout_slices


def stash_membrane_corrections(
    hybrid: HybridHardCoreMapping,
    stage: Any,
    membranes: np.ndarray,
    corrections: Dict[int, np.ndarray],
    *,
    half_step_charge: float,
) -> None:
    """Stash ``m_T/theta - half_step`` per eligible node slice (count units)."""
    for node_id, start, end in membrane_readout_slices(hybrid, stage):
        corrections[node_id] = (
            np.asarray(membranes[:, start:end], dtype=np.float64)
            - float(half_step_charge)
        )


def flat_membrane_slices(mapping: Any, *, armed: bool) -> list[tuple[int, int, int]]:
    """Eligible ``(node_id, start, end)`` output slices of a single-neural-stage
    hybrid mapping; empty when the gate is off or the runner was handed a bare
    ``HardCoreMapping`` (no node structure to gate eligibility on)."""
    if not armed:
        return []
    if not isinstance(mapping, HybridHardCoreMapping):
        return []
    neural_stages = [s for s in mapping.stages if s.kind == "neural"]
    if len(neural_stages) != 1:
        return []
    return membrane_readout_slices(mapping, neural_stages[0])

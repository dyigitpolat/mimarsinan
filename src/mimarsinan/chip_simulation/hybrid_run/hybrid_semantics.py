"""Inter-stage signal contract for hybrid neural segments (all backends)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import store_segment_output_numpy
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing


@dataclass
class NeuralSegmentResult:
    """Canonical neural-segment outputs for hybrid state-buffer propagation."""

    inter_stage: np.ndarray
    per_core_activations: Optional[List[np.ndarray]] = None
    per_core_spike_counts: Optional[List[np.ndarray]] = None


def is_ttfs_spiking_mode(spiking_mode: str) -> bool:
    return requires_ttfs_firing(spiking_mode)


def store_neural_segment_output(
    spiking_mode: str,
    output_map,
    state_buffer: Dict[int, np.ndarray],
    result: NeuralSegmentResult,
) -> None:
    """Write segment outputs into the hybrid state buffer using mode semantics."""
    if is_ttfs_spiking_mode(spiking_mode):
        if result.per_core_activations is None:
            raise ValueError(
                "TTFS store_neural_segment_output requires per_core_activations"
            )
    store_segment_output_numpy(output_map, state_buffer, result.inter_stage)

"""Inter-stage signal contract for hybrid neural segments (all backends)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mimarsinan.chip_simulation.hybrid_execution import store_segment_output_numpy
from mimarsinan.chip_simulation.spiking_semantics import TTFS_MODES


@dataclass
class NeuralSegmentResult:
    """Canonical neural-segment outputs for hybrid state-buffer propagation."""

    inter_stage: np.ndarray
    per_core_activations: Optional[List[np.ndarray]] = None
    per_core_spike_counts: Optional[List[np.ndarray]] = None


def is_ttfs_spiking_mode(spiking_mode: str) -> bool:
    return str(spiking_mode) in TTFS_MODES


def lif_inter_stage_from_spike_counts(
    seg_out_spike_count: np.ndarray,
    simulation_length: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """LIF / rate: between-stage values are spike counts normalized by ``T``."""
    t = max(int(simulation_length), 1)
    return (
        np.asarray(seg_out_spike_count, dtype=dtype).reshape(1, -1) / np.asarray(t, dtype=dtype)
    )


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

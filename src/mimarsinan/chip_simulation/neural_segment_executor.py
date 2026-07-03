"""Dispatch neural-segment execution by backend and spiking mode."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import assemble_segment_input_numpy
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    is_ttfs_spiking_mode,
)
from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor


def execute_neural_segment_analytical(
    hcm: Any,
    stage_input_map,
    state_buffer: Dict[int, np.ndarray],
    *,
    num_samples: int,
    simulation_length: int,
    spiking_mode: str,
    dtype: npt.DTypeLike = np.float64,
) -> NeuralSegmentResult:
    """Canonical TTFS/LIF analytical path for references and buffer propagation."""
    seg_in = assemble_segment_input_numpy(
        stage_input_map, state_buffer, num_samples, dtype=dtype,
    )
    if is_ttfs_spiking_mode(spiking_mode):
        return TtfsAnalyticalExecutor().run_segment(
            hcm, seg_in,
            simulation_length=simulation_length,
            spiking_mode=spiking_mode,
        )
    raise NotImplementedError(
        "execute_neural_segment_analytical supports TTFS modes only; "
        "use backend-specific LIF runners for rate/LIF."
    )

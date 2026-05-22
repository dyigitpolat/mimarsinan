"""Canonical TTFS neural-segment execution (HCM, SANA-FE, Nevresim alignment)."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from mimarsinan.chip_simulation.hybrid_semantics import NeuralSegmentResult, is_ttfs_spiking_mode
from mimarsinan.chip_simulation.ttfs_segment import (
    gather_segment_ttfs_output_from_cores,
    run_ttfs_continuous_segment,
    run_ttfs_quantized_segment,
    segment_ttfs_arrays_from_mapping,
)
from mimarsinan.mapping.core_geometry import used_neurons


class TtfsAnalyticalExecutor:
    """Closed-form TTFS segment semantics shared across simulators."""

    def run_segment(
        self,
        hcm: Any,
        seg_input: np.ndarray,
        *,
        simulation_length: int,
        spiking_mode: str,
    ) -> NeuralSegmentResult:
        if not is_ttfs_spiking_mode(spiking_mode):
            raise ValueError(
                f"TtfsAnalyticalExecutor requires TTFS mode, got {spiking_mode!r}"
            )
        seg_in = np.asarray(seg_input, dtype=np.float64)
        if seg_in.ndim != 2:
            raise ValueError(f"seg_input must be 2-D; got shape {seg_in.shape}")
        seg_arrays = segment_ttfs_arrays_from_mapping(hcm)
        quantized = spiking_mode == "ttfs_quantized"
        if quantized:
            seg_out, bufs = run_ttfs_quantized_segment(
                seg_arrays, seg_in, int(simulation_length),
            )
        else:
            seg_out, bufs = run_ttfs_continuous_segment(seg_arrays, seg_in)

        per_core: List[np.ndarray] = []
        for ci, core in enumerate(hcm.cores):
            n_out = used_neurons(core, min_one=True)
            if n_out <= 0:
                per_core.append(np.zeros(0, dtype=np.float64))
                continue
            per_core.append(bufs[ci][0, :n_out].astype(np.float64, copy=False))

        inter = gather_segment_ttfs_output_from_cores(seg_arrays, seg_in, per_core)
        return NeuralSegmentResult(
            inter_stage=inter.astype(np.float64, copy=False),
            per_core_activations=per_core,
        )

    def membrane_voltages(
        self,
        hcm: Any,
        seg_input: np.ndarray,
    ) -> List[np.ndarray]:
        """Per-core ``V = W @ a + b`` before activation (SANA-FE preset injection)."""
        from mimarsinan.chip_simulation.ttfs_segment import ttfs_core_membrane_voltages

        seg_arrays = segment_ttfs_arrays_from_mapping(hcm)
        return ttfs_core_membrane_voltages(
            seg_arrays, np.asarray(seg_input, dtype=np.float64),
        )

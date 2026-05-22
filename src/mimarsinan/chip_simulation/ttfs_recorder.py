"""TTFS activation records for HCM↔SANA-FE parity (not spike-count based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CoreTtfsActivations:
    core_index: int
    n_out_used: int
    output_activation: np.ndarray


def normalize_core_output_activation(
    activation: np.ndarray,
    *,
    n_out_used: int | None = None,
    sample_index: int = 0,
) -> np.ndarray:
    """Return a 1-D per-neuron activation vector for parity records (sample 0)."""
    arr = np.asarray(activation, dtype=np.float64)
    if arr.ndim >= 2:
        vec = arr[int(sample_index)]
    else:
        vec = arr.ravel()
    n = int(n_out_used) if n_out_used is not None else int(vec.size)
    n = min(n, int(vec.size))
    return np.asarray(vec[:n], dtype=np.float64).reshape(-1)


@dataclass
class SegmentTtfsRecord:
    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]
    seg_output: np.ndarray
    cores: List[CoreTtfsActivations] = field(default_factory=list)


@dataclass
class TtfsRunRecord:
    sample_index: int
    simulation_length: int
    spiking_mode: str
    segments: Dict[int, SegmentTtfsRecord] = field(default_factory=dict)
    compute_outputs: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class TtfsDiff:
    stage_index: int
    stage_name: str
    core_index: int
    neuron_index: int
    ref_value: float
    actual_value: float


# Contract-vs-contract: same numpy/float64 analytical runner on both sides.
TTFS_CONTRACT_ATOL = 1e-12
TTFS_CONTRACT_RTOL = 1e-9

# Hardware/plugin readout: float32 compute-op gather and plugin trace ULP drift.
TTFS_HARDWARE_ATOL = 1e-6
TTFS_HARDWARE_RTOL = 1e-5


def compare_ttfs_records(
    ref: TtfsRunRecord,
    actual: TtfsRunRecord,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> List[TtfsDiff]:
    diffs: List[TtfsDiff] = []
    for stage_index, ref_seg in ref.segments.items():
        act_seg = actual.segments.get(stage_index)
        if act_seg is None:
            continue
        ref_by = {c.core_index: c for c in ref_seg.cores}
        for act_core in act_seg.cores:
            ref_core = ref_by.get(act_core.core_index)
            if ref_core is None:
                continue
            ref_vec = normalize_core_output_activation(
                ref_core.output_activation, n_out_used=ref_core.n_out_used,
            )
            act_vec = normalize_core_output_activation(
                act_core.output_activation, n_out_used=act_core.n_out_used,
            )
            n = min(ref_core.n_out_used, act_core.n_out_used, ref_vec.size, act_vec.size)
            for ni in range(n):
                rv = float(ref_vec[ni])
                av = float(act_vec[ni])
                if not np.isclose(rv, av, rtol=rtol, atol=atol):
                    diffs.append(TtfsDiff(
                        stage_index=stage_index,
                        stage_name=ref_seg.stage_name,
                        core_index=act_core.core_index,
                        neuron_index=ni,
                        ref_value=rv,
                        actual_value=av,
                    ))
    return diffs


def compare_ttfs_contract_records(
    ref: TtfsRunRecord,
    actual: TtfsRunRecord,
) -> List[TtfsDiff]:
    """Parity for analytical contract paths (tight tolerance)."""
    return compare_ttfs_records(
        ref, actual, atol=TTFS_CONTRACT_ATOL, rtol=TTFS_CONTRACT_RTOL,
    )


def compare_ttfs_hardware_records(
    ref: TtfsRunRecord,
    actual: TtfsRunRecord,
) -> List[TtfsDiff]:
    """Parity for plugin/hardware activations vs analytical reference."""
    return compare_ttfs_records(
        ref, actual, atol=TTFS_HARDWARE_ATOL, rtol=TTFS_HARDWARE_RTOL,
    )


def format_first_ttfs_diff(
    diffs: List[TtfsDiff],
    *,
    layer: str = "",
) -> str:
    if not diffs:
        return ""
    d = diffs[0]
    label = f"TTFS {layer} parity".strip() if layer else "TTFS parity"
    return (
        f"{label} mismatch at stage {d.stage_index} ({d.stage_name!r}), "
        f"core {d.core_index} neuron {d.neuron_index}: "
        f"ref={d.ref_value:.8g} actual={d.actual_value:.8g}"
    )

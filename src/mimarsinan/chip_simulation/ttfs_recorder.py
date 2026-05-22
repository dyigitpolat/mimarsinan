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
            n = min(ref_core.n_out_used, act_core.n_out_used,
                    ref_core.output_activation.size, act_core.output_activation.size)
            for ni in range(n):
                rv = float(ref_core.output_activation[ni])
                av = float(act_core.output_activation[ni])
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

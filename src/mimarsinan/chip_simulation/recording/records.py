"""Per-segment spike-count records for HCM↔Loihi parity verification."""
# Subtractive reset with no voltage decay ⇒ output spike counts depend only on total integrated input (order-independent), so equal input counts + weights + threshold ⇒ equal output counts.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CoreSpikeCounts:
    """Summed spike activity for one hard core over the sample window."""

    core_index: int
    n_in_used: int
    n_out_used: int
    core_latency: int
    has_hardware_bias: bool
    n_always_on_axons: int

    input_spike_count: np.ndarray
    output_spike_count: np.ndarray


@dataclass
class SegmentSpikeRecord:
    """Spike-count snapshot for one ``HybridStage`` of kind ``"neural"``."""

    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]

    seg_input_rates: np.ndarray
    seg_input_spike_count: np.ndarray
    seg_output_spike_count: np.ndarray

    cores: List[CoreSpikeCounts] = field(default_factory=list)


@dataclass
class RunRecord:
    """All per-segment records produced by a single forward pass.

    ``segments`` keyed by ``stage_index``; compute stages store float outputs in
    ``compute_outputs`` keyed by ``ComputeOp.id`` (consumed by Loihi harness mode)."""

    sample_index: int
    T: int
    segments: Dict[int, SegmentSpikeRecord] = field(default_factory=dict)
    compute_outputs: Dict[int, np.ndarray] = field(default_factory=dict)


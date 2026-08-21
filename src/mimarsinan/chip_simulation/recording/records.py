"""Per-segment spike-count records for HCM↔Loihi parity verification."""
# Order-independence is a PROPERTY OF THE LAW, not of counts: under a lossless
# subtractive reset with no decay and at most one spike per cycle, a window's
# output count depends only on the total integrated input. A per-event law
# (firing_granularity='per_event') denies that hypothesis — arrival order and
# adjacency change the count — so under it these counts are only a projection
# of the record, and the per-cycle raster below is the load-bearing half.

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
    # (T, n_out_used) per-cycle emission multiplicities in PRODUCER-LOCAL time;
    # None wherever the executing law makes the count a complete record.
    output_spike_raster: Optional[np.ndarray] = None


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


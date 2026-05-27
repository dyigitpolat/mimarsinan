"""Per-segment spike-count records for HCM↔Loihi parity verification.

Both ``SpikingHybridCoreFlow`` (HCM) and ``LavaLoihiRunner`` (Loihi)
populate the same ``RunRecord`` shape on a single sample so the two
simulators can be diffed core-by-core.  We compare *spike counts*
(integer totals over the simulation window) rather than full per-cycle
spike trains.  With ``firing_mode='Default'`` (subtractive reset, no
voltage decay), output spike counts are determined by total integrated
input — order doesn't matter — so equal input counts + equal weights +
equal threshold ⇒ equal output counts.  Counts therefore catch every
class of bug we care about (encoding, routing, weights, threshold,
hardware bias) without dragging cycle-alignment edge cases into the
comparison.

Per-segment fields are the user's stated comparison surface: total
spikes entering / leaving each neural segment.  Per-core fields are a
drill-down used by ``compare_records`` to localise the first divergent
core, axon, or neuron when the segment-level totals disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CoreSpikeCounts:
    """Summed spike activity for one hard core over the sample window.

    Both ``input_spike_count`` and ``output_spike_count`` are integer-valued
    even when stored as ``int64`` — the underlying spikes are 0/1 and we
    sum them over the cycles a given simulator considers "real" for this
    core.
    """

    core_index: int
    n_in_used: int
    n_out_used: int
    core_latency: int
    has_hardware_bias: bool
    n_always_on_axons: int

    input_spike_count: np.ndarray   # (n_in_used,) int64 — total per-axon
    output_spike_count: np.ndarray  # (n_out_used,) int64 — total per-neuron


@dataclass
class SegmentSpikeRecord:
    """Spike-count snapshot for one ``HybridStage`` of kind ``"neural"``.

    ``seg_input_rates`` is the float driver assembled from the state
    buffer prior to encoding; harness-mode Loihi runs use it to feed
    each segment in isolation, decoupling per-stage divergence from
    upstream cascading.
    """

    stage_index: int
    stage_name: str
    schedule_segment_index: Optional[int]
    schedule_pass_index: Optional[int]

    seg_input_rates: np.ndarray            # (1, seg_in_size) float32
    seg_input_spike_count: np.ndarray      # (seg_in_size,) int64 — encoded train summed over T
    seg_output_spike_count: np.ndarray     # (seg_out_size,) int64 — gathered output sums

    cores: List[CoreSpikeCounts] = field(default_factory=list)


@dataclass
class RunRecord:
    """All per-segment records produced by a single forward.

    ``segments`` is keyed by ``stage_index`` in mapping order.  Compute
    stages do not produce a ``SegmentSpikeRecord``; their float outputs
    live in ``compute_outputs`` (keyed by ``ComputeOp.id``) and are
    consumed by Loihi harness mode to feed downstream neural stages
    without re-running the host module.
    """

    sample_index: int
    T: int
    segments: Dict[int, SegmentSpikeRecord] = field(default_factory=dict)
    compute_outputs: Dict[int, np.ndarray] = field(default_factory=dict)


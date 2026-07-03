from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.mapping.support.activation_scales import NodeScale


_FINAL_OUTPUT_SENTINEL = -999


@dataclass
class SegmentIOSlice:
    """Maps a contiguous slice of a neural segment's I/O buffer to a state-buffer entry."""
    node_id: int
    offset: int
    size: int


@dataclass
class HybridStage:
    """A single stage in a hybrid runtime program (neural or compute)."""

    kind: Literal["neural", "compute"]
    name: str
    hard_core_mapping: HardCoreMapping | None = None
    compute_op: ComputeOp | None = None
    input_map: list[SegmentIOSlice] = field(default_factory=list)
    output_map: list[SegmentIOSlice] = field(default_factory=list)
    schedule_segment_index: int | None = None
    schedule_pass_index: int | None = None


@dataclass
class HybridHardCoreMapping:
    """Deployable hybrid program: neural segments interleaved with ComputeOp barriers."""

    stages: List[HybridStage]
    output_sources: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    # Scalar float, or per-channel np.ndarray for ttfs_theta_cotrain nodes.
    node_activation_scales: dict[int, NodeScale] = field(default_factory=dict)
    node_input_activation_scales: dict[int, NodeScale] = field(default_factory=dict)
    # Per-producer per-channel rate shift applied before the [0,1] clamp; the consumer core's bias is pre-corrected (B' = B - W·s) so the on-chip result is unchanged.
    node_output_shifts: dict[int, np.ndarray] = field(default_factory=dict)
    # Lazy consumer-refcount cache filled by SpikingHybridCoreFlow._build_consumer_counts.
    _consumer_counts_cache: dict[int, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def get_compute_ops(self) -> List[ComputeOp]:
        return [s.compute_op for s in self.stages if s.kind == "compute" and s.compute_op is not None]

    def get_neural_segments(self) -> List[HardCoreMapping]:
        return [s.hard_core_mapping for s in self.stages if s.kind == "neural" and s.hard_core_mapping is not None]


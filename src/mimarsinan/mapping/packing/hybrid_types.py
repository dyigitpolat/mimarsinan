from __future__ import annotations

import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np

from mimarsinan.mapping.activation_scales import (
    compute_node_input_scales as _compute_node_input_activation_scales,
    compute_node_output_scales as _compute_node_activation_scales,
)
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping, compact_soft_core_mapping


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
    # Scheduling metadata (None when scheduling is off / single-pass segments)
    schedule_segment_index: int | None = None
    schedule_pass_index: int | None = None


@dataclass
class HybridHardCoreMapping:
    """Deployable hybrid program: neural segments interleaved with ComputeOp barriers."""

    stages: List[HybridStage]
    output_sources: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    node_activation_scales: dict[int, float] = field(default_factory=dict)
    node_input_activation_scales: dict[int, float] = field(default_factory=dict)

    def get_compute_ops(self) -> List[ComputeOp]:
        return [s.compute_op for s in self.stages if s.kind == "compute" and s.compute_op is not None]

    def get_neural_segments(self) -> List[HardCoreMapping]:
        return [s.hard_core_mapping for s in self.stages if s.kind == "neural" and s.hard_core_mapping is not None]


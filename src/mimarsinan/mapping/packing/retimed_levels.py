"""Per-depth-level execution stages for windowed-lif fused segments [C3 fused].

The mapping stays FUSED (one honest neural stage per maximal run); when per-hop
re-timing is armed, each multi-level stage carries ``retimed_level_stages`` —
built by the SAME flush path the historical per-hop split used, so they are
structurally the split builder's hop stages (real node-id I/O maps, identical
compaction chaining). The shared stage loop executes them through each
backend's existing per-stage handler; the fused stage remains the program
artifact reports and capacity read.
"""

from __future__ import annotations

from typing import Sequence

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.segmentation import segment_depth_groups
from mimarsinan.mapping.packing.hybrid_segment import _flush_neural_segment
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _make_available_hardware_cores,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_types import HybridStage
from mimarsinan.mapping.platform.platform_constraints import pool_sign_granularity


def attach_retimed_level_stages(
    stage: HybridStage,
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    weight_banks: dict,
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    identity: bool = False,
    level_units: "Sequence[list[NeuralCore]] | None" = None,
) -> None:
    """Attach per-depth-level execution stages to a fused neural stage.

    No-op (stage executes directly, streaming) for single-node, single-depth,
    or coalescing/psum-group segments — exactly the historical split refusals.
    Level chips place on a private pool: they are execution vehicles like the
    per-segment compiled binaries, not capacity-bearing program structure.
    [U1 R5] ``level_units``: a pass of a multi-level fused segment executes
    the fused segment's levels it holds, one level unit each, even a single one.
    """
    groups = segment_depth_groups(current_neural) if level_units is None else list(level_units)
    if groups is None or (level_units is None and len(groups) <= 1):
        return

    pool = [] if identity else _make_available_hardware_cores(cores_config)
    # Compaction reindex chains ACROSS levels exactly as the split build
    # chained it across hop segments; boundary-facing (output) nodes keep
    # every neuron in both views, so downstream fused indexing agrees.
    level_reindex: dict[int, dict[int, int]] = {}
    levels: list[HybridStage] = []
    for depth, group in enumerate(groups):
        nodes = _reindex_nodes(group, level_reindex) if level_reindex else group
        level_stage, seg_reindex = _flush_neural_segment(
            current_neural=nodes,
            consumed_by=consumed_by,
            shared_pool=pool,
            weight_banks=weight_banks,
            name=f"{stage.name}_hop{depth}",
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            identity=identity,
            sign_granularity=pool_sign_granularity(cores_config),
        )
        level_reindex.update(seg_reindex)
        # [nevresim parity] a retimed hop's input is the COUNT re-encode by
        # definition; boundary encodes must never pass a producer's raw
        # emitted rhythm through the train cache (the t0_04 s32 catch:
        # same counts, shifted comb, ±1 fires downstream).
        level_stage.is_retimed_level = True
        levels.append(level_stage)
    stage.retimed_level_stages = levels

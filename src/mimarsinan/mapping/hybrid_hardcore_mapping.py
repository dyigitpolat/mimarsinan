from __future__ import annotations

import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore, ir_graph_to_soft_core_mapping
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
    """
    A single stage in a hybrid runtime program.

    - "neural": A HardCoreMapping that can be executed on the chip runtime.
    - "compute": A ComputeOp that must be executed as a sync barrier.

    Neural stages carry ``input_map`` / ``output_map`` metadata so the
    runtime can assemble the segment's input from a global state buffer
    and store the segment's output back into that buffer.
    """

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
    """
    A deployable *hybrid* program representation:

    neural segment (HardCoreMapping) -> ComputeOp barrier -> neural segment -> ...

    Each neural segment can be packed/codegen'ed as a standalone chip program.
    ComputeOps represent host-side (or auxiliary) computation between chip runs.

    ``output_sources`` mirrors the original IRGraph output_sources so the
    runtime can assemble the final network output from the state buffer.

    ``node_activation_scales`` maps original IR ``node_id`` to the *output*
    ``activation_scale`` (the divisor that normalises the op's training-range
    output back to TTFS [0, 1] for downstream NeuralCores).

    ``node_input_activation_scales`` maps original IR ``node_id`` to the
    *input* rescale factor (the multiplier that brings gathered inputs from
    TTFS [0, 1] back to training range before running the op's module).
    Distinct from ``node_activation_scales`` for encoding-layer ComputeOps
    whose sources are the raw model input (already in training range, so
    in_scale=1.0) but whose output must still be divided by the wrapped
    Perceptron's activation_scale.
    """

    stages: List[HybridStage]
    output_sources: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    node_activation_scales: dict[int, float] = field(default_factory=dict)
    node_input_activation_scales: dict[int, float] = field(default_factory=dict)

    def get_compute_ops(self) -> List[ComputeOp]:
        return [s.compute_op for s in self.stages if s.kind == "compute" and s.compute_op is not None]

    def get_neural_segments(self) -> List[HardCoreMapping]:
        return [s.hard_core_mapping for s in self.stages if s.kind == "neural" and s.hard_core_mapping is not None]


def _make_available_hardware_cores(cores_config: Sequence[dict]) -> list[HardCore]:
    available_hardware_cores: list[HardCore] = []
    for core_type in cores_config:
        count = int(core_type["count"])
        max_axons = int(core_type["max_axons"])
        max_neurons = int(core_type["max_neurons"])
        has_bias = bool(core_type.get("has_bias", True))
        for _ in range(count):
            available_hardware_cores.append(
                HardCore(max_axons, max_neurons, has_bias_capability=has_bias)
            )
    return available_hardware_cores


def _remap_external_sources_to_segment_inputs(
    *,
    nodes: list[NeuralCore],
    output_sources: np.ndarray,
    weight_banks: dict | None = None,
) -> tuple[IRGraph, list[SegmentIOSlice]]:
    """
    Build a neural-only IRGraph for a segment with support for
    multiple external source nodes (skip connections).

    External IRSource references (node_id >= 0, not in this segment) are
    remapped to segment-local inputs (IRSource(node_id=-2, index=...)).
    Multiple external nodes are packed into a composite input buffer::

        [original_input 0..max_orig] [ext_A 0..sA-1] [ext_B 0..sB-1] ...

    Returns
    -------
    segment_graph : IRGraph
        Neural-only graph with remapped sources.
    input_map : list[SegmentIOSlice]
        Describes how to assemble the composite input buffer from the
        global state buffer at runtime.
    """
    if weight_banks is None:
        weight_banks = {}
    node_ids = {n.id for n in nodes}

    for src in output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
            raise ValueError(
                "Segment output_sources reference external node "
                f"(node_id={src.node_id})."
            )

    max_input_idx = -1
    for n in nodes:
        for src in n.input_sources.flatten():
            if isinstance(src, IRSource) and src.is_input():
                max_input_idx = max(max_input_idx, src.index)

    external_max_idx: dict[int, int] = {}
    for n in nodes:
        for src in n.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
                external_max_idx[src.node_id] = max(
                    external_max_idx.get(src.node_id, 0), src.index
                )

    input_map: list[SegmentIOSlice] = []
    current_offset = (max_input_idx + 1) if max_input_idx >= 0 else 0

    if max_input_idx >= 0:
        input_map.append(SegmentIOSlice(node_id=-2, offset=0, size=max_input_idx + 1))

    offsets: dict[int, int] = {}
    for ext_id in sorted(external_max_idx.keys()):
        size = external_max_idx[ext_id] + 1
        offsets[ext_id] = current_offset
        input_map.append(SegmentIOSlice(node_id=ext_id, offset=current_offset, size=size))
        current_offset += size

    def remap_src(src: IRSource) -> IRSource:
        if src.node_id >= 0 and src.node_id not in node_ids:
            return IRSource(node_id=-2, index=offsets[src.node_id] + src.index)
        return src

    new_nodes: list[NeuralCore] = []
    for n in nodes:
        # Shallow copy with fresh input_sources — avoids deep-copying
        # ``core_matrix`` (the hot cost) since it is immutable after
        # quantization and safely shareable across reindexed copies.
        n2 = copy.copy(n)
        flat = [remap_src(src) for src in n.input_sources.flatten()]
        n2.input_sources = np.array(flat, dtype=object).reshape(n.input_sources.shape)
        new_nodes.append(n2)

    graph = IRGraph(
        nodes=new_nodes,
        output_sources=np.array(output_sources, dtype=object),
        weight_banks=weight_banks,
    )
    return graph, input_map


def _check_no_split_coalescing_groups(nodes: list) -> None:
    """Assert that no coalescing group is split at a segment boundary.

    Each coalescing group that has partial cores in this segment must also
    have its accumulator core in this segment.  A missing accumulator means
    a ComputeOp was inserted between a partial core and its accumulator,
    which would break the coalescing semantics.
    """
    group_roles: dict[int, list[str]] = {}
    for n in nodes:
        if isinstance(n, NeuralCore):
            gid = getattr(n, "coalescing_group_id", None)
            role = getattr(n, "coalescing_role", None)
            if gid is not None and role is not None:
                group_roles.setdefault(gid, []).append(role)

    for gid, roles in group_roles.items():
        if "master" in roles:
            continue # Unified wide core, no psums needed.

        if "accum" not in roles:
            raise ValueError(
                f"Coalescing group {gid} has partial cores in this neural segment "
                f"but its accumulator is missing (roles found: {roles}). "
                f"A ComputeOp must not be inserted between coalescing partial cores "
                f"and their accumulator."
            )


def _reindex_nodes(
    nodes: list[NeuralCore],
    reindex_maps: dict[int, dict[int, int]],
) -> list[NeuralCore]:
    """Clone nodes with fresh input_sources and apply compaction reindex.

    Uses a shallow copy plus an independent ``input_sources`` ndarray so we
    can mutate sources without touching the originals.  Weight tensors
    (``core_matrix``, ``hardware_bias``, heatmaps, masks) are shared by
    reference — they are immutable after quantization/compaction and
    deep-copying them dominated HCM build time at ViT scale
    (numpy deep-copy was 2.7 s out of 4 s on a 708-core workload).
    """
    result = []
    for n in nodes:
        n2 = copy.copy(n)
        # Fresh object-array for input_sources so reindex mutations do not
        # leak into the original.  Element objects (IRSource) are small and
        # safe to share — we only overwrite entries via .flat[i] assignment.
        n2.input_sources = np.array(
            n.input_sources.flatten(),
            dtype=object,
        ).reshape(n.input_sources.shape)
        _apply_reindex_to_ir_sources(n2.input_sources, reindex_maps)
        result.append(n2)
    return result


def _apply_reindex_to_ir_sources(
    sources: np.ndarray,
    reindex_maps: dict[int, dict[int, int]],
) -> None:
    """Apply compaction reindex maps to an array of IRSource objects in place.

    For each IRSource referencing a compacted core, updates ``.index``
    to the new (post-compaction) neuron index.  Sources referencing
    pruned neurons are replaced with off-sources.
    """
    for i, src in enumerate(sources.flatten()):
        if not isinstance(src, IRSource) or src.node_id < 0:
            continue
        remap = reindex_maps.get(src.node_id)
        if remap is None:
            continue  # core not in any compacted segment
        new_idx = remap.get(src.index)
        if new_idx is not None:
            sources.flat[i] = IRSource(node_id=src.node_id, index=new_idx)
        else:
            # Neuron was pruned — replace with off-source
            sources.flat[i] = IRSource(node_id=-1, index=0)


def _flush_neural_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    shared_pool: list[HardCore],
    weight_banks: dict,
    name: str,
    allow_neuron_splitting: bool = False,
    skip_coalescing_check: bool = False,
) -> tuple[HybridStage, dict[int, dict[int, int]]]:
    """Pack a neural segment using cores drawn from *shared_pool*.

    ``consumed_by`` is the pre-scanned map of *node_id → set of consumer
    node_ids* over the full IR graph.  It is used to determine which
    segment-internal nodes need to appear in the segment's output buffer
    (i.e. those consumed by any node *outside* this segment).

    When ``skip_coalescing_check`` is True, the coalescing group integrity
    check is skipped.  This is used by scheduled passes where coalescing
    fragments may be distributed across passes with the state buffer
    handling inter-pass data flow.

    Returns:
        (stage, reindex_maps) where reindex_maps is
        ``{core_id: {old_neuron_idx: new_neuron_idx}}`` from compaction.
        Callers must apply these maps to any external references that
        point into this segment's cores (e.g. final output_sources,
        ComputeOp input_sources).
    """
    segment_node_ids = {n.id for n in current_neural}

    if not skip_coalescing_check:
        _check_no_split_coalescing_groups(current_neural)

    output_nodes: list[NeuralCore] = []
    for n in current_neural:
        consumers = consumed_by.get(n.id, set())
        if any(c not in segment_node_ids for c in consumers):
            output_nodes.append(n)

    output_sources_list: list[IRSource] = []
    for n in output_nodes:
        for idx in range(n.get_output_count()):
            output_sources_list.append(IRSource(node_id=n.id, index=idx))

    if output_nodes and not output_sources_list:
        raise ValueError(
            "Segment has output_nodes but 0 output refs (all get_output_count() are 0). "
            "Output layer was over-pruned; at least one output neuron must remain."
        )

    output_sources = np.array(output_sources_list, dtype=object)

    seg_graph, input_map = _remap_external_sources_to_segment_inputs(
        nodes=current_neural,
        output_sources=output_sources,
        weight_banks=weight_banks,
    )
    soft = ir_graph_to_soft_core_mapping(seg_graph)

    # Compact soft cores: remove all-zero rows/columns and reindex spans so
    # hardware mapping shows only utilized structure (pruning reflected).
    reindex_maps = compact_soft_core_mapping(soft.cores, soft.output_sources)

    # Rebuild output_map from compacted output sizes
    output_map = []
    current_offset = 0
    output_core_ids = [n.id for n in output_nodes]
    core_by_id = {c.id: c for c in soft.cores}
    for nid in output_core_ids:
        core = core_by_id.get(nid)
        size = core.get_output_count() if core else 0
        output_map.append(SegmentIOSlice(node_id=nid, offset=current_offset, size=size))
        current_offset += size

    hard = HardCoreMapping(shared_pool)
    try:
        hard.map(soft, allow_neuron_splitting=allow_neuron_splitting)
    except RuntimeError as e:
        # Rich diagnostic — the layout mapper pre-validated this sub-segment
        # via ``pack_layout`` but the real greedy packer on ``SoftCore``s
        # disagreed.  Expose the exact softcore shapes + their threshold
        # groups so we can triangulate the divergence (typically a mismatch
        # between ``SoftCore.threshold_group_id`` and its ``LayoutSoftCoreSpec``
        # counterpart, or pruning changing axon counts between layout time
        # and hard-core mapping time).
        group_ids: Dict[object, int] = {}
        rows = []
        for sc in soft.cores:
            tg = getattr(sc, "threshold_group_id", None)
            pi = getattr(sc, "perceptron_index", None)
            tg_val = tg if tg is not None else f"fallback(-{sc.id + 1})"
            group_ids.setdefault(tg_val, 0)
            group_ids[tg_val] += 1
            rows.append(
                f"    id={sc.id} name={sc.name!r} axons={sc.get_input_count()} "
                f"neurons={sc.get_output_count()} perc_idx={pi} tg={tg}"
            )
        hw_summary = ", ".join(
            f"{hc.axons_per_core}x{hc.neurons_per_core} count={hc.axons_per_core}"
            for hc in shared_pool[:3]
        )
        diag = (
            f"Hard-core packing failed in segment '{name}': {e}.\n"
            f"  Sub-segment softcores: {len(soft.cores)}\n"
            f"  Distinct threshold groups: {len(group_ids)} "
            f"(most: {sorted(group_ids.items(), key=lambda kv: -kv[1])[:3]})\n"
            f"  Pool size: {len(shared_pool)} (types head: {hw_summary})\n"
            f"  Softcore heads:\n" + "\n".join(rows[:5])
        )
        raise RuntimeError(diag) from e

    return HybridStage(
        kind="neural",
        name=name,
        hard_core_mapping=hard,
        input_map=input_map,
        output_map=output_map,
    ), reindex_maps


def _validate_coalescing_budget(
    cores: list[NeuralCore],
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool,
) -> None:
    """Verify every wide NeuralCore's coalescing group fits in one core type's count.

    Coalescing cores for a single NeuralCore cannot be distributed across
    schedule passes because the hardware lacks membrane potential
    initialization — partial sums from different passes would need to be
    accumulated before activation, which is not yet supported.

    Raises ``RuntimeError`` with a descriptive message when no core type
    has a sufficient count.
    """
    import math

    for core in cores:
        if core.core_matrix is None:
            continue
        n_weight_axons = core.get_input_count()
        if core.hardware_bias is None:
            src_flat = core.input_sources.flatten()
            if (len(src_flat) > 0
                    and isinstance(src_flat[-1], IRSource)
                    and src_flat[-1].is_always_on()):
                n_weight_axons -= 1
        n_neurons = core.get_output_count()

        fits_any = False
        best_needed = 0
        for ct in cores_config:
            ct_axons = int(ct["max_axons"])
            ct_neurons = int(ct["max_neurons"])
            ct_count = int(ct["count"])
            neuron_ok = allow_neuron_splitting or ct_neurons >= n_neurons
            if not neuron_ok:
                continue
            n_coalesce = math.ceil(n_weight_axons / ct_axons)
            if ct_count >= n_coalesce:
                fits_any = True
                break
            best_needed = max(best_needed, n_coalesce)

        if not fits_any and best_needed > 0:
            raise RuntimeError(
                f"Core '{core.name}' has {n_weight_axons} weight axons and "
                f"{n_neurons} neurons, requiring at least {best_needed} "
                f"coalescing cores in a single pass. No hardware core type "
                f"has sufficient count. Increase the count of a suitable "
                f"core type to at least {best_needed}, or use a core type "
                f"with wider axons."
            )


def _flush_scheduled_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    cores_config: Sequence[dict],
    weight_banks: dict,
    segment_index: int,
    segment_label: str,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> tuple[list[HybridStage], dict[int, dict[int, int]]]:
    """Flush a neural segment as a single scheduled pass.

    One segment → one ``HybridStage``.  The layout mapper has already
    inserted sync barriers (= segment boundaries) wherever the hardware
    cannot hold the combined cores, so any segment reaching this call
    must fit the physical core pool.  We call ``_flush_neural_segment``
    exactly once against a fresh pool; if the packer cannot place every
    core, we let the ``RuntimeError`` propagate — that is an infeasible
    template the user must fix, not a case to silently split.

    Latency-group pass splitting was previously performed here.  It
    counted each IR NeuralCore as ≥ 1 hardware core and therefore
    over-partitioned segments that the layout packer would fit in one
    pass.  The simulator then treated every sub-pass as a sync barrier
    (rate-level handoff between cores), breaking cycle-accurate LIF
    semantics and causing SCM / HCM / nevresim accuracy drift.
    """
    if allow_coalescing:
        _validate_coalescing_budget(
            current_neural, cores_config, allow_neuron_splitting,
        )

    shared_pool = _make_available_hardware_cores(cores_config)
    stage, seg_reindex_all = _flush_neural_segment(
        current_neural=current_neural,
        consumed_by=consumed_by,
        shared_pool=shared_pool,
        weight_banks=weight_banks,
        name=segment_label,
        allow_neuron_splitting=allow_neuron_splitting,
        skip_coalescing_check=True,
    )
    stage.schedule_segment_index = segment_index
    stage.schedule_pass_index = 0

    return [stage], seg_reindex_all


def _compute_node_activation_scales(ir_graph: IRGraph) -> dict[int, float]:
    """Extract per-node *output* scales from an IRGraph.

    This is the scale downstream consumers assume — the divisor applied to
    the op's training-range output before storing it in the state buffer,
    so that values land in the TTFS [0, 1] convention used by NeuralCore
    effective weights.

    See also :func:`_compute_node_input_activation_scales` for the scale
    applied when *rescaling* gathered inputs back to training range.

    * NeuralCores: their own ``activation_scale``.
    * ComputeOps wrapping a Perceptron / PerceptronMapper: the wrapped
      Perceptron's ``activation_scale`` (its output is in
      ``[0, activation_scale]`` after the decorator chain, and the
      downstream NeuralCore's ``per_input_scales`` equals that same value).
    * Generic ComputeOps: average of source scales (pass-through heuristic
      that matches ``compute_per_source_scales``'s downstream per-input
      assignment for scale-equivariant and bias-carrying ops).
    """
    scales: dict[int, float] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            s = node.activation_scale
            scales[node.id] = float(s.item() if hasattr(s, "item") else s)
        elif isinstance(node, ComputeOp):
            module = (node.params or {}).get("module")
            wrapped_scale = _perceptron_wrapped_activation_scale(module)
            if wrapped_scale is not None:
                scales[node.id] = wrapped_scale
                continue

            src_scales: list[float] = []
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    src_scales.append(scales.get(src.node_id, 1.0))
            scales[node.id] = (
                sum(src_scales) / len(src_scales) if src_scales else 1.0
            )
    return scales


def _compute_node_input_activation_scales(ir_graph: IRGraph) -> dict[int, float]:
    """Input-rescale factors for ComputeOps.

    Applied by consumers as ``gathered = gathered * in_scale`` before running
    the op's module.  Differs from the output scale for encoding-layer
    ComputeOps whose sources are the raw model input (already in training
    range — no rescale needed).

    * Source-from-node ComputeOps: average of source (output) scales.
    * All-raw-input ComputeOps (encoding path): ``1.0`` (no rescale).
    * NeuralCores: their activation_scale (unused by the compute-op path,
      populated for API symmetry).
    """
    out_scales = _compute_node_activation_scales(ir_graph)
    in_scales: dict[int, float] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            in_scales[node.id] = out_scales[node.id]
        elif isinstance(node, ComputeOp):
            src_scales: list[float] = []
            all_raw = True
            for src in node.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    all_raw = False
                    src_scales.append(out_scales.get(src.node_id, 1.0))
            if all_raw:
                in_scales[node.id] = 1.0
            else:
                in_scales[node.id] = (
                    sum(src_scales) / len(src_scales) if src_scales else 1.0
                )
    return in_scales


def _perceptron_wrapped_activation_scale(module) -> float | None:
    """Return the activation_scale of a Perceptron / PerceptronMapper wrapped
    as a ``ComputeOp(module)``, or ``None`` if *module* doesn't carry one.

    Encoding-layer perceptrons (first layer of a neural segment) are mapped
    to ``ComputeOp(op_type="module")`` that holds either the Perceptron
    directly (from ``PerceptronMapper._map_to_ir_as_encoding_compute_op``)
    or the Conv2DPerceptronMapper (from ``Conv2DPerceptronMapper._map_to_ir``).
    Both expose ``activation_scale`` via ``module.activation_scale`` or
    ``module.perceptron.activation_scale`` respectively.
    """
    if module is None:
        return None
    s = getattr(module, "activation_scale", None)
    if s is None:
        perceptron = getattr(module, "perceptron", None)
        if perceptron is not None:
            s = getattr(perceptron, "activation_scale", None)
    if s is None:
        return None
    try:
        return float(s.item() if hasattr(s, "item") else s)
    except (TypeError, ValueError):
        return None


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool = False,
    allow_scheduling: bool = False,
    allow_coalescing: bool = False,
) -> HybridHardCoreMapping:
    """
    Compile a unified IRGraph into a HybridHardCoreMapping.

    Supports skip connections / residual paths through a state-buffer
    approach: each neural stage carries ``input_map`` / ``output_map``
    metadata so the runtime can maintain a ``Dict[int, Tensor]`` state
    buffer keyed by original IR node_id.

    When ``allow_scheduling`` is False (default), a **single** pool of
    hardware cores is allocated upfront and shared across all neural
    segments so the total core budget is respected.

    When ``allow_scheduling`` is True, each neural segment (or pass within
    a segment) gets a **fresh** hardware core pool — the same physical
    cores are reprogrammed between passes.  This allows mapping models
    that need more cores than available, trading latency for chip area.

    ``allow_coalescing`` is threaded to the schedule partitioner so the
    hardware cost model matches the wizard verifier.
    """

    consumed_by: dict[int, set[int]] = defaultdict(set)
    for node in ir_graph.nodes:
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                consumed_by[src.node_id].add(node.id)
    for src in ir_graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            consumed_by[src.node_id].add(_FINAL_OUTPUT_SENTINEL)

    stages: list[HybridStage] = []

    # Collect compaction reindex maps from all segments so we can fix up
    # external references (final output_sources, ComputeOp input_sources).
    all_reindex_maps: dict[int, dict[int, int]] = {}

    if allow_scheduling:
        _build_scheduled(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
    else:
        _build_single_pool(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
        )

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    # Apply compaction reindex to final output_sources.
    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    # Build per-node scales for TTFS ComputeOp input rescaling and output
    # normalisation.  See dataclass docstring for semantics.
    node_activation_scales = _compute_node_activation_scales(ir_graph)
    node_input_activation_scales = _compute_node_input_activation_scales(ir_graph)

    return HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
        node_activation_scales=node_activation_scales,
        node_input_activation_scales=node_input_activation_scales,
    )


def _build_single_pool(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
) -> None:
    """Original single-shared-pool compilation path."""
    shared_pool: list[HardCore] = _make_available_hardware_cores(cores_config)

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                # Apply accumulated reindex maps to nodes' input_sources so
                # references to previously-compacted cores use correct indices.
                if all_reindex_maps:
                    current_neural = _reindex_nodes(current_neural, all_reindex_maps)
                stage, seg_reindex = _flush_neural_segment(
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    shared_pool=shared_pool,
                    weight_banks=ir_graph.weight_banks,
                    name=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                )
                stages.append(stage)
                all_reindex_maps.update(seg_reindex)
                current_neural = []

            # Apply compaction reindex to ComputeOp input_sources so they
            # reference the compacted neuron indices in the state buffer.
            # Use a shallow copy + fresh input_sources ndarray — deep-copying
            # a ComputeOp also copies its ``params["module"]`` (torch nn.Module
            # with weight tensors), which dominated HCM build at ViT scale.
            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        if all_reindex_maps:
            current_neural = _reindex_nodes(current_neural, all_reindex_maps)
        stage, seg_reindex = _flush_neural_segment(
            current_neural=current_neural,
            consumed_by=consumed_by,
            shared_pool=shared_pool,
            weight_banks=ir_graph.weight_banks,
            name="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
        )
        stages.append(stage)
        all_reindex_maps.update(seg_reindex)


def _split_segment_by_capacity(
    cores: list[NeuralCore],
    cores_config: Sequence[dict],
    *,
    allow_coalescing: bool,
    allow_neuron_splitting: bool,
) -> list[list[NeuralCore]]:
    """Wrap :func:`split_softcores_by_capacity` for a list of IR NeuralCores.

    Converts each NeuralCore to a ``LayoutSoftCoreSpec`` shape, delegates
    to the shared layout-side splitter (so the wizard's capacity analysis
    and the hard-core mapper's capacity analysis always agree), then maps
    the resulting sub-segment groupings back onto the original NeuralCore
    list order.
    """
    if not cores:
        return []

    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
    from mimarsinan.mapping.schedule_partitioner import split_softcores_by_capacity

    hw_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]

    # Mirror ``LayoutIRMapping._finalize_softcores``: the threshold group a
    # softcore belongs to is its owning Perceptron, so all NeuralCores tiled
    # out of one Perceptron (conv positions, psum fragments, output tiling)
    # share a group id and pack together under ``pack_layout``'s group
    # constraint.  Using ``NeuralCore.threshold_group_id`` directly would
    # miss this — that attribute is often ``None`` on IR cores, which
    # previously made ``_to_spec`` fall back to unique-per-index groups and
    # fragmented the capacity split into one pass per softcore.  Fall back
    # to a unique negative id only when there is no perceptron_index, so
    # standalone non-Perceptron cores (e.g. synthesised accumulators) are
    # kept isolated as before.
    specs: list[LayoutSoftCoreSpec] = []
    spec_to_core: dict[int, NeuralCore] = {}
    for idx, core in enumerate(cores):
        lat = int(core.latency) if core.latency is not None else 0
        pi = getattr(core, "perceptron_index", None)
        tg = int(pi) if pi is not None else -(idx + 1)
        spec = LayoutSoftCoreSpec(
            input_count=int(len(core.input_sources.flatten())),
            output_count=int(core.get_output_count()),
            threshold_group_id=tg,
            latency_tag=lat,
            segment_id=0,
            name=core.name,
        )
        specs.append(spec)
        spec_to_core[id(spec)] = core

    sub_specs = split_softcores_by_capacity(
        specs,
        hw_types,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_neuron_splitting,
    )
    return [[spec_to_core[id(s)] for s in sub] for sub in sub_specs]


def _flush_scheduled_subsegments(
    *,
    cores: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    cores_config: Sequence[dict],
    weight_banks: dict,
    segment_index_start: int,
    segment_label_base: str,
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    all_reindex_maps: dict[int, dict[int, int]],
    stages: list[HybridStage],
) -> int:
    """Flush one IR segment, splitting by capacity into one or more stages.

    The layout-side splitter (``split_softcores_by_capacity``) produces an
    initial partition validated via ``pack_layout``.  The real hard-core
    packer (``SoftCoreMapping.map`` → ``greedy_pack_softcores`` with
    ``fuse_hardcores``) is authoritative for deployment: if it fails to
    pack a sub-segment the layout thought was fine, we halve that
    sub-segment and retry each half with a fresh pool, recursively, until
    it fits or it is a singleton softcore.  Singleton failure = genuine
    infeasibility; the error propagates unchanged.

    This halving fallback is *only* for layout-vs-real-packer divergence
    (e.g. greedy ordering choosing different core types on tight
    configs).  It never re-introduces latency-group splitting — each
    halved piece still contains the full latency stack that lived inside
    the sub-segment the layout produced.  If this fallback fires often,
    the fix is to make the layout-side estimator use the same packer
    path as the real flusher; this wrapper keeps the pipeline running in
    the meantime.
    """
    sub_segments = _split_segment_by_capacity(
        cores,
        cores_config,
        allow_coalescing=allow_coalescing,
        allow_neuron_splitting=allow_neuron_splitting,
    )
    if not sub_segments:
        return segment_index_start

    def _flush_or_halve(sub_cores, sub_label, sub_idx):
        if all_reindex_maps:
            sub_cores_reindexed = _reindex_nodes(sub_cores, all_reindex_maps)
        else:
            sub_cores_reindexed = sub_cores
        try:
            seg_stages, seg_reindex = _flush_scheduled_segment(
                current_neural=sub_cores_reindexed,
                consumed_by=consumed_by,
                cores_config=cores_config,
                weight_banks=weight_banks,
                segment_index=segment_index_start + sub_idx,
                segment_label=sub_label,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_coalescing=allow_coalescing,
            )
            stages.extend(seg_stages)
            all_reindex_maps.update(seg_reindex)
            return 1
        except RuntimeError:
            if len(sub_cores) <= 1:
                raise
            mid = len(sub_cores) // 2
            left_count = _flush_or_halve(sub_cores[:mid], f"{sub_label}_h0", sub_idx)
            right_count = _flush_or_halve(sub_cores[mid:], f"{sub_label}_h1", sub_idx + left_count)
            return left_count + right_count

    offset = 0
    for sub_idx, sub_cores in enumerate(sub_segments):
        label = segment_label_base if len(sub_segments) == 1 else f"{segment_label_base}_cap{sub_idx}"
        offset += _flush_or_halve(sub_cores, label, offset)

    return segment_index_start + offset


def _build_scheduled(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
    allow_coalescing: bool = False,
) -> None:
    """Scheduled compilation: fresh core pool per pass, over-sized segments
    are split at capacity boundaries so each emitted neural stage fits in a
    single pass."""
    segment_index = 0

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                segment_index = _flush_scheduled_subsegments(
                    cores=current_neural,
                    consumed_by=consumed_by,
                    cores_config=cores_config,
                    weight_banks=ir_graph.weight_banks,
                    segment_index_start=segment_index,
                    segment_label_base=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                    all_reindex_maps=all_reindex_maps,
                    stages=stages,
                )
                current_neural = []

            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        _flush_scheduled_subsegments(
            cores=current_neural,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=ir_graph.weight_banks,
            segment_index_start=segment_index,
            segment_label_base="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            all_reindex_maps=all_reindex_maps,
            stages=stages,
        )

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
    """Build a neural-only IRGraph with external sources remapped to segment inputs."""
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
    """Shallow-copy nodes; fresh input_sources; share immutable weight tensors."""
    result = []
    for n in nodes:
        n2 = copy.copy(n)
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
    """Apply compaction reindex maps to IRSource objects in place."""
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
    use_legacy_softcore_flush: bool = False,
) -> tuple[HybridStage, dict[int, dict[int, int]]]:
    """Pack a neural segment using cores drawn from shared_pool."""
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
    if use_legacy_softcore_flush:
        soft = ir_graph_to_soft_core_mapping(seg_graph)
    else:
        from mimarsinan.mapping.neural_segment_packing import (
            neural_segment_to_soft_core_mapping,
        )

        soft = neural_segment_to_soft_core_mapping(seg_graph, weight_banks)

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
        group_ids: dict[object, int] = {}
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
    """Verify every wide NeuralCore's coalescing group fits in one core type's count."""
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
            from mimarsinan.mapping.coalescing import coalescing_fragment_count

            n_coalesce = coalescing_fragment_count(n_weight_axons, ct_axons)
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
    use_legacy_softcore_flush: bool = False,
) -> tuple[list[HybridStage], dict[int, dict[int, int]]]:
    """One segment → one HybridStage on a fresh hardware pool."""
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
        use_legacy_softcore_flush=use_legacy_softcore_flush,
    )
    stage.schedule_segment_index = segment_index
    stage.schedule_pass_index = 0

    return [stage], seg_reindex_all


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool = False,
    allow_scheduling: bool = False,
    allow_coalescing: bool = False,
    use_legacy_softcore_flush: bool = False,
) -> HybridHardCoreMapping:
    """Compile a unified IRGraph into a HybridHardCoreMapping."""

    from mimarsinan.mapping.ir_segmentation import build_ir_consumed_by

    consumed_by = build_ir_consumed_by(ir_graph)

    stages: list[HybridStage] = []

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
            use_legacy_softcore_flush=use_legacy_softcore_flush,
        )
    else:
        _build_single_pool(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
            use_legacy_softcore_flush=use_legacy_softcore_flush,
        )

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

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
    use_legacy_softcore_flush: bool = False,
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
                if all_reindex_maps:
                    current_neural = _reindex_nodes(current_neural, all_reindex_maps)
                stage, seg_reindex = _flush_neural_segment(
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    shared_pool=shared_pool,
                    weight_banks=ir_graph.weight_banks,
                    name=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                    use_legacy_softcore_flush=use_legacy_softcore_flush,
                )
                stages.append(stage)
                all_reindex_maps.update(seg_reindex)
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
        if all_reindex_maps:
            current_neural = _reindex_nodes(current_neural, all_reindex_maps)
        stage, seg_reindex = _flush_neural_segment(
            current_neural=current_neural,
            consumed_by=consumed_by,
            shared_pool=shared_pool,
            weight_banks=ir_graph.weight_banks,
            name="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
            use_legacy_softcore_flush=use_legacy_softcore_flush,
        )
        stages.append(stage)
        all_reindex_maps.update(seg_reindex)


_SPLIT_FALLBACK_LOGGED = False


def _split_segment_by_capacity(
    cores: list[NeuralCore],
    cores_config: Sequence[dict],
    *,
    allow_coalescing: bool,
    allow_neuron_splitting: bool,
    ir_graph: IRGraph | None = None,
    hardware_bias: bool = False,
) -> list[list[NeuralCore]]:
    """Split IR NeuralCores by hardware capacity via split_softcores_by_capacity."""
    global _SPLIT_FALLBACK_LOGGED
    if not cores:
        return []

    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
    from mimarsinan.mapping.mapping_structure import compute_core_input_count
    from mimarsinan.mapping.schedule_partitioner import split_softcores_by_capacity

    hw_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]

    layout_specs = list(getattr(ir_graph, "layout_softcores", None) or [])
    use_layout = bool(layout_specs) and all(
        getattr(c, "layout_softcore_index", None) is not None for c in cores
    )

    specs: list[LayoutSoftCoreSpec] = []
    spec_to_core: dict[int, NeuralCore] = {}
    for idx, core in enumerate(cores):
        if use_layout:
            sc_idx = int(core.layout_softcore_index)  # type: ignore[arg-type]
            spec = layout_specs[sc_idx]
        else:
            if not _SPLIT_FALLBACK_LOGGED:
                import logging

                logging.getLogger(__name__).warning(
                    "Scheduled split: reconstructing LayoutSoftCoreSpec from NeuralCore "
                    "(missing IRGraph.layout_softcores); input_count may diverge from SCM."
                )
                _SPLIT_FALLBACK_LOGGED = True
            lat = int(core.latency) if core.latency is not None else 0
            pi = getattr(core, "perceptron_index", None)
            tg = int(pi) if pi is not None else -(idx + 1)
            n_sources = int(len(core.input_sources.flatten()))
            has_bias_axon = core.hardware_bias is None and any(
                getattr(s, "is_always_on", lambda: False)() for s in core.input_sources.flatten()
            )
            in_count = compute_core_input_count(
                n_sources - (1 if has_bias_axon else 0),
                has_bias=has_bias_axon,
                hardware_bias=hardware_bias,
            )
            spec = LayoutSoftCoreSpec(
                input_count=in_count,
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
    ir_graph: IRGraph | None = None,
    hardware_bias: bool = False,
    use_legacy_softcore_flush: bool = False,
) -> int:
    """Flush one IR segment; split by capacity when scheduling."""
    sub_segments = _split_segment_by_capacity(
        cores,
        cores_config,
        allow_coalescing=allow_coalescing,
        allow_neuron_splitting=allow_neuron_splitting,
        ir_graph=ir_graph,
        hardware_bias=hardware_bias,
    )
    if not sub_segments:
        return segment_index_start

    for sub_idx, sub_cores in enumerate(sub_segments):
        label = segment_label_base if len(sub_segments) == 1 else f"{segment_label_base}_cap{sub_idx}"
        sub_cores_reindexed = (
            _reindex_nodes(sub_cores, all_reindex_maps) if all_reindex_maps else sub_cores
        )
        seg_stages, seg_reindex = _flush_scheduled_segment(
            current_neural=sub_cores_reindexed,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=weight_banks,
            segment_index=segment_index_start + sub_idx,
            segment_label=label,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            use_legacy_softcore_flush=use_legacy_softcore_flush,
        )
        stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)

    return segment_index_start + len(sub_segments)


def _build_scheduled(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
    allow_coalescing: bool = False,
    use_legacy_softcore_flush: bool = False,
) -> None:
    """Scheduled compilation: fresh core pool per pass."""
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
                    ir_graph=ir_graph,
                    use_legacy_softcore_flush=use_legacy_softcore_flush,
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
            ir_graph=ir_graph,
            use_legacy_softcore_flush=use_legacy_softcore_flush,
        )

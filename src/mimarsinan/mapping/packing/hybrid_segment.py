from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping, compact_soft_core_mapping
from mimarsinan.mapping.packing.hybrid_types import HybridStage, SegmentIOSlice


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
    )
    stage.schedule_segment_index = segment_index
    stage.schedule_pass_index = 0

    return [stage], seg_reindex_all



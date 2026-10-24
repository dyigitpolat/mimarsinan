from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCore
from mimarsinan.mapping.packing.hybrid_types import SegmentIOSlice
from mimarsinan.mapping.platform.platform_constraints import PER_AXON, SIGN_GRANULARITY_FIELD


def _make_available_hardware_cores(cores_config: Sequence[dict]) -> list[HardCore]:
    """One HardCore per declared core; a per-axon type's cores carry the declared-only ``weight_sign_granularity``."""
    available_hardware_cores: list[HardCore] = []
    for core_type in cores_config:
        count = int(core_type["count"])
        max_axons = int(core_type["max_axons"])
        max_neurons = int(core_type["max_neurons"])
        has_bias = bool(core_type.get("has_bias", True))
        per_axon = core_type.get(SIGN_GRANULARITY_FIELD) == PER_AXON
        for _ in range(count):
            core = HardCore(max_axons, max_neurons, has_bias_capability=has_bias)
            if per_axon:
                core.weight_sign_granularity = PER_AXON
            available_hardware_cores.append(core)
    return available_hardware_cores


def _remap_external_sources_to_segment_inputs(
    *,
    nodes: list[NeuralCore],
    segment_output_refs: np.ndarray,
    weight_banks: dict | None = None,
) -> tuple[IRGraph, list[SegmentIOSlice]]:
    """Build a neural-only IRGraph with external sources remapped to segment inputs."""
    if weight_banks is None:
        weight_banks = {}
    node_ids = {n.id for n in nodes}

    for src in segment_output_refs.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
            raise ValueError(
                "Segment output refs reference external node "
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

    new_nodes: list[IRNode] = []
    for n in nodes:
        n2 = copy.copy(n)
        flat = [remap_src(src) for src in n.input_sources.flatten()]
        n2.input_sources = np.array(flat, dtype=object).reshape(n.input_sources.shape)
        new_nodes.append(n2)

    graph = IRGraph(
        nodes=new_nodes,
        output_sources=np.array(segment_output_refs, dtype=object),
        weight_banks=weight_banks,
    )
    graph._is_segment_subgraph = True  # pyright: ignore[reportAttributeAccessIssue] — dynamic marker read via getattr in boundary_policy
    return graph, input_map


def _check_no_split_coalescing_groups(nodes: list) -> None:
    """Assert every coalescing group with partial cores in this segment also has its accumulator here."""
    group_roles: dict[int, list[str]] = {}
    for n in nodes:
        if isinstance(n, NeuralCore):
            gid = getattr(n, "coalescing_group_id", None)
            role = getattr(n, "coalescing_role", None)
            if gid is not None and role is not None:
                group_roles.setdefault(gid, []).append(role)

    for gid, roles in group_roles.items():
        if "master" in roles:
            continue

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
    """Apply compaction reindex maps to IRSource objects in place; an entry the map keeps at its index keeps its object
    (no IRSource is written in place anywhere, so sharing it is sharing a value: coverage W9)."""
    for i, src in enumerate(sources.flatten()):
        if not isinstance(src, IRSource) or src.node_id < 0:
            continue
        remap = reindex_maps.get(src.node_id)
        if remap is None:
            continue
        new_idx = remap.get(src.index)
        if new_idx is None:
            sources.flat[i] = IRSource(node_id=-1, index=0)
        elif new_idx != src.index:
            sources.flat[i] = IRSource(node_id=src.node_id, index=new_idx)


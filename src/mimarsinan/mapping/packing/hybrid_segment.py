from __future__ import annotations

from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping, compact_soft_core_mapping
from mimarsinan.mapping.packing.hybrid_types import HybridStage, SegmentIOSlice

from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _check_no_split_coalescing_groups,
    _make_available_hardware_cores,
    _remap_external_sources_to_segment_inputs,
)

def _flush_neural_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    shared_pool: list[HardCore],
    weight_banks: dict,
    name: str,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = True,
    skip_coalescing_check: bool = False,
    identity: bool = False,
) -> tuple[HybridStage, dict[int, dict[int, int]]]:
    """Pack a neural segment using cores drawn from shared_pool.

    ``identity=True`` skips the pool entirely: each soft core gets its own
    exactly-sized hard core (``HardCoreMapping.map_identity``).
    """
    segment_node_ids = {n.id for n in current_neural}

    if not skip_coalescing_check:
        _check_no_split_coalescing_groups(current_neural)

    output_nodes: list[NeuralCore] = []
    for n in current_neural:
        consumers = consumed_by.get(n.id, set())
        if any(c not in segment_node_ids for c in consumers):
            output_nodes.append(n)

    segment_output_refs_list: list[IRSource] = []
    for n in output_nodes:
        for idx in range(n.get_output_count()):
            segment_output_refs_list.append(IRSource(node_id=n.id, index=idx))

    if output_nodes and not segment_output_refs_list:
        raise ValueError(
            "Segment has output_nodes but 0 output refs (all get_output_count() are 0). "
            "Output layer was over-pruned; at least one output neuron must remain."
        )

    segment_output_refs = np.array(segment_output_refs_list, dtype=object)

    seg_graph, input_map = _remap_external_sources_to_segment_inputs(
        nodes=current_neural,
        segment_output_refs=segment_output_refs,
        weight_banks=weight_banks,
    )
    from mimarsinan.mapping.packing.neural_segment_packing import (
        neural_segment_to_soft_core_mapping,
    )

    soft = neural_segment_to_soft_core_mapping(seg_graph, weight_banks)

    reindex_maps = compact_soft_core_mapping(soft.cores, soft.output_sources)

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
        if identity:
            hard.map_identity(soft)
        else:
            hard.map(soft, allow_neuron_splitting=allow_neuron_splitting,
                     allow_coalescing=allow_coalescing)
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
            from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count

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
        allow_coalescing=allow_coalescing,
        skip_coalescing_check=True,
    )
    stage.schedule_segment_index = segment_index
    stage.schedule_pass_index = 0

    return [stage], seg_reindex_all



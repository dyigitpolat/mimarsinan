"""Pack a neural IR segment into softcores for hard-core bin-packing."""

from __future__ import annotations

from typing import Any, Dict

from mimarsinan.mapping.ir import IRGraph, NeuralCore, ir_source_to_spike_source
from mimarsinan.mapping.support.spike_source_columns import SpikeSourceColumns


def neural_segment_to_soft_core_mapping(
    seg_graph: IRGraph,
    weight_banks: Dict[int, Any],
    *,
    columnar: bool = False,
):
    """Convert a neural-only segment ``IRGraph`` to ``SoftCoreMapping`` via ``neural_core_to_soft_core``
    (``columnar``: every soft core's axon sources and the segment's output sources as ``SpikeSourceColumns``)."""
    from mimarsinan.mapping.ir import neural_core_to_soft_core
    from mimarsinan.mapping.packing.softcore.soft_core_mapper import SoftCoreMapping

    compute_ops = seg_graph.get_compute_ops()
    if compute_ops:
        raise ValueError(
            f"neural_segment_to_soft_core_mapping: segment has {len(compute_ops)} ComputeOp nodes"
        )

    soft = SoftCoreMapping()
    soft.weight_banks = {
        int(bid): bank.core_matrix
        for bid, bank in (weight_banks or {}).items()
        if hasattr(bank, "core_matrix")
    }

    for node in seg_graph.nodes:
        if isinstance(node, NeuralCore):
            sc = neural_core_to_soft_core(node, graph=seg_graph, columnar=columnar)
            sc.threshold = node.threshold
            sc.latency = node.latency
            soft.cores.append(sc)

    outputs = seg_graph.output_sources.flatten()
    soft.output_sources = (
        SpikeSourceColumns.from_ir_sources(outputs) if columnar
        else [ir_source_to_spike_source(src) for src in outputs]
    )
    return soft

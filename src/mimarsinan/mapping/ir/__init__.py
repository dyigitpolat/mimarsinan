"""Unified IR: types, graph container, and legacy conversions."""

from mimarsinan.mapping.ir.graph import IRGraph
from mimarsinan.mapping.ir.legacy_convert import (
    ir_graph_to_soft_core_mapping,
    ir_source_to_spike_source,
    neural_core_to_soft_core,
    soft_core_mapping_to_ir_graph,
    soft_core_to_neural_core,
    spike_source_to_ir_source,
)
from mimarsinan.mapping.ir.types import (
    ComputeOp,
    IRNode,
    IRSource,
    NeuralCore,
    WeightBank,
)

__all__ = [
    "IRGraph",
    "ir_graph_to_soft_core_mapping",
    "ir_source_to_spike_source",
    "neural_core_to_soft_core",
    "soft_core_mapping_to_ir_graph",
    "soft_core_to_neural_core",
    "spike_source_to_ir_source",
    "ComputeOp",
    "IRNode",
    "IRSource",
    "NeuralCore",
    "WeightBank",
]

"""Model-to-hardware mapping: IR, mapper graph, softcore/hardcore mappings."""

from mimarsinan.mapping.ir import (
    IRSource,
    IRNode,
    NeuralCore,
    ComputeOp,
    IRGraph,
    WeightBank,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
    compute_psum_params,
)
from mimarsinan.mapping.packing.softcore import SoftCore, HardCore, HardCoreMapping
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    SegmentIOSlice,
    HybridStage,
    HybridHardCoreMapping,
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.graph.pruning_propagation import compute_propagated_pruned_rows_cols
from mimarsinan.mapping.pruning.graph import (
    GlobalPruningResult,
    compute_global_pruned_sets,
)
from mimarsinan.mapping.pruning.ir_pruning_analysis import compute_graph_io_exemption


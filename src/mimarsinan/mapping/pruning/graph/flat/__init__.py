"""Flat wave engine: the elimination analysis over arrays instead of objects.

The production fixpoint (docs/elimination_flat_engine_plan.md). Semantics are
defined by the reference loop in ``pruning_graph_core``; the unit suite holds
both engines to bit-identical fingerprints, and the reference stays available
in production as the opt-in ``elimination_analysis_cross_check`` debug tool.
"""

from mimarsinan.mapping.pruning.graph.flat.state import FlatState, build_flat_state
from mimarsinan.mapping.pruning.graph.flat.kernels import (
    flat_cross_core_dead_axons,
    flat_orphan_neurons,
)

__all__ = [
    "FlatState",
    "build_flat_state",
    "flat_cross_core_dead_axons",
    "flat_orphan_neurons",
]

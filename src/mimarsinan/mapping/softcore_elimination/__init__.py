"""[W6b] Weight-cell elimination in the MAPPED SOFTCORES — the headline metric.

Total model compression is not what structured elimination targets: the claim
is about the weights that actually occupy crossbar softcores. This package
measures that directly, per run, in two labelled views (per-crossbar occupancy
and physical weight storage) and under every propagation arm that ran, so the
paper table and the C1 arm-increment evidence are one artifact.
"""

from mimarsinan.mapping.softcore_elimination.build import (
    REALIZED_ARM,
    build_view,
    report_from_arms,
    report_from_pruned_ir_graph,
)
from mimarsinan.mapping.softcore_elimination.facts import (
    BankFacts,
    InstanceFacts,
    SoftcoreEliminationError,
    SoftcoreFacts,
    facts_from_masks,
    facts_from_pruning_result,
)
from mimarsinan.mapping.softcore_elimination.markdown import (
    render_softcore_elimination_markdown,
    write_softcore_elimination_markdown,
)
from mimarsinan.mapping.softcore_elimination.naming import (
    softcore_group_name,
    softcore_layer_name,
)
from mimarsinan.mapping.softcore_elimination.report import (
    EliminationView,
    SoftcoreEliminationReport,
    summarize_softcore_elimination,
    write_softcore_elimination_record,
)
from mimarsinan.mapping.softcore_elimination.types import (
    GEOMETRY_AS_STORED,
    GEOMETRY_PRE_ELIMINATION,
    SOFTCORE_ELIMINATION_RECORD_FILENAME,
    SOFTCORE_ELIMINATION_TABLE_FILENAME,
    GroupElimination,
    StorageElimination,
)

__all__ = [
    "BankFacts",
    "EliminationView",
    "GEOMETRY_AS_STORED",
    "GEOMETRY_PRE_ELIMINATION",
    "GroupElimination",
    "InstanceFacts",
    "REALIZED_ARM",
    "SOFTCORE_ELIMINATION_RECORD_FILENAME",
    "SOFTCORE_ELIMINATION_TABLE_FILENAME",
    "SoftcoreEliminationError",
    "SoftcoreEliminationReport",
    "SoftcoreFacts",
    "StorageElimination",
    "build_view",
    "facts_from_masks",
    "facts_from_pruning_result",
    "render_softcore_elimination_markdown",
    "report_from_arms",
    "report_from_pruned_ir_graph",
    "softcore_group_name",
    "softcore_layer_name",
    "summarize_softcore_elimination",
    "write_softcore_elimination_markdown",
    "write_softcore_elimination_record",
]

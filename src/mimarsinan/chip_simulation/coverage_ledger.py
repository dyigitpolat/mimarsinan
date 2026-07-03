"""Hypervolume coverage ledger: genericity as a measured covered/claimed fraction over ledger rows.

Compatibility façade — the implementation lives in ``hypervolume_axes`` (axis model),
``hypervolume_cells`` (cell key + claims), ``coverage_rows`` (row classification), and
``coverage_reporting`` (aggregation).
"""

from mimarsinan.chip_simulation.coverage_reporting import (
    AttributionFidelity,
    CoverageReport,
    FlagMetadata,
    KNOWN_CRACKED_REGIONS,
    coverage_report,
)
from mimarsinan.chip_simulation.coverage_rows import (
    PLACEMENT_FIXABLE_DEFAULT_OWNER,
    PLACEMENT_FIXABLE_FIX_PATH,
    CoverageStatus,
    classify_validity_tier,
    row_to_cell,
    row_to_cells,
)
from mimarsinan.chip_simulation.hypervolume_axes import (
    AXES,
    AXIS_WILDCARD,
    AxisKind,
    HypervolumeAxis,
    ScreeningStatus,
    active_axes,
    collapse_orthogonal_axes,
    collapsed_axis_representatives,
    interacting_axes,
)
from mimarsinan.chip_simulation.hypervolume_cells import (
    HypervolumeCell,
    cell_covers,
    claimed_subproduct,
    honest_claimed_subproduct,
)

__all__ = [
    "AxisKind",
    "ScreeningStatus",
    "AttributionFidelity",
    "HypervolumeAxis",
    "AXES",
    "AXIS_WILDCARD",
    "collapse_orthogonal_axes",
    "collapsed_axis_representatives",
    "interacting_axes",
    "active_axes",
    "HypervolumeCell",
    "CoverageStatus",
    "FlagMetadata",
    "PLACEMENT_FIXABLE_DEFAULT_OWNER",
    "PLACEMENT_FIXABLE_FIX_PATH",
    "KNOWN_CRACKED_REGIONS",
    "classify_validity_tier",
    "claimed_subproduct",
    "honest_claimed_subproduct",
    "cell_covers",
    "row_to_cell",
    "row_to_cells",
    "CoverageReport",
    "coverage_report",
]

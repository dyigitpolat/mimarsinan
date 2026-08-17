from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
from mimarsinan.mapping.layout.layout_plan import LayoutPlan


def stats_dict_from_hybrid_mapping(
    mapping: Any, core_types: Optional[Sequence[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Wizard-shaped mapping performance stats from a compiled hybrid mapping.

    Delegates to :meth:`LayoutPlan.from_hybrid_mapping`, the same stats engine
    as the wizard layout path (single source of truth). ``core_types`` is the
    DECLARED chip: without it, chip-relative figures (``chip_occupancy_pct``)
    degenerate to allocated-relative ones — the R0 finding.
    """
    plan = LayoutPlan.from_hybrid_mapping(mapping, core_types=core_types)
    if plan is None:
        return None
    return plan.stats.to_dict()

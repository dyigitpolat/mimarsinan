from __future__ import annotations
from typing import Any, Dict, Optional


def stats_dict_from_hybrid_mapping(mapping: Any) -> Optional[Dict[str, Any]]:
    """Wizard-shaped mapping performance stats from a compiled hybrid mapping.

    Delegates to :meth:`LayoutPlan.from_hybrid_mapping`, so the deployment
    mapping reports statistics through the exact same
    ``build_stats_from_packing_result`` engine as the wizard layout path
    (single source of truth).  Coalescing-group and split-softcore counts are
    derived from placement provenance rather than hardcoded to zero.
    """
    from mimarsinan.mapping.layout.layout_plan import LayoutPlan

    plan = LayoutPlan.from_hybrid_mapping(mapping)
    if plan is None:
        return None
    return plan.stats.to_dict()

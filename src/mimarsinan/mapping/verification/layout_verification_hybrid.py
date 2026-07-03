from __future__ import annotations
from typing import Any, Dict, Optional
from mimarsinan.mapping.layout.layout_plan import LayoutPlan


def stats_dict_from_hybrid_mapping(mapping: Any) -> Optional[Dict[str, Any]]:
    """Wizard-shaped mapping performance stats from a compiled hybrid mapping.

    Delegates to :meth:`LayoutPlan.from_hybrid_mapping`, the same stats engine
    as the wizard layout path (single source of truth).
    """
    plan = LayoutPlan.from_hybrid_mapping(mapping)
    if plan is None:
        return None
    return plan.stats.to_dict()

"""Phase 3/4: deployment-side mapping stats are derived from the same
``build_stats_from_packing_result`` engine as the wizard layout path (via
``LayoutPlan.from_hybrid_mapping``), so the GUI's real-mapping stats use one
schema and no longer hardcode coalescing / split counts to zero."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from integration._placement_signature import build_hybrid_for_config  # noqa: E402

from mimarsinan.mapping.layout.layout_plan import LayoutPlan  # noqa: E402
from mimarsinan.mapping.verification.layout_verification_hybrid import (  # noqa: E402
    stats_dict_from_hybrid_mapping,
)
from mimarsinan.mapping.verification.layout_verification_types import (  # noqa: E402
    LayoutVerificationStats,
)


_FULL_SCHEMA = set(LayoutVerificationStats.__dataclass_fields__.keys())


def test_hybrid_stats_use_full_layout_schema():
    for name in ("dense_two_core", "neuron_split", "scheduled_wide", "pruned", "multi_segment"):
        stats = stats_dict_from_hybrid_mapping(build_hybrid_for_config(name))
        assert stats is not None
        assert set(stats) == _FULL_SCHEMA, f"{name}: schema drift"
        assert stats["feasible"] is True


def test_split_softcore_count_no_longer_hardcoded_zero():
    """The previous hybrid stats hardcoded split_softcore_count=0; the unified
    path derives it from placement provenance."""
    stats = stats_dict_from_hybrid_mapping(build_hybrid_for_config("neuron_split"))
    assert stats["split_softcore_count"] >= 1
    assert stats["split_cores"] >= 1


def test_stats_dict_matches_layout_plan():
    for name in ("dense_two_core", "neuron_split", "multi_segment"):
        hm = build_hybrid_for_config(name)
        plan = LayoutPlan.from_hybrid_mapping(hm)
        assert plan is not None
        assert stats_dict_from_hybrid_mapping(hm) == plan.stats.to_dict()

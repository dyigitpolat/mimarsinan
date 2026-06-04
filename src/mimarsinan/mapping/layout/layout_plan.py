"""``LayoutPlan`` -- the single shape-only artifact tying segmentation, soft-core
specs, packing, and stats together.

A ``LayoutPlan`` can be built two ways and both feed the *same*
``LayoutVerificationStats`` computation, so the wizard miniview and the
deployed hybrid mapping can never report divergent placement statistics:

- :func:`build_layout_plan` -- from a shape-only verification result + hardware
  core types (the wizard / NAS / snapshot "planned" path).
- :meth:`LayoutPlan.from_hybrid_mapping` -- from a compiled
  ``HybridHardCoreMapping`` by deriving each used hardcore's snapshot and the
  coalescing / split groups from placement provenance (the deployment path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.mapping.layout.layout_types import (
    LayoutCoreSnapshot,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)


@dataclass(frozen=True)
class LayoutPlan:
    """Shape-only placement plan + derived statistics."""

    feasible: bool
    stats: LayoutVerificationStats
    layout_preview: Optional[dict] = None
    host_side_segment_count: int = 0
    packing_result: Optional[LayoutPackingResult] = None
    schedule_info: Optional[dict] = None
    errors: Tuple[str, ...] = ()
    field_errors: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_hybrid_mapping(cls, mapping: Any) -> Optional["LayoutPlan"]:
        """Derive a ``LayoutPlan`` from a compiled ``HybridHardCoreMapping``.

        Snapshots come from each neural segment's *used* hardcores; coalescing
        and split distributions come from the per-softcore placement provenance
        recorded by ``HardCoreMapping.merge_softcore_into``.  The resulting
        ``LayoutPackingResult`` is fed through the shared
        :func:`build_stats_from_packing_result`, so deployment stats use the
        identical formulas as the wizard layout path.
        """
        from mimarsinan.mapping.verification.layout_verification_packing import (
            build_stats_from_packing_result,
        )

        stages = getattr(mapping, "stages", None)
        if not stages:
            return None

        snapshots: List[LayoutCoreSnapshot] = []
        coalescing_counts: Dict[Any, int] = {}
        split_fragments_per_sc: Dict[Any, int] = {}
        total_placements = 0
        schedule_pass_present = False
        max_pass_by_segment: Dict[int, int] = {}

        for stage in stages:
            if getattr(stage, "kind", None) != "neural":
                continue
            hcm = getattr(stage, "hard_core_mapping", None)
            if hcm is None:
                continue

            sched_idx = getattr(stage, "schedule_pass_index", None)
            seg_idx = getattr(stage, "schedule_segment_index", None) or 0
            if sched_idx is not None:
                schedule_pass_present = True
                max_pass_by_segment[seg_idx] = max(
                    max_pass_by_segment.get(seg_idx, 0), sched_idx + 1
                )

            placements_per_core = getattr(hcm, "soft_core_placements_per_hard_core", []) or []
            for core_idx, hc in enumerate(getattr(hcm, "cores", []) or []):
                placements = (
                    placements_per_core[core_idx]
                    if core_idx < len(placements_per_core)
                    else []
                )
                total_placements += len(placements)
                used_area = sum(
                    int(p.get("axons", 0)) * int(p.get("neurons", 0))
                    for p in placements
                )
                ax_total = int(getattr(hc, "axons_per_core", 0))
                neu_total = int(getattr(hc, "neurons_per_core", 0))
                snapshots.append(
                    LayoutCoreSnapshot(
                        axons_per_core=ax_total,
                        neurons_per_core=neu_total,
                        used_axons=max(0, ax_total - int(getattr(hc, "available_axons", 0))),
                        used_neurons=max(0, neu_total - int(getattr(hc, "available_neurons", 0))),
                        used_area=int(used_area),
                        softcore_count=len(placements),
                    )
                )
                for p in placements:
                    cg = p.get("coalescing_group_id")
                    if cg is not None:
                        coalescing_counts[cg] = coalescing_counts.get(cg, 0) + 1
                    sg = p.get("split_group_id")
                    if sg is not None:
                        key = (p.get("ir_node_id"), p.get("split_group_id"))
                        # Count fragments per original split softcore (by ir node).
                        node_key = p.get("ir_node_id")
                        split_fragments_per_sc[node_key] = (
                            split_fragments_per_sc.get(node_key, 0) + 1
                        )

        if not snapshots:
            return None

        coalescing_group_sizes = tuple(
            c for c in coalescing_counts.values() if c > 1
        )
        # A softcore split into N fragments was split N-1 times.
        split_counts_per_sc = tuple(
            n - 1 for n in split_fragments_per_sc.values() if n > 1
        )

        cores_used = len(snapshots)
        total_capacity = sum(s.capacity for s in snapshots)
        used_area = sum(s.used_area for s in snapshots)
        unused_total = int(total_capacity - used_area)

        packing = LayoutPackingResult(
            feasible=True,
            cores_used=cores_used,
            total_capacity=int(total_capacity),
            used_area=int(used_area),
            unused_area_total=unused_total,
            avg_unused_area_per_core=(
                float(unused_total / cores_used) if cores_used else float("inf")
            ),
            unusable_space_total=0,
            avg_unusable_space_per_core=0.0,
            used_core_softcore_counts=tuple(s.softcore_count for s in snapshots),
            used_core_snapshots=tuple(snapshots),
            coalesced_fragment_count=sum(coalescing_group_sizes),
            split_fragment_count=sum(split_counts_per_sc),
            coalescing_group_sizes=coalescing_group_sizes or None,
            split_counts_per_sc=split_counts_per_sc or None,
        )

        stats = build_stats_from_packing_result(
            packing,
            num_original_softcores=total_placements or cores_used,
            softcores=None,
            core_types=None,
        )
        stats_dict = stats.to_dict()

        # Hybrid-specific fields the shape-only packing cannot know.
        # No core_types are available from a compiled mapping, so chip-level
        # accounting falls back to the used-core total (legacy behaviour).
        stats_dict["total_hw_cores"] = cores_used
        stats_dict["neural_segment_count"] = len(mapping.get_neural_segments())
        if schedule_pass_present:
            schedule_pass_count = sum(max_pass_by_segment.values())
            stats_dict["schedule_pass_count"] = int(schedule_pass_count)
            stats_dict["schedule_sync_count"] = max(
                0, schedule_pass_count - len(max_pass_by_segment)
            )

        return cls(
            feasible=True,
            stats=LayoutVerificationStats(**stats_dict),
            packing_result=packing,
        )


def build_layout_plan(
    verification: Any,
    core_types: List[Dict[str, Any]],
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    allow_scheduling: bool = False,
) -> LayoutPlan:
    """Build a ``LayoutPlan`` from a shape-only ``MappingVerificationResult``.

    Routes through :func:`verify_hardware_config` (which packs and computes
    stats via the shared ``build_stats_from_packing_result``), so the wizard
    miniview consumes exactly the same stats engine as the deployment path.
    """
    from mimarsinan.mapping.verification.verifier import verify_hardware_config

    hw = verify_hardware_config(
        verification.softcores,
        core_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
        allow_scheduling=allow_scheduling,
    )

    stats_dict = dict(hw.get("stats") or {})
    return LayoutPlan(
        feasible=bool(hw.get("feasible", False)),
        stats=LayoutVerificationStats(
            **{k: v for k, v in stats_dict.items() if k in _STATS_FIELDS}
        )
        if stats_dict
        else _empty_layout_stats(),
        layout_preview=getattr(verification, "layout_preview", None),
        host_side_segment_count=getattr(verification, "host_side_segment_count", 0),
        packing_result=hw.get("packing_result"),
        schedule_info=hw.get("schedule_info"),
        errors=tuple(hw.get("errors") or ()),
        field_errors=dict(hw.get("field_errors") or {}),
    )


_STATS_FIELDS = set(LayoutVerificationStats.__dataclass_fields__.keys())


def _empty_layout_stats() -> LayoutVerificationStats:
    from mimarsinan.mapping.verification.layout_verification_packing import _empty_stats

    return _empty_stats(feasible=False)

"""The registered introspection surface: seven payloads over two view kinds.

Each payload declares which views can answer it; a view that cannot simply does
not serve it (``placement`` needs a sealed record; ``softcores`` is the
shape-only candidate's own table, whose sealed counterpart IS ``placement``).
"""

from __future__ import annotations

from typing import Any, Optional

from mimarsinan.deployment_record.introspection import builders
from mimarsinan.deployment_record.introspection.payloads import (
    BankCompositionPayload,
    CapabilitiesPayload,
    LayerRollupPayload,
    LayoutStatsPayload,
    PlacementPayload,
    SchedulePayload,
    SoftcoresPayload,
)
from mimarsinan.deployment_record.introspection.registry import (
    IntrospectionRegistry,
    IntrospectionSpec,
)
from mimarsinan.deployment_record.introspection.views import (
    CANDIDATE_LAYOUT,
    DEPLOYMENT_RECORD,
    capability_bits_of,
)


def _candidate_rows(view: Any):
    return builders.softcore_rows_from_specs(view.softcores)


def _record_rows(view: Any):
    return builders.softcore_rows_from_placements(view.record.placement.softcores)


def _candidate_softcores(view: Any) -> Optional[SoftcoresPayload]:
    if not view.softcores:
        return None
    return SoftcoresPayload(softcores=_candidate_rows(view))


def _candidate_rollup(view: Any) -> Optional[LayerRollupPayload]:
    if not view.softcores:
        return None
    return LayerRollupPayload(layers=builders.rollup_rows(_candidate_rows(view)))


def _record_rollup(view: Any) -> Optional[LayerRollupPayload]:
    if not view.record.placement.softcores:
        return None
    return LayerRollupPayload(layers=builders.rollup_rows(_record_rows(view)))


def _candidate_banks(view: Any) -> Optional[BankCompositionPayload]:
    if not view.softcores:
        return None
    banks, unbanked = builders.bank_rows(_candidate_rows(view))
    return BankCompositionPayload(banks=banks, unbanked_softcore_count=unbanked)


def _record_banks(view: Any) -> Optional[BankCompositionPayload]:
    placement = view.record.placement
    if not placement.softcores:
        return None
    banks, unbanked = builders.bank_rows(_record_rows(view), placement.banks)
    return BankCompositionPayload(banks=banks, unbanked_softcore_count=unbanked)


def _record_placement(view: Any) -> Optional[PlacementPayload]:
    placements = view.record.placement.softcores
    if not placements:
        return None
    return PlacementPayload(placements=builders.placement_rows(placements))


def _candidate_schedule(view: Any) -> Optional[SchedulePayload]:
    stats = view.layout
    if stats is None:
        return None
    return SchedulePayload(
        max_schedule_passes=view.capabilities.max_schedule_passes,
        pass_count=int(stats.schedule_pass_count),
        sync_count=int(stats.schedule_sync_count),
        max_cores_per_pass=int(stats.max_cores_per_pass),
        segments=builders.candidate_segment_rows(_candidate_rows(view)),
    )


def _record_schedule(view: Any) -> Optional[SchedulePayload]:
    schedule = view.record.schedule
    bits = capability_bits_of(view) or {}
    return SchedulePayload(
        max_schedule_passes=bits.get("max_schedule_passes"),
        pass_count=int(schedule.pass_count),
        sync_count=int(schedule.sync_count),
        max_cores_per_pass=None,
        segments=builders.record_segment_rows(schedule),
        reprogram_passes=int(schedule.reprogram_passes),
        reuse_passes=int(schedule.reuse_passes),
        compute_op_count=int(schedule.compute_op_count),
    )


def _capabilities(view: Any) -> Optional[CapabilitiesPayload]:
    bits = capability_bits_of(view)
    return None if bits is None else CapabilitiesPayload(bits=dict(bits))


def _candidate_layout_stats(view: Any) -> Optional[LayoutStatsPayload]:
    if view.layout is None:
        return None
    return LayoutStatsPayload(stats=view.layout.to_dict(), error=view.layout_error)


def _record_layout_stats(view: Any) -> Optional[LayoutStatsPayload]:
    return LayoutStatsPayload(stats=view.record.utilization.layout.to_dict())


def build_registry() -> IntrospectionRegistry:
    """The registered catalogue (a fresh registry; the module holds the shared one)."""
    registry = IntrospectionRegistry()
    registry.register(IntrospectionSpec(
        name="softcores",
        payload_type=SoftcoresPayload,
        requires="shape-only layout softcores (a search candidate)",
        doc=(
            "Every soft core the candidate emits: shape, residency class, latency "
            "tag, segment, the shared bank it reads, and the source layer that "
            "emitted it."
        ),
        builders={CANDIDATE_LAYOUT: _candidate_softcores},
    ))
    registry.register(IntrospectionSpec(
        name="layer_rollup",
        payload_type=LayerRollupPayload,
        requires="softcores or placements carrying perceptron identity",
        doc=(
            "Cores aggregated by REAL source-layer identity (perceptron_index): "
            "how many cores a layer emits, their area, extents, and banks."
        ),
        builders={CANDIDATE_LAYOUT: _candidate_rollup, DEPLOYMENT_RECORD: _record_rollup},
    ))
    registry.register(IntrospectionSpec(
        name="bank_composition",
        payload_type=BankCompositionPayload,
        requires="softcores or placements carrying bank identity",
        doc=(
            "Per shared weight bank: how many cores read it (the sharing degree "
            "weight-stationary scheduling exploits) and, on a sealed record, its "
            "real size in parameters."
        ),
        builders={CANDIDATE_LAYOUT: _candidate_banks, DEPLOYMENT_RECORD: _record_banks},
    ))
    registry.register(IntrospectionSpec(
        name="placement",
        payload_type=PlacementPayload,
        requires="a sealed record's placement fragment",
        doc=(
            "The softcore→hard-core assignment as deployed: which physical core, "
            "at which axon/neuron offset, in which pass of which segment."
        ),
        builders={DEPLOYMENT_RECORD: _record_placement},
    ))
    registry.register(IntrospectionSpec(
        name="schedule",
        payload_type=SchedulePayload,
        requires="a layout answer or a sealed schedule fragment",
        doc=(
            "The composed pass structure (residency-first, capacity fallback): "
            "total passes, sync barriers, per-segment structure, and (sealed) "
            "how many passes reprogram versus reuse resident weights."
        ),
        builders={
            CANDIDATE_LAYOUT: _candidate_schedule, DEPLOYMENT_RECORD: _record_schedule,
        },
    ))
    registry.register(IntrospectionSpec(
        name="capabilities",
        payload_type=CapabilitiesPayload,
        requires="a declared platform",
        doc=(
            "Every capability bit the platform declares — the complete set, so a "
            "reader can tell which permissions and scheduler produced this layout. "
            "max_axons/max_neurons are the EFFECTIVE per-core limits the mapper "
            "budgets against (the declared grid maximum, minus the bias axon when "
            "the cores carry no hardware bias); a platform declaring no core grid "
            "serves them as null."
        ),
        builders={CANDIDATE_LAYOUT: _capabilities, DEPLOYMENT_RECORD: _capabilities},
    ))
    registry.register(IntrospectionSpec(
        name="layout_stats",
        payload_type=LayoutStatsPayload,
        requires="a layout verification census",
        doc=(
            "The full layout census: utilisation, wasted axons/neurons, "
            "fragmentation, coalescing/split groups, schedule totals."
        ),
        builders={
            CANDIDATE_LAYOUT: _candidate_layout_stats,
            DEPLOYMENT_RECORD: _record_layout_stats,
        },
    ))
    return registry


INTROSPECTION_REGISTRY = build_registry()

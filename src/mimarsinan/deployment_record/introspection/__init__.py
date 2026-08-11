"""The introspection channel: typed, versioned payloads over the deployment artifact.

Optimizers (LLM-driven or classical) depend on THESE types and nothing else —
never on ``mapping`` internals — so the questions an agent may ask are a
declared, versioned surface rather than whatever a producer happens to expose.
"""

from mimarsinan.deployment_record.introspection.catalog import (
    INTROSPECTION_REGISTRY,
    build_registry,
)
from mimarsinan.deployment_record.introspection.payloads import (
    INTROSPECTION_FORMAT_VERSION,
    BankCompositionPayload,
    BankRow,
    CapabilitiesPayload,
    IntrospectionPayload,
    IntrospectionRow,
    LayerRollupPayload,
    LayerRollupRow,
    LayoutStatsPayload,
    PlacementPayload,
    PlacementRow,
    SchedulePayload,
    SegmentPassRow,
    SoftcoreRow,
    SoftcoresPayload,
    channel_envelope,
)
from mimarsinan.deployment_record.introspection.registry import (
    IntrospectionRegistry,
    IntrospectionSpec,
)
from mimarsinan.deployment_record.introspection.views import (
    CANDIDATE_LAYOUT,
    DEPLOYMENT_RECORD,
    CandidateLayoutView,
    IntrospectionView,
    RecordIntrospectionView,
)

__all__ = [
    "CANDIDATE_LAYOUT",
    "DEPLOYMENT_RECORD",
    "INTROSPECTION_FORMAT_VERSION",
    "INTROSPECTION_REGISTRY",
    "BankCompositionPayload",
    "BankRow",
    "CandidateLayoutView",
    "CapabilitiesPayload",
    "IntrospectionPayload",
    "IntrospectionRegistry",
    "IntrospectionRow",
    "IntrospectionSpec",
    "IntrospectionView",
    "LayerRollupPayload",
    "LayerRollupRow",
    "LayoutStatsPayload",
    "PlacementPayload",
    "PlacementRow",
    "RecordIntrospectionView",
    "SchedulePayload",
    "SegmentPassRow",
    "SoftcoreRow",
    "SoftcoresPayload",
    "build_registry",
    "channel_envelope",
]

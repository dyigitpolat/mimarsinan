"""Typed, versioned, provenance-carrying deployment-record fragment schemas."""

from mimarsinan.deployment_record.schema.accuracy import (
    AccuracyReadRecord,
    AccuracyRecord,
    AdaptationRecord,
    CertificateRecord,
    FtPassWallRecord,
)
from mimarsinan.deployment_record.schema.physics import (
    EnergyRecord,
    EnergyTermRecord,
    LatencyDecomposition,
    SegmentTimingRecord,
    TimingRecord,
)
from mimarsinan.deployment_record.schema.placement import (
    BankRecord,
    FloorplanRecord,
    PlacementRecord,
    SoftcorePlacementRecord,
    TileRecord,
)
from mimarsinan.deployment_record.schema.provenance import (
    Band,
    ModeledValue,
    Provenance,
)
from mimarsinan.deployment_record.schema.record import (
    DEPLOYMENT_RECORD_FILENAME,
    DEPLOYMENT_RECORD_FORMAT_VERSION,
    DeploymentRecord,
    RecordIdentity,
    load_deployment_record,
    save_deployment_record,
)
from mimarsinan.deployment_record.schema.schedule import (
    ComputeOpRecord,
    PassCarryRecord,
    ScheduleRecord,
    SegmentCoreRecord,
    SegmentRecord,
)
from mimarsinan.deployment_record.schema.traffic import (
    BoundaryTrafficRecord,
    NocLinkLoadRecord,
    NocTrafficRecord,
    TrafficRecord,
)
from mimarsinan.deployment_record.schema.utilization import (
    ComputePartitionRecord,
    CrossbarUtilizationRecord,
    LayoutStatsRecord,
    UtilizationRecord,
)

__all__ = [
    "AccuracyReadRecord",
    "AccuracyRecord",
    "AdaptationRecord",
    "BankRecord",
    "Band",
    "BoundaryTrafficRecord",
    "CertificateRecord",
    "ComputeOpRecord",
    "ComputePartitionRecord",
    "CrossbarUtilizationRecord",
    "DEPLOYMENT_RECORD_FILENAME",
    "DEPLOYMENT_RECORD_FORMAT_VERSION",
    "DeploymentRecord",
    "EnergyRecord",
    "EnergyTermRecord",
    "FloorplanRecord",
    "FtPassWallRecord",
    "LatencyDecomposition",
    "LayoutStatsRecord",
    "ModeledValue",
    "NocLinkLoadRecord",
    "NocTrafficRecord",
    "PlacementRecord",
    "Provenance",
    "RecordIdentity",
    "PassCarryRecord",
    "ScheduleRecord",
    "SegmentCoreRecord",
    "SegmentRecord",
    "SegmentTimingRecord",
    "SoftcorePlacementRecord",
    "TileRecord",
    "TimingRecord",
    "TrafficRecord",
    "UtilizationRecord",
    "load_deployment_record",
    "save_deployment_record",
]

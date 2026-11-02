"""SANA-FE run-record dataclasses."""
from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown
from mimarsinan.chip_simulation.sanafe.records.hardware import (
    SanafeArchGeometry,
    SanafeCascadePoint,
    SanafeConnectivityEdge,
    SanafeCoreRecord,
    SanafeCriticalCore,
    SanafeCycleEnergyPoint,
    SanafeNocLink,
    SanafeNocLinkLoad,
    SanafeTileRecord,
)
from mimarsinan.chip_simulation.sanafe.records.run import (
    SanafeCoreDiff,
    SanafeRunRecord,
    SanafeSegmentRecord,
)
from mimarsinan.chip_simulation.sanafe.records.persisted import (
    SanafeCoreCounts,
    SanafeSegmentFile,
    SanafeSegmentFileRefusal,
    SanafeSegmentStore,
)

__all__ = [
    "SanafeArchGeometry",
    "SanafeCascadePoint",
    "SanafeConnectivityEdge",
    "SanafeCoreCounts",
    "SanafeCoreDiff",
    "SanafeCoreRecord",
    "SanafeCriticalCore",
    "SanafeCycleEnergyPoint",
    "SanafeEnergyBreakdown",
    "SanafeNocLink",
    "SanafeNocLinkLoad",
    "SanafeRunRecord",
    "SanafeSegmentFile",
    "SanafeSegmentFileRefusal",
    "SanafeSegmentRecord",
    "SanafeSegmentStore",
    "SanafeTileRecord",
]

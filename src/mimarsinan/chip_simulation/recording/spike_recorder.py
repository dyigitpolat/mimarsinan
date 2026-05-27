"""Per-segment spike-count records for HCM↔Loihi parity verification."""
from mimarsinan.chip_simulation.recording.compare import Diff, compare_records, format_first_diff
from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord

__all__ = [
    "CoreSpikeCounts",
    "Diff",
    "RunRecord",
    "SegmentSpikeRecord",
    "compare_records",
    "format_first_diff",
]

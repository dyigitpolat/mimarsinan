"""What one ODIN device run measured: the counts record plus the transport's walls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from mimarsinan.chip_simulation.recording.records import RunRecord

#: The backend name every registry, capability table and certificate uses.
BACKEND_NAME = "odin_fpga"


@dataclass(frozen=True)
class OdinSegmentTiming:
    """The measured cost of executing ONE neural segment on the device.

    ``program_wall_s`` is the reprogramming physics the plan asks the record to
    carry (§5.4): a chip whose weights do not fit is reprogrammed per pass, and
    that wall is a deployment cost, not an overhead to hide.
    """

    stage_index: int
    stage_name: str
    cores: int
    program_ops: int
    program_bytes: int
    program_wall_s: float
    program_basis: str
    run_wall_s: float
    device_cycles: int
    samples: int
    cycles_per_sample: int
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_wall_s(self) -> float:
        return float(self.program_wall_s) + float(self.run_wall_s)


@dataclass
class OdinFpgaRunRecord:
    """One sample on the device: the count record, the walls, the raw counts."""

    transport: str
    record: RunRecord
    timings: List[OdinSegmentTiming] = field(default_factory=list)
    compute_stage_walls: List[Dict[str, Any]] = field(default_factory=list)
    per_cycle_counts: Dict[int, Dict[Tuple[int, int, int, int], int]] = field(
        default_factory=dict)

    @property
    def sample_index(self) -> int:
        return int(self.record.sample_index)

    @property
    def program_wall_s(self) -> float:
        return sum(t.program_wall_s for t in self.timings)

    @property
    def run_wall_s(self) -> float:
        return sum(t.run_wall_s for t in self.timings)

    @property
    def device_cycles(self) -> int:
        return sum(int(t.device_cycles) for t in self.timings)

    def to_hcm_subset(self) -> RunRecord:
        """The projection a count certificate compares against the HCM reference."""
        return self.record


def aggregate_walls(records: List[OdinFpgaRunRecord]) -> Dict[str, float]:
    """Run-level measured walls: programming, execution, and their sum."""
    programming = sum(r.program_wall_s for r in records)
    execution = sum(r.run_wall_s for r in records)
    return {
        "programming_s": programming,
        "execution_s": execution,
        "total_s": programming + execution,
        "device_cycles": float(sum(r.device_cycles for r in records)),
        "samples": float(len(records)),
    }

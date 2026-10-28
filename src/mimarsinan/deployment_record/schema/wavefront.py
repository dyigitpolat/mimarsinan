"""[U-F4] The wavefront schedule a pipelined program runs: one row per group, and the two per-pass core fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.schema.serde import strict_kwargs


@dataclass(frozen=True)
class WavefrontRecord:
    """One wavefront group of the schedule: its segment, its first pass and pass count, the levels that share them,
    the layout cores its first pass loads (every copy of the group's banks, once per group), and the census of the
    values it produces and reads itself (the dense crossings, out and in, and the peak live carry over its pass
    boundaries, in the transfer's buffered bytes and in carried values): what a declared on-chip carry buffer holds
    instead of the boundary."""

    segment_index: int
    first_pass: int
    passes: int
    levels: Tuple[int, ...]
    layout_cores: int
    onchip_out_bytes: int = 0
    onchip_in_bytes: int = 0
    onchip_peak_live_bytes: int = 0
    onchip_peak_live_elements: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(int(level) for level in self.levels))
        if min(int(self.passes), int(self.layout_cores)) < 1 or min(int(self.first_pass), *self.levels, 0) < 0:
            raise ValueError(f"WavefrontRecord states {self.passes} passes on {self.layout_cores} layout cores from "
                             f"pass {self.first_pass} over levels {self.levels}: none may be negative or empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WavefrontRecord":
        return cls(**strict_kwargs(cls, data))


def wavefronts_from(value: Any) -> Optional[Tuple[WavefrontRecord, ...]]:
    """A sealed ``ScheduleRecord.wavefronts`` (dicts or records) as records, None when unset."""
    if value is None:
        return None
    return tuple(v if isinstance(v, WavefrontRecord) else WavefrontRecord.from_dict(v) for v in value)


def settle_core_fields(name: str, cores: int, dark: Optional[Sequence[int]],
                       ordinals: Optional[Sequence[int]]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """``SegmentRecord``'s dark cores (indices of its own cores) and head ordinals (one per core, non-negative), as
    tuples; any other value refuses by name."""
    dark_t = None if dark is None else tuple(int(i) for i in dark)
    if dark_t is not None and any(not 0 <= i < cores for i in dark_t):
        raise ValueError(f"{name}.dark_core_indices {dark_t} must index the {cores} cores of its pass")
    ord_t = None if ordinals is None else tuple(int(o) for o in ordinals)
    if ord_t is not None and (len(ord_t) != cores or any(o < 0 for o in ord_t)):
        raise ValueError(f"{name}.core_ordinals must state one non-negative head ordinal per core ({cores}), got "
                         f"{ord_t}")
    return dark_t, ord_t

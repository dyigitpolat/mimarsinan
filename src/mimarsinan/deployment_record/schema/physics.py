"""Timing and energy: measured SANA-FE quantities plus banded modeled terms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema.provenance import ModeledValue
from mimarsinan.deployment_record.schema.serde import (
    float_pair,
    optional,
    require_choice,
    strict_kwargs,
    tuple_of,
)

ENERGY_TERM_KINDS = frozenset({"measured", "modeled"})


@dataclass(frozen=True)
class SegmentTimingRecord:
    """One neural segment's measured execution: ``sim_time_s`` INCLUDES NoC hop latency."""

    stage_index: int
    timesteps_executed: int
    sim_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentTimingRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class LatencyDecomposition:
    """End-to-end latency terms; ``note`` states the no-double-count discipline."""

    programming_s: Optional[ModeledValue]
    compute_steps: int
    compute_sim_time_s: Optional[float]
    # host_ops_s is the raw measured total over the whole run; _per_pass is the
    # same measurement on the per-sample axis compute_sim_time_s lives on (a
    # sample-0 census), so the two are only ever summed after normalization.
    host_ops_s: Optional[float]
    host_ops_s_per_pass: Optional[float]
    sync_s: Optional[ModeledValue]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatencyDecomposition":
        kwargs = strict_kwargs(cls, data)
        kwargs["programming_s"] = optional(ModeledValue.from_dict, kwargs["programming_s"])
        kwargs["sync_s"] = optional(ModeledValue.from_dict, kwargs["sync_s"])
        return cls(**kwargs)


@dataclass(frozen=True)
class TimingRecord:
    """The timing fragment: static part always; ``per_segment`` empty without SANA-FE."""

    s_global: int
    depth: int
    per_segment: Tuple[SegmentTimingRecord, ...]
    latency: LatencyDecomposition

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimingRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["per_segment"] = tuple_of(
            SegmentTimingRecord.from_dict, kwargs["per_segment"]
        )
        kwargs["latency"] = LatencyDecomposition.from_dict(kwargs["latency"])
        return cls(**kwargs)


@dataclass(frozen=True)
class EnergyTermRecord:
    """One energy term: measured terms carry ``band_mj=None``; modeled terms a range."""

    name: str
    mj: float
    kind: str
    band_mj: Optional[Tuple[float, float]]
    basis: str

    def __post_init__(self) -> None:
        require_choice("EnergyTermRecord", "kind", self.kind, ENERGY_TERM_KINDS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnergyTermRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["band_mj"] = optional(float_pair, kwargs["band_mj"])
        return cls(**kwargs)


@dataclass(frozen=True)
class EnergyRecord:
    """The energy fragment: SANA-FE measured totals plus the term breakdown."""

    total_energy_mj: float
    mj_per_sample: float
    sample_count: int
    breakdown: Tuple[EnergyTermRecord, ...]
    energy_proxy_neuron_steps: int
    total_spikes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnergyRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["breakdown"] = tuple_of(EnergyTermRecord.from_dict, kwargs["breakdown"])
        return cls(**kwargs)

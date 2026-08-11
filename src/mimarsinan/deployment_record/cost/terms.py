"""The cost report's types: every term banded, sourced, and JSON-safe.

A :class:`CostTerm` is the model's atom: what it is, in which unit, its
nominal value, the ``(low, nominal, high)`` band it was drawn from (``None``
for a measurement — a measured number never wears a modeled band), the
epistemic kind, and the SOURCE it came from (a record field path, or the model
surface that produced it). The invariants are enforced at construction: a
measured term may not carry a band, a modeled term MUST, and any band must
bracket the value it explains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.schema import Band
from mimarsinan.deployment_record.schema.serde import (
    optional,
    require_choice,
    strict_kwargs,
    tuple_of,
)

COST_TERM_KINDS = frozenset({"measured", "modeled", "derived"})
COST_REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CostTerm:
    """One cost contribution: value + band + epistemic kind + provenance."""

    name: str
    unit: str
    value: float
    band: Optional[Band]
    kind: str
    source: str

    def __post_init__(self) -> None:
        require_choice("CostTerm", "kind", self.kind, COST_TERM_KINDS)
        for field_name in ("name", "unit", "source"):
            if not getattr(self, field_name):
                raise ValueError(f"CostTerm.{field_name} must be stated, got empty")
        if self.kind == "measured" and self.band is not None:
            raise ValueError(
                f"measured term {self.name!r} carries a band; a measurement is "
                f"never presented with modeled uncertainty"
            )
        if self.kind == "modeled" and self.band is None:
            raise ValueError(
                f"modeled term {self.name!r} has no band; every modeled value "
                f"ships (low, nominal, high) with a written basis"
            )
        if self.band is not None and not (self.band.low <= self.value <= self.band.high):
            raise ValueError(
                f"term {self.name!r} value {self.value} is outside its band "
                f"({self.band.low}, {self.band.high})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "band": None if self.band is None else self.band.to_dict(),
            "kind": self.kind,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostTerm":
        kwargs = strict_kwargs(cls, data)
        kwargs["band"] = optional(Band.from_dict, kwargs["band"])
        return cls(**kwargs)


def find_term(terms: Sequence[CostTerm], name: str) -> CostTerm:
    """The named term, or a loud failure listing what is actually there."""
    for term in terms:
        if term.name == name:
            return term
    raise KeyError(f"no cost term named {name!r}; have {[t.name for t in terms]}")


@dataclass(frozen=True)
class SegmentInitCost:
    """One scheduled segment's initialization cost (reset constant + payload)."""

    stage_index: int
    segment_index: int
    pass_index: int
    programming: str
    core_count: int
    terms: Tuple[CostTerm, ...]

    def term(self, name: str) -> CostTerm:
        return find_term(self.terms, name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "segment_index": self.segment_index,
            "pass_index": self.pass_index,
            "programming": self.programming,
            "core_count": self.core_count,
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentInitCost":
        kwargs = strict_kwargs(cls, data)
        kwargs["terms"] = tuple_of(CostTerm.from_dict, kwargs["terms"])
        return cls(**kwargs)


@dataclass(frozen=True)
class DeploymentCostReport:
    """The whole cost surface over one sealed record: grouped, banded terms."""

    segments: Tuple[SegmentInitCost, ...]
    energy: Tuple[CostTerm, ...]
    latency: Tuple[CostTerm, ...]
    area: Tuple[CostTerm, ...]
    throughput: Tuple[CostTerm, ...]
    notes: Tuple[str, ...]
    format_version: int = COST_REPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if int(self.format_version) != COST_REPORT_FORMAT_VERSION:
            raise ValueError(
                f"DeploymentCostReport format_version {self.format_version} != "
                f"{COST_REPORT_FORMAT_VERSION}; migrate explicitly, never "
                f"tolerate silently"
            )

    def all_terms(self) -> Tuple[CostTerm, ...]:
        """Every term in the report, segment terms included."""
        segment_terms = tuple(
            term for segment in self.segments for term in segment.terms
        )
        return segment_terms + self.energy + self.latency + self.area + self.throughput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "energy": [term.to_dict() for term in self.energy],
            "latency": [term.to_dict() for term in self.latency],
            "area": [term.to_dict() for term in self.area],
            "throughput": [term.to_dict() for term in self.throughput],
            "notes": list(self.notes),
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeploymentCostReport":
        kwargs = strict_kwargs(cls, data)
        kwargs["segments"] = tuple_of(SegmentInitCost.from_dict, kwargs["segments"])
        for group in ("energy", "latency", "area", "throughput"):
            kwargs[group] = tuple_of(CostTerm.from_dict, kwargs[group])
        kwargs["notes"] = tuple(str(note) for note in kwargs["notes"])
        return cls(**kwargs)


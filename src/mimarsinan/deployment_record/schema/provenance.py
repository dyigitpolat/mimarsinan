"""Structural provenance: no proxy is ever presented as a measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from mimarsinan.deployment_record.schema.serde import require_choice, strict_kwargs

PROVENANCE_KINDS = frozenset({"measured", "modeled", "derived", "declared"})


@dataclass(frozen=True)
class Provenance:
    """Who produced a fragment group, at which step, and with what epistemic kind."""

    kind: str
    producer: str
    step: str
    detail: str = ""

    def __post_init__(self) -> None:
        require_choice("Provenance", "kind", self.kind, PROVENANCE_KINDS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class Band:
    """A ``(low, nominal, high)`` coefficient range with a written basis."""

    low: float
    nominal: float
    high: float
    basis: str

    def __post_init__(self) -> None:
        if not (self.low <= self.nominal <= self.high):
            raise ValueError(
                f"Band must satisfy low <= nominal <= high, got "
                f"({self.low}, {self.nominal}, {self.high})"
            )
        if not self.basis:
            raise ValueError("Band.basis must state the written basis, got empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Band":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class ModeledValue:
    """A modeled quantity: the nominal value plus the band it was drawn from."""

    value: float
    band: Band

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "band": self.band.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModeledValue":
        kwargs = strict_kwargs(cls, data)
        kwargs["band"] = Band.from_dict(kwargs["band"])
        return cls(**kwargs)

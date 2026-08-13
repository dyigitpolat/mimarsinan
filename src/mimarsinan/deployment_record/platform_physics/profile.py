"""A target's declared physics: banded, evidenced values over the closed vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.units import to_canonical, unit_for
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.schema.serde import require_choice, strict_kwargs

PHYSICS_FORMAT_VERSION = 1

EVIDENCE_KINDS = frozenset({"published", "datasheet", "derived", "estimated"})
MEASUREMENT_KINDS = frozenset({"silicon", "simulation", "projection", "mixed"})

#: What each evidence kind must carry for the value to be readable years later.
_EVIDENCE_REQUIREMENTS = {
    "published": ("citation", "the paper and table/figure the number came from"),
    "datasheet": ("citation", "the datasheet and revision the number came from"),
    "derived": ("derivation", "the arithmetic that produced this per-unit number"),
    "estimated": ("note", "why this estimate is reasonable and its expected error"),
}


@dataclass(frozen=True)
class PhysicsConstantValue:
    """One declared value: a band in its declared unit, plus the evidence behind it.

    The SI band is a PROPERTY, never a stored field — there is exactly one number in
    the profile, so the declared and canonical forms cannot drift apart.
    """

    key: str
    low: float
    nominal: float
    high: float
    unit: str
    evidence_kind: str
    citation: str = ""
    derivation: str = ""
    note: str = ""
    overridden: bool = False

    def __post_init__(self) -> None:
        spec = spec_for(self.key)
        require_choice("PhysicsConstantValue", "evidence_kind", self.evidence_kind,
                       EVIDENCE_KINDS)
        if unit_for(self.unit).dimension != spec.dimension:
            raise ValueError(
                f"{self.key}: declared unit {self.unit!r} has the wrong dimension — "
                f"{self.key} is a {spec.dimension} constant"
            )
        if not (self.low <= self.nominal <= self.high):
            raise ValueError(
                f"{self.key}: band must satisfy low <= nominal <= high, got "
                f"({self.low}, {self.nominal}, {self.high})"
            )
        required, why = _EVIDENCE_REQUIREMENTS[self.evidence_kind]
        if not str(getattr(self, required)).strip():
            raise ValueError(
                f"{self.key}: evidence_kind {self.evidence_kind!r} requires "
                f"{required!r} — {why}"
            )

    @property
    def basis(self) -> str:
        """The written basis carried into every Band and CostTerm built from this value."""
        detail = self.citation or self.derivation or self.note
        marker = " (operator override)" if self.overridden else ""
        return f"{self.evidence_kind}{marker}: {detail}"

    @property
    def band(self) -> Band:
        """The declared band in the dimension's SI base unit."""
        return Band(
            to_canonical(self.low, self.unit),
            to_canonical(self.nominal, self.unit),
            to_canonical(self.high, self.unit),
            basis=self.basis,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "low": self.low,
            "nominal": self.nominal,
            "high": self.high,
            "unit": self.unit,
            "evidence_kind": self.evidence_kind,
            "citation": self.citation,
            "derivation": self.derivation,
            "note": self.note,
            "overridden": self.overridden,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhysicsConstantValue":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class PlatformPhysicsValidity:
    """The operating point the declared numbers hold at."""

    measurement_kind: str
    technology_node_nm: Optional[float] = None
    supply_v: Optional[float] = None
    temperature_c: Optional[float] = None
    array_size_assumed: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        require_choice("PlatformPhysicsValidity", "measurement_kind",
                       self.measurement_kind, MEASUREMENT_KINDS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_kind": self.measurement_kind,
            "technology_node_nm": self.technology_node_nm,
            "supply_v": self.supply_v,
            "temperature_c": self.temperature_c,
            "array_size_assumed": self.array_size_assumed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformPhysicsValidity":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class PlatformPhysics:
    """A named target's physics. An undeclared constant is ABSENT, never defaulted."""

    name: str
    display_name: str
    description_file: str
    validity: PlatformPhysicsValidity
    constants: Mapping[str, PhysicsConstantValue] = field(default_factory=dict)
    #: The target's DATAFLOW, for the quantities no record counts (ADC
    #: conversions). Absent = the digital model, which converts nothing.
    conversion_model: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = PHYSICS_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != PHYSICS_FORMAT_VERSION:
            raise ValueError(
                f"PlatformPhysics format_version {self.format_version} != "
                f"{PHYSICS_FORMAT_VERSION}; migrate explicitly, never tolerate silently"
            )
        for key, value in self.constants.items():
            if key != value.key:
                raise ValueError(
                    f"{self.name}: constant keyed {key!r} carries key {value.key!r}"
                )

    def has(self, key: str) -> bool:
        """Whether this target declares ``key`` at all."""
        return key in self.constants

    def band(self, key: str) -> Band:
        """The SI band for ``key``; absence raises — there is no default to fall back on."""
        if key not in self.constants:
            raise KeyError(
                f"{self.name} declares no value for {key!r}; the objectives that need it "
                f"are unavailable, and no default may stand in for vendor data"
            )
        return self.constants[key].band

    def missing(self, keys: Iterable[str]) -> Tuple[str, ...]:
        """Which of ``keys`` this target does not declare, in the order asked."""
        return tuple(key for key in keys if key not in self.constants)

    def declares_all(self, keys: Iterable[str]) -> bool:
        """The availability predicate every absolute objective gates on."""
        return not self.missing(keys)

    def evidence_kinds(self) -> Dict[str, str]:
        """Per-constant evidence, for reports that must disclose what is estimated."""
        return {key: value.evidence_kind for key, value in self.constants.items()}

    def overridden_keys(self) -> Tuple[str, ...]:
        """Constants an operator replaced, so a run states the physics it actually used."""
        return tuple(sorted(k for k, v in self.constants.items() if v.overridden))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "display_name": self.display_name,
            "description_file": self.description_file,
            "validity": self.validity.to_dict(),
            "conversion_model": dict(self.conversion_model),
            "constants": {k: v.to_dict() for k, v in sorted(self.constants.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformPhysics":
        kwargs = strict_kwargs(cls, data)
        kwargs["validity"] = PlatformPhysicsValidity.from_dict(kwargs["validity"])
        kwargs["constants"] = {
            key: PhysicsConstantValue.from_dict(value)
            for key, value in (kwargs.get("constants") or {}).items()
        }
        kwargs["conversion_model"] = dict(kwargs.get("conversion_model") or {})
        return cls(**kwargs)

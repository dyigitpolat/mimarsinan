"""A target's declared physics: banded, evidenced values over the closed vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.units import to_canonical, unit_for
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.schema.serde import require_choice, strict_kwargs

PHYSICS_FORMAT_VERSION = 1

EVIDENCE_KINDS = frozenset(
    {"published", "datasheet", "derived", "estimated", "measured"}
)
MEASUREMENT_KINDS = frozenset({"silicon", "simulation", "projection", "mixed"})

#: What each evidence kind must carry for the value to be readable years later.
_EVIDENCE_REQUIREMENTS = {
    "published": ("citation", "the paper and table/figure the number came from"),
    "datasheet": ("citation", "the datasheet and revision the number came from"),
    "derived": ("derivation", "the arithmetic that produced this per-unit number"),
    "estimated": ("note", "why this estimate is reasonable and its expected error"),
    # [H3b] measured on THIS deployment stack (host calibration, bench runs) —
    # the strongest kind, and the one that must say where and on what machine.
    "measured": ("note", "what was measured, on which machine, and how to redo it"),
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


#: How an aggregate constant may be read outside its measured core: ``cells`` scales it by the
#: core's cell count relative to the validity core's.
SCALING_RULES = frozenset({"cells"})


@dataclass(frozen=True)
class ValidityGeometry:
    """[E190] The core the aggregate constants were measured on: geometry, cell bits, core counts."""

    axons: int
    neurons: int
    #: Weight bits one cell of that core holds; a program with wider weights lies outside it.
    cell_bits: Optional[int] = None
    #: ``(min, max)`` cores of the measured chip, where the source bounds it.
    core_count: Optional[Tuple[int, int]] = None
    scaling: Optional[str] = None

    def __post_init__(self) -> None:
        if self.scaling is not None:
            require_choice("ValidityGeometry", "scaling", self.scaling, SCALING_RULES)
        if self.core_count is not None:
            low, high = (int(v) for v in self.core_count)
            if not 1 <= low <= high:
                raise ValueError(f"ValidityGeometry.core_count must be 1 <= min <= max, got {self.core_count}")
            object.__setattr__(self, "core_count", (low, high))

    def to_dict(self) -> Dict[str, Any]:
        return {"axons": self.axons, "neurons": self.neurons, "cell_bits": self.cell_bits,
                "core_count": None if self.core_count is None else list(self.core_count),
                "scaling": self.scaling}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidityGeometry":
        kwargs = strict_kwargs(cls, data)
        if kwargs.get("core_count") is not None:
            kwargs["core_count"] = tuple(kwargs["core_count"])
        return cls(**kwargs)


@dataclass(frozen=True)
class PlatformPhysicsValidity:
    """The operating point the declared numbers hold at."""

    measurement_kind: str
    technology_node_nm: Optional[float] = None
    supply_v: Optional[float] = None
    temperature_c: Optional[float] = None
    array_size_assumed: str = ""
    notes: str = ""
    #: The structured validity of the aggregate constants (absent on physics that predates it).
    geometry: Optional[ValidityGeometry] = None

    def __post_init__(self) -> None:
        require_choice("PlatformPhysicsValidity", "measurement_kind",
                       self.measurement_kind, MEASUREMENT_KINDS)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "measurement_kind": self.measurement_kind,
            "technology_node_nm": self.technology_node_nm,
            "supply_v": self.supply_v,
            "temperature_c": self.temperature_c,
            "array_size_assumed": self.array_size_assumed,
            "notes": self.notes,
        }
        if self.geometry is not None:
            data["geometry"] = self.geometry.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformPhysicsValidity":
        kwargs = strict_kwargs(cls, data)
        if kwargs.get("geometry") is not None:
            kwargs["geometry"] = ValidityGeometry.from_dict(kwargs["geometry"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ExternalComponent:
    """[CL2] A component the priced system needs and the profile does not describe (an SRAM-CIM macro's backing
    store): a run declares its ``constants`` beside the profile, or nothing absolute is read."""

    name: str
    constants: Tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name.isspace():
            raise ValueError("ExternalComponent needs a name")
        if not self.constants:
            raise ValueError(f"external component {self.name!r} names no constant")
        for key in self.constants:
            spec_for(key)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "constants": list(self.constants), "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalComponent":
        kwargs = strict_kwargs(cls, data)
        kwargs["constants"] = tuple(kwargs.get("constants") or ())
        return cls(**kwargs)


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
    #: [CL2] Components the profile leaves out; serialized only when present.
    external_components: Tuple[ExternalComponent, ...] = ()

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

    def undeclared_external(self) -> Tuple[ExternalComponent, ...]:
        """The external components with a constant this physics does not declare yet."""
        return tuple(component for component in self.external_components if self.missing(component.constants))

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
            **({"external_components": [c.to_dict() for c in self.external_components]}
               if self.external_components else {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformPhysics":
        kwargs = strict_kwargs(cls, data)
        kwargs["external_components"] = tuple(
            ExternalComponent.from_dict(c) for c in kwargs.get("external_components") or ())
        kwargs["validity"] = PlatformPhysicsValidity.from_dict(kwargs["validity"])
        kwargs["constants"] = {
            key: PhysicsConstantValue.from_dict(value)
            for key, value in (kwargs.get("constants") or {}).items()
        }
        kwargs["conversion_model"] = dict(kwargs.get("conversion_model") or {})
        return cls(**kwargs)

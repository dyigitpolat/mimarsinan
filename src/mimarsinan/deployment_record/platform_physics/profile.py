"""A target's declared physics: banded, evidenced values over the closed vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
# [U-E3] The value type and its evidence kinds live below the converter (``constants``), re-exported here.
from mimarsinan.deployment_record.platform_physics.constants import (
    EVIDENCE_KINDS as EVIDENCE_KINDS, PhysicsConstantValue as PhysicsConstantValue, constant_from_declaration, spec_for,
)
from mimarsinan.deployment_record.platform_physics.converter import ConverterTable
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.schema.serde import require_choice, strict_kwargs

PHYSICS_FORMAT_VERSION = 1

MEASUREMENT_KINDS = frozenset({"silicon", "simulation", "projection", "mixed"})

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


#: [U-G] The constants of a declared on-chip carry buffer (``carry_buffer_bytes``) a profile may state over capacity.
CARRY_BUFFER_KEYS = ("e_carry_buffer_per_byte", "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte")


class CarryBufferDeclarationError(ToolchainRefusal, ValueError):
    """[U-G] A carry-buffer table the loader refuses, by the name of what it states wrong."""


@dataclass(frozen=True)
class CarryBufferTable:
    """[U-G] An on-chip carry buffer's constants over its declared capacity (``carry_buffer_bytes``): an SRAM's
    access energy (and, where its source says so, its leakage and area per byte) depends on its size, so a profile
    states them per capacity; a buffer constant that holds at every size is stated flat in its constants instead."""

    rows: Mapping[int, Mapping[str, PhysicsConstantValue]]
    source: Mapping[str, Any] = field(default_factory=dict)

    @property
    def capacities(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rows))

    def keys(self) -> Tuple[str, ...]:
        return tuple(k for k in CARRY_BUFFER_KEYS if any(k in row for row in self.rows.values()))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CarryBufferTable) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:  # pragma: no cover - frozen dataclasses of mappings are not hashed in use
        return hash(self.capacities)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CarryBufferTable":
        if not isinstance(data, Mapping):
            raise CarryBufferDeclarationError("carry_buffer must be an object of rows and an optional note")
        if unknown := sorted(set(data) - {"rows", "note"}):
            raise CarryBufferDeclarationError(f"carry_buffer declares unknown fields {unknown}")
        rows: Dict[int, Dict[str, PhysicsConstantValue]] = {}
        for raw in data.get("rows") or ():
            size = raw.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise CarryBufferDeclarationError(f"carry_buffer row bytes {size!r} is no positive integer")
            if size in rows:
                raise CarryBufferDeclarationError(f"carry_buffer states the capacity {size} B twice")
            if stray := sorted(k for k in raw if k not in ("bytes",) + CARRY_BUFFER_KEYS):
                raise CarryBufferDeclarationError(f"carry_buffer row at {size} B states {stray}, outside "
                                                  f"{list(CARRY_BUFFER_KEYS)}")
            if not any(k in raw for k in CARRY_BUFFER_KEYS):
                raise CarryBufferDeclarationError(f"carry_buffer row at {size} B states no buffer constant")
            rows[size] = {k: constant_from_declaration(k, raw[k]) for k in CARRY_BUFFER_KEYS if k in raw}
        if not rows:
            raise CarryBufferDeclarationError("carry_buffer states no row")
        return cls(rows=rows, source=dict(data))


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
    #: [E-5] The converter over resolution (``readout_law.adc_bits``, ``converter.py``); serialized only when present.
    converter: Optional[ConverterTable] = None
    #: [U-G] The on-chip carry buffer over capacity (``carry_buffer_bytes``); serialized only when present.
    carry_buffer: Optional[CarryBufferTable] = None

    def __post_init__(self) -> None:
        if self.format_version != PHYSICS_FORMAT_VERSION:
            raise ValueError(f"PlatformPhysics format_version {self.format_version} != "
                             f"{PHYSICS_FORMAT_VERSION}; migrate explicitly, never tolerate silently")
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

    def at_carry_buffer(self, capacity: Optional[int]) -> Tuple["PlatformPhysics", Dict[str, str]]:
        """[U-G] (this physics at a declared carry-buffer ``capacity`` in bytes, {buffer constant: why the table leaves
        it unstated there}): each row constant at the capacity's row; a flat constant (an operator's override) stands
        at every capacity. Unchanged without a table or a declared capacity."""
        table = self.carry_buffer
        if table is None or capacity is None:
            return self, {}
        row, constants, unstated = table.rows.get(int(capacity), {}), dict(self.constants), {}
        for key in (k for k in table.keys() if k not in self.constants):
            if key in row:
                constants[key] = row[key]
            else:
                unstated[key] = (f"{key} at carry_buffer_bytes {int(capacity)}: the chip's carry_buffer table states "
                                 f"({', '.join(str(c) for c in table.capacities)}) B")
        return replace(self, constants=constants), unstated

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
            **({"converter": self.converter.to_dict()} if self.converter is not None else {}),
            **({"carry_buffer": self.carry_buffer.to_dict()} if self.carry_buffer is not None else {}),
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
        kwargs["converter"] = None if kwargs.get("converter") is None else ConverterTable.from_dict(kwargs["converter"])
        kwargs["carry_buffer"] = (None if kwargs.get("carry_buffer") is None
                                  else CarryBufferTable.from_dict(kwargs["carry_buffer"]))
        return cls(**kwargs)

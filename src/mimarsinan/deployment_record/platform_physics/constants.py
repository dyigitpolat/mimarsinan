"""The physics-constant vocabulary: what a target MAY declare, declared once; one declared value
(``PhysicsConstantValue``) and the one judge of a declaration against the vocabulary
(``constant_from_declaration``). [U-E3] The value and its judge live here, below the converter and the
loader: ``profile`` imports the converter (a profile carries its converter table), and the converter judges
its rows as profile constants, so a judge above the converter would close an init cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.deployment_record.platform_physics import vocabulary
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.schema.serde import require_choice, strict_kwargs
from mimarsinan.deployment_record.units import DIMENSIONS, to_canonical, unit_for

PHYSICS_GROUPS: Tuple[str, ...] = vocabulary.GROUPS


@dataclass(frozen=True)
class PhysicsConstantSpec:
    """One declarable constant: its group, dimension, expected unit and multiplicand."""

    key: str
    group: str
    dimension: str
    display_unit: str
    multiplicand: str
    doc: str

    def __post_init__(self) -> None:
        if self.group not in PHYSICS_GROUPS:
            raise ValueError(
                f"{self.key}: group {self.group!r} is not one of {list(PHYSICS_GROUPS)}"
            )
        if self.dimension not in DIMENSIONS:
            raise ValueError(
                f"{self.key}: dimension {self.dimension!r} is not one of {list(DIMENSIONS)}"
            )
        if unit_for(self.display_unit).dimension != self.dimension:
            raise ValueError(
                f"{self.key}: display unit {self.display_unit!r} is not a "
                f"{self.dimension} unit"
            )


def _build() -> Dict[str, PhysicsConstantSpec]:
    specs: Dict[str, PhysicsConstantSpec] = {}
    for key, group, dimension, unit, multiplicand, doc in vocabulary.ROWS:
        if key in specs:
            raise ValueError(f"physics constant {key!r} is declared twice")
        specs[key] = PhysicsConstantSpec(key, group, dimension, unit, multiplicand, doc)
    return specs


PHYSICS_CONSTANTS: Mapping[str, PhysicsConstantSpec] = _build()

#: Aggregate -> the decomposed constants it already contains (see vocabulary.SUPERSEDES).
SUPERSEDES: Mapping[str, Tuple[str, ...]] = vocabulary.SUPERSEDES


def superseded_by(key: str) -> Tuple[str, ...]:
    """Constants ``key`` already accounts for, which must not be priced beside it."""
    return SUPERSEDES.get(key, ())


def resolve_supersessions(keys: Iterable[str]) -> Tuple[str, ...]:
    """Which of ``keys`` a pricer may charge, once aggregates absorb what they contain.

    A profile may legitimately declare an aggregate AND its decomposition — both are
    published facts. Charging both double counts, so the aggregate wins and the terms it
    already contains drop out. Declaration order never matters.
    """
    declared = set(keys)
    absorbed = {
        member
        for aggregate in declared & set(SUPERSEDES)
        for member in superseded_by(aggregate)
    }
    return tuple(key for key in sorted(declared) if key not in absorbed)


def spec_for(key: str) -> PhysicsConstantSpec:
    """The declaration for ``key``, or a loud error naming the whole vocabulary."""
    try:
        return PHYSICS_CONSTANTS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a declarable physics constant; the vocabulary is "
            f"{sorted(PHYSICS_CONSTANTS)}"
        ) from None


def keys_in_group(group: str) -> Tuple[str, ...]:
    """Vocabulary order within one group — also the wizard panel's row order."""
    if group not in PHYSICS_GROUPS:
        raise KeyError(f"{group!r} is not a physics group; expected {list(PHYSICS_GROUPS)}")
    return tuple(key for key, spec in PHYSICS_CONSTANTS.items() if spec.group == group)


EVIDENCE_KINDS = frozenset(
    {"published", "datasheet", "derived", "estimated", "measured"}
)

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


SANAFE_PRESET_REF = "sanafe_preset"

#: The SANA-FE per-event presets are stored in SI already, and their suffix names the
#: unit — which is what lets a mis-bound reference (an energy priced with a latency) hit
#: the value's own dimension guard instead of resolving silently.
_PRESET_SUFFIX_UNITS = {"_energy_j": "J", "_latency_s": "s"}


def _resolve_source_ref(key: str, ref: str) -> Dict[str, Any]:
    """Materialize a constant from an existing in-repo declaration, never a copy of it.

    One indirection form: ``sanafe_preset:<preset>:<field>``. The numbers keep exactly
    one home, so editing the preset moves the profile with it.
    """
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != SANAFE_PRESET_REF:
        raise ValueError(
            f"{key}: source_ref {ref!r} is not of the form "
            f"'{SANAFE_PRESET_REF}:<preset>:<field>'"
        )
    _, preset_name, field_name = parts
    if preset_name not in PRESETS:
        raise KeyError(
            f"{key}: source_ref names unknown preset {preset_name!r}; known presets are "
            f"{sorted(PRESETS)}"
        )
    preset = PRESETS[preset_name]
    if field_name not in preset:
        raise KeyError(
            f"{key}: source_ref names unknown preset field {field_name!r}; "
            f"{preset_name} declares {sorted(preset)}"
        )
    unit = next(
        (unit for suffix, unit in _PRESET_SUFFIX_UNITS.items()
         if field_name.endswith(suffix)),
        None,
    )
    if unit is None:
        raise ValueError(
            f"{key}: preset field {field_name!r} carries no unit suffix "
            f"({sorted(_PRESET_SUFFIX_UNITS)}), so its dimension is unknown"
        )
    value = float(preset[field_name])  # type: ignore[literal-required]
    return {
        "low": value,
        "nominal": value,
        "high": value,
        "unit": unit,
        "evidence_kind": "derived",
        "derivation": f"resolved from {ref} (the SANA-FE per-event preset this target "
                      f"simulates with); the number keeps its single home there",
    }


def constant_from_declaration(key: str, data: Mapping[str, Any]) -> PhysicsConstantValue:
    """One constant from its JSON declaration, judged against the vocabulary: a profile's constants, a lane
    block's rows (W08) and a converter table's rows (E-5) alike. [U-E3] The one judge, below the converter and the
    loader, so both import it at top."""
    spec = spec_for(key)
    declares_band = "nominal" in data
    declares_ref = "source_ref" in data
    if declares_band == declares_ref:
        raise ValueError(
            f"{key}: a constant declares exactly one of a band ('nominal', with optional "
            f"'low'/'high') or a 'source_ref'"
        )
    if declares_ref:
        fields = _resolve_source_ref(key, str(data["source_ref"]))
        fields["citation"] = str(data.get("citation", ""))
        fields["note"] = str(data.get("note", ""))
    else:
        nominal = float(data["nominal"])
        fields = {
            "low": float(data.get("low", nominal)),
            "nominal": nominal,
            "high": float(data.get("high", nominal)),
            "unit": str(data.get("unit", spec.display_unit)),
            "evidence_kind": str(data.get("evidence_kind", "estimated")),
            "citation": str(data.get("citation", "")),
            "derivation": str(data.get("derivation", "")),
            "note": str(data.get("note", "")),
        }
    unknown = set(data) - {"low", "nominal", "high", "unit", "evidence_kind", "citation",
                           "derivation", "note", "source_ref"}
    if unknown:
        raise ValueError(f"{key}: unknown fields {sorted(unknown)} in the declaration")
    try:
        return PhysicsConstantValue(key=key, **fields)
    except ValueError as exc:
        if not declares_ref:
            raise
        raise ValueError(f"{key}: source_ref {data['source_ref']!r} — {exc}") from exc

"""Reading a physics profile: strict, description-backed, and unable to fork a number."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.profile import (
    PHYSICS_FORMAT_VERSION,
    PhysicsConstantValue,
    PlatformPhysics,
    PlatformPhysicsValidity,
)

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


def _constant_from_dict(key: str, data: Mapping[str, Any]) -> PhysicsConstantValue:
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


def profile_from_dict(data: Mapping[str, Any]) -> PlatformPhysics:
    """Build a profile from its JSON body, refusing anything the vocabulary rejects."""
    version = int(data.get("format_version", -1))
    if version != PHYSICS_FORMAT_VERSION:
        raise ValueError(
            f"physics profile format_version {version} != {PHYSICS_FORMAT_VERSION}; "
            f"migrate explicitly, never tolerate silently"
        )
    constants = {
        key: _constant_from_dict(key, value)
        for key, value in (data.get("constants") or {}).items()
    }
    conversion_model = dict(data.get("conversion_model") or {})
    # Validate the dataflow HERE: an undescribable one must fail at declaration,
    # not as a conversion count nobody can defend at pricing time.
    conversion_model_for(conversion_model)
    return PlatformPhysics(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description_file=str(data["description_file"]),
        validity=PlatformPhysicsValidity.from_dict(data["validity"]),
        constants=constants,
        conversion_model=conversion_model,
    )


def load_profile(path: Path) -> PlatformPhysics:
    """Load ``<name>.json``, requiring its description file to exist beside it."""
    path = Path(path)
    physics = profile_from_dict(json.loads(path.read_text()))
    if physics.name != path.stem:
        raise ValueError(
            f"{path.name} declares name {physics.name!r}; a profile's file name is its "
            f"name, so this profile would resolve under {path.stem!r} and never load"
        )
    description = path.with_name(physics.description_file)
    if not description.is_file():
        raise FileNotFoundError(
            f"{physics.name}: description file {physics.description_file!r} is missing "
            f"beside {path} — every profile states its citations and estimate rationales "
            f"in prose, so a profile without one may not ship"
        )
    return physics


def apply_overrides(
    physics: PlatformPhysics, overrides: Mapping[str, Mapping[str, Any]]
) -> PlatformPhysics:
    """A new profile with operator-declared values replacing (or adding) constants.

    An override must say why it deviates; the replacement is marked so reports and the
    wizard can show it as a departure from the declared profile.
    """
    if not overrides:
        return physics
    constants = dict(physics.constants)
    for key, raw in overrides.items():
        spec = spec_for(key)
        existing = physics.constants.get(key)
        data = dict(raw)
        data.setdefault("unit", existing.unit if existing else spec.display_unit)
        data.setdefault("evidence_kind", "estimated")
        constants[key] = PhysicsConstantValue(
            key=key,
            low=float(data.get("low", data["nominal"])),
            nominal=float(data["nominal"]),
            high=float(data.get("high", data["nominal"])),
            unit=str(data["unit"]),
            evidence_kind=str(data["evidence_kind"]),
            citation=str(data.get("citation", "")),
            derivation=str(data.get("derivation", "")),
            note=str(data.get("note", "")),
            overridden=True,
        )
    return PlatformPhysics(
        name=physics.name,
        display_name=physics.display_name,
        description_file=physics.description_file,
        validity=physics.validity,
        constants=constants,
        conversion_model=physics.conversion_model,
    )

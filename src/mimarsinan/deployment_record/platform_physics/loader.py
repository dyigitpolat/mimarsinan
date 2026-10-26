"""Reading a physics profile: strict, description-backed, and unable to fork a number."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping

from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.profile import (
    PHYSICS_FORMAT_VERSION,
    ExternalComponent,
    PhysicsConstantValue,
    PlatformPhysics,
    PlatformPhysicsValidity,
)
from mimarsinan.deployment_record.units import TIME, to_canonical

SANAFE_PRESET_REF = "sanafe_preset"

#: [consts] How an override declaration composes with the profile's own value of its key (the declaration's
#: ``combine``): ``replace`` (the default: an operator's departure replaces the profile's value) or ``slower`` (the
#: declaration is one stage of a series link whose other stage the profile declares, as a backing store's channel
#: beside the chip's own programming port: a streamed payload moves no faster than its slower stage, so the composed
#: time is the larger of the two corner by corner, the pipelined bound of the link (a store-and-forward link, which
#: adds its stages, is slower); a profile without its own stage takes the declared one).
COMBINE_RULES = ("replace", "slower")
#: [consts fix round] The closed field set of an override declaration; a field outside it refuses by name.
OVERRIDE_FIELDS = frozenset({"nominal", "low", "high", "unit", "evidence_kind", "citation", "derivation", "note",
                             "combine"})
#: Evidence classes weaker than a shown derivation: a composition (arithmetic) cannot strengthen such a stage.
_BELOW_DERIVED = frozenset({"estimated"})

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


def constant_from_declaration(key: str, data: Mapping[str, Any]) -> PhysicsConstantValue:
    """One constant from its JSON declaration, judged as a profile's are (a lane block's rows, W08)."""
    return _constant_from_dict(key, data)


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
    external = tuple(ExternalComponent.from_dict(c) for c in data.get("external_components") or ())
    for component in external:
        if declared := [key for key in component.constants if key in constants]:
            raise ValueError(f"{data['name']}: external component {component.name!r} names {declared}, which the "
                             "profile declares itself; an external component's constants come from beside it")
    return PlatformPhysics(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description_file=str(data["description_file"]),
        validity=PlatformPhysicsValidity.from_dict(data["validity"]),
        constants=constants,
        conversion_model=conversion_model,
        external_components=external,
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
    wizard can show it as a departure from the declared profile. A declaration stating
    ``combine: slower`` composes with the profile's own value instead (``COMBINE_RULES``).
    """
    if not overrides:
        return physics
    constants = dict(physics.constants)
    for key, raw in overrides.items():
        spec = spec_for(key)
        existing = physics.constants.get(key)
        data = dict(raw)
        unknown = set(data) - OVERRIDE_FIELDS
        if unknown:
            raise ValueError(f"{key}: unknown fields {sorted(unknown)} in the declaration (an override declares "
                             f"{sorted(OVERRIDE_FIELDS)})")
        combine = str(data.pop("combine", "replace"))
        if combine not in COMBINE_RULES:
            raise ValueError(f"{key}: combine {combine!r} is not one of {list(COMBINE_RULES)}")
        if combine == "slower" and spec.dimension != TIME:
            raise ValueError(f"{key}: combine 'slower' composes a time (the slower stage of a series link), and "
                             f"{key} is a {spec.dimension} constant")
        data.setdefault("unit", existing.unit if existing else spec.display_unit)
        data.setdefault("evidence_kind", "estimated")
        value = PhysicsConstantValue(
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
        constants[key] = _slower(value, existing) if combine == "slower" and existing is not None else value
    return PlatformPhysics(
        name=physics.name,
        display_name=physics.display_name,
        description_file=physics.description_file,
        validity=physics.validity,
        constants=constants,
        conversion_model=physics.conversion_model,
        external_components=physics.external_components,
    )


def _stage(value: PhysicsConstantValue) -> str:
    return "; ".join(part for part in (value.evidence_kind, value.citation, value.derivation, value.note) if part)


def _slower(declared: PhysicsConstantValue, own: PhysicsConstantValue) -> PhysicsConstantValue:
    """The slower stage of a series link, corner by corner, in the declared unit, stating both stages: the pipelined
    bound of the link (a store-and-forward link is slower). Its class is derived (the arithmetic shown) unless a stage
    that prevails at some corner is weaker, which the composition keeps (an estimated port stays estimated)."""
    scale = to_canonical(1.0, declared.unit)
    ends = ("low", "nominal", "high")
    prevailing = {(declared if getattr(declared.band, end) >= getattr(own.band, end) else own).evidence_kind
                  for end in ends}
    low, nominal, high = (max(getattr(declared.band, end), getattr(own.band, end)) / scale for end in ends)
    derivation = (f"the slower stage of a series link, corner by corner (combine: slower; the pipelined bound, a "
                  f"store-and-forward link is slower): the declared stage [{_stage(declared)}] and the profile's own "
                  f"[{_stage(own)}]")
    weak = sorted(prevailing & _BELOW_DERIVED)
    return replace(declared, low=low, nominal=nominal, high=high, evidence_kind=weak[0] if weak else "derived",
                   citation="; ".join(c for c in (declared.citation, own.citation) if c), derivation=derivation,
                   note=declared.note or ((own.note or derivation) if weak else ""))

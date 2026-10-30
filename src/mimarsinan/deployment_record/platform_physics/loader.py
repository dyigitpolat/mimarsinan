"""Reading a physics profile: strict, description-backed, and unable to fork a number."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

from mimarsinan.deployment_record.platform_physics.area_components import resolve_area_declarations
from mimarsinan.deployment_record.platform_physics.constants import (  # [U-E3] the one constant judge
    constant_from_declaration, spec_for,
)
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.converter import ConverterTable
from mimarsinan.deployment_record.platform_physics.hosts.description import HOST_IDENTITY_FIELD, require_one_host
from mimarsinan.deployment_record.platform_physics.profile import (
    PHYSICS_FORMAT_VERSION,
    CarryBufferDeclarationError,
    CarryBufferTable,
    ExternalComponent,
    PhysicsConstantValue,
    PlatformPhysics,
    PlatformPhysicsValidity,
)
from mimarsinan.deployment_record.units import TIME, to_canonical

#: [consts] How an override declaration composes with the profile's own value of its key (the declaration's
#: ``combine``): ``replace`` (the default: an operator's departure replaces the profile's value) or ``slower`` (the
#: declaration is one stage of a series link whose other stage the profile declares, as a backing store's channel
#: beside the chip's own programming port: a streamed payload moves no faster than its slower stage, so the composed
#: time is the larger of the two corner by corner, the pipelined bound of the link (a store-and-forward link, which
#: adds its stages, is slower); a profile without its own stage takes the declared one).
COMBINE_RULES = ("replace", "slower")
#: [consts fix round] The closed field set of an override declaration; a field outside it refuses by name. [U-J]
#: ``host``: the host a host row describes (``hosts.description``), one host per deployment.
OVERRIDE_FIELDS = frozenset({"nominal", "low", "high", "unit", "evidence_kind", "citation", "derivation", "note",
                             "combine", HOST_IDENTITY_FIELD})
#: Evidence classes weaker than a shown derivation: a composition (arithmetic) cannot strengthen such a stage.
_BELOW_DERIVED = frozenset({"estimated"})

def profile_from_dict(data: Mapping[str, Any]) -> PlatformPhysics:
    """Build a profile from its JSON body, refusing anything the vocabulary rejects."""
    version = int(data.get("format_version", -1))
    if version != PHYSICS_FORMAT_VERSION:
        raise ValueError(
            f"physics profile format_version {version} != {PHYSICS_FORMAT_VERSION}; "
            f"migrate explicitly, never tolerate silently"
        )
    validity = PlatformPhysicsValidity.from_dict(data["validity"])
    # [U-A] Component-named area rows and the chip components' sum (area_global_fixed) at the chip's own node.
    declared, chip = resolve_area_declarations(data.get("constants") or {}, data.get("chip_components") or (),
                                               validity.technology_node_nm)
    constants = {key: constant_from_declaration(key, value) for key, value in declared.items()}
    conversion_model = dict(data.get("conversion_model") or {})
    # Validate the dataflow HERE: an undescribable one must fail at declaration,
    # not as a conversion count nobody can defend at pricing time.
    conversion_model_for(conversion_model)
    external = tuple(ExternalComponent.from_dict(c) for c in data.get("external_components") or ())
    for component in external:
        if declared := [key for key in component.constants if key in constants]:
            raise ValueError(f"{data['name']}: external component {component.name!r} names {declared}, which the "
                             "profile declares itself; an external component's constants come from beside it")
    return with_carry_buffer(PlatformPhysics(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description_file=str(data["description_file"]),
        validity=validity,
        constants=constants,
        conversion_model=conversion_model,
        external_components=external,
        converter=None if data.get("converter") is None else _converter(data["converter"]),
        chip_components=chip,
    ), data.get("carry_buffer"))


def with_carry_buffer(physics: PlatformPhysics, declaration: Optional[Mapping[str, Any]]) -> PlatformPhysics:
    """[U-G] The physics with ``declaration`` as its carry-buffer table over capacity (None removes it), judged as a
    profile's: a buffer constant the physics states flat may not be stated in a row as well (two values at once)."""
    table = None if declaration is None else CarryBufferTable.from_dict(declaration)
    if both := sorted(k for k in (table.keys() if table else ()) if k in physics.constants):
        raise CarryBufferDeclarationError(f"{physics.name}: {both} stated both flat and in carry_buffer rows")
    return replace(physics, carry_buffer=table)


def _converter(data: Mapping[str, Any]) -> ConverterTable:
    """[E-5] The profile's converter over resolution, judged as its constants are."""
    return ConverterTable.from_dict(data)


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
    ``combine: slower`` composes with the profile's own value instead (``COMBINE_RULES``). [U-J] Host rows naming
    their host (``HOST_IDENTITY_FIELD``) replace the profile's host rows whole, and host rows of two hosts refuse by
    name (``hosts.description.require_one_host``).
    """
    if not overrides:
        return physics
    replaced = require_one_host(overrides, physics.constants)
    constants = {key: value for key, value in physics.constants.items() if key not in replaced}
    for key, raw in overrides.items():
        spec = spec_for(key)
        existing = physics.constants.get(key)
        data = dict(raw)
        unknown = set(data) - OVERRIDE_FIELDS
        if unknown:
            raise ValueError(f"{key}: unknown fields {sorted(unknown)} in the declaration (an override declares "
                             f"{sorted(OVERRIDE_FIELDS)})")
        combine = str(data.pop("combine", "replace"))
        host = data.pop(HOST_IDENTITY_FIELD, None)
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
            note=(f"[host {host}] " if host else "") + str(data.get("note", "")),
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
        converter=physics.converter,
        carry_buffer=physics.carry_buffer,
        chip_components=physics.chip_components,
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

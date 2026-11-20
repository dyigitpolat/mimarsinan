"""Reading a physics profile: strict, description-backed, and unable to fork a number."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from mimarsinan.common.refusal.format_version import exact_version
from mimarsinan.deployment_record.platform_physics.area_components import (
    FIXED, fixed_area_declaration, resolve_area_declarations,
)
from mimarsinan.deployment_record.platform_physics.area_rule.declaration import (  # [U-AREA] the block and its judge
    AREA_RULE, AreaRuleBlock, converter_area_stated, judge_beside, judge_block)
from mimarsinan.deployment_record.platform_physics.area_rule.registry import load_registry
from mimarsinan.deployment_record.platform_physics.constants import (  # [U-E3] the one constant judge
    constant_from_declaration, spec_for,
)
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.converter import ConverterTable
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    ENERGY_PLANE_FIELD, HOST_IDENTITY_FIELD, SYSTEM, energy_plane_refusal, require_one_host,
)
from mimarsinan.deployment_record.platform_physics.hosts.admission import admit_host_rows, refuse_named, value_refusal
from mimarsinan.deployment_record.platform_physics.hosts.derivation import DERIVATION_FIELD, canonical
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD, judged_link_ends
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.hosts.store import (
    STORE_FIELD, BackingStoreError, attach_refusal, restated_store, store_from_declaration,
)
from mimarsinan.deployment_record.platform_physics.hosts.transfer import TRANSFER_FIELD, transfer_rows_from_declaration
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
#: ``host``: the host a host row describes (``hosts.description``), one host per deployment. [U-J2]
#: ``energy_plane``: whose energy a host energy row is (``hosts.description.ENERGY_PLANES``). [U-J5 L-2] ``link_ends``:
#: the ends of the host-chip link a link row counts at each corner (``hosts.link_ends``). [U-HOST2] ``derived_from``:
#: a row's declared derivation from its host description's rows (``hosts.derivation``).
OVERRIDE_FIELDS = frozenset({"nominal", "low", "high", "unit", "evidence_kind", "citation", "derivation", "note",
                             "combine", HOST_IDENTITY_FIELD, ENERGY_PLANE_FIELD, LINK_ENDS_FIELD, DERIVATION_FIELD})
#: Evidence classes weaker than a shown derivation: a composition (arithmetic) cannot strengthen such a stage.
_BELOW_DERIVED = frozenset({"estimated"})
#: [U-HOST2] A host row is its description's row or a declared derivation of its rows: its combination with another
#: stage is declared in its derivation (``hosts.derivation``), never by the loader's combine.
_HOST_COMBINE = ("a host row composes with no other stage through the loader's combine: its combination is declared "
                 "in its derivation (hosts.derivation's 'combine' over the description's rows)")

def profile_from_dict(data: Mapping[str, Any]) -> PlatformPhysics:
    """Build a profile from its JSON body, refusing anything the vocabulary rejects."""
    version = exact_version(data.get("format_version", -1), f"physics profile {data.get('name')!r}")  # [U-HOST3]
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
    admit_host_rows(constants, f"profile {data['name']!r}")  # [U-HOST] a profile declares no host row of its own
    conversion_model = dict(data.get("conversion_model") or {})
    # Validate the dataflow HERE: an undescribable one must fail at declaration,
    # not as a conversion count nobody can defend at pricing time.
    conversion_model_for(conversion_model)
    external = tuple(ExternalComponent.from_dict(c) for c in data.get("external_components") or ())
    for component in external:
        if declared := [key for key in component.constants if key in constants]:
            raise ValueError(f"{data['name']}: external component {component.name!r} names {declared}, which the "
                             "profile declares itself; an external component's constants come from beside it")
    converter = None if data.get("converter") is None else _converter(data["converter"])
    block = None if data.get(AREA_RULE) is None else AreaRuleBlock.from_dict(data[AREA_RULE])
    if block is not None:  # [U-AREA] judged after the area declarations, beside the profile's own rows
        judge_block(block, load_registry(), fixed_stated={"profile": _nominal(constants.get(FIXED))},
                    converter_table=converter_area_stated(constants, converter), validity=validity.geometry)
    return with_carry_buffer(PlatformPhysics(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description_file=str(data["description_file"]),
        validity=validity,
        constants=constants,
        conversion_model=conversion_model,
        external_components=external,
        converter=converter,
        chip_components=chip,
        backing_store=(None if data.get(STORE_FIELD) is None
                       else store_from_declaration(data[STORE_FIELD], str(data["name"]))),
        transfer_rows=transfer_rows_from_declaration(data.get(TRANSFER_FIELD), str(data["name"])),  # [U-J5 L-1]
        area_rule=block,
    ), data.get("carry_buffer"))


def _nominal(value: Optional[PhysicsConstantValue]) -> Optional[float]:
    return None if value is None else value.nominal


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
    name (``hosts.description.require_one_host``). [U-J3] A ``backing_store`` entry restates where the chip's store
    lives (``hosts.store``), a declared departure: a host-owned store drops the chip's memory PHYs and re-derives its
    ``area_global_fixed``, and a host attach row on a chip-owned store refuses by name. [U-HOST2] Every declared row
    naming a host, and every host row, is judged on its value before any composition (its description's own row or a
    declared derivation of its rows, ``hosts.admission``), and every host row of the result again, else refused by
    name; a host row's combination is declared in its derivation, never by ``combine``.
    """
    if not overrides:
        return physics
    overrides = dict(overrides)
    store, components, constants_of_store = _store(physics, overrides.pop(STORE_FIELD, None), overrides)
    replaced = require_one_host(overrides, physics.constants)
    constants = {key: value for key, value in physics.constants.items() if key not in replaced}
    # [U-J2] an override replaces its row whole, its stated energy plane with it
    system = {key for key in physics.system_energy_rows if key not in replaced and key not in overrides}
    ends = {key: value for key, value in physics.link_ends.items() if key not in replaced and key not in overrides}
    refused: Dict[str, str] = {}
    named: Set[str] = set()
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
        derived = data.pop(DERIVATION_FIELD, None)
        if (plane := data.pop(ENERGY_PLANE_FIELD, None)) is not None:
            if why := energy_plane_refusal(key, plane):
                raise ValueError(why)
            system |= {key} if plane == SYSTEM else set()
        if (counted := data.pop(LINK_ENDS_FIELD, None)) is not None:  # [U-J5 L-2] the link row's ends, per corner
            ends[key] = judged_link_ends(key, counted, f"{physics.name} (override {key})")
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
            host=str(host) if host else "",
            derived_from="" if derived is None else canonical(key, derived),
        )
        if host or spec.group == HOST:  # [U-HOST2] a row naming a host is judged on its declared value, before any
            named |= {str(host)} if host else set()  # composition, so no combine relabels what it declares
            why = [r for r in (value_refusal(key, value, key in system, ends.get(key)),
                               _HOST_COMBINE if spec.group == HOST and combine != "replace" else None) if r]
            if why:
                refused[key] = "; ".join(why)
                continue
            if spec.group != HOST:  # a chip row fed over the host's link is the chip's row once composed
                value = replace(value, host="", derived_from="")
        constants[key] = _slower(value, existing) if combine == "slower" and existing is not None else value
    refuse_named(refused, f"{physics.name}: the run's declared rows", named)
    constants.update(constants_of_store)
    if physics.area_rule is not None:  # [U-AREA] an override or a store restatement beside the block (M2, N-6)
        own = physics.constants.get(FIXED)
        judge_beside(physics.area_rule, converter_table="area_per_adc" in overrides, fixed_stated={
            "profile": None if own is None or own.overridden or FIXED in overrides else own.nominal,
            "override": _nominal(constants.get(FIXED)) if FIXED in overrides else None,
            "store": _nominal(constants_of_store.get(FIXED))})
    origin = admit_host_rows(constants, f"{physics.name} with the run's overrides", system, ends, named,  # [U-HOST2]
                             judged=[key for key in overrides if spec_for(key).group == HOST],
                             origin=physics.host_origin)  # [U-HOST3] kept rows by the origin they were judged under
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
        chip_components=components,
        system_energy_rows=tuple(sorted(system)),
        backing_store=store,
        transfer_rows=physics.transfer_rows,
        link_ends=ends,
        area_rule=physics.area_rule,
        host_origin=origin,
    )


def _store(physics: PlatformPhysics, declaration: Any, overrides: Mapping[str, Mapping[str, Any]]):
    """[U-J3] (the store, the chip components it keeps, the constants it re-derives): the profile's, or a run's
    restatement; a host attach row (a non-host row naming a host) on a chip-owned store refuses by name."""
    store, components, derived = physics.backing_store, physics.chip_components, {}
    if declaration is not None:
        store, components = restated_store(components, declaration, f"{physics.name} (a run's restatement)")
        if components != physics.chip_components:
            if FIXED in overrides:
                raise BackingStoreError(f"{physics.name}: {FIXED} is declared beside a {STORE_FIELD} restatement "
                                        "that changes the chip components, whose sum it is: the area is stated twice")
            row = fixed_area_declaration(components, physics.validity.technology_node_nm)
            derived = {FIXED: replace(constant_from_declaration(FIXED, row), overridden=True)}
    fed = [key for key, row in overrides.items() if row.get(HOST_IDENTITY_FIELD) and spec_for(key).group != HOST]
    if why := attach_refusal(store, fed):
        raise BackingStoreError(f"{physics.name}: {why}")
    return store, components, derived


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

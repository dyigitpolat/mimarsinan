"""Profile name + overrides -> the concrete physics a run is priced with."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.lane_blocks import TargetNode, lane_block_rows
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides, constant_from_declaration
from mimarsinan.deployment_record.platform_physics.profile import (
    PlatformPhysics,
    PlatformPhysicsValidity,
)
from mimarsinan.deployment_record.platform_physics.registry import get_platform_physics

#: The profile an operator gets when they declare constants for a target we ship none for.
CUSTOM_PROFILE_NAME = "custom"

_CUSTOM = PlatformPhysics(
    name=CUSTOM_PROFILE_NAME,
    display_name="Operator-declared physics",
    description_file="",
    validity=PlatformPhysicsValidity(
        measurement_kind="mixed",
        notes="Operator-declared constants only; no profile was selected, so the "
              "validity domain is whatever the operator's own numbers hold at.",
    ),
)


def resolve_platform_physics(
    profile_name: Optional[str], overrides: Optional[Mapping[str, Mapping[str, Any]]],
    lane_block: Optional[str] = None,
) -> Optional[PlatformPhysics]:
    """The physics for a run, or ``None`` when it declares none.

    ``None`` is meaningful: a run that names no profile and overrides nothing must
    report no absolute area/energy number at all, rather than one computed from
    framework defaults wearing a vendor's name. A declared ``lane_block`` (W08) adds
    the lane rows the physics lacks; it never makes a chip priceable on its own.
    """
    overrides = overrides or {}
    if not profile_name and not overrides:
        return None
    base = get_platform_physics(profile_name) if profile_name else _CUSTOM
    physics = apply_overrides(base, overrides)
    validity = physics.validity
    rows = lane_block_rows(lane_block, physics.constants,
                           TargetNode(technology_node_nm=validity.technology_node_nm, supply_v=validity.supply_v))
    if not rows:
        return physics
    constants = dict(physics.constants)
    constants.update({key: constant_from_declaration(key, data) for key, data in rows.items()})
    return replace(physics, constants=constants)


class OverrideBlockConflictError(ToolchainRefusal, ValueError):
    """Two named override blocks declare the same constant."""


def merge_override_blocks(blocks: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Named override blocks (a host, lane or backing-store block) merged into one
    ``platform_physics_overrides`` value, constant by constant.

    The resolver reads that key whole and a target override replaces a key's whole value, so blocks
    declared apart would silently lose one; merged here, each constant keeps its block's name in
    its note, and a constant two blocks declare refuses by name.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    owner: Dict[str, str] = {}
    for block, constants in blocks.items():
        for key, declaration in constants.items():
            if key in merged:
                raise OverrideBlockConflictError(
                    f"constant {key!r} is declared by override blocks {owner[key]!r} and {block!r}; "
                    "declare it in one block")
            entry = dict(declaration)
            entry["note"] = f"[{block}] {entry['note']}" if entry.get("note") else f"[{block}]"
            merged[key] = entry
            owner[key] = block
    return merged


#: N2b: the write-technology constants (CH-A1's rows and the per-byte commit energy), what a core type's own write
#: physics declares for its cells. The transfer channel (``e_dma_per_byte``, ``t_program_per_byte``) and every other
#: constant stay the chip's.
WRITE_TECHNOLOGY_KEYS = ("e_write_per_cell", "e_verify_read_per_cell", "write_verify_iterations", "t_write_per_unit",
                         "t_write_setup_per_event", "endurance_cycles", "e_program_per_byte")


class WritePhysicsError(ToolchainRefusal, ValueError):
    """A core type's ``write_physics`` that names no shipped physics profile."""


def write_physics_profile(name: str, sealed: Optional[Mapping[str, Mapping[str, Any]]] = None) -> PlatformPhysics:
    """The write physics a core type names: the copy its run sealed (``write_physics_resolved``), else the shipped
    profile; a name no profile ships refuses by name, listing the shipped ones."""
    if sealed and name in sealed:
        return PlatformPhysics.from_dict(sealed[name])
    try:
        return get_platform_physics(name)
    except KeyError as missing:
        raise WritePhysicsError(f"write_physics {name!r}: {missing.args[0]}") from None


def technology_physics(chip: PlatformPhysics, technology: Any) -> PlatformPhysics:
    """The physics a core type's cells are written under (``write_technology.CoreTechnology``): the chip's where it
    names no write physics, else the chip's with the named physics' write rows (its sealed copy, else the profile)."""
    if technology.write_physics is None:
        return chip
    named = technology.physics if technology.physics is not None else write_physics_profile(technology.write_physics)
    return with_write_technology(chip, named)


def with_write_technology(chip: PlatformPhysics, technology: PlatformPhysics) -> PlatformPhysics:
    """The chip's physics with every write-technology constant taken from ``technology``, absent where it declares
    none (never the chip's): the physics one core type's cells are written under."""
    constants = {k: v for k, v in chip.constants.items() if k not in WRITE_TECHNOLOGY_KEYS}
    constants.update({k: v for k, v in technology.constants.items() if k in WRITE_TECHNOLOGY_KEYS})
    return replace(chip, name=f"{chip.name} (writes: {technology.name})", constants=constants)

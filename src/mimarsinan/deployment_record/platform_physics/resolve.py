"""Profile name + overrides -> the concrete physics a run is priced with."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
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
    profile_name: Optional[str], overrides: Optional[Mapping[str, Mapping[str, Any]]]
) -> Optional[PlatformPhysics]:
    """The physics for a run, or ``None`` when it declares none.

    ``None`` is meaningful: a run that names no profile and overrides nothing must
    report no absolute area/energy number at all, rather than one computed from
    framework defaults wearing a vendor's name.
    """
    overrides = overrides or {}
    if not profile_name and not overrides:
        return None
    base = get_platform_physics(profile_name) if profile_name else _CUSTOM
    return apply_overrides(base, overrides)


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

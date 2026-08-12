"""Profile name + overrides -> the concrete physics a run is priced with."""

from __future__ import annotations

from typing import Any, Mapping, Optional

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

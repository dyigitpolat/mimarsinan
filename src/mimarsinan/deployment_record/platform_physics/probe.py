"""The all-constants physics probe: 'could a target ever back this axis'."""

from __future__ import annotations

from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS
from mimarsinan.deployment_record.platform_physics.profile import (
    PhysicsConstantValue,
    PlatformPhysics,
    PlatformPhysicsValidity,
)

#: Strictly positive: a zero constant would make a band touch zero (uninvertible for
#: throughput) and would read as a real "this costs nothing" declaration.
_PROBE_VALUE = 1.0

_PROBE_NOTE = (
    "capability probe placeholder — not a declaration about any real target"
)


def probe_physics() -> PlatformPhysics:
    """Every vocabulary constant declared, so availability reflects the AXIS only.

    Aggregates are declared too, which is deliberate: the probe then exercises the
    same supersession path a published-aggregate target takes.
    """
    return PlatformPhysics(
        name="__probe__",
        display_name="Capability probe",
        description_file="",
        validity=PlatformPhysicsValidity(
            measurement_kind="mixed", notes=_PROBE_NOTE,
        ),
        constants={
            key: PhysicsConstantValue(
                key=key,
                low=_PROBE_VALUE,
                nominal=_PROBE_VALUE,
                high=_PROBE_VALUE,
                unit=spec.display_unit,
                evidence_kind="estimated",
                note=_PROBE_NOTE,
            )
            for key, spec in PHYSICS_CONSTANTS.items()
        },
    )

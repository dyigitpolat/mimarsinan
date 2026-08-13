"""The Co-Design physics panel's payload: profiles, constants, completeness.

Everything here is a projection of declarations that already exist — the profile
registry, the constant vocabulary, and the objectives registry's own availability
predicates — so the panel and the launch behaviour cannot disagree about what a
target can back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from mimarsinan.deployment_record.objectives import OBJECTIVES
from mimarsinan.deployment_record.objectives.probes import run_capability_probe
from mimarsinan.deployment_record.platform_physics import (
    PHYSICS_CONSTANTS,
    PHYSICS_GROUPS,
    PlatformPhysics,
    available_profiles,
    get_platform_physics,
    keys_in_group,
)
from mimarsinan.deployment_record.platform_physics.resolve import (
    resolve_platform_physics,
)
from mimarsinan.gui.wizard.schema import _objective_label

#: The capability question is asked of the widest mode, so the readout describes the
#: PHYSICS rather than a search mode the user has not chosen yet.
_WIDEST_MODE = "joint"


def physics_profile_options() -> List[Dict[str, str]]:
    """The shipped profiles, as the selector's option rows."""
    return [
        {"id": name, "label": get_platform_physics(name).display_name}
        for name in available_profiles()
    ]


def _resolved(
    profile: str, overrides: Mapping[str, Any]
) -> Optional[PlatformPhysics]:
    return resolve_platform_physics(profile or "", dict(overrides or {}))


def physics_completeness(
    profile: str, overrides: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Which absolute objectives this declaration can back, and what is missing.

    The availability answer is the registry's own, so a green row here is exactly a
    run that will resolve, and a grey row is exactly one that will be refused.
    """
    physics = _resolved(profile, overrides)
    probe = run_capability_probe(_WIDEST_MODE, physics)
    rows: List[Dict[str, Any]] = []
    for spec in OBJECTIVES.all():
        if not OBJECTIVES.requires_physics(spec.key):
            continue
        available = spec.available(probe)
        rows.append({
            "key": spec.key,
            # The SAME name the objective picker shows: a readout that said
            # `chip_area_mm2` beside a chip reading "Chip Area (mm²)" would make
            # the user match them up by hand.
            "label": _objective_label(spec.key),
            "unit": spec.unit,
            "direction": spec.direction,
            "available": bool(available),
            "missing": "" if available else _first_missing(physics, spec.key),
        })
    return rows


#: Which constant each absolute axis needs FIRST, for a readout that names one thing
#: rather than a list nobody reads. Derived from the pricer's own refusal order.
_AXIS_HEAD_CONSTANT = {
    "chip_area_mm2": ("area_per_core_total", "area_per_cell"),
    "energy_per_inference_mj": ("e_synaptic_event_total", "e_mac"),
    "e2e_latency_s": ("t_cycle",),
    "throughput_inferences_s": ("t_cycle",),
}


def _first_missing(physics: Optional[PlatformPhysics], axis: str) -> str:
    """The constant this axis needs and does not have — named, not enumerated."""
    candidates = _AXIS_HEAD_CONSTANT.get(axis, ())
    if physics is None:
        return candidates[0] if candidates else "a declared physics profile"
    for key in candidates:
        if physics.has(key):
            continue
        return key
    return "a quantity this run does not produce"


def _constant_row(
    physics: Optional[PlatformPhysics], key: str, overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    spec = PHYSICS_CONSTANTS[key]
    row: Dict[str, Any] = {
        "key": key,
        "group": spec.group,
        "unit": spec.display_unit,
        "doc": spec.doc,
        "multiplicand": spec.multiplicand,
        "declared": False,
        "overridden": False,
        "banded": False,
        "low": None,
        "nominal": None,
        "high": None,
        "evidence_kind": None,
        "evidence_detail": "",
    }
    if physics is None or not physics.has(key):
        return row
    value = physics.constants[key]
    row.update({
        "declared": True,
        "overridden": bool(value.overridden),
        "unit": value.unit,
        "low": value.low,
        "nominal": value.nominal,
        "high": value.high,
        "banded": value.low != value.high,
        "evidence_kind": value.evidence_kind,
        "evidence_detail": value.citation or value.derivation or value.note,
    })
    return row


def physics_profile_panel(
    profile: str, overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """The whole panel: header, completeness readout, and grouped constant rows.

    EVERY vocabulary constant appears, declared or not — what a target has NOT said
    is what disables objectives, so hiding it would hide the reason.
    """
    physics = _resolved(profile, overrides)
    groups: List[Dict[str, Any]] = []
    if physics is not None:
        for group in PHYSICS_GROUPS:
            rows = [_constant_row(physics, key, overrides) for key in keys_in_group(group)]
            groups.append({
                "group": group,
                "declared_count": sum(1 for row in rows if row["declared"]),
                "total_count": len(rows),
                "constants": rows,
            })
    return {
        "selected": profile or "",
        "display_name": physics.display_name if physics else "",
        "validity": physics.validity.to_dict() if physics else None,
        "description_file": physics.description_file if physics else "",
        "groups": groups,
        "completeness": physics_completeness(profile, overrides),
    }

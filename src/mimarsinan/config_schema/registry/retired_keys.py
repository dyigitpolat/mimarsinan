"""Scoped retirement: keyed migration errors for documents declaring retired keys.

A retired key belongs to exactly one config sub-document — its SCOPE
(``deployment_parameters`` | ``platform_constraints``) — and every error row
and remedy op carries that scope, so the wizard applies clear/set in the
correct sub-document even after the key is gone from the registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from mimarsinan.chip_simulation.activation_semantics import axes_from_legacy

RETIREMENT_SCOPES = ("deployment_parameters", "platform_constraints")

_DP = "deployment_parameters"


def _retired_row(
    scope: str, key: str, message: str, remedies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {"key": key, "scope": scope, "message": message,
            "rule_id": "retired_key", "remedies": remedies}


def _clear_remedy(scope: str, key: str, label: str) -> Dict[str, Any]:
    return {"label": label, "action": "clear", "key": key, "scope": scope}


def _spiking_mode_migration_row(dp: Mapping[str, Any]) -> Dict[str, Any]:
    """The one-click migration for a document still declaring spiking_mode."""
    mode = dp.get("spiking_mode")
    schedule = dp.get("ttfs_cycle_schedule")
    try:
        family, variant = axes_from_legacy(mode, schedule)
    except ValueError as exc:
        return _retired_row(
            _DP, "spiking_mode",
            f"spiking_mode is retired — activation semantics are authored as "
            f"(spiking_family, spiking_variant) — and {exc}",
            [_clear_remedy(_DP, "spiking_mode", "Remove spiking_mode")],
        )
    ops = [
        {"action": "set", "key": "spiking_family", "value": family, "scope": _DP},
        {"action": "set", "key": "spiking_variant", "value": variant, "scope": _DP},
        {"action": "clear", "key": "spiking_mode", "scope": _DP},
        {"action": "clear", "key": "ttfs_cycle_schedule", "scope": _DP},
    ]
    return _retired_row(
        _DP, "spiking_mode",
        f"spiking_mode is retired: activation semantics are authored as "
        f"(spiking_family, spiking_variant). spiking_mode={mode!r} maps to "
        f"spiking_family={family!r}, spiking_variant={variant!r} — the exact "
        f"historical semantics (old 'lif' was the windowed discipline).",
        [{"label": f"Migrate to spiking_family='{family}' + "
                   f"spiking_variant='{variant}'", "ops": ops}],
    )


_RETIRED_STANDALONE_MESSAGES: Dict[str, Dict[str, str]] = {
    "deployment_parameters": {
        "ttfs_cycle_schedule": (
            "ttfs_cycle_schedule is retired: the schedule is the spiking_variant "
            "axis ('synchronized' or 'cascaded' under spiking_family='ttfs'). "
            "Remove the key."
        ),
        "lif_execution_discipline": (
            "lif_execution_discipline is retired: the temporal discipline is the "
            "spiking_variant axis; the internal executor evaluation form is "
            "derivation-owned. Remove the key."
        ),
        "lif_per_hop_retiming": (
            "lif_per_hop_retiming is retired: per-hop re-timing is part of the "
            "windowed-lif (lif_sync) semantics via the exact-QAT recipe pairing, "
            "never a knob. Remove the key."
        ),
    },
    "platform_constraints": {
        "allow_weight_reuse": (
            "allow_weight_reuse is retired: weight reuse is always on — weight "
            "banks are built unconditionally and the reuse-phase report always "
            "prints; bank-resident scheduling is chosen by schedule_policy. "
            "Remove the key."
        ),
    },
}


def retired_keys_in_scope(scope: str) -> frozenset:
    """Every retired document key of one scope (spiking_mode carries its own
    migration row instead of a standalone message)."""
    keys = set(_RETIRED_STANDALONE_MESSAGES.get(scope, {}))
    if scope == _DP:
        keys.add("spiking_mode")
    return frozenset(keys)


def retired_scope_of(key: str) -> Optional[str]:
    """The scope a retired key belongs to, or ``None`` for a non-retired key."""
    for scope in RETIREMENT_SCOPES:
        if key in retired_keys_in_scope(scope):
            return scope
    return None


def retired_key_errors(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Keyed migration rows for a config-shaped mapping declaring retired keys.

    Inspects BOTH scopes; one row per declared retired key, each carrying its
    scope. The spiking_mode row carries the full one-click axes migration (its
    ops also clear a declared schedule).
    """
    rows: List[Dict[str, Any]] = []
    for scope in RETIREMENT_SCOPES:
        body = config.get(scope)
        if not isinstance(body, Mapping):
            continue
        if scope == _DP and "spiking_mode" in body:
            rows.append(_spiking_mode_migration_row(body))
        for key, message in _RETIRED_STANDALONE_MESSAGES[scope].items():
            if key in body:
                rows.append(_retired_row(
                    scope, key, message,
                    [_clear_remedy(scope, key, f"Remove {key}")],
                ))
    return rows

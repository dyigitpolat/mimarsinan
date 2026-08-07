"""Keyed migration errors for documents declaring the retired taxonomy keys."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.activation_semantics import axes_from_legacy


def _retired_row(
    key: str, message: str, remedies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {"key": key, "message": message, "rule_id": "retired_key",
            "remedies": remedies}


def _clear_remedy(key: str, label: str) -> Dict[str, Any]:
    return {"label": label, "action": "clear", "key": key}


def _spiking_mode_migration_row(dp: Mapping[str, Any]) -> Dict[str, Any]:
    """The one-click migration for a document still declaring spiking_mode."""
    mode = dp.get("spiking_mode")
    schedule = dp.get("ttfs_cycle_schedule")
    try:
        family, variant = axes_from_legacy(mode, schedule)
    except ValueError as exc:
        return _retired_row(
            "spiking_mode",
            f"spiking_mode is retired — activation semantics are authored as "
            f"(spiking_family, spiking_variant) — and {exc}",
            [_clear_remedy("spiking_mode", "Remove spiking_mode")],
        )
    ops = [
        {"action": "set", "key": "spiking_family", "value": family},
        {"action": "set", "key": "spiking_variant", "value": variant},
        {"action": "clear", "key": "spiking_mode"},
        {"action": "clear", "key": "ttfs_cycle_schedule"},
    ]
    return _retired_row(
        "spiking_mode",
        f"spiking_mode is retired: activation semantics are authored as "
        f"(spiking_family, spiking_variant). spiking_mode={mode!r} maps to "
        f"spiking_family={family!r}, spiking_variant={variant!r} — the exact "
        f"historical semantics (old 'lif' was the windowed discipline).",
        [{"label": f"Migrate to spiking_family='{family}' + "
                   f"spiking_variant='{variant}'", "ops": ops}],
    )


_RETIRED_STANDALONE_MESSAGES = {
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
}


def retired_spiking_key_errors(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Keyed migration rows for documents declaring the retired taxonomy keys.

    One row per declared retired key; the spiking_mode row carries the full
    one-click axes migration (its ops also clear a declared schedule).
    """
    dp = config.get("deployment_parameters")
    if not isinstance(dp, Mapping):
        return []
    rows: List[Dict[str, Any]] = []
    if "spiking_mode" in dp:
        rows.append(_spiking_mode_migration_row(dp))
    for key, message in _RETIRED_STANDALONE_MESSAGES.items():
        if key in dp:
            rows.append(_retired_row(
                key, message, [_clear_remedy(key, f"Remove {key}")],
            ))
    return rows

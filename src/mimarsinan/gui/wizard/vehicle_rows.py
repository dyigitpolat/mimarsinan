"""Per-vehicle support/on/why rows, computable for EVERY draft."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mimarsinan.chip_simulation.activation_semantics import (
    resolve_activation_semantics,
)
from mimarsinan.config_schema.registry import (
    Category,
    FieldType,
    REGISTRY,
    parse_deployment_document,
)
from mimarsinan.config_schema.recipe_fold import resolve_backend_enable
from mimarsinan.config_schema.resolve import effective_view
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy


def vehicle_rows(draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-vehicle state rows: unrelated errors must never remove the rows
    the on/off toggles render from."""
    dp = parse_deployment_document(draft or {}).dp
    effective = effective_view(dp)

    sim_enables: Optional[Dict[str, bool]]
    if str(effective.get("core_semantics") or "spiking") == "mvm":
        # Value-domain deployment: spiking simulators are structurally off —
        # the rows say so instead of pretending mode support.
        rows = []
        for key, entry in REGISTRY.items():
            if (entry.group != "deployment_target"
                    or entry.category is not Category.DERIVED
                    or entry.type is not FieldType.BOOL):
                continue
            rows.append({
                "key": key, "label": entry.label,
                "declared": isinstance(dp.get(key), bool),
                "supported": False, "on": False,
                "why": "off — spiking simulators cannot run the value-domain "
                       "(core_semantics='mvm') family",
            })
        return rows
    try:
        semantics = resolve_activation_semantics(effective)
        mode = semantics.legacy_spiking_mode
        schedule = semantics.legacy_ttfs_cycle_schedule
        recipe = ConversionPolicy.derive(
            mode, schedule, spiking_variant=semantics.variant)
        sim_enables = dict(recipe.sim_enables)
        opt_in = set(recipe.sim_opt_in)
    except ValueError:
        mode = str(effective.get("spiking_family"))
        schedule = None
        sim_enables = None
        opt_in = set()

    rows: List[Dict[str, Any]] = []
    for key, entry in REGISTRY.items():
        if (entry.group != "deployment_target"
                or entry.category is not Category.DERIVED
                or entry.type is not FieldType.BOOL):
            continue
        declared = dp.get(key)
        row: Dict[str, Any] = {
            "key": key,
            "label": entry.label,
            "declared": isinstance(declared, bool),
        }
        if sim_enables is None:
            row.update(supported=None, on=None,
                       why=f"unknown spiking family/variant {mode!r} — fix "
                           "the semantics to see vehicle support")
        else:
            supported = bool(sim_enables.get(key, False))
            on = resolve_backend_enable(
                supported=supported, declared=declared, opt_in=key in opt_in)
            why = None
            if entry.why is not None:
                why = entry.why({
                    "spiking_mode": mode, "ttfs_cycle_schedule": schedule, key: on,
                })
            row.update(supported=supported, on=on, why=why)
        rows.append(row)
    return rows

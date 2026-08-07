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
from mimarsinan.config_schema.resolve import effective_view
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy


def vehicle_rows(draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-vehicle state rows: unrelated errors must never remove the rows
    the on/off toggles render from."""
    dp = parse_deployment_document(draft or {}).dp
    effective = effective_view(dp)

    sim_enables: Optional[Dict[str, bool]]
    try:
        semantics = resolve_activation_semantics(effective)
        mode = semantics.legacy_spiking_mode
        schedule = semantics.legacy_ttfs_cycle_schedule
        sim_enables = dict(ConversionPolicy.derive(mode, schedule).sim_enables)
    except ValueError:
        mode = str(effective.get("spiking_family"))
        schedule = None
        sim_enables = None

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
            on = bool(supported and declared is not False)
            why = None
            if entry.why is not None:
                why = entry.why({
                    "spiking_mode": mode, "ttfs_cycle_schedule": schedule, key: on,
                })
            row.update(supported=supported, on=on, why=why)
        rows.append(row)
    return rows

"""Normalized campaign-ledger science-row schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from mimarsinan.chip_simulation.coverage_ledger import (
    HypervolumeCell,
    classify_validity_tier,
)
from mimarsinan.chip_simulation.hypervolume_axis_encoder import (
    cell_coordinates_from_row,
    syncs_from_row,
)
from mimarsinan.chip_simulation.parity_contract import parity_contract_metadata


class LedgerSchemaError(ValueError):
    """A ledger record cannot be normalized into a science row."""


LEDGER_SCHEMA_VERSION = "mimarsinan-ledger-v1"
DEFAULT_TIMING_PHASES = (
    "total",
    "tuning",
    "ramp",
    "recovery",
    "stabilization",
    "evaluation",
)
_TIMING_FIELDS = (
    "wall_s",
    "tuning_wall_s",
    "total_deployment_wall_s",
    "max_ft_pass_wall_s",
    "gradual_s",
)


def _copy_if_present(out: dict[str, Any], src: Mapping[str, Any], key: str) -> None:
    if key in src and src[key] is not None:
        out[key] = src[key]


def _wall_s(row: Mapping[str, Any]) -> float | None:
    for key in ("run_wall_s", "wall_s", "total_wall_s"):
        value = row.get(key)
        if value is not None:
            return float(value)
    result = row.get("result")
    if isinstance(result, Mapping) and result.get("wall_s") is not None:
        return float(result["wall_s"])
    timing = row.get("timing")
    if isinstance(timing, Mapping) and timing.get("total_wall_s") is not None:
        return float(timing["total_wall_s"])
    return None


def fastest_successful_baseline_wall_s(rows: Iterable[Mapping[str, Any]]) -> float:
    """Return the fastest successful deployed row wall time."""
    candidates: list[float] = []
    for row in rows:
        if int(row.get("returncode", 0)) != 0:
            continue
        if row.get("deployed_acc") is None and row.get("deployed_acc_mean") is None:
            continue
        wall = _wall_s(row)
        if wall is not None:
            candidates.append(wall)
    if not candidates:
        raise ValueError("no successful baseline row with deployed accuracy and wall time")
    return min(candidates)


def with_relative_timing(row: Mapping[str, Any], baseline_wall_s: float) -> dict[str, Any]:
    """Return ``row`` annotated with relative timing against ``baseline_wall_s``."""
    baseline = float(baseline_wall_s)
    if baseline <= 0.0:
        raise ValueError("baseline_wall_s must be positive")
    out = dict(row)
    wall = _wall_s(out)
    out["timing_baseline_wall_s"] = baseline
    out["relative_time"] = None if wall is None else float(wall) / baseline
    out["faster_than_baseline"] = (
        None if out["relative_time"] is None else out["relative_time"] < 1.0
    )
    return out


def default_timing_schema() -> dict[str, Any]:
    return {
        phase: {"wall_s": None, "relative_to_baseline": None}
        for phase in DEFAULT_TIMING_PHASES
    }


def normalize_step_metrics(step_metrics: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize per-step metric/timing records to a stable JSON shape."""
    normalized: list[dict[str, Any]] = []
    for step in step_metrics:
        name = str(step.get("name", ""))
        if not name:
            raise ValueError("step metric record missing non-empty name")
        timing = dict(step.get("timing") or {})
        timing.setdefault("wall_s", None)
        timing.setdefault("relative_to_baseline", None)
        normalized.append(
            {
                "name": name,
                "status": str(step.get("status", "planned")),
                "metrics": dict(step.get("metrics") or {}),
                "timing": timing,
            }
        )
    return normalized


def normalize_planned_ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a planned campaign row before any GPU result exists."""
    out = deepcopy(dict(row))
    out.setdefault("schema", LEDGER_SCHEMA_VERSION)
    out.setdefault("row_type", "planned")
    out.setdefault("returncode", None)
    out.setdefault("deployed_acc", None)
    out.setdefault("run_wall_s", None)
    out.setdefault("relative_time", None)
    out.setdefault("timing_baseline_wall_s", None)
    out.setdefault("faster_than_baseline", None)
    out.setdefault("timing", default_timing_schema())
    out.setdefault("probes", {})
    out.setdefault("axes", {})
    out.setdefault("acceptance", {})
    out.setdefault("budget_schedule", {})
    if out.get("axes"):
        out.setdefault("parity_contracts", parity_contract_metadata(out["axes"]))
    out["step_metrics"] = normalize_step_metrics(out.get("step_metrics") or [])
    return out


def _timing(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in _TIMING_FIELDS:
        value = row.get(key)
        if value is not None:
            values[key] = float(value)
    return values


def _provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dict(row.get("provenance") or {})
    for key in ("source", "source_file", "config", "experiment", "run_dir", "run_id"):
        if key in row and row[key] is not None and key not in provenance:
            provenance[key] = row[key]
    return provenance


def _cost_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("cost_record") is not None:
        return {"kind": "measured_cost_record"}
    if row.get("cost_proxy") is not None or row.get("cost_model") is not None:
        return {"kind": "proxy"}
    return {"kind": "missing"}


def normalize_ledger_record(
    row: Mapping[str, Any], *, require_science: bool = True,
) -> dict[str, Any]:
    """Return a normalized science ledger row.

    The normal form keeps the original measurement fields, but stamps the row with
    explicit hypervolume axes, a canonical cell key, canonical validity tier,
    timing sub-record, and provenance. Rows without ``deployment_validity`` can
    pass through unchanged when ``require_science`` is false.
    """
    raw_tier = row.get("deployment_validity")
    tier = classify_validity_tier(raw_tier)
    if tier is None:
        if require_science:
            raise LedgerSchemaError("science ledger rows require deployment_validity")
        return dict(row)
    vehicle = row.get("model") or row.get("model_type")
    if not vehicle:
        raise LedgerSchemaError("science ledger rows require model or model_type")

    syncs = syncs_from_row(row)
    coords = cell_coordinates_from_row(row, sync=syncs[0])
    data = coords.as_cell_kwargs()
    data["vehicle"] = str(vehicle)
    cell = HypervolumeCell(**data)

    out = dict(row)
    out.update(
        {
            "model": data["vehicle"],
            "dataset": data["dataset"],
            "spiking_mode": data["firing"],
            "sync": data["sync"],
            "syncs": list(syncs),
            "backend": data["backend"],
            "regime": data["regime"],
            "quantization": data["quantization"],
            "pruning": data["pruning"],
            "mapping_strategy": data["mapping_strategy"],
            "S": data["s"],
            "depth": data["depth"],
            "deployment_validity": str(raw_tier),
            "deployment_validity_tier": tier.name,
            "validity": {
                "raw": str(raw_tier),
                "tier": tier.name,
                "owner": row.get("flag_owner"),
                "fix_path": row.get("flag_fix_path") or row.get("fix_path"),
            },
            "hypervolume_cell_key": cell.cell_key,
            "cost_provenance": _cost_provenance(row),
        }
    )
    if row.get("schedule") in ("cascaded", "synchronized"):
        out["schedule"] = data["sync"]
    else:
        out.pop("schedule", None)
    for key in (
        "deployed_acc",
        "deployed_acc_mean",
        "ann_acc",
        "ann_test_acc_mean",
        "retention_pp",
        "verdict",
        "n_eval",
        "seed",
        "study",
    ):
        _copy_if_present(out, row, key)
    timing = _timing(row)
    if timing:
        out["timing"] = timing
    provenance = _provenance(row)
    if provenance:
        out["provenance"] = provenance
    return out

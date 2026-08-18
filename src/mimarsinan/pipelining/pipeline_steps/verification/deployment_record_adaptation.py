"""The run directory's adaptation artifacts, read into the record's fragments.

Two readers over what the tuner-hosting steps left behind: the AC5 wall bundle
(``ft_pass_walls.json`` + the ``retention_ledger.json`` detail line) and the
[TS5] controller-ledger roll-up over every ``<Step>.adaptation_ledger.json``.
FAIL LOUD — a malformed artifact is a defect, never a shrug.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.chip_simulation.cost_extraction import FT_PASS_WALLS_FILENAME
from mimarsinan.deployment_record.schema import (
    AdaptationRecord,
    AdaptationSummaryRecord,
    FtPassWallRecord,
    Provenance,
)
from mimarsinan.tuning.orchestration.adaptation_ledger import LEDGER_ENTRY_KEY
from mimarsinan.tuning.orchestration.run_instrumentation import (
    RETENTION_LEDGER_FILENAME,
)

LEDGER_ARTIFACT_SUFFIX = f".{LEDGER_ENTRY_KEY}.json"
"""What the cache's ``basic`` strategy names a step's sealed ledger entry."""

_TOTALS = (
    "proposed", "accepted", "rejected", "retries", "recovery_steps",
    "probe_evals", "endpoint_steps", "total_steps",
)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def adaptation_from_run_dir(
    working_directory: str, *, tuner_steps_resolved: bool
) -> Tuple[Optional[AdaptationRecord], str]:
    """Read the run's adaptation artifacts into the fragment (+ ledger detail).

    ``ft_pass_walls.json`` populates the wall bundle; a resolved tuner-hosting
    step that recorded no FT passes still attaches an (empty) fragment so the
    seal's availability matrix holds honestly. Returns ``(None, "")`` when the
    run neither hosts tuners nor recorded walls.
    """
    walls_path = os.path.join(working_directory, FT_PASS_WALLS_FILENAME)
    record: Optional[AdaptationRecord] = None
    if os.path.exists(walls_path):
        data = _read_json(walls_path) or {}
        record = AdaptationRecord(
            max_ft_pass_wall_s=float(data.get("max_ft_pass_wall_s", 0.0)),
            ft_pass_walls=tuple(
                FtPassWallRecord(
                    label=str(entry["label"]), wall_s=float(entry["wall_s"])
                )
                for entry in data.get("passes") or ()
            ),
        )
    elif tuner_steps_resolved:
        record = AdaptationRecord(max_ft_pass_wall_s=0.0, ft_pass_walls=())

    detail_parts: List[str] = []
    if record is not None and not record.ft_pass_walls:
        detail_parts.append("no FT passes recorded (ft_pass_walls.json absent)")
    ledger_path = os.path.join(working_directory, RETENTION_LEDGER_FILENAME)
    if os.path.exists(ledger_path):
        entries = (_read_json(ledger_path) or {}).get("entries") or []
        detail_parts.append(
            f"retention_ledger.json: {len(entries)} tuner-step entries")
    return record, "; ".join(detail_parts)


def adaptation_summary_from_run_dir(
    working_directory: str,
) -> Optional[AdaptationSummaryRecord]:
    """[TS5] Fold every step's sealed controller ledger into the record's totals.

    Totals and stall counts SUM across the adaptation steps; ``completed_via``
    keys the path each step's rate search exited through by step name (a step
    whose search never completed is simply absent). ``None`` when the run sealed
    no ledger at all — the record then carries no fragment, exactly as before.
    """
    paths = sorted(glob.glob(os.path.join(
        working_directory, f"*{LEDGER_ARTIFACT_SUFFIX}"
    )))
    if not paths:
        return None
    totals: Dict[str, int] = {name: 0 for name in _TOTALS}
    stalls: Dict[str, int] = {}
    completed: Dict[str, str] = {}
    for path in paths:
        payload = _read_json(path) or {}
        step = os.path.basename(path)[: -len(LEDGER_ARTIFACT_SUFFIX)]
        for name, value in (payload.get("totals") or {}).items():
            totals[name] = totals.get(name, 0) + int(value)
        for name, count in (payload.get("stalls_by_path") or {}).items():
            stalls[name] = stalls.get(name, 0) + int(count)
        if payload.get("completed_via") is not None:
            completed[step] = str(payload["completed_via"])
    return AdaptationSummaryRecord(
        **totals, stalls_by_path=stalls, completed_via=completed,
    )


def attach_adaptation_fragments(
    builder: Any, working_directory: str, *, tuner_steps_resolved: bool,
    step_name: str,
) -> bool:
    """Attach both run-dir adaptation fragments; returns whether walls attached.

    Both are optional by construction — a run with no tuner-hosting step
    attaches neither and seals exactly as it did before either existed.
    """
    walls, detail = adaptation_from_run_dir(
        working_directory, tuner_steps_resolved=tuner_steps_resolved,
    )
    if walls is not None:
        builder.attach("adaptation", walls, Provenance(
            kind="measured", producer="tuner ft_pass_wall_metrics",
            step=step_name, detail=detail,
        ))
    summary = adaptation_summary_from_run_dir(working_directory)
    if summary is not None:
        builder.attach("adaptation_ledger", summary, Provenance(
            kind="measured", producer="tuner-step adaptation_ledger artifacts",
            step=step_name,
            detail=f"completed_via: {sorted(summary.completed_via)}",
        ))
    return walls is not None

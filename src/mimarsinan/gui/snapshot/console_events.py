"""Best-effort [TAG] console-line parser: legacy-run backfill for events.jsonl.

Live runs never parse their own prints — ``reporter.event`` is the primary
channel; this exists only so runs recorded before events.jsonl still get
annotation lanes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

_GATE_ACTION_RE = re.compile(r"\[MBH-GATE\]\s+(?:tuner=(\S+)\s+)?(\w+)")


def _coerce(raw: str) -> Any:
    text = raw.strip().strip("'\"").rstrip("),")
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _kv_payload(line: str) -> Dict[str, Any]:
    return {key: _coerce(value) for key, value in _KV_RE.findall(line)}


def _parse_line(line: str) -> Optional[Dict[str, Any]]:
    if "[MBH-GATE]" in line:
        match = _GATE_ACTION_RE.search(line)
        if not match:
            return None
        tuner, action = match.groups()
        if action == "constructive_stall":
            action = "stall"
        if action not in ("entry", "accept", "reject", "stall"):
            return None
        payload = _kv_payload(line.split("]", 1)[1])
        payload["action"] = action
        if tuner:
            payload["tuner"] = tuner
        return {"kind": "mbh_gate", "payload": payload}
    if "[MBH-ENDPOINT]" in line:
        return {"kind": "mbh_endpoint", "payload": _kv_payload(line)}
    if "[PROFILE]" in line:
        payload = _kv_payload(line)
        step = re.search(r"step='([^']+)'", line)
        if step:
            payload["step"] = step.group(1)
        wall = re.search(r"wall=\s*([0-9.]+)s", line)
        if wall:
            payload["wall_s"] = float(wall.group(1))
        metric = re.search(r"metric=([0-9.]+)", line)
        if metric:
            payload["metric"] = float(metric.group(1))
        return {"kind": "profile", "payload": payload}
    if "[LR-REFUSE]" in line:
        return {"kind": "lr_refusal", "payload": {"message": line.strip()}}
    return None


def parse_console_events(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse console.jsonl records into event records (same shape as events.jsonl).

    ``step`` attribution is unavailable in console logs, so events carry an
    empty step name except [PROFILE] lines (which name their step).
    """
    events: List[Dict[str, Any]] = []
    seq = 0
    for record in records:
        parsed = _parse_line(str(record.get("line", "")))
        if parsed is None:
            continue
        seq += 1
        events.append({
            "seq": seq,
            "step": parsed["payload"].get("step", ""),
            "kind": parsed["kind"],
            "payload": parsed["payload"],
            "timestamp": record.get("ts", 0.0),
            "backfilled": True,
        })
    return events

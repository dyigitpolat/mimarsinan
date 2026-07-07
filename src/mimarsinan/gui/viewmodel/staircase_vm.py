"""The D-hat ratchet staircase: the M-invariant made visible (and enforced)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


class StaircaseMonotonicityError(ValueError):
    """The accepted best-D-hat sequence decreased — the M-invariant is broken."""


def build_staircase(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Gate events -> per-tuner staircases of accepted best-D-hat + probe dots.

    ``best_full_acc`` only ever rises within a tuner; a downward step is a
    ratchet violation and raises rather than rendering a lie.
    """
    tuners: Dict[str, Dict[str, Any]] = {}
    for record in events:
        if record.get("kind") != "mbh_gate":
            continue
        payload = record.get("payload") or {}
        tuner = str(payload.get("tuner", ""))
        lane = tuners.setdefault(tuner, {
            "tuner": tuner, "step": record.get("step"),
            "probes": [], "staircase": [], "stalled": False, "entry": None,
        })
        action = payload.get("action")
        index = len(lane["probes"])
        if action == "entry":
            lane["entry"] = payload.get("best_full_acc")
            lane["staircase"].append({"i": index, "best": payload.get("best_full_acc")})
        elif action in ("accept", "reject"):
            lane["probes"].append({
                "i": index,
                "rung": payload.get("rung"),
                "full_acc": payload.get("full_acc"),
                "accepted": action == "accept",
            })
            lane["staircase"].append({"i": index + 1, "best": payload.get("best_full_acc")})
        elif action == "stall":
            lane["stalled"] = True

    for lane in tuners.values():
        best_values = [
            point["best"] for point in lane["staircase"] if point["best"] is not None
        ]
        for previous, current in zip(best_values, best_values[1:]):
            if current < previous:
                raise StaircaseMonotonicityError(
                    f"tuner {lane['tuner']!r}: best_full_acc fell "
                    f"{previous:.6f} -> {current:.6f}; the D-hat ratchet only rises"
                )
    return {"tuners": list(tuners.values())}


def highwater(events: Sequence[Mapping[str, Any]]) -> Any:
    """The run-level D-hat high-water implied by the gate/endpoint events."""
    best = None
    for record in events:
        payload = record.get("payload") or {}
        candidates: List[Any] = []
        if record.get("kind") == "mbh_gate":
            candidates = [payload.get("best_full_acc"), payload.get("full_acc")]
        elif record.get("kind") == "mbh_endpoint":
            candidates = [payload.get("exit"), payload.get("entry")]
        for value in candidates:
            if value is not None and (best is None or value > best):
                best = value
    return best

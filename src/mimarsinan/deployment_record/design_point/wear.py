"""[fix_b] Wear per declared core type: a row's lifetime is the shortest of its core types' lifetimes, each from its
endurance band, its most-written cell's writes per inference at the row's batch and the declared rate.

A row of two write technologies (resistive static tiles written per period beside SRAM operand cores written every
inference) has no single writes-per-cell value or endurance band: one mixed value (4/64 static plus 1 operand write
read as 1.0625 writes per inference against one NVM band) neither derives nor checks its verdict. Each core type
states its own wear here; the row's hours are the corner-wise minimum and the binding type is named, and the reader
(``schema.validate_instance``) recomputes every entry from endurance, wear and rate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

S_PER_H = 3600.0
#: The verdict order of the row: its worst core type.
ORDER = {"admitted": 0, "straddles": 1, "inadmissible": 2}
RTOL = 1e-6


def hours_of(endurance: Sequence[float], writes: float, rate_hz: float) -> Optional[List[float]]:
    """Endurance over wear, corner-wise: E / (w r 3600) hours; None where no cell of the type is written."""
    if not writes:
        return None
    return [float(e) / (float(writes) * float(rate_hz) * S_PER_H) for e in endurance]


def verdict_of(hours: Optional[Sequence[float]], floor: float) -> str:
    """The write law's rule: inadmissible when the longest lifetime fails the floor, admitted when the shortest clears
    it, else straddles; a type that writes no cell is admitted."""
    if hours is None:
        return "admitted"
    return "inadmissible" if hours[2] < floor else "admitted" if hours[0] >= floor else "straddles"


def label(entry: Mapping[str, Any]) -> str:
    return f"{entry['role']} ({entry.get('write_technology')})"


def binding(entries: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The core type whose lifetime binds at the low corner, by name; None where no type is written."""
    stated = [e for e in entries if e.get("hours") is not None]
    return None if not stated else label(min(stated, key=lambda e: e["hours"][0]))


def _close(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(abs(x - y) <= RTOL * max(abs(y), 1e-300) for x, y in zip(a, b))


def wear_problems(row: Mapping[str, Any], declarations: Mapping[str, Any]) -> List[str]:
    """Every violation of the per-core-type rules in a judged row: a row of several write technologies states its
    lifetime per core type; the entries are the chip's core types; each entry's hours and verdict are endurance / wear
    / rate and the floor's verdict; the row's hours are the corner-wise minimum, its verdict the worst, its binding
    type named."""
    declared = {(c["role"], c.get("write_technology")) for c in row["chip"]["core_types"]}
    life = row.get("lifetime") or {}
    entries = life.get("per_core_type")
    if entries is None:
        return ["a row of several write technologies states its lifetime per core type"] \
            if len({t for _, t in declared}) > 1 else []
    out: List[str] = []
    stated = {(e["role"], e.get("write_technology")) for e in entries}
    if declared and stated != declared:
        out.append(f"lifetime.per_core_type {sorted(map(str, stated))} is not the chip's core types "
                   f"{sorted(map(str, declared))}")
    rate, floor = float(declarations["inference_rate_hz"]), float(declarations["min_lifetime_hours"])
    for index, entry in enumerate(entries):
        writes, endurance = entry["max_writes_per_cell_per_inference"], entry["endurance"]
        if writes is None or (endurance is None and writes):
            continue
        want = None if not writes or endurance is None else hours_of(endurance["cycles"], writes, rate)
        if (want is None) != (entry["hours"] is None) or (want is not None and not _close(entry["hours"], want)):
            out.append(f"lifetime.per_core_type[{index}]: hours {entry['hours']} are not endurance / wear / rate "
                       f"{want}")
        if entry["verdict"] != verdict_of(want, floor):
            out.append(f"lifetime.per_core_type[{index}]: verdict {entry['verdict']!r} is not the floor's "
                       f"{verdict_of(want, floor)!r}")
    hours = [e["hours"] for e in entries if e.get("hours") is not None]
    if hours and (life.get("hours") is None or not _close(life["hours"], [min(h[c] for h in hours) for c in range(3)])):
        out.append("lifetime.hours are not the corner-wise minimum over the core types")
    if life.get("binding") != binding(entries):
        out.append(f"lifetime.binding {life.get('binding')!r} is not the binding core type {binding(entries)!r}")
    verdicts = [e["verdict"] for e in entries if e.get("verdict") in ORDER]
    if verdicts and life.get("verdict") != max(verdicts, key=ORDER.__getitem__):
        out.append(f"lifetime.verdict {life.get('verdict')!r} is not the worst core type's")
    return out


def single_type_wear(writes: Mapping[str, Any], lifetime: Mapping[str, Any], role: str,
                     technology: Optional[str]) -> Dict[str, Any]:
    """The per-core-type blocks of a row of one core type: its writes and its lifetime as one entry each."""
    entry_writes = {"role": role, "write_technology": technology,
                    "static_written_cells_per_period": int(writes["static_written_cells_per_period"]),
                    "operand_written_cells_per_inference": int(writes["operand_written_cells_per_inference"]),
                    "max_writes_per_cell_per_inference": writes["max_writes_per_cell_per_inference"]}
    entry_life = {"role": role, "write_technology": technology,
                  "max_writes_per_cell_per_inference": writes["max_writes_per_cell_per_inference"],
                  "endurance": lifetime.get("endurance"), "hours": lifetime.get("hours"),
                  "verdict": lifetime.get("verdict")}
    return {"writes": [entry_writes], "lifetime": [entry_life], "binding": binding([entry_life])}

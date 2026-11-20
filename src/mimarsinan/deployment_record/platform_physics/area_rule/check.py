"""[U-AREA] The registry's own evidence check (T-9): every value object of the data files is checked.

- Every cited id is an evidence entry; a published value equals its entry.
- A derived or fitted value states its arithmetic; a fitted value states its held-out report.
- Every derived value is re-evaluated from its expression, every fitted value re-aggregated from its points (a law
  re-fitted, the node range re-read, a band re-derived from its per-chip values), and compared with its stored value.
- No key of any data file names a published chip (the published points' names, by their first word).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut


@dataclass(frozen=True)
class RegistryCheck:
    values_checked: int
    recomputed: int
    derived_or_fitted: int
    problems: Tuple[str, ...]
    chip_names_in_keys: Tuple[str, ...]


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b))


def _recompute(registry: Any, path: str, obj: Mapping[str, Any]) -> Any:
    """The value an object's recompute spec gives (a pair for a band), or None where it states none."""
    recompute = obj.get("recompute") or {}
    aggregate = recompute.get("aggregate")
    if "expression" in recompute:
        return registry.evaluate(recompute["expression"])
    if aggregate == "range_of_nodes":
        return list(registry.range_nm())
    if aggregate == "log_linear_least_squares":
        a_ref, alpha, bits = registry.law(obj)
        if not (_close(alpha, obj["alpha_per_bit"], u.RECOMPUTE_TOLERANCE) and list(bits) == obj["bits_range"]):
            raise AreaRuleRefusal(f"{path}: alpha or bits range recomputes to {alpha}, {list(bits)}")
        return a_ref
    if aggregate == "band_wider_of_held_out_and_strict":
        return list(band_factors(registry, recompute.get("types")))
    if aggregate in ("geometric_mean", "max"):
        return registry.fitted(obj, (), path)[u.FIRST]
    return None


def band_factors(registry: Any, types: Any) -> Tuple[float, float]:
    """L = 1 / max and H = 1 / min of the wider of the held-out and the strict per-chip spreads (review M4)."""
    points = [p for p in registry.band_points() if types is None or p["type"] in types]
    held = [p["held_out"] for p in points if p["held_out"] is not None]
    strict = [v for p in points for v in p["strict"]] or held
    return min(u.UNITY / max(held), u.UNITY / max(strict)), max(u.UNITY / min(held), u.UNITY / min(strict))


def _same(got: Any, stored: Any) -> bool:
    if isinstance(stored, list):
        return isinstance(got, list) and len(got) == len(stored) and all(
            _close(a, b, u.RECOMPUTE_TOLERANCE) for a, b in zip(got, stored))
    return _close(got, stored, u.RECOMPUTE_TOLERANCE)


def _keys(obj: Any) -> Iterable[str]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys(value)


def unknown_citations(registry: Any) -> Tuple[str, ...]:
    """Every id the data files or the published points cite that the evidence file does not hold."""
    cited = {i for _p, obj in registry.walk() for i in (obj.get("citation") or ())}
    for point in registry.points:
        cited |= set(point.get("held_out_ids") or ()) | set(point.get("own_units") or ())
        for scope in point["scopes"]:
            cited |= set(scope.get("published_ids") or ())
    return tuple(sorted(i for i in cited if i not in registry.evidence.entries))


def chip_names(registry: Any) -> Tuple[str, ...]:
    """The published chips' names by their first word (data of the published points; no module names a chip)."""
    return tuple(sorted({re.split(r"[\s(]", p["name"])[u.FIRST] for p in registry.published_points()}))


def check_registry(registry: Any) -> RegistryCheck:
    problems: List[str] = []
    walked = registry.walk()
    recomputed: List[str] = []
    derived_or_fitted: List[str] = []
    for path, obj in walked:
        kind = obj["evidence_kind"]
        for entry_id in obj.get("citation") or ():
            if entry_id not in registry.evidence.entries:
                problems.append(f"{path}: unknown id {entry_id}")
        if kind == "published":
            entry = obj["entry_value"]
            value = registry.evidence.value(entry["id"], entry["index"], entry["key"])
            if not _close(value, obj["value"], u.ENTRY_TOLERANCE):
                problems.append(f"{path}: {obj['value']} is not {entry['id']}'s {value}")
            continue
        derived_or_fitted.append(path)
        if not obj.get("derivation"):
            problems.append(f"{path}: {kind} without its arithmetic")
        if kind == "fitted" and not obj.get("held_out"):
            problems.append(f"{path}: fitted without a held-out report")
        try:
            got = _recompute(registry, path, obj)
        except (AreaRuleRefusal, HeldOut, KeyError, ZeroDivisionError) as caught:
            problems.append(f"{path}: does not recompute ({caught})")
            continue
        if got is None:
            problems.append(f"{path}: states no recompute")
            continue
        recomputed.append(path)
        if not _same(got, obj["value"]):
            problems.append(f"{path}: recomputes to {got}, not {obj['value']}")
    problems += [f"unknown citation {i}" for i in registry.unknown_citations()]
    names = chip_names(registry)
    found = set()
    for name in sorted(registry.files):
        for key in _keys(registry.files[name]):
            found |= {n for n in names if re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", key, re.IGNORECASE)}
    return RegistryCheck(values_checked=len(walked), recomputed=len(recomputed),
                         derived_or_fitted=len(derived_or_fitted), problems=tuple(problems),
                         chip_names_in_keys=tuple(sorted(found)))

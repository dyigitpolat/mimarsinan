""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


class RuleError(Exception):
    ""


@dataclass(frozen=True)
class Component:
    ""
    name: str
    watts_per_unit: float
    units: float
    source: str

    def watts(self) -> float:
        if self.watts_per_unit < 0 or self.units < 0:
            raise RuleError(f"{self.name}: a negative power or size is refused")
        return self.watts_per_unit * self.units


def static_watts(components: Iterable[Component]) -> float:
    ""
    return float(sum(c.watts() for c in components))


def static_energy(components: Iterable[Component], seconds: float) -> float:
    ""
    if seconds < 0 or not math.isfinite(seconds):
        raise RuleError(f"powered time {seconds!r} refused")
    return static_watts(components) * seconds


def scaled_energy(e_c0_dyn: float, e_dyn: float, e_static: float, f: float) -> float:
    ""
    return e_c0_dyn + f * (e_dyn - e_c0_dyn) + e_static


def choose_step(objectives: Sequence[float]) -> int:
    ""
    best, bi = None, -1
    for i, v in enumerate(objectives):
        if best is None or v < best:
            best, bi = v, i
    if bi < 0:
        raise RuleError("no step to choose")
    return bi


def priced_unit(dyn_unit: float, step_dyn: Sequence[float], step_static: Sequence[float], accounting: str = "A",
                compute_only_static: float = 0.0) -> Tuple[float, int]:
    ""
    if len(step_dyn) != len(step_static) or not step_dyn:
        raise RuleError("the reference needs one static leg per step")
    i = choose_step([a + b for a, b in zip(step_dyn, step_static)])
    if accounting == "A":
        return dyn_unit + step_static[i], i
    if accounting == "D":
        return dyn_unit + compute_only_static, i
    if accounting == "B":
        return step_dyn[i] + step_static[i], i
    raise RuleError(f"accounting {accounting!r} is not declared (A of record, D and B sensitivities)")


def priced_budget(beta: float, unit: float) -> float:
    return beta * unit


SI = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12, "W/bit": 1.0, "s": 1.0}
REQUIRED = ("unit", "low", "nominal", "high", "source_low", "source_nominal", "source_high", "class_low",
            "class_nominal", "class_high")
SOURCE_FIELDS = ("quote", "where", "copy_sha256")


def check_components(name: str, comps: Sequence[Dict]) -> None:
    for c in comps:
        st = c.get("status", "")
        if st.startswith("sourced"):
            srcs = c.get("sources") or []
            if not srcs:
                raise RuleError(f"{name}: component {c.get('component')} is sourced without a source")
            for s in srcs:
                miss = [k for k in SOURCE_FIELDS if not str(s.get(k, "")).strip()]
                if miss:
                    raise RuleError(f"{name}: component {c.get('component')} source lacks {miss}")
            pu = c.get("per_unit_mW") or {}
            if not all(isinstance(v, (int, float)) and v >= 0 for v in pu.values()) or not pu:
                raise RuleError(f"{name}: component {c.get('component')} has no per-unit power")
        elif st == "unsourced":
            if not str(c.get("searched", "")).strip():
                raise RuleError(f"{name}: unsourced component {c.get('component')} states no search")
        else:
            raise RuleError(f"{name}: component {c.get('component')} has no status")


def load_params(path: str) -> Dict[str, Dict]:
    ""
    with open(path) as fh:
        doc = json.load(fh)
    out = {}
    for name, p in doc["parameters"].items():
        miss = [k for k in REQUIRED if k not in p]
        if miss:
            raise RuleError(f"parameter {name}: missing {miss}")
        if p["unit"] not in SI:
            raise RuleError(f"parameter {name}: unit {p['unit']!r} not declared")
        for end in ("low", "nominal", "high"):
            v = p[end]
            if v is None or not isinstance(v, (int, float)) or v < 0:
                raise RuleError(f"parameter {name} {end}: {v!r} refused")
            if v == 0 and not p.get(f"zero_reason_{end}"):
                raise RuleError(f"parameter {name} {end}: zero without a stated reason")
            if not str(p[f"source_{end}"]).strip() or not str(p[f"class_{end}"]).strip():
                raise RuleError(f"parameter {name} {end}: source or class empty")
        if not p["low"] <= p["nominal"] <= p["high"]:
            raise RuleError(f"parameter {name}: band not ordered")
        if "components" in p:
            check_components(name, p["components"])
        out[name] = p
    return out


def component_basis(p: Dict) -> Dict:
    ""
    lo = hi = 0.0
    unsourced = []
    for c in p.get("components", []):
        if c["status"] == "unsourced":
            unsourced.append(c["component"])
            continue
        pu = c["per_unit_mW"]
        lo += c["count"] * min(pu.values())
        hi += c["count"] * max(pu.values())
    basis = "declared sensitivity" if (unsourced or p.get("basis") == "declared sensitivity") else "sourced"
    return dict(sourced_part_mW=(lo, hi), unsourced=unsourced, basis=basis)


def band_value(p: Dict, end: str) -> float:
    ""
    if end == "zero":
        return 0.0
    return float(p[end]) * SI[p["unit"]]


def rests_on(params: Dict[str, Dict], point: Dict[str, float]) -> List[str]:
    ""
    return sorted(k for k, v in point.items() if v and k in params
                  and params[k].get("basis") == "declared sensitivity")


def bisect_change(sig, lo: float, hi: float, rel: float = 1e-4, geometric: bool = True) -> Tuple[float, float]:
    ""
    s0 = sig(lo)
    a, b = lo, hi
    while (b - a) > rel * max(abs(b), 1e-300):
        m = math.sqrt(a * b) if (geometric and a > 0) else 0.5 * (a + b)
        if sig(m) == s0:
            a = m
        else:
            b = m
    return a, b

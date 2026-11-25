""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

NVM = ("neurram", "hermes_64", "isaac_like")


def med(values):
    values = [v for v in values if v is not None]
    return None if not values else statistics.median(values)


def main() -> int:
    out = Path(sys.argv[1])
    regime = json.load(open(out / "regime_map.json"))
    counts = json.load(open(out / "capacity_counts.json"))
    points = regime["points"]
    s = {"points": len(points)}
    status = Counter()
    for p in points:
        status[(p["target"], "refused" if p.get("refusal") else "above budget" if p.get("above_budget") else "priced")] += 1
    s["status_per_target"] = {f"{t}|{k}": v for (t, k), v in sorted(status.items())}
    refusals = Counter((p["target"], p["refusal"]["name"]) for p in points if p.get("refusal"))
    s["refusals"] = {f"{t}|{n}": v for (t, n), v in sorted(refusals.items())}
    regimes = Counter()
    for p in points:
        for axis in ("energy", "time"):
            r = (p.get(axis) or {}).get("regime")
            if r and p.get("arm") in (None, "absent") and p.get("variant") in (None, "host") + tuple(
                    v for v in [p.get("variant")] if v and v.startswith("host+")):
                regimes[(p["target"] + ("" if p.get("variant") in (None, "host") else "+" + p["variant"][5:]),
                         axis, p["inference_batch_K"], r["reading"])] += 1
    s["regime_readings_u1"] = {"|".join(map(str, k)): v for k, v in sorted(regimes.items())}
    m_rel = defaultdict(list)
    for p in points:
        for axis in ("energy", "time"):
            a = p.get(axis) or {}
            rel = a.get("m_rel_diff_floor")
            if rel and p.get("arm") == "floor":
                m_rel[axis].append(rel[1])
    s["m_rel_diff_floor_nominal_floor_arm"] = {a: {"n": len(v), "median": med(v), "min": min(v) if v else None,
                                                   "max": max(v) if v else None} for a, v in m_rel.items()}
    ks = []
    for p in points:
        if p["target"] in NVM and p.get("arm") in (None, "absent") and p.get("inference_batch_K") == 1 \
                and not p.get("refusal") and (p.get("k_star") or {}).get("k_star_census") is not None:
            k = p["k_star"]
            ks.append({"network": p["network"], "target": p["target"], "budget": p["budget"], "cores": p["cores"],
                       "x_fluid": p["x_fluid"], "x_packed": p["x_packed"], "max_writes": k.get("max_writes_per_cell_per_inference"),
                       "k_star_census_nominal_E": k.get("k_star_census"), "k_star_closed_nominal_E": k.get("k_star_closed"),
                       "ratio": k.get("ratio"), "factor_packing_replication": k.get("factor_packing_replication"),
                       "factor_rotation": k.get("factor_rotation"),
                       "min_batch": {x: (p.get("min_batch") or {}).get(x) for x in (
                           "k_endurance_census", "k_bandwidth", "k_min_census", "binding", "e_cross_census", "verdict")}})
    s["k_star_nvm_u1_k1"] = ks
    arms = []
    for row in regime.get("side_by_side", []):
        a = row["arms"]
        if all(x in a for x in ("absent", "floor", "stateless")) and row["inference_batch_K"] == 1 and \
                row.get("variant") in (None, "host"):
            arms.append(row)
    s["arms_rows"] = len(arms)
    s["floor_over_stateless_bytes"] = {"median": med([r.get("floor_over_stateless_bytes") for r in arms]),
                                       "n": sum(1 for r in arms if r.get("floor_over_stateless_bytes") is not None)}
    s["floor_over_u1_bytes"] = {"median": med([r.get("floor_over_u1_bytes") for r in arms]),
                                "n": sum(1 for r in arms if r.get("floor_over_u1_bytes") is not None),
                                "floor_below_u1": sum(1 for r in arms if (r.get("floor_over_u1_bytes") or 9) < 1),
                                "floor_above_u1": sum(1 for r in arms if (r.get("floor_over_u1_bytes") or 0) > 1)}


    reuse = defaultdict(lambda: defaultdict(set))
    for p in points:
        r = (p.get("reuse") or {}).get("macs_per_weight")
        if r is not None:
            res = p.get("resolution")
            key = p["network"] if not res else f"{p['network']}|{'x'.join(str(x) for x in res)}"
            reuse[key][round(float(r), 9)].add(p["target"])
    s["macs_per_weight"] = {k: (next(iter(v)) if len(v) == 1 else {f"{x:.9g}": sorted(t) for x, t in sorted(v.items())})
                            for k, v in sorted(reuse.items())}
    s["capacity"] = {k: counts.get(k) for k in ("probe_110", "probe_estimate", "full_set")}
    s["provisional"] = regime["meta"]["provisional"]
    (out / "summary.json").write_text(json.dumps(s, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: s[k] for k in ("points", "floor_over_stateless_bytes", "floor_over_u1_bytes",
                                         "m_rel_diff_floor_nominal_floor_arm", "macs_per_weight")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

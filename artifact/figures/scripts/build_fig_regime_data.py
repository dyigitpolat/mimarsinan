""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
sys.path.insert(0, str(F3.parent / "figures_v2" / "scripts"))
import _ledger as LG

ANCHOR = "clfivepoints"

READING = {
    "resident-dominant": "write_dominated",
    "informative": "informative",
    "capacity-free": "capacity_free",
    "straddles": "straddling",
    "resident (no per-inference weight writes)": "no_weight_writes",
}
LABEL = {
    "write_dominated": "Write-dominated",
    "informative": "Mixed",
    "capacity_free": "Capacity-free",
    "straddling": "Corner-dependent regime",
    "no_weight_writes": "Single pass, no rewrites",
}
AXES = {"time": "Latency", "energy": "Energy"}
REGIMES = ("write_dominated", "informative", "capacity_free")
DIAMOND = "no_weight_writes"


def sig(v: float, n: int = 4) -> float:
    return float(f"{v:.{n}g}")


def main():
    path = LG.record(ANCHOR, "out/regime_map.json")
    rm = json.loads(path.read_text())
    src = f"{LG.rel(path)} (sha256 {LG.sha256(path)[:16]})"

    thr = rm["meta"]["regime_thresholds"]
    bounds = {
        "capacity_free_max": dict(LG.value("regimefree"), record=thr["capacity_free_max"],
                                  record_path="meta/regime_thresholds/capacity_free_max"),
        "write_dominated_min": dict(LG.value("regimewrite"), record=thr["resident_dominant_min"],
                                    record_path="meta/regime_thresholds/resident_dominant_min"),
    }
    for name, b in bounds.items():
        if abs(float(b["value"]) - float(b["record"])) > 1e-9 * max(1.0, abs(float(b["record"]))):
            raise SystemExit(f"{name}: the ledger's {b['value']} is not the record's {b['record']}")
    lo_b = float(bounds["capacity_free_max"]["value"])
    hi_b = float(bounds["write_dominated_min"]["value"])

    def band_of(y: float) -> str:
        ""
        return "capacity_free" if y <= lo_b else ("informative" if y < hi_b else "write_dominated")

    pts = {a: defaultdict(lambda: {"x": [], "pi_k": []}) for a in AXES}
    pos = {a: defaultdict(set) for a in AXES}
    count = {a: Counter() for a in AXES}
    ext = {a: {} for a in AXES}
    chips = {a: set() for a in AXES}
    n_points = 0
    for p in rm["points"]:
        if p.get("refusal") or not p.get("x_packed") or p["x_packed"] <= 1.0:
            continue
        had = False
        for a in AXES:
            ax = p.get(a)
            if not isinstance(ax, dict) or not ax.get("pi_over_k"):
                continue
            reading = (ax.get("regime") or {}).get("reading")
            if reading is None:
                continue
            r = READING[reading]
            x, y = float(p["x_packed"]), float(ax["pi_over_k"][1])
            if r in REGIMES and r != band_of(y):
                raise SystemExit(f"{a}: a point reads {r} at intensity {y:g}, in the band of {band_of(y)}")
            pts[a][r]["x"].append(x); pts[a][r]["pi_k"].append(y)
            pos[a][r].add((sig(x), sig(y)))
            count[a][r] += 1
            e = ext[a].setdefault(r, [x, x, y, y])
            e[0], e[1], e[2], e[3] = min(e[0], x), max(e[1], x), min(e[2], y), max(e[3], y)
            chips[a].add(p["target"])
            had = True
        n_points += had

    axes = {}
    for a, title in AXES.items():
        series = {}
        for r in LABEL:
            if not count[a][r] and r not in REGIMES:
                continue


            e = ext[a].get(r) or [None] * 4
            series[r] = {
                "label": LABEL[r],
                "points": count[a][r],
                "positions": len(pos[a][r]),
                **dict(zip(("x_min", "x_max", "pi_k_min", "pi_k_max"), (None if v is None else sig(v) for v in e))),
                "source": f"{src} points[*]: x_packed, {a}.pi_over_k[1] (nominal corner), "
                          f"{a}.regime.reading = {[k for k, v in READING.items() if v == r][0]!r}",
                "mark": "one mark per point at its recorded position",
                "x": pts[a][r]["x"], "pi_k": pts[a][r]["pi_k"],
            }
            if len(series[r]["x"]) != count[a][r]:
                raise SystemExit(f"{a}/{r}: {len(series[r]['x'])} positions for {count[a][r]} points")
        held = [s for s in series.values() if s["points"]]
        axes[a] = {"title": title, "points": sum(count[a].values()), "chips": sorted(chips[a]), "series": series,
                   "x_min": min(s["x_min"] for s in held), "x_max": max(s["x_max"] for s in held),
                   "pi_k_min": min(s["pi_k_min"] for s in held),
                   "pi_k_max": max(s["pi_k_max"] for s in held)}

    def slot(key):
        return LG.value(key) if LG.has(key) else None

    data = {
        "figure": "fig_regime",
        "label": "fig:regime",
        "message": "Overflow and programming intensity place a configuration in one of three cost regimes.",
        "built_by": "records/figures_v3/scripts/build_fig_regime_data.py",
        "source": src,
        "selection": "points beyond capacity (x_packed > 1), not refused, with a regime reading on the axis",
        "status": "estimated",
        "points_with_a_reading": n_points,
        "partial": rm["meta"].get("partial", {}).get("label"),
        "anchor": LG.value(ANCHOR),
        "bounds": bounds,
        "tolerance": dict(LG.value("regimetau"), what="the tolerance of the capacity-free reading: running beyond "
                          "capacity changes a capacity-free configuration's cost by at most this"),
        "placement": {
            "x": "x_packed (overflow after packing), as recorded",
            "y": "<axis>.pi_over_k[1] (programming intensity per inference, nominal corner), as recorded",
            "rule": "one mark per point at its recorded position; no binning, no snapping, no rounding",
        },
        "axes": axes,
        "ledger": LG.values("clfivepoints", "clfiverefused", "regimechips", "macsperweight", "regimesram",
                            "regimeisaac"),
        "not_drawn": {
            "anchor_pack": {"why": "drawn by fig_chipprops (the endurance panel); no intensity is read at a refused "
                                   "schedule", "value": slot("inetpackx"), "refusal": slot("inetschedrefusal")},
            "headline_pairs": {"why": "a boundary: overflow after packing not read by its run (priced in Section "
                                      "VI-C)", "value": slot("kpairsanchored")},
        },
    }
    out = LG.write("fig_regime", data, F3 / "data")
    print(f"wrote {out}: {n_points} points with a reading, each at its recorded position")
    for a in AXES:
        print(f"  {a}: {axes[a]['points']} points; " +
              "; ".join(f"{r}: {s['points']} pts at {s['positions']} distinct positions, x to {s['x_max']:g}, "
                        f"Pi/K {s['pi_k_min']:g} to {s['pi_k_max']:g}" if s["points"] else f"{r}: no point"
                        for r, s in axes[a]["series"].items()))


if __name__ == "__main__":
    main()

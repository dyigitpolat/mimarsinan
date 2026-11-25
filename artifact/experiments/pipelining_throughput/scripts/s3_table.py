""
import glob
import json
import math
import sys
from pathlib import Path

SCR = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
MEMBER = "R18-IN224"
CHIPS = {144: "S", 256: "M", 576: "L"}
KS = (1, 2, 4, 8, 16)
VARS = {"study:shipped": "465", "study:sync100": "100", "study:sync0": "0"}
COMPASS = {("M", 1): 430.0, ("M", 2): 497.0, ("M", 4): 493.0, ("M", 8): 601.0, ("M", 16): 1 / 1.32e-3,
           ("S", 16): 750.0, ("L", 16): 661.0}


def cell(p, b):
    if not isinstance(p, dict) or p.get("e2e_latency_s") is None:
        return {"refused": json.dumps((p or {}).get("e2e_latency_s_refusal") or p, default=str)[:400]}
    lat = p["e2e_latency_s"]
    co = p["compute_only"]["e2e_latency_s"][1]
    ss = p["steady_state_no_writes"]["e2e_latency_s"][1]
    we = (p.get("latency_write_exposed_s") or [None, 0.0])[1] or 0.0
    st, ht = p.get("store_traffic") or {}, p.get("host_time_at_K") or {}
    return {"thr": 1 / lat[1], "band": [1 / lat[2], 1 / lat[0]], "passes_per_run": p["passes"],
            "written_passes": p.get("written_passes"), "reprogram_passes": p.get("reprogram_passes"),
            "us": {"compute": co * 1e6, "barriers": (ss - co) * 1e6, "writes": we * 1e6,
                   "residual": (lat[1] - ss - we) * 1e6},
            "write_payload_bytes_per_inference": st.get("write_payload_bytes"),
            "carry_in_bytes_per_inference": st.get("carry_in_bytes"),
            "carry_link_time_s": st.get("carry_link_time_s"), "carry_store": st.get("store"),
            "host_time_s": ht.get("latency_host_s"), "host_link_time_s": ht.get("latency_host_link_s")}


def arm_cells(arm_out, chip, b):
    cand = arm_out.get("candidate") or {}
    sched = cand.get("schedule") or {}
    if "refusal" in sched:
        r = sched["refusal"]
        return {"refused": f"{r.get('name')}: {r.get('reason')}"[:400]}, {}
    out = {}
    for key, c in cand.items():
        if "|K" not in key:
            continue
        k = int(key.split("|K")[1])
        c = cell(c, b)
        if "thr" in c:
            c["passes_per_inference"] = c["passes_per_run"] * math.ceil(k / b) / k if b else c["passes_per_run"]
            ref = COMPASS.get((chip, k))
            c["factor"] = None if ref is None else c["thr"] / ref
        out[key] = c
    return None, out


grid = {"branch": "tcad/throughput-replication", "identity": None, "A": {}, "B1": {}, "B2": {}, "check": {},
        "compass": {f"{c}|K{k}": v for (c, k), v in COMPASS.items()}}
heads = set()
idf = SCR / "identity.json"
if idf.exists():
    grid["identity"] = json.loads(idf.read_text())
for n, chip in CHIPS.items():
    f = SCR / "run" / f"w0_c{n}_B.json"
    if not f.exists():
        continue
    a = json.loads(f.read_text())
    for arm, v in a["members"][MEMBER]["arms"].items():
        refused, cells = arm_cells(v, chip, 0)
        grid["A"][f"{chip}|{arm}"] = refused or cells
for f in sorted(glob.glob(str(SCR / "run" / "e2_c*_s*.json"))):
    d = json.loads(Path(f).read_text())
    heads.add(d["head"])
    chip, b = CHIPS[d["cores"]], int(d.get("slots") or 0)
    arm_out = d["members"][MEMBER]["arms"][d["arm"]]
    refused, cells = arm_cells(arm_out, chip, b)
    pipelined = [g for p in d["plans"] for g in p["groups"] if g["pipelined"]]
    key = f"{chip}|b{d['budget']}|g{d['limit']}|buf{int(d['buffer'])}" + (f"|s{b}" if b else "")
    arm = "A" if d["limit"] == "A" else ("B2" if b else "B1")
    grid[arm][key] = {
        "arm": d["arm"], "declare": d["declare"], "resolved": d["resolved"], "refused": refused, "cells": cells,
        "segments": [{k: p[k] for k in ("segment", "specs", "passes", "wavefront")} for p in d["plans"]],
        "groups": len(pipelined), "copies": sum(g["copies"] for g in pipelined),
        "group_passes": [g["passes"] for g in pipelined], "group_copies": [g["copies"] for g in pipelined],
        "live_peak_bytes": (d["live_peaks"] or {}).get(MEMBER) if isinstance(d["live_peaks"], dict) else None,
        "live_peaks_raw": d["live_peaks"], "wall_s": d["wall_s"]}
for f in sorted(glob.glob(str(SCR / "check" / "check_*.json"))):
    grid["check"][Path(f).stem] = json.loads(Path(f).read_text())
grid["heads"] = sorted(heads)
(HERE / "e3_grid.json").write_text(json.dumps(grid, indent=1, sort_keys=True, default=str) + "\n")
E2 = json.loads((HERE.parent / "e2" / "e2_grid.json").read_text())
WIDE, R3, SHIP = "b131072", "b8192", "study:shipped"


def best(g, arm, chip, budget, buf, variant, k):
    ""
    top = None
    for key, e in g[arm].items():
        parts = key.split("|")
        if parts[0] != chip or parts[1] != budget or parts[3] != f"buf{buf}" or e["refused"]:
            continue
        if arm == "B2" and parts[4] != f"s{k}":
            continue
        c = e["cells"].get(f"{variant}|K{k}")
        if c and "thr" in c and (top is None or c["thr"] > top[1]["thr"]):
            top = (parts[2][1:], c, e)
    return top


def fmt(c):
    return "none" if c is None else f"{c['thr']:,.1f}" + ("" if c.get("factor") is None else f" ({c['factor']:.2f})")


print("heads", grid["heads"], "identity", json.dumps((grid["identity"] or {}).get("summary")))
print("\n## head to head, budget 131,072, no buffer, 465 / 100 / 0 ns: COMPASS | A (e2) | B1 (e2) | B2 old (e2) | B2 now")
for (chip, k), ref in sorted(COMPASS.items(), key=lambda x: ("MLS".index(x[0][0]), x[0][1])):
    row = [f"{chip} K{k}", f"{ref:.0f}"]
    a = E2["A"].get(f"{chip}|B5m@131072+pipelined")
    row.append(" / ".join(fmt(a.get(f"{v}|K{k}")) if isinstance(a, dict) and f"{v}|K{k}" in a else "none" for v in VARS))
    for g, arm in ((E2, "B1"), (E2, "B2"), (grid, "B2")):
        got = [best(g, arm, chip, WIDE, 0, v, k) for v in VARS]
        row.append(" / ".join(f"{t[0]}:{fmt(t[1])}" if t else "none" for t in got))
    print(" | ".join(row))
print("\n## best B2 limit (now) per chip over every K, barrier, budget and buffer")
top = {}
for key, e in grid["B2"].items():
    chip, budget, g, buf, s = key.split("|")
    if e["refused"]:
        continue
    for ck, x in e["cells"].items():
        if "thr" in x and ((chip, budget, buf, ck) not in top or x["thr"] > top[(chip, budget, buf, ck)][1]):
            top[(chip, budget, buf, ck)] = (g, x["thr"])
print(sorted({(k[0], v[0]) for k, v in top.items()}))
print("\n## B2 over limits (465, wide, no buffer): chip limit | ppi K16 now (old, B1) | groups copies | thr K2 K4 K8 K16 now (old) | peak K2..K16")
for chip, limits in (("M", (160, 192, 224, 256)), ("L", (216, 288, 360, 432, 504, 576))):
    for gl in limits:
        row, peaks, ppi, gc = [], [], None, None
        for k in (2, 4, 8, 16):
            key = f"{chip}|{WIDE}|g{gl}|buf0|s{k}"
            e, o = grid["B2"].get(key), E2["B2"].get(key)
            x = None if e is None or e["refused"] else e["cells"].get(f"{SHIP}|K{k}")
            y = None if o is None or o["refused"] else o["cells"].get(f"{SHIP}|K{k}")
            row.append(("none" if x is None or "thr" not in x else f"{x['thr']:,.0f}")
                       + ("" if y is None or "thr" not in y else f" ({y['thr']:,.0f})"))
            peaks.append(None if e is None else e["live_peak_bytes"])
            if k == 16 and x is not None and "thr" in x:
                ppi = (round(x["passes_per_inference"], 1), round(y["passes_per_inference"], 1))
                gc = (e["groups"], e["copies"])
        b1 = E2["B1"].get(f"{chip}|{WIDE}|g{gl}|buf0")
        b1x = b1["cells"].get(f"{SHIP}|K16") if b1 and not b1["refused"] else None
        print(chip, gl, "| ppi", ppi, "B1", b1x and b1x["passes_per_inference"], "|", gc, "|", row, "| B1 K16",
              b1x and round(b1x["thr"]), "|", peaks)
print("\n## buffer and budget against the wide budget without the buffer, every B2 cell and barrier (now)")
moved, refused, n = [], [], 0
for key, e in grid["B2"].items():
    chip, budget, g, buf, s = key.split("|")
    if budget == WIDE and buf == "buf0":
        continue
    base = grid["B2"].get(f"{chip}|{WIDE}|{g}|buf0|{s}")
    if e["refused"]:
        refused.append((key, e["refused"]["refused"][:200]))
        continue
    for ck, x in e["cells"].items():
        y = base["cells"].get(ck)
        if "thr" not in x:
            refused.append((key, ck, x["refused"][:200]))
            continue
        n += 1
        if y is None or "thr" not in y or abs(x["thr"] - y["thr"]) > 1e-9 * y["thr"]:
            moved.append((key, ck, x["thr"], y and y.get("thr")))
print("cells compared", n, "moved", len(moved))
for m in moved[:20]:
    print("moved", m)
for key, e in sorted(grid["B2"].items()):
    if key.split("|")[1] == WIDE and key.split("|")[3] == "buf0":
        if e["refused"]:
            refused.append((key, e["refused"]["refused"][:200]))
        refused += [(key, ck, x["refused"][:200]) for ck, x in e["cells"].items() if "thr" not in x]
for r in sorted(set(map(str, refused))):
    print("refused", r)
print("\n## time's parts (us per inference, 465): B2 now beside B2 old (e2) and B1 (e2)")
for chip, gl, k in (("M", 256, 16), ("L", 432, 16), ("M", 256, 2), ("M", 256, 4), ("M", 256, 8), ("L", 432, 2),
                    ("L", 432, 4), ("L", 432, 8)):
    key = f"{chip}|{WIDE}|g{gl}|buf0|s{k}"
    x = grid["B2"][key]["cells"][f"{SHIP}|K{k}"]
    y = E2["B2"][key]["cells"][f"{SHIP}|K{k}"]
    z = E2["B1"][f"{chip}|{WIDE}|g{gl}|buf0"]["cells"][f"{SHIP}|K{k}"]
    fall = y["write_payload_bytes_per_inference"] - x["write_payload_bytes_per_inference"]
    for name, c in (("B2 now", x), ("B2 old", y), ("B1", z)):
        tot = 1e6 / c["thr"]
        print(chip, gl, f"K{k}", name, "ppi", round(c["passes_per_inference"], 2), "written/run", c["written_passes"],
              {p: f"{v:.1f} ({100 * v / tot:.0f}%)" for p, v in c["us"].items() if p != "residual"},
              "residual", round(c["us"]["residual"], 4), "total", round(tot, 1), "wbytes/inf",
              c["write_payload_bytes_per_inference"], "carry time", c["carry_link_time_s"], "host", c["host_time_s"])
    print("   written bytes per inference fall", fall, "expected (B-1)/B x 256,000 =", (k - 1) / k * 256000,
          "equal" if abs(fall - (k - 1) / k * 256000) < 1e-6 else "DIFFERS",
          "| B2 now - B1 wbytes", x["write_payload_bytes_per_inference"] - z["write_payload_bytes_per_inference"])
print("\n## B2 now against B2 old, every cell: passes per run and compute and barrier parts equal?")
same, other = 0, []
for key, e in grid["B2"].items():
    o = E2["B2"].get(key)
    if e["refused"] or o is None or o["refused"]:
        if bool(e["refused"]) != bool(o and o["refused"]):
            other.append((key, "refusal differs"))
        continue
    for ck, x in e["cells"].items():
        y = o["cells"].get(ck)
        if "thr" not in x or y is None or "thr" not in y:
            if ("thr" in x) != (y is not None and "thr" in y):
                other.append((key, ck, "refusal differs"))
            continue
        k = int(ck.split("|K")[1])
        ok = (x["passes_per_run"] == y["passes_per_run"] and abs(x["us"]["compute"] - y["us"]["compute"]) < 1e-6
              and abs(x["us"]["barriers"] - y["us"]["barriers"]) < 1e-6
              and abs(y["write_payload_bytes_per_inference"] - x["write_payload_bytes_per_inference"]
                      - (k - 1) / k * 256000) < 1e-6)
        same += ok
        if not ok:
            other.append((key, ck, x["passes_per_run"], y["passes_per_run"],
                          y["write_payload_bytes_per_inference"] - x["write_payload_bytes_per_inference"]))
print("cells with equal passes, compute, barriers and the write fall (B-1)/B x 256,000:", same, "other", len(other))
for o in other[:30]:
    print("other", o)

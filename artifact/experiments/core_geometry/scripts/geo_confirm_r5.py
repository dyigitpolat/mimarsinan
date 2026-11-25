""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys

import numpy as np

RA = "records/research_artifacts"
G = f"{RA}/13_geometry_study"
HL = f"{RA}/06_build/headline_v46/headline_result.json"
PINS = {f"{G}/plan_r5.json": "4dbdd6bc2185b5065f4f1b232766806be5ac5df0a272aa44d012bcf57e0a82eb",
        f"{G}/plan_r3.json": "a1b1ec3af2f28da5d5a6f1ac57128a806a97aefdb832ae51708bf38f9b32268c",
        f"{G}/scripts/geo_readings.py": "a310d51914c6bed57dc48c067080627b1c8ade74c1438dfcf2c711884dc6c805",
        HL: "9ee39790c03f5dad98bec73aa5ddb6dbd133666cd22e0ba873884acdae1607e4",
        f"{G}/fits/heldout_predictions_r4.json": "9e7b5efbc29c22b8f20b8e0880777b27b02cd143d9f7c2ae355fb0181f0f91e3",
        f"{G}/fits/heldout_predictions_r5.json": "d668d149424b79e285fe4d907b208c30d1d76d10d21d19cf734d830a1e867fa3"}
EXPLORATORY_PINS = {f"{G}/sweep/designs.jsonl": "d9a03bd7bf5db064f42b6a084f48b55edb64b885ac56d71040048f6c4589a4b7",
                    f"{G}/analysis/beside_anchors.csv": "7e24019292a2f9f46cc45b167688fcbefa0860514b8989be22054c54a2609276"}
LABEL = ("CONFIRMATORY (pre-registered, EXPLORATORY study): {what} of the confirmatory run (plan_r5.json, "
         "RA/13_geometry_study); not a record")
CAP = {"H10.1": 0.5 * 66.528, "K64": 0.5 * 66.528, "R1'": 0.5 * 66.528, "R1L": 0.25 * 66.528}
KCELL = {"H10.1": 16, "K64": 64, "R1'": 1, "R1L": 1}
VAR = {"study:declared_read+edge_host": 0, "study:declared_read+edge_host_low": 1,
       "study:declared_read+edge_host_high": 2}
SETS = {"S1": {4}, "S2": {3}, "S3": {2}, "S4": {1}, "S13": {1, 2, 3, 4}}
REF = "R18 w1 b4444"
TAU = 55
FIT = 0.025
TWO = (1 + FIT) / (1 - FIT) - 1
LEAK = 0.20
EQ = 1e-9
FLOOR = 0.05
Q_BH = 0.10
SIG = 50 * 0.9083
DRAWS = 4000
SEED_RUN = 20261108
SEED_EXPLORATORY = 20261107

_spec = importlib.util.spec_from_file_location("geo_readings", f"{G}/scripts/geo_readings.py")
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def pin(pins):
    out = {}
    for p, want in pins.items():
        got = sha(p)
        if got != want:
            raise SystemExit(f"REFUSED: INPUT_DIGEST {p} is {got}")
        out[p] = got
    return out


def load_counts():
    mt = json.load(open(HL))["measured_table"]
    if mt.get("total") != 5000 or mt.get("label") != "MEASURED single seed s0, E01 (5,000 images)":
        raise SystemExit(f"REFUSED: ACCURACY_SOURCE {mt.get('label')}")
    return mt["correct"], mt["label"]


def load_designs(path, cnt, keep=None, drop=frozenset()):
    ""
    designs = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            x = json.loads(line)
            if x.get("outcome") != "priced" or not x["rung_tags"]["main"]:
                continue
            g = (x["R"], x["C"], x["s"])
            if (keep is not None and g not in keep) or g in drop:
                continue
            key = (x["part"], x["cores"], x.get("routed_block_sha256"))
            d = designs.get(key)
            if d is None:
                d = {"g": g, "members": tuple(x["members"]), "arm": x["arm"], "cores": x["cores"],
                     "area": x["design_area_mm2"], "e": {}, "t": {}, "st": {},
                     "maxc": max(cnt[m] for m in x["members"]), "bits": x["cores"] * x["R"] * x["C"] * x["s"],
                     "isref": REF in x["members"], "routed": x.get("routed_block_sha256")}
                designs[key] = d
            vk = (VAR[x["variant"]], x["K"])
            d["e"][vk] = x["energy_mj"]
            d["t"][vk] = x["time_s"]
            d["st"][vk] = (x.get("energy_parts_mj") or {}).get("energy_static_mj")
    return list(designs.values())


def val(d, met, v, K, e, lk=0.0):
    if met == "area":
        return d["area"][e]
    if met == "energy":
        arr = d["e"].get((v, K))
        if arr is None:
            return None
        if lk:
            st = d["st"].get((v, K))
            if st is None:
                return None
            return arr[e] + lk * st[e]
        return arr[e]
    return d["t"][(v, K)][e]


def gname(g):
    return f"{g[0]}x{g[1]}x{g[2]}"


def env(counts, k, t):
    mask = counts >= t
    if not mask.any():
        return np.full(k.shape[1], np.inf)
    return k[mask].min(axis=0)


def margin(J, S):
    if not (np.isfinite(J).all() and np.isfinite(S).all()):
        return None
    return float(((S - J) / J).min())


def read_test(D, f, setting, counts, members, Z, order, statistic="r5"):
    ""
    met, t, cell = f["metric"], float(f["target_images"]), f["cell"]
    K = KCELL[cell]
    plane = [d for d in D if d["g"][2] in SETS[setting] and d["area"][1] <= CAP[cell] + 1e-9]
    nm = len(members)
    mi = {m: i for i, m in enumerate(members)}
    kJ = np.full((nm, 9), np.inf)
    kS = np.full((nm, 9), np.inf)
    per = []
    for v in range(3):
        for e in range(3):
            r = 3 * v + e
            pl = [d for d in plane if val(d, met, v, K, e) is not None]
            refd = [d for d in pl if d["isref"]]
            rec = {"r": r, "v": v, "e": e}
            reach_t = [d for d in pl if d["maxc"] >= t]
            reach_s = [d for d in pl if d["maxc"] >= t - TAU]
            if not refd:
                rec["determinable"] = False
                per.append(rec)
                continue

            def tk(d, lk=0.0):
                en = val(d, "energy", v, K, e, lk)
                return (val(d, met, v, K, e, lk), en if en is not None else float("inf"), d["area"][e],
                        d["t"][(v, K)][e], d["bits"], order[d["g"]])
            gr = min(refd, key=tk)
            kref = val(gr, met, v, K, e)
            refbest = {}
            for d in refd:
                x0 = val(d, met, v, K, e)
                refbest[d["g"]] = min(refbest.get(d["g"], math.inf), x0)
            others = [x0 for g, x0 in refbest.items() if g != gr["g"]]
            band_nom = {g for g, x0 in refbest.items() if x0 <= kref * (1 + FIT)}
            band_two = {g for g, x0 in refbest.items() if x0 <= kref * (1 + TWO)}
            if met == "energy" and statistic == "r5":
                lkJ, lkS = +LEAK, -LEAK
                reflo, khi = {}, min(val(d, met, v, K, e, +LEAK) for d in refd)
                for d in refd:
                    x0 = val(d, met, v, K, e, -LEAK)
                    reflo[d["g"]] = min(reflo.get(d["g"], math.inf), x0)
                band = {g for g, x0 in reflo.items() if x0 <= (1 + FIT) * khi}
                band2 = {g for g, x0 in reflo.items() if x0 <= (1 + TWO) * khi}
                kref_ceiling = min(reflo.values())
            else:
                lkJ = lkS = 0.0
                band = band_nom if statistic == "r5" else {gr["g"]}
                band2 = band_two
                kref_ceiling = kref
            for d in pl:
                cJ = val(d, met, v, K, e, lkJ)
                cS = val(d, met, v, K, e, lkS)
                for m in d["members"]:
                    i = mi[m]
                    if cJ is not None and cJ < kJ[i, r]:
                        kJ[i, r] = cJ
                    if cS is not None and d["g"] in band and cS < kS[i, r]:
                        kS[i, r] = cS
            cand_J = [d for d in reach_t if val(d, met, v, K, e, lkJ) is not None]
            cand_S = [d for d in reach_s if d["g"] in band and val(d, met, v, K, e, lkS) is not None]
            if not cand_J or not cand_S:
                rec.update(determinable=False, band=sorted(gname(g) for g in band))
                per.append(rec)
                continue
            J = min(cand_J, key=lambda d: tk(d, lkJ))
            S = min(cand_S, key=lambda d: tk(d, lkS))
            jv, sv = val(J, met, v, K, e, lkJ), val(S, met, v, K, e, lkS)
            c2 = [d for d in reach_s if d["g"] in band2 and val(d, met, v, K, e, lkS) is not None]
            s2 = val(min(c2, key=lambda d: tk(d, lkS)), met, v, K, e, lkS) if c2 else None
            cg = [d for d in reach_s if d["g"] == gr["g"] and val(d, met, v, K, e, lkS) is not None]
            sg = val(min(cg, key=lambda d: tk(d, lkS)), met, v, K, e, lkS) if cg else None

            own = None
            if met == "energy" and statistic == "r5":
                cJ0 = [d for d in reach_t if val(d, met, v, K, e) is not None]
                cS0 = [d for d in reach_s if d["g"] in band_nom and val(d, met, v, K, e) is not None]
                if cJ0 and cS0:
                    j0 = val(min(cJ0, key=tk), met, v, K, e)
                    own = (val(min(cS0, key=tk), met, v, K, e) - j0) / j0
            rec.update(determinable=True, J=f"{J['members'][0]}@{gname(J['g'])}/{J['cores']}/{J['arm']}",
                       J_geometry=gname(J["g"]), J_routed=J["routed"], J_value=jv,
                       S=f"{S['members'][0]}@{gname(S['g'])}/{S['cores']}/{S['arm']}", S_geometry=gname(S["g"]),
                       S_routed=S["routed"], S_value=sv, T=(sv - jv) / jv, G_ref=gname(gr["g"]), k_ref=kref,
                       band=sorted(gname(g) for g in band), band_size=len(band),
                       runner_up_gap=(min(others) / kref - 1) if others else None,
                       T_two_sided=(s2 - jv) / jv if s2 is not None else None,
                       T_declared_G_ref=(sg - jv) / jv if sg is not None else None,
                       T_own_leakage=own, ceiling=(kref_ceiling - jv) / jv)
            per.append(rec)
    det = all(r["determinable"] for r in per)
    out = {"id": f["id"], "metric": met, "family": f["family"], "target_images": int(t), "cell": cell,
           "setting": setting, "determinable": det, "per_reading": per}
    if not det:
        out.update(T=None, beyond=False, p=1.0)
        return out, None
    m5 = min(per, key=lambda r: r["T"])
    tm = [r["T_two_sided"] for r in per]
    tg = [r["T_declared_G_ref"] for r in per]
    out.update(T=m5["T"], beyond=m5["T"] > EQ, set_at={"r": m5["r"], "v": m5["v"], "e": m5["e"], "J": m5["J"],
                                                       "S": m5["S"], "S_geometry": m5["S_geometry"]},
               T_closed_clock=min(r["T"] for r in per if r["e"] in (1, 2)),
               T_two_sided=min(tm) if all(x is not None for x in tm) else None,
               T_declared_G_ref=min(tg) if all(x is not None for x in tg) else None,
               T_own_leakage=(min(r["T_own_leakage"] for r in per) if met == "energy" and statistic == "r5"
                              and all(r["T_own_leakage"] is not None for r in per) else None),
               ceiling=min(r["ceiling"] for r in per),
               gap_min=min((r["runner_up_gap"] for r in per if r["runner_up_gap"] is not None), default=None),
               cost_geometries_J=sorted({r["J_geometry"] for r in per}),
               cost_geometries_S=sorted({r["S_geometry"] for r in per}))

    T_env = margin(env(counts, kJ, t), env(counts, kS, t - TAU))
    Sat = np.stack([env(counts, kS, c) for c in counts])
    base = np.minimum(kJ, kS)
    k0 = np.maximum(base, Sat)
    k0[~np.isfinite(base) | ~np.isfinite(Sat)] = np.inf
    bad = 0
    for tt in np.arange(counts.min(), counts.max() + 1):
        a, c = env(counts, k0, tt), env(counts, kS, tt)
        if (np.isfinite(a) != np.isfinite(c)).any() or (np.isfinite(c) & (np.abs(a - c) > EQ * np.abs(c))).any():
            bad += 1
    out["null_identity_failures"] = bad
    out["T_equals_member_level_T"] = T_env is not None and abs(T_env - out["T"]) <= 1e-12 * max(1.0, abs(out["T"]))
    res = {}
    for seed, z in Z.items():
        cd = counts[None, :] + SIG * z
        Jd = np.where((cd >= t)[:, :, None], k0[None], np.inf).min(axis=1)
        Sd = np.where((cd >= t - TAU)[:, :, None], kS[None], np.inf).min(axis=1)
        ok = np.isfinite(Jd).all(axis=1) & np.isfinite(Sd).all(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Td = ((Sd - Jd) / Jd).min(axis=1)
        draws = [float(x) if o else None for x, o in zip(Td, ok)]
        res[seed] = GR.margin_p(out["T"], out["beyond"], draws)
    out["p_by_seed"] = {str(s): r for s, r in res.items()}
    return out, res


def decide(tests, seed, unconfirmable=()):
    ""
    p1 = [o["p_by_seed"][str(seed)]["p"] if o["determinable"] and o["beyond"] else 1.0 for o in tests]
    a1 = GR.bh_adjusted(p1)
    p2 = [1.0 if o["id"] in unconfirmable else p for o, p in zip(tests, p1)]
    a2 = GR.bh_adjusted(p2)
    cands = [o["id"] for o, a in zip(tests, a1) if o["determinable"] and o["beyond"] and o["T"] >= FLOOR and a <= Q_BH]
    conf = [o["id"] for o, a in zip(tests, a2) if o["determinable"] and o["beyond"] and o["T"] >= FLOOR and a <= Q_BH
            and o["id"] not in unconfirmable]
    return p1, a1, a2, cands, conf


def fams(ids):
    return {k: sum(1 for i in ids if i.startswith(k)) for k in ("C-time", "C-energy", "C-area")}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def common():
    p5 = json.load(open(f"{G}/plan_r5.json"))
    p3 = json.load(open(f"{G}/plan_r3.json"))
    members = [m["label"] for m in p3["members"]]
    if [m["label"] for m in p5["members"]] != members:
        raise SystemExit("REFUSED: MEMBER_ORDER")
    cnt, label = load_counts()
    counts = np.array([float(cnt[m]) for m in members])
    return p5, p3, members, cnt, label, counts


def selftest(out_path):
    digests = pin({**PINS, **EXPLORATORY_PINS})
    p5, p3, members, cnt, label, counts = common()
    order = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(p3["grid"]["geometry_table"])}
    D = load_designs(f"{G}/sweep/designs.jsonl", cnt)
    Z = {s: np.random.default_rng(s).standard_normal((DRAWS, len(members))) for s in (SEED_EXPLORATORY, SEED_RUN)}
    rows = list(csv.DictReader(l for l in open(f"{G}/analysis/beside_anchors.csv") if not l.startswith("#")))
    idx = {(r["reading"], r["reference"], r["metric"], r["t"], r["setting"], r["cell"]): r for r in rows}
    fails, tests, tests4 = [], [], []

    def check(name, ok, detail=""):
        if not ok:
            fails.append(f"{name}: {detail}")
    for f in p5["family"]:
        o, _ = read_test(D, f, f["exploratory_setting"], counts, members, Z, order, "r5")
        o4, _ = read_test(D, f, f["exploratory_setting"], counts, members, {SEED_EXPLORATORY: Z[SEED_EXPLORATORY]},
                          order, "r4")
        x = f["exploratory_reading_r5"]
        check(f"{f['id']}.T_r5", o["determinable"] and abs(o["T"] - x["T_r5"]) <= 1e-12 * max(1, abs(x["T_r5"])),
              (o["T"], x["T_r5"]))
        check(f"{f['id']}.set_at", (o["set_at"]["v"], o["set_at"]["e"], o["set_at"]["S_geometry"])
              == (x["set_at"]["v"], x["set_at"]["e"], x["set_at"]["S_geometry"]), (o["set_at"], x["set_at"]))
        check(f"{f['id']}.cost_J", o["cost_geometries_J"] == sorted(x["cost_geometries_J"]))
        check(f"{f['id']}.cost_S", o["cost_geometries_S"] == sorted(x["cost_geometries_S"]))
        check(f"{f['id']}.T_closed_clock", abs(o["T_closed_clock"] - x["T_r5_closed_clock_beside"]) <= 1e-12 * max(
            1, abs(x["T_r5_closed_clock_beside"])))
        check(f"{f['id']}.T_two_sided", abs(o["T_two_sided"] - x["T_r5_two_sided_band_beside"]) <= 1e-12 * max(
            1, abs(x["T_r5_two_sided_band_beside"])))
        check(f"{f['id']}.T_declared_G_ref_r4", abs(o4["T"] - x["T_declared_G_ref_r4"]) <= 1e-12 * max(
            1, abs(x["T_declared_G_ref_r4"])), (o4["T"], x["T_declared_G_ref_r4"]))
        check(f"{f['id']}.member_level_T", o["T_equals_member_level_T"] and o4["T_equals_member_level_T"])
        check(f"{f['id']}.null_identity_r5", o["null_identity_failures"] == 0, o["null_identity_failures"])
        check(f"{f['id']}.null_identity_r4", o4["null_identity_failures"] == 0, o4["null_identity_failures"])
        b = idx[("main", REF, f["metric"], str(f["target_images"]), f["exploratory_setting"], f["cell"])]
        check(f"{f['id']}.p_r4_equals_beside_anchors", abs(o4["p_by_seed"][str(SEED_EXPLORATORY)]["p"] - float(b["p"]))
              <= 1e-15, (o4["p_by_seed"][str(SEED_EXPLORATORY)]["p"], b["p"]))
        tests.append(o)
        tests4.append(o4)
    want = sorted(f["id"] for f in p5["family"] if f["expected_to_confirm_r5"])
    dec = {}
    for seed in (SEED_EXPLORATORY, SEED_RUN):
        _p1, a1, _a2, cands, conf = decide(tests, seed)
        dec[str(seed)] = {"count": len(conf), "by_family": fams(conf), "tests": conf,
                          "max_bh_adjusted_of_confirmed": max(a for o, a in zip(tests, a1) if o["id"] in conf)}
        check(f"expected_17_seed_{seed}", sorted(conf) == want and len(conf) == 17, (len(conf), conf))
    res = {"confirmatory": LABEL.format(what="geo_confirm_r5.py's selftest on the exploratory reading"),
           "ok": not fails, "failures": fails, "tests_reproduced": len(tests), "designs_main_tagged": len(D),
           "decision_on_the_exploratory_reading": dec, "expected_r5": {"count": len(want), "by_family": fams(want)},
           "accuracy_source": label, "input_digests": digests,
           "tests": [{k: o[k] for k in ("id", "T", "T_closed_clock", "T_two_sided", "set_at", "null_identity_failures",
                                        "cost_geometries_J", "cost_geometries_S")} | {"p": o["p_by_seed"]}
                     for o in tests]}
    with open(out_path, "x") as fh:
        json.dump(clean(res), fh, indent=1)
    print(json.dumps({k: res[k] for k in ("ok", "tests_reproduced", "decision_on_the_exploratory_reading")}, indent=1))
    for x in fails[:40]:
        print("FAIL", x)
    return 0 if not fails else 3


def tile_state(tiles_dir):
    ""
    st = {}
    if not tiles_dir or not os.path.isdir(tiles_dir):
        return st
    for name in sorted(os.listdir(tiles_dir)):
        f = os.path.join(tiles_dir, name, "score.json")
        if not name.startswith("h_ng45_") or not os.path.exists(f):
            continue
        sc = json.load(open(f))
        g = sc["geometry"]
        cur = st.get(g)
        rec = {"tile": name, "has_result": sc["has_result"], "fallback": sc["fallback"], "meets": sc.get("meets"),
               "measured": sc.get("measured"), "score_sha256": sha(f)}
        if cur is None or (not cur["has_result"] and sc["has_result"]):
            st[g] = rec
        st[g].setdefault("attempts", [])
    for name in sorted(os.listdir(tiles_dir)):
        if name.startswith("h_ng45_") and os.path.isdir(os.path.join(tiles_dir, name)):
            g = name[len("h_ng45_"):].replace("_fb", "")
            st.setdefault(g, {"tile": None, "has_result": False, "measured": None, "attempts": []})
            st[g]["attempts"].append(name)
    return st


def needs(metric):
    return {"area": ("area",), "time": ("area", "study_clock_closed"),
            "energy": ("area", "energy_leakage", "study_clock_closed")}[metric]


def read(args):
    digests = pin(PINS)
    p5, p3, members, cnt, label, counts = common()
    order = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(p5["grid"]["geometry_table"])}
    digests[args.designs] = sha(args.designs)
    replaced = []
    if args.routed:
        digests[args.routed] = sha(args.routed)
        complete = json.load(open(args.routed_complete)) if args.routed_complete else None
        keep = {tuple(int(z) for z in g.split("x")) for g in (complete or [])}
        R = load_designs(args.routed, cnt, keep=keep)
        replaced = sorted(keep)
        D = load_designs(args.designs, cnt, drop=frozenset(keep)) + R
    else:
        D = load_designs(args.designs, cnt)
    Z = {SEED_RUN: np.random.default_rng(SEED_RUN).standard_normal((DRAWS, len(members)))}
    tests = []
    for f in p5["family"]:
        o, _ = read_test(D, f, f["setting"][:-1], counts, members, Z, order, "r5")
        if o["determinable"] and o["null_identity_failures"]:
            raise SystemExit(f"REFUSED: NULL_IDENTITY {f['id']} ({o['null_identity_failures']} targets)")
        tests.append(o)
    p1, a1, a2, cands, _ = decide(tests, SEED_RUN)
    st = tile_state(args.tiles)
    pred4 = json.load(open(f"{G}/fits/heldout_predictions_r4.json"))
    pred5 = json.load(open(f"{G}/fits/heldout_predictions_r5.json"))
    routable = {k.replace("h_ng45_", "") for k in pred5["address"]}
    rate = statistics.median(p5["wall"]["routing"]["rates_s_per_1000_um2"].values())
    serve = {}
    for o, p, a in zip(tests, p1, a1):
        o["p"] = p
        o["bh_adjusted_pass1"] = a
        o["candidate"] = o["id"] in cands
        if o["determinable"]:
            geos = sorted(set(o["cost_geometries_J"]) | set(o["cost_geometries_S"]))
            o["cost_geometries"] = [{"geometry": g, "routable": g in routable, "state": st.get(g),
                                     "measured_for_the_test": bool(st.get(g) and st[g]["has_result"] and all(
                                         (st[g]["measured"] or {}).get(k) for k in needs(o["metric"])))}
                                    for g in geos]
            if o["candidate"]:
                for g in geos:
                    serve.setdefault(g, []).append(o["id"])
    queue = []
    for g, ids in serve.items():
        if g in st and st[g]["attempts"]:
            continue
        if g not in routable:
            continue
        logic = pred4["tiles"][f"h_ng45_{g}"]["logic_um2_F2"]
        queue.append({"geometry": g, "tile": f"h_ng45_{g}", "candidates_served": sorted(ids),
                      "predicted_central_wall_s": 30 + rate * logic / 1000,
                      "grid_order": order[tuple(int(z) for z in g.split("x"))]})
    queue.sort(key=lambda q: (-len(q["candidates_served"]), q["predicted_central_wall_s"], q["grid_order"]))
    res = {"confirmatory": LABEL.format(what="a held-out reading (geo_confirm_r5.py read)"),
           "designs": args.designs, "routed": args.routed, "geometries_replaced_by_routed_parts": [gname(g) for g in replaced],
           "designs_main_tagged": len(D), "seed": SEED_RUN, "accuracy_source": label,
           "determinable": sum(1 for o in tests if o["determinable"]),
           "beyond_the_noise": [o["id"] for o in tests if o["beyond"]],
           "candidates": {"count": len(cands), "by_family": fams(cands), "tests": cands},
           "post_sweep_queue": queue, "tile_state": st, "input_digests": digests, "tests": tests}
    with open(args.out, "x") as fh:
        json.dump(clean(res), fh, indent=1)
    if args.queue_out:
        with open(args.queue_out, "x") as fh:
            fh.write("# " + LABEL.format(what="the post-sweep routing queue") + "\n")
            fh.write("".join(q["tile"] + "\n" for q in queue))
    print(json.dumps({"determinable": res["determinable"], "beyond": len(res["beyond_the_noise"]),
                      "candidates": res["candidates"], "post_sweep_queue": [q["tile"] for q in queue],
                      "replaced": res["geometries_replaced_by_routed_parts"]}, indent=1))
    for o in tests:
        if o["determinable"]:
            print(o["id"], "T %.4f p %.6f bh1 %.6f cand %s set %s/%s S %s J %s" % (
                o["T"], o["p"], o["bh_adjusted_pass1"], o["candidate"], o["set_at"]["v"], o["set_at"]["e"],
                o["set_at"]["S_geometry"], ",".join(o["cost_geometries_J"])))
        else:
            print(o["id"], "not determinable")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--out", required=True)
    r = sub.add_parser("read")
    r.add_argument("--designs", required=True)
    r.add_argument("--routed", default=None)
    r.add_argument("--routed-complete", default=None, help="a JSON list of the routed geometries whose parts all ended")
    r.add_argument("--tiles", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--queue-out", default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a.out)
    return read(a)


if __name__ == "__main__":
    raise SystemExit(main())

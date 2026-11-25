""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

RA = "records/research_artifacts"
G = f"{RA}/13_geometry_study"
LABEL = ("CONFIRMATORY (pre-registered, EXPLORATORY study): {what} of the confirmatory analysis (plan_r5.json, "
         "RA/13_geometry_study); not a record")
PINS = {
    f"{G}/scripts/geo_confirm_r5.py": "42b486d34ed95b1768481e0b225174f70ae38e851291e71c6c1a76159614fb10",
    f"{G}/scripts/geo_readings.py": "a310d51914c6bed57dc48c067080627b1c8ade74c1438dfcf2c711884dc6c805",
    f"{G}/scripts/geo_analysis.py": "5bfd87d8c5e6e16b63b7a60b4e59d0550fecb8383e622000651f7efe2792308e",
    f"{G}/plan_r5.json": "4dbdd6bc2185b5065f4f1b232766806be5ac5df0a272aa44d012bcf57e0a82eb",
    f"{G}/plan_r3.json": "a1b1ec3af2f28da5d5a6f1ac57128a806a97aefdb832ae51708bf38f9b32268c",
    f"{RA}/06_build/headline_v46/headline_result.json": "9ee39790c03f5dad98bec73aa5ddb6dbd133666cd22e0ba873884acdae1607e4",
    f"{RA}/06_build/headline_v46/headline_rules.json": "4752567d1a8caf323c8211bbc24c1461ad5dd199d270ee11ab261df57bae20d2",
    f"{RA}/12_tcad_readiness/e2e/iso_accuracy/rules.json": "63f716b1e0d0e42f7bf554ffa4a7a8af8404b27533fb1eecdfabe981b409c200",
}
EXPLORATORY_PINS = {
    f"{G}/sweep/designs.jsonl": "d9a03bd7bf5db064f42b6a084f48b55edb64b885ac56d71040048f6c4589a4b7",
    f"{G}/analysis_fix/beside_env_anchors_union.csv": "be8e744777451d87602009333844468317a2e142d8f705e9ec6c5974c2a7335c",
    f"{G}/analysis/rule_scores.csv": "189f7657e625b01d662c12cea839d9ef468d1f60bd0a0822d8e93b776174818d",
}
HELDOUT_PINS = {
    f"{G}/sweep_r5/designs.jsonl": "b4079f77bb3ae45b7e00514d46b9170914f9b79711d9c02b55e23e24e11d6fe5",
    f"{G}/sweep_r5/routed_vlast/designs_routed.jsonl": "5010e2d2c7e516b4121fa2b3828a5e46a0856ef6832ee739e4271d6dbb676181",
    f"{G}/confirm_r5/readings/reading_vlast.json": "fb17603a6bc881c0c8c1a1b4b53bbe46462f3b1e388e93f4508dbd2f54148c20",
    f"{G}/confirm_r5/readings/reading_fits.json": "3807ec49ed4cbe0b72657e09f2b25d48c4a8c770d6b637fe5afb09fa7cf017c0",
    f"{G}/pd_points_r5/SHA256SUMS": "592ff92ebb53f439885df549eae17521586a4105d1171a1f7e55ec1cfa81c0c7",
    f"{G}/fits/heldout_predictions_r4.json": "9e7b5efbc29c22b8f20b8e0880777b27b02cd143d9f7c2ae355fb0181f0f91e3",
}
REPLACED = ((384, 96, 4),)
SECOND_REF = "R18 w1/2 b4444"
REF10, REF34 = "R10 w1 b4444", "R34 w1 b4444"
FOUR = ("main", "A_rp_s", "T_rs", "E_ops")
FOUR_KEY = {"main": "main", "T_rs": "T_rs", "E_ops": "E_ops", "A_rp_s": "A_rp_s (second reference)"}


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


def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pin({k: v for k, v in PINS.items() if k.endswith(".py")})
GC = by_path("geo_confirm_r5", f"{G}/scripts/geo_confirm_r5.py")
GR = GC.GR
sys.path.insert(0, f"{G}/scripts")
GA = by_path("geo_analysis", f"{G}/scripts/geo_analysis.py")


_CAPT = {}
_margin_p = GR.margin_p


def _capture(T, beyond, draws):
    _CAPT["draws"] = list(draws)
    return _margin_p(T, beyond, draws)


GR.margin_p = _capture


def clean(x):
    return GC.clean(x)


def consts_of(p3):
    fs = p3["fits_summary"]
    inc = fs["G_C1_row_slot_increment"]
    return {"i0_um2": inc["cell_bits_line"]["coef"]["i0_um2"], "i1_um2_per_bit": inc["cell_bits_line"]["coef"]["i1_um2_per_bit"],
            "inc_um2": inc["ruled_increment_um2"], "rho_w": fs["F3_cell_bits_main"]["rho_W"],
            "rho_w2": fs["F3_rpc2"]["rho_W2"], "rho_r": fs["F3_E_RC"]["rho_R_vs_256"]}


def load_reading(path, cnt, rd, ref, consts, keep=None, drop=frozenset()):
    ""
    tag = GR.READING_RUNGS[rd]
    designs = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            x = json.loads(line)
            if x.get("outcome") != "priced" or not x["rung_tags"][tag]:
                continue
            g = (x["R"], x["C"], x["s"])
            if (keep is not None and g not in keep) or g in drop:
                continue
            q = x["quantities"]
            key = (x["part"], x["cores"], x.get("routed_block_sha256"))
            d = designs.get(key)
            if d is None:
                dd = {"design_area_mm2": x["design_area_mm2"], "cores_physical": q["cores_physical"], "R": x["R"],
                      "C_built": x["C"], "s": x["s"], "tiles": q["tiles"], "lanes_per_core": x["lanes_per_core"],
                      "weight_columns": x["weight_columns"]}
                d = {"g": g, "members": tuple(x["members"]), "arm": x["arm"], "cores": x["cores"],
                     "area": GR.reading_area_mm2("A_rp_s" if rd == "A_rp_s" else "main", dd, consts),
                     "e": {}, "t": {}, "st": {}, "maxc": max(cnt[m] for m in x["members"]),
                     "bits": x["cores"] * x["R"] * x["C"] * x["s"], "isref": ref in x["members"],
                     "routed": x.get("routed_block_sha256"), "arch": x["arch"], "b_classes": tuple(x["b_classes"])}
                designs[key] = d
            vk = (GC.VAR[x["variant"]], x["K"])
            dq = {"R": x["R"], "C_built": x["C"], "s": x["s"], "cells_per_weight": x["k"], "time_s": x["time_s"],
                  "compute_time_s": q["compute_time_s"], "energy_mj": x["energy_mj"],
                  "static_power_w": q["static_power_w"], "synaptic_events": q["synaptic_events"],
                  "write_overlap_mode": q.get("write_overlap_mode", "serial")}
            d["t"][vk] = GR.reading_time_s("T_rs" if rd == "T_rs" else "main", dq)
            st = (x.get("energy_parts_mj") or {}).get("energy_static_mj")
            if x["energy_mj"] is None:
                d["e"][vk] = None
                d["st"][vk] = None
                continue
            d["e"][vk] = GR.reading_energy_mj(rd if rd in ("T_rs", "E_ops") else "main", dq, consts)
            if st is not None and rd == "T_rs":
                f = x["R"] / 256 - 1
                st = [a + p * c * f * 1e3 for a, p, c in zip(st, q["static_power_w"], q["compute_time_s"])]
            d["st"][vk] = st
    return list(designs.values())


def read_one(D, f, setting, counts, members, Z, order, statistic):
    _CAPT.pop("draws", None)
    o, _ = GC.read_test(D, f, setting, counts, members, Z, order, statistic)
    return o, _CAPT.pop("draws", None)


class Side:
    ""

    def __init__(self, plan, members, counts):
        hr = json.load(open(f"{RA}/06_build/headline_v46/headline_rules.json"))
        arch_info = {a["id"]: a for a in hr["architectures"]}
        hm = {m["label"]: m for m in hr["members"]}
        if [m["label"] for m in hr["members"]] != members:
            raise SystemExit("REFUSED: FAMILY_ORDER")
        sm = json.load(open(f"{RA}/12_tcad_readiness/e2e/iso_accuracy/rules.json"))["size_measures"]
        self.m_arch = [m["arch"] for m in plan["members"]]
        self.m_bits = [m["bits"] for m in plan["members"]]
        self.m_xbar = np.array([float(hm[m]["census_params"]) for m in members])
        self.m_pbits = np.array([float(sum(arch_info[self.m_arch[i]]["census_stage_params"][str(s + 1)] * self.m_bits[i][s]
                                           for s in range(4))) for i in range(len(members))])
        self.m_macs = np.array([float(sm["macs_per_inference"]["values"][a]) for a in self.m_arch])
        self.m_act = np.array([float(sm["activation_elements"]["values"][a]) for a in self.m_arch])
        self.arch_info = arch_info
        self.counts = counts


def tk_of(met, v, K, e, order):
    def tk(d, lk=0.0):
        en = GC.val(d, "energy", v, K, e, lk)
        return (GC.val(d, met, v, K, e, lk), en if en is not None else float("inf"), d["area"][e],
                d["t"][(v, K)][e], d["bits"], order[d["g"]])
    return tk


def beside(D, f, setting, counts, members, order, side, geoms, consts, tiles, bands=None):
    ""
    met, t, cell = f["metric"], float(f["target_images"]), f["cell"]
    K = GC.KCELL[cell]
    svals = GC.SETS[setting]
    plane = [d for d in D if d["g"][2] in svals and d["area"][1] <= GC.CAP[cell] + 1e-9]
    nm = len(members)
    mi = {m: i for i, m in enumerate(members)}
    has = np.zeros(nm, dtype=bool)
    for d in plane:
        for m in d["members"]:
            has[mi[m]] = True
    inf = np.inf
    names = ["G-ref", "CF-ref", f"G-ref({REF10})", f"G-ref({REF34})", "G-dense"]
    gs = [i for i, row in enumerate(geoms) if row["s"] in svals]
    gdense = GA.g_dense(SimpleNamespace(geoms=geoms, consts=consts), gs)
    gdense = None if gdense is None else (geoms[gdense]["R"], geoms[gdense]["C_cells"], geoms[gdense]["s"])
    tl = [(n.split(" (")[0], tuple(v)) for n, v in tiles.items() if tuple(v)[2] in svals]
    names += [f"tile {n}" for n, _ in tl]
    if bands is not None:
        names.append("G-ref band")
    rules = {n: np.full((nm, 9), inf) for n in names}
    kJ = np.full((nm, 9), inf)
    gref_by_r = []
    for r in range(9):
        v, e = divmod(r, 3)
        tk = tk_of(met, v, K, e, order)
        pl = [d for d in plane if GC.val(d, met, v, K, e) is not None]
        kG = {}
        for d in pl:
            c = GC.val(d, met, v, K, e)
            col = kG.setdefault(d["g"], np.full(nm, inf))
            for m in d["members"]:
                i = mi[m]
                if c < col[i]:
                    col[i] = c
                if c < kJ[i, r]:
                    kJ[i, r] = c
        refd = [d for d in pl if GC.REF in d["members"]]
        if not refd:
            gref_by_r.append(None)
            continue
        gr = min(refd, key=tk)
        gref_by_r.append(GC.gname(gr["g"]))
        rules["G-ref"][:, r] = kG[gr["g"]]
        cf = np.full(nm, inf)
        for d in pl:
            if d["g"] == gr["g"] and d["cores"] == gr["cores"]:
                c = GC.val(d, met, v, K, e)
                for m in d["members"]:
                    cf[mi[m]] = min(cf[mi[m]], c)
        rules["CF-ref"][:, r] = cf
        for lab, ref in ((f"G-ref({REF10})", REF10), (f"G-ref({REF34})", REF34)):
            rd_ = [d for d in pl if ref in d["members"]]
            if rd_:
                rules[lab][:, r] = kG[min(rd_, key=tk)["g"]]
        if gdense is not None and gdense in kG:
            rules["G-dense"][:, r] = kG[gdense]
        for n, g in tl:
            if g in kG:
                rules[f"tile {n}"][:, r] = kG[g]
        if bands is not None:
            col = np.full(nm, inf)
            for gn in bands[r]:
                g = tuple(int(z) for z in gn.split("x"))
                if g in kG:
                    col = np.minimum(col, kG[g])
            rules["G-ref band"][:, r] = col
    ref = mi[GC.REF]
    nf = {"NF-ref": [ref] if has[ref] else []}
    b4 = np.where(has & np.array([b == [4, 4, 4, 4] for b in side.m_bits]))[0]
    nf["NF-top"] = [int(b4[np.argmax(counts[b4])])] if len(b4) else []
    ns = SimpleNamespace(counts=counts)
    nf["NF-bits"] = GA.nf_picks(ns, has, side.m_pbits, side.m_xbar)
    nf["NF-size"] = GA.nf_picks(ns, has, side.m_xbar, side.m_pbits)
    nf["NF-MAC"] = GA.nf_picks(ns, has, side.m_macs, side.m_pbits)
    nf["NF-act"] = GA.nf_picks(ns, has, side.m_act, side.m_pbits)
    for lab, ms in nf.items():
        k = np.full((nm, 9), inf)
        if ms:
            k[ms] = kJ[ms]
        rules[lab] = k
    Jt = GC.env(counts, kJ, t)
    out = {"G_ref_by_reading": gref_by_r, "G_dense": None if gdense is None else GC.gname(gdense),
           "network_first_members": {lab: [members[m] for m in ms] for lab, ms in nf.items()}, "rules": {}}
    kenv = np.full((nm, 9), inf)
    for lab, k in rules.items():
        kenv = np.minimum(kenv, k)
        T = GC.margin(Jt, GC.env(counts, k, t - GC.TAU))
        out["rules"][lab] = {"T": T, "beyond": T is not None and T > GC.EQ}
    T = GC.margin(Jt, GC.env(counts, kenv, t - GC.TAU))
    out["rules"]["S_env"] = {"T": T, "beyond": T is not None and T > GC.EQ}
    return out


def rule_block(D, plan, geoms, members, side, consts, cells=("H10.1", "K64", "R1'", "R1L")):
    nets = [(a, b) for a in plan["architectures"] for b in plan["arch_bit_classes"][a]]
    nidx = {n: i for i, n in enumerate(nets)}
    gidx = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(geoms)}
    NG = len(geoms)
    ns = SimpleNamespace(networks=nets, arch_info=side.arch_info, m_macs=side.m_macs, m_arch=side.m_arch, geoms=geoms,
                         NG=NG, consts=consts, net_label=[f"{a} b{b}" for a, b in nets],
                         gname=[f"{g['R']}x{g['C_cells']}x{g['s']}" for g in geoms],
                         gR=np.array([g["R"] for g in geoms]), gs=np.array([g["s"] for g in geoms]),
                         gC=np.array([g["C_cells"] for g in geoms]))
    pred = GA.rule_predictions(ns, "main")
    gs13 = list(range(NG))
    out = {}
    for cell in cells:
        K = GC.KCELL[cell]
        cN = {m: np.full((len(nets), NG), np.inf) for m in ("area", "energy")}
        for d in D:
            if d["area"][1] > GC.CAP[cell] + 1e-9:
                continue
            g = gidx[d["g"]]
            for b in d["b_classes"]:
                n = nidx.get((d["arch"], b))
                if n is None:
                    continue
                a = d["area"][1]
                if a < cN["area"][n, g]:
                    cN["area"][n, g] = a
                en = GC.val(d, "energy", 0, K, 1)
                if en is not None and en < cN["energy"][n, g]:
                    cN["energy"][n, g] = en
        for met in ("area", "energy"):
            sc = GA.rule_scores(ns, cN[met], pred[met], gs13)
            rec = {k: sc[k] for k in ("networks", "mean_spearman", "hit_rate_geometry", "hit_rate_geometry_set",
                                      "hit_rate_R", "hit_rate_s", "hit_rate_C")}
            if met == "energy":

                hits, per = 0, []
                for n, (a, b) in enumerate(nets):
                    row = cN["energy"][n]
                    ok = np.isfinite(row)
                    if not ok.any():
                        continue
                    best = GA.argmin_set(row, ok)
                    bset = {ns.gname[i] for i in best}
                    rule = {f"384x48x{b}", f"384x96x{b}"}
                    hit = bool(bset & rule)
                    hits += hit
                    per.append({"network": ns.net_label[n], "priced_best": sorted(bset), "rule_set": sorted(rule),
                                "hit": hit})
                rec["post_hoc_energy_rule_heldout"] = {"networks": len(per), "hits": hits,
                                                       "hit_rate": hits / len(per) if per else None, "per_network": per}
            rec["per_network"] = sc["per_network"]
            out[f"{cell}|{met}"] = rec
    return out


def decide_labels(tests, tiles, replaced, metric_of, unconf_energy_reason):
    ""
    ids = list(tests)
    p1 = [tests[i]["p"] if tests[i]["determinable"] and tests[i]["beyond"] else 1.0 for i in ids]
    a1 = GR.bh_adjusted(p1)
    cand = [i for i, a in zip(ids, a1) if tests[i]["determinable"] and tests[i]["beyond"] and tests[i]["T"] >= GC.FLOOR
            and a <= GC.Q_BH]
    unconf = {}
    for i in ids:
        need = GC.needs(metric_of[i])
        if metric_of[i] == "energy" and unconf_energy_reason:
            unconf[i] = unconf_energy_reason
            continue
        if i in cand:
            miss = [g for g in tests[i]["cost_geometries"] if not (tiles.get(g) and tiles[g]["has_result"] and all(
                (tiles[g]["measured"] or {}).get(k) for k in need))]
            if miss:
                unconf[i] = "cost geometries not measured for " + ", ".join(need) + ": " + ", ".join(miss)
    p2 = [1.0 if i in unconf else p for i, p in zip(ids, p1)]
    a2 = GR.bh_adjusted(p2)
    conf = [i for i, a in zip(ids, a2) if tests[i]["determinable"] and tests[i]["beyond"] and tests[i]["T"] >= GC.FLOOR
            and a <= GC.Q_BH and i not in unconf]
    lab = {}
    for i, a in zip(ids, a2):
        o = tests[i]
        names = []
        if i in conf:
            pr = [g for g in o["cost_geometries"] if g in replaced]
            names.append("CONFIRMED, priced from routed tiles" if pr else "CONFIRMED, the fits validated at its tiles")
            if metric_of[i] in ("time", "energy") and not all(
                    (tiles.get(g, {}).get("measured") or {}).get("low_end_closed") for g in o["cost_geometries"]):
                names.append("low end UNCONFIRMABLE (a reading label)")
            decision = "CONFIRMED"
        elif i in unconf:
            decision = "UNCONFIRMABLE"
            names.append(f"UNCONFIRMABLE ({unconf[i]})")
        else:
            decision = "NOT CONFIRMED"
            if not o["determinable"]:
                why = "not determinable"
            elif not o["beyond"]:
                why = "not beyond the noise"
            elif o["T"] < GC.FLOOR:
                why = "T below the size floor"
            else:
                why = "pass-2 BH-adjusted p above 0.10"
            names.append(f"NOT CONFIRMED ({why})")
        if decision != "CONFIRMED" and o["determinable"]:
            if o.get("ceiling") is not None and o["ceiling"] < GC.FLOOR:
                names.append("BLOCKED (reference-anchor identity)")
            if (o["T"] < GC.FLOOR and o.get("T_declared_G_ref") is not None and o["T_declared_G_ref"] >= GC.FLOOR):
                names.append(f"BLOCKED (sizing band), set by {o.get('S_geometry_set')}")
            if (metric_of[i] == "energy" and o["T"] < GC.FLOOR and o.get("T_own_leakage") is not None
                    and o["T_own_leakage"] >= GC.FLOOR):
                names.append("BLOCKED (leakage at its worst)")
        lab[i] = {"decision": decision, "labels": names}
    return {"p_pass1": dict(zip(ids, p1)), "bh_pass1": dict(zip(ids, a1)), "candidates": cand,
            "unconfirmable": unconf, "p_pass2": dict(zip(ids, p2)), "bh_pass2": dict(zip(ids, a2)),
            "confirmed": conf, "labels": lab}


def selftest(out_path):
    digests = pin({**PINS, **EXPLORATORY_PINS})
    p5, p3, members, cnt, label, counts = GC.common()
    consts = consts_of(p3)
    order = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(p3["grid"]["geometry_table"])}
    fails = []

    def check(name, ok, detail=""):
        if not ok:
            fails.append(f"{name}: {detail}")
    path = f"{G}/sweep/designs.jsonl"
    D0 = GC.load_designs(path, cnt)
    DR = {rd: load_reading(path, cnt, rd, SECOND_REF if rd == "A_rp_s" else GC.REF, consts) for rd in FOUR}
    four = {}
    for f in p5["family"]:
        st = f["exploratory_setting"]
        o0, _ = read_one(D0, f, st, counts, members, {}, order, "r4")
        om, _ = read_one(DR["main"], f, st, counts, members, {}, order, "r4")
        check(f"{f['id']}.loader_main_equals_geo_confirm", o0["T"] == om["T"], (o0["T"], om["T"]))
        want = f["exploratory_reading_post_hoc"]["four_readings_T"]
        four[f["id"]] = {}
        for rd in FOUR:
            o, _ = read_one(DR[rd], f, st, counts, members, {}, order, "r4")
            w = want[FOUR_KEY[rd]]
            four[f["id"]][rd] = o["T"]
            ok = (o["T"] is None and w is None) or (o["T"] is not None and w is not None
                                                     and abs(o["T"] - w) <= 1e-9 * max(1, abs(w)))
            check(f"{f['id']}.four.{rd}", ok, (o["T"], w))

    side = Side(p3, members, counts)
    rows = []
    with open(f"{G}/analysis_fix/beside_env_anchors_union.csv") as fh:
        fh.readline()
        rows = list(csv.DictReader(fh))
    idx = {(r["cell"], r["setting"], r["metric"], r["t"]): r for r in rows}
    tiles3 = p3["declared"]["study_tiles_S14"]["tiles"]
    nb = 0
    for f in p5["family"]:
        st = f["exploratory_setting"]
        b = beside(D0, f, st, counts, members, order, side, p3["grid"]["geometry_table"], consts, tiles3)
        row = idx[(f["cell"], st, f["metric"], str(f["target_images"]))]
        for lab, v in b["rules"].items():
            col = f"T {lab}"
            if col not in row:
                continue
            w = None if row[col] == "" else float(row[col])
            ok = (v["T"] is None and w is None) or (v["T"] is not None and w is not None
                                                     and abs(v["T"] - w) <= 1e-9 * max(1, abs(w)))
            check(f"{f['id']}.beside.{lab}", ok, (v["T"], w))
            nb += 1

    with open(f"{G}/analysis/rule_scores.csv") as fh:
        fh.readline()
        rs = [r for r in csv.DictReader(fh) if r["reading"] == "main"]
    rb = rule_block(DR["main"], p3, p3["grid"]["geometry_table"], members, side, consts)
    nr = 0
    for r in rs:
        mine = rb[f"{r['cell']}|{r['metric']}"]
        for k_csv, k_mine in (("networks", "networks"), ("mean_spearman", "mean_spearman"),
                              ("hit_rate_geometry_unique", "hit_rate_geometry"),
                              ("hit_rate_geometry_in_rule_set", "hit_rate_geometry_set"), ("hit_rate_R", "hit_rate_R"),
                              ("hit_rate_s", "hit_rate_s"), ("hit_rate_C", "hit_rate_C")):
            w = float(r[k_csv]) if r[k_csv] != "" else None
            v = mine[k_mine]
            ok = (v is None and w is None) or (v is not None and w is not None and abs(v - w) <= 1e-9 * max(1, abs(w)))
            check(f"rule.{r['cell']}.{r['metric']}.{k_csv}", ok, (v, w))
            nr += 1

    syn = {"A": {"determinable": True, "beyond": True, "T": 0.2, "p": 1 / 4001, "cost_geometries": ["g1", "g2"],
                 "T_declared_G_ref": 0.2, "ceiling": 1.0},
           "B": {"determinable": True, "beyond": True, "T": 0.2, "p": 1 / 4001, "cost_geometries": ["g1", "g3"],
                 "T_declared_G_ref": 0.2, "ceiling": 1.0},
           "C": {"determinable": True, "beyond": False, "T": 0.0, "p": 1.0, "cost_geometries": ["g1"],
                 "T_declared_G_ref": 0.3, "ceiling": 1.0, "S_geometry_set": "g9"},
           "D": {"determinable": False, "beyond": False, "T": None, "p": 1.0, "cost_geometries": []},
           "E": {"determinable": True, "beyond": True, "T": 0.03, "p": 1 / 4001, "cost_geometries": ["g1"],
                 "T_declared_G_ref": 0.03, "ceiling": 0.04}}
    tl = {"g1": {"has_result": True, "meets": True, "measured": {"area": True, "study_clock_closed": True,
                                                                  "low_end_closed": False}},
          "g2": {"has_result": True, "meets": False, "measured": {"area": True, "study_clock_closed": True,
                                                                   "low_end_closed": False}},
          "g3": {"has_result": False, "meets": None, "measured": None}}
    dl = decide_labels(syn, tl, {"g2"}, {k: "time" for k in syn}, None)
    check("syn.A", dl["labels"]["A"]["labels"][:2] == ["CONFIRMED, priced from routed tiles",
                                                       "low end UNCONFIRMABLE (a reading label)"], dl["labels"]["A"])
    check("syn.B", dl["labels"]["B"]["decision"] == "UNCONFIRMABLE" and dl["p_pass2"]["B"] == 1.0, dl["labels"]["B"])
    check("syn.C", dl["labels"]["C"]["labels"] == ["NOT CONFIRMED (not beyond the noise)",
                                                   "BLOCKED (sizing band), set by g9"], dl["labels"]["C"])
    check("syn.D", dl["labels"]["D"]["labels"] == ["NOT CONFIRMED (not determinable)"], dl["labels"]["D"])
    check("syn.E", dl["labels"]["E"]["labels"] == ["NOT CONFIRMED (T below the size floor)",
                                                   "BLOCKED (reference-anchor identity)"], dl["labels"]["E"])
    check("syn.bh", abs(dl["bh_pass1"]["A"] - (1 / 4001) * 5 / 3) < 1e-15 and abs(dl["bh_pass2"]["A"] - (1 / 4001) * 5 / 2)
          < 1e-15, (dl["bh_pass1"]["A"], dl["bh_pass2"]["A"]))
    dle = decide_labels({"X": syn["A"]}, tl, set(), {"X": "energy"}, "energy measured at no tile")
    check("syn.energy", dle["labels"]["X"]["decision"] == "UNCONFIRMABLE", dle["labels"]["X"])
    res = {"confirmatory": LABEL.format(what="the selftest on the exploratory reading"), "ok": not fails,
           "failures": fails, "tests": len(p5["family"]), "four_readings_checked": 4 * len(p5["family"]),
           "beside_values_checked": nb, "rule_values_checked": nr, "input_digests": digests,
           "four_readings_exploratory": four}
    with open(out_path, "x") as fh:
        json.dump(clean(res), fh, indent=1)
    print(json.dumps({k: res[k] for k in ("ok", "tests", "four_readings_checked", "beside_values_checked",
                                          "rule_values_checked")}))
    for x in fails[:60]:
        print("FAIL", x)
    return 0 if not fails else 3


def verify_sums(root, sums):
    bad = []
    n = 0
    with open(sums) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            h, p = line.rstrip("\n").split(None, 1)
            p = p.lstrip("*")
            n += 1
            if sha(os.path.join(root, p)) != h:
                bad.append(p)
    return n, bad


def analyze(out_dir):
    if os.path.exists(out_dir):
        raise SystemExit(f"REFUSED: OUT_EXISTS {out_dir}")
    digests = pin({**PINS, **HELDOUT_PINS})
    n_sums, bad = verify_sums(f"{G}/pd_points_r5", f"{G}/pd_points_r5/SHA256SUMS")
    if bad:
        raise SystemExit(f"REFUSED: TILE_SUMS {bad[:5]}")
    p5, p3, members, cnt, label, counts = GC.common()
    consts = consts_of(p3)
    for row in p5["grid"]["geometry_table"]:
        mine = GR.reading_core_add_um2("A_rp_s", row["R"], row["C_cells"], row["s"], consts)
        if abs(mine - row["A_rp_s_core_add_um2"]) > 1e-9 * max(1.0, abs(row["A_rp_s_core_add_um2"])):
            raise SystemExit(f"REFUSED: CORE_ADD {row['R']}x{row['C_cells']}x{row['s']}")
    f3 = p5["fits_summary"]["F3_adopted"]
    for s in (1, 2, 3, 4):
        if abs(GR.rho_w(s, f3["a_pj"], f3["b_pj_per_bit"]) - consts["rho_w"][str(s)]) > 1e-12:
            raise SystemExit(f"REFUSED: RHO_W s {s}")
    geoms = p5["grid"]["geometry_table"]
    order = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(geoms)}
    dpath = f"{G}/sweep_r5/designs.jsonl"
    rpath = f"{G}/sweep_r5/routed_vlast/designs_routed.jsonl"
    keep = set(REPLACED)
    D = GC.load_designs(dpath, cnt, drop=frozenset(keep)) + GC.load_designs(rpath, cnt, keep=keep)
    vlast = json.load(open(f"{G}/confirm_r5/readings/reading_vlast.json"))
    vfits = json.load(open(f"{G}/confirm_r5/readings/reading_fits.json"))
    V = {o["id"]: o for o in vlast["tests"]}
    VF = {o["id"]: o for o in vfits["tests"]}
    Z = {GC.SEED_RUN: np.random.default_rng(GC.SEED_RUN).standard_normal((GC.DRAWS, len(members)))}
    tests, draws = {}, {}
    for f in p5["family"]:
        o, dr = read_one(D, f, f["setting"][:-1], counts, members, Z, order, "r5")
        if o["determinable"] and o["null_identity_failures"]:
            raise SystemExit(f"REFUSED: NULL_IDENTITY {f['id']}")
        v = V[f["id"]]
        same = o["determinable"] == v["determinable"]
        if o["determinable"]:
            same &= abs(o["T"] - v["T"]) <= 1e-12 * max(1, abs(v["T"]))
            same &= o["p_by_seed"][str(GC.SEED_RUN)]["p"] == v["p"]
            same &= o["set_at"] == v["set_at"] and o["cost_geometries_J"] == v["cost_geometries_J"]
            same &= o["cost_geometries_S"] == v["cost_geometries_S"]
        if not same:
            raise SystemExit(f"REFUSED: REPRO_VLAST {f['id']}")
        o["p"] = o["p_by_seed"][str(GC.SEED_RUN)]["p"] if o["determinable"] else 1.0
        o["cost_geometries"] = sorted(set(o.get("cost_geometries_J", [])) | set(o.get("cost_geometries_S", [])))
        o["S_geometry_set"] = (o.get("set_at") or {}).get("S_geometry")
        o["f"] = f
        tests[f["id"]] = o
        draws[f["id"]] = dr

    tiles = GC.tile_state(f"{G}/pd_points_r5")
    scores = {}
    for name in sorted(os.listdir(f"{G}/pd_points_r5")):
        fp = f"{G}/pd_points_r5/{name}/score.json"
        if name.startswith("h_ng45_") and os.path.exists(fp):
            scores[name] = json.load(open(fp))
    anchor = scores.get("h_ng45_384x192x4", {})
    energy_reason = None if anchor.get("has_result") else (
        "energy and leakage measured at no tile: the anchor's primary 384x192x4 has no result (RD-3)")
    metric_of = {i: tests[i]["metric"] for i in tests}
    dec = decide_labels(tests, tiles, {GC.gname(g) for g in REPLACED}, metric_of, energy_reason)

    side = Side(p3, members, counts)
    four = {}
    four_ids = [i for i in tests if i in dec["confirmed"] or tests[i]["beyond"]]
    if four_ids:
        DR = {"main": D}
        for rd in ("A_rp_s", "T_rs", "E_ops"):
            ref = SECOND_REF if rd == "A_rp_s" else GC.REF
            DR[rd] = (load_reading(dpath, cnt, rd, ref, consts, drop=frozenset(keep))
                      + load_reading(rpath, cnt, rd, ref, consts, keep=keep))
        for i in four_ids:
            f = tests[i]["f"]
            rec = {}
            for rd in FOUR:
                o = tests[i] if rd == "main" else read_one(DR[rd], f, f["setting"][:-1], counts, members, {}, order,
                                                           "r5")[0]
                rec[rd] = {"T": o["T"], "beyond": bool(o["beyond"]), "determinable": o["determinable"],
                           "set_at": o.get("set_at"), "T_closed_clock": o.get("T_closed_clock"),
                           "T_two_sided": o.get("T_two_sided"), "T_declared_G_ref": o.get("T_declared_G_ref"),
                           "band_at_set_reading": None if not o["determinable"] else
                           o["per_reading"][o["set_at"]["r"]]["band"],
                           "J_at_main_nominal": None if not o["determinable"] else o["per_reading"][1]["J"],
                           "S_at_main_nominal": None if not o["determinable"] else o["per_reading"][1]["S"],
                           "null_identity_failures": o.get("null_identity_failures")}
            vals = {rd: rec[rd]["T"] for rd in FOUR}
            if all(v is not None for v in vals.values()):
                arg = min(vals, key=lambda k: vals[k])
                small = vals[arg]
            else:
                arg, small = None, None
            four[i] = {"readings": rec, "smallest": small, "set_by": arg,
                       "beyond_under_all_four": all(rec[rd]["beyond"] for rd in FOUR),
                       "by_the_floor_under_all_four": small is not None and small >= GC.FLOOR and all(
                           rec[rd]["beyond"] for rd in FOUR),
                       "saving_on_separate_base_smallest": None if small is None else 1 - 1 / (1 + small)}
    bes = {}
    for i, o in tests.items():
        if not o["determinable"]:
            continue
        bands = [r["band"] for r in o["per_reading"]] if o["metric"] != "energy" else None
        bes[i] = beside(D, o["f"], o["f"]["setting"][:-1], counts, members, order, side, geoms, consts, {}, bands)

    det = [i for i in tests if tests[i]["determinable"]]
    rows = []
    for d_ in range(GC.DRAWS):
        rows.append([None if draws[i][d_] is None else draws[i][d_] > GC.EQ for i in det])
    chance = GR.chance_count(rows)
    if p5["arch_bit_classes"] != p3["arch_bit_classes"]:
        raise SystemExit("REFUSED: ARCH_BIT_CLASSES")
    DM = (load_reading(dpath, cnt, "main", GC.REF, consts, drop=frozenset(keep))
          + load_reading(rpath, cnt, "main", GC.REF, consts, keep=keep))
    rule = rule_block(DM, p3, geoms, members, side, consts)

    tile_rows = []
    for name, sc in scores.items():
        a, tm, el = sc.get("area") or {}, sc.get("timing") or {}, sc.get("energy_leakage_raw") or {}
        tile_rows.append({"tile": name, "geometry": sc["geometry"], "fallback": sc["fallback"], "view": sc.get("view"),
                          "idle_rows": sc.get("idle_rows"), "has_result": sc["has_result"],
                          "no_result_because": sc.get("no_result_because"), "meets": sc.get("meets"),
                          "A_fit_over_A_rt_minus_1": a.get("rel_error_A_fit_over_A_rt"),
                          "logic_rel_error_beside": a.get("logic_rel_error_beside"),
                          "logic_within_5p4pct": None if a.get("logic_rel_error_beside") is None else
                          abs(a["logic_rel_error_beside"]) <= 0.054,
                          "closes_2ns": tm.get("closes_2ns"), "closes_91MHz": tm.get("closes_study_clock_91MHz"),
                          "closes_1p5GHz": tm.get("closes_low_end_1500MHz"), "fmax_mhz": tm.get("fmax_mhz"),
                          "E_rt_over_E_fit_beside": (el["E_rt_pj_per_mac"] / el["E_fit_pj_per_mac"])
                          if el.get("E_rt_pj_per_mac") and el.get("E_fit_pj_per_mac") else None,
                          "L_rt_over_L_fit_beside": (el["L_rt_mw"] / el["L_fit_mw"])
                          if el.get("L_rt_mw") and el.get("L_fit_mw") else None})

    summary = []
    for i, o in tests.items():
        f = o["f"]
        lab = dec["labels"][i]
        pr = o["per_reading"] if o["determinable"] else []
        nom = pr[1] if pr else {}
        s = {"id": i, "family": f["family"], "metric": o["metric"], "setting": f["setting"], "cell": f["cell"],
             "target_images": f["target_images"], "target_member": f["target_member"],
             "determinable": o["determinable"], "T": o["T"], "beyond": bool(o["beyond"]),
             "p": dec["p_pass1"][i], "bh_pass1": dec["bh_pass1"][i], "p_pass2": dec["p_pass2"][i],
             "bh_pass2": dec["bh_pass2"][i], "meets_floor": o["T"] is not None and o["T"] >= GC.FLOOR,
             "decision": lab["decision"], "labels": lab["labels"],
             "expected_r5": f["expected_to_confirm_r5"], "expected_r4_decision": f["expected_to_confirm_r4_decision"],
             "exploratory_T_r5": f["exploratory_reading_r5"]["T_r5"],
             "exploratory_named": f.get("named"),
             "T_closed_clock": o.get("T_closed_clock"), "T_two_sided": o.get("T_two_sided"),
             "T_declared_G_ref": o.get("T_declared_G_ref"), "T_own_leakage": o.get("T_own_leakage"),
             "ceiling": o.get("ceiling"), "gap_min": o.get("gap_min"), "set_at": o.get("set_at"),
             "cost_geometries": o["cost_geometries"],
             "cost_geometry_states": {g: tiles.get(g) for g in o["cost_geometries"]},
             "band_at_set_reading": None if not pr else pr[o["set_at"]["r"]]["band"],
             "band_sizes": None if not pr else [r["band_size"] for r in pr],
             "J_main_nominal": nom.get("J"), "S_main_nominal": nom.get("S"),
             "nominal_ratio_S_over_J": (nom["S_value"] / nom["J_value"]) if nom else None,
             "saving_on_separate_base": None if o["T"] is None else 1 - 1 / (1 + o["T"]),
             "fits_reading_T": VF[i]["T"], "fits_reading_p": VF[i]["p"],
             "reason_not_determinable": None if o["determinable"] else [
                 {"r": r["r"], "band": r.get("band")} for r in o["per_reading"] if not r["determinable"]][:3],
             "beside": bes.get(i), "four_readings": four.get(i)}
        summary.append(s)
    os.makedirs(out_dir)
    res = {"confirmatory": LABEL.format(what="the analysis's machine record"), "accuracy_source": label,
           "decision_reading": {"designs": dpath, "routed": rpath, "replaced": [GC.gname(g) for g in REPLACED],
                                "designs_main_tagged": len(D), "equal_to_reading_vlast": True},
           "tile_sums": {"files": n_sums, "bad": bad}, "energy_unconfirmable_reason": energy_reason,
           "counts": {"determinable": len(det), "beyond": sum(1 for i in tests if tests[i]["beyond"]),
                      "candidates": dec["candidates"], "unconfirmable": dec["unconfirmable"],
                      "confirmed": dec["confirmed"], "confirmed_by_family": GC.fams(dec["confirmed"]),
                      "expected_r5": [i for i in tests if tests[i]["f"]["expected_to_confirm_r5"]]},
           "chance_count_O_1": {"tests": len(det), **chance,
                                "observed_beyond": sum(1 for i in det if tests[i]["beyond"])},
           "predictive_rule_and_post_hoc_energy_rule": rule, "routed_tiles": tile_rows,
           "tests": summary, "input_digests": digests}
    with open(f"{out_dir}/analysis.json", "x") as fh:
        json.dump(clean(res), fh, indent=1)
    hdr = ["id", "setting", "cell", "metric", "target_images", "determinable", "T", "T_closed_clock", "T_two_sided",
           "T_declared_G_ref", "ceiling", "gap_min", "p", "bh_pass1", "p_pass2", "bh_pass2", "meets_floor", "decision",
           "labels", "expected_r5", "exploratory_T_r5", "cost_geometries", "J_main_nominal", "S_main_nominal",
           "nominal_ratio_S_over_J", "saving_on_separate_base", "S_env_T", "four_smallest", "four_set_by"]
    with open(f"{out_dir}/tests.csv", "x", newline="") as fh:
        fh.write("# " + LABEL.format(what="the 36 tests' table") + "\n")
        w = csv.writer(fh)
        w.writerow(hdr)
        for s in summary:
            se = ((s["beside"] or {}).get("rules") or {}).get("S_env", {}).get("T")
            fr = s["four_readings"] or {}
            w.writerow([s["id"], s["setting"], s["cell"], s["metric"], s["target_images"], s["determinable"], s["T"],
                        s["T_closed_clock"], s["T_two_sided"], s["T_declared_G_ref"], s["ceiling"], s["gap_min"],
                        s["p"], s["bh_pass1"], s["p_pass2"], s["bh_pass2"], s["meets_floor"], s["decision"],
                        " | ".join(s["labels"]), s["expected_r5"], s["exploratory_T_r5"], " ".join(s["cost_geometries"]),
                        s["J_main_nominal"], s["S_main_nominal"], s["nominal_ratio_S_over_J"],
                        s["saving_on_separate_base"], se, fr.get("smallest"), fr.get("set_by")])
    print(json.dumps({"determinable": len(det), "candidates": dec["candidates"], "unconfirmable": dec["unconfirmable"],
                      "confirmed": dec["confirmed"], "chance": {k: chance[k] for k in ("E0", "q95")}}, indent=1))
    for s in summary:
        fr = s["four_readings"] or {}
        print(s["id"], s["decision"], "T", None if s["T"] is None else round(s["T"], 5), "bh2", round(s["bh_pass2"], 5),
              "| four", {k: (None if v["T"] is None else round(v["T"], 5)) for k, v in (fr.get("readings") or {}).items()},
              "| S_env", ((s["beside"] or {}).get("rules") or {}).get("S_env"), "|", "; ".join(s["labels"]))
    return 0


def refs(out_path, ids):
    ""
    if os.path.exists(out_path):
        raise SystemExit(f"REFUSED: OUT_EXISTS {out_path}")
    digests = pin({**PINS, **HELDOUT_PINS})
    p5, p3, members, cnt, label, counts = GC.common()
    consts = consts_of(p3)
    order = {(g["R"], g["C_cells"], g["s"]): i for i, g in enumerate(p5["grid"]["geometry_table"])}
    dpath = f"{G}/sweep_r5/designs.jsonl"
    rpath = f"{G}/sweep_r5/routed_vlast/designs_routed.jsonl"
    keep = set(REPLACED)
    fam = {f["id"]: f for f in p5["family"]}
    res = {"confirmatory": LABEL.format(what="the reference swap beside AC-1 (deciding nothing)"),
           "input_digests": digests, "tests": {}}
    for rd in FOUR:
        for ref in (GC.REF, SECOND_REF):
            D = (load_reading(dpath, cnt, rd, ref, consts, drop=frozenset(keep))
                 + load_reading(rpath, cnt, rd, ref, consts, keep=keep))
            for i in ids:
                f = fam[i]
                o = read_one(D, f, f["setting"][:-1], counts, members, {}, order, "r5")[0]
                pr = o["per_reading"]
                res["tests"].setdefault(i, {})[f"{rd} | {ref}"] = {
                    "determinable": o["determinable"], "T": o["T"], "beyond": bool(o["beyond"]),
                    "undeterminable_readings": [r["r"] for r in pr if not r["determinable"]],
                    "set_at": o.get("set_at"), "band_at_set_reading": None if not o["determinable"] else
                    pr[o["set_at"]["r"]]["band"], "G_ref_main_nominal": pr[1].get("G_ref") if pr[1]["determinable"] else None}
    with open(out_path, "x") as fh:
        json.dump(clean(res), fh, indent=1)
    for i, v in res["tests"].items():
        print(i, {k: (None if x["T"] is None else round(x["T"], 5)) for k, x in v.items()})
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--out", required=True)
    a_ = sub.add_parser("analyze")
    a_.add_argument("--out", required=True)
    r_ = sub.add_parser("refs")
    r_.add_argument("--out", required=True)
    r_.add_argument("--ids", required=True, help="comma-separated test ids")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a.out)
    if a.mode == "refs":
        return refs(a.out, a.ids.split(","))
    return analyze(a.out)


if __name__ == "__main__":
    raise SystemExit(main())

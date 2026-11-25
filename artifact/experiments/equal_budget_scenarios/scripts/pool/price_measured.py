""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
CD = os.path.join(BUILD, "codesign")
P_CD43 = os.path.join(CD, "v43", "cd43_common.py")
P_RR45 = os.path.join(CD, "v45", "cd45_rr.py")
P_POOL = os.path.join(CD, "v43_pool", "pool_common.py")
P_FINE45 = os.path.join(CD, "v45", "cd45_fine.py")
REC_FINE = os.path.join(CD, "v45", "runs", "s0", "fine.json")
REC_RR = os.path.join(CD, "v45", "runs", "s0", "rr.json")
REC_POOL = os.path.join(CD, "v43", "runs", "s0", "pool_plan.json")
PINNED = OrderedDict([
    (P_CD43, "4912db9a7d35339e"), (P_RR45, "390ffd9bcd29d5bb"), (P_POOL, "75bd6fb28f1421cc"),
    (P_FINE45, "6cdee835cb206137"), (os.path.join(CD, "v42", "cd42_common.py"), "f7c3a4d379fece70"),
    (os.path.join(CD, "cd_family.py"), "a90a000d837a04bb"), (os.path.join(BUILD, "x9", "x9_0_v3.py"), "4f2b52a540f271e2"),
    (REC_FINE, "c78be638ff744fb0"), (REC_RR, "27e01281126ad262"), (REC_POOL, "c5dafc1108448d76"),
])


CELLS = OrderedDict([("H10.1", ("SRAM-DDR", 16, 0.5, 2.0)), ("R1'", ("SRAM-DDR", 1, 0.5, 4.0)),
                     ("R1L", ("SRAM-DDR", 1, 0.25, 2.0)), ("K64", ("SRAM-DDR", 64, 0.5, 2.0))])
AREA_MODELS = ("fakeram", "equal area per bit")
PORT_BAND = (0.005, 0.05)
DECLARED_READ = (2.0, 0, 0.05)
S_AU_CONVENTION = 1.13
TECH = "SRAM-DDR"


def sha16(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def skey(s: float) -> str:
    return f"{s:g}"


def check_pins() -> OrderedDict:
    out = OrderedDict()
    for path, want in PINNED.items():
        got = sha16(path)
        if got != want:
            raise SystemExit(f"REFUSED: {os.path.relpath(path, BUILD)} sha256 {got} != pinned {want}")
        out[os.path.relpath(path, BUILD)] = got
    return out


def _literal_cells() -> OrderedDict:
    ""
    found = OrderedDict()
    tree = ast.parse(open(P_POOL).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CELLS" for t in node.targets):
            for k, v in ast.literal_eval(node.value.args[0]):
                found[k] = tuple(v)
    tree = ast.parse(open(P_FINE45).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                and ast.unparse(node.targets[0]) == "FR.PC.CELLS['R1L']"):
            found["R1L"] = tuple(ast.literal_eval(node.value))
    rr = OrderedDict()
    tree = ast.parse(open(P_RR45).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CELLS" for t in node.targets):
            for k, v in ast.literal_eval(node.value.args[0]):
                rr[k] = tuple(v)
    for k, (am, b) in rr.items():
        if found.get(k) is not None and found[k][2:] != (am, b):
            raise SystemExit(f"REFUSED: cd45_rr.py cell {k} {am, b} differs from {found[k]}")
    return found


def cells_checked() -> OrderedDict:
    lit = _literal_cells()
    for k, v in CELLS.items():
        if lit.get(k) != v:
            raise SystemExit(f"REFUSED: declared cell {k} {v} differs from the v4.3/v4.5 source {lit.get(k)}")
    return lit


_RR = None


def estimators():
    ""
    global _RR
    if _RR is None:
        check_pins()
        argv = list(sys.argv)
        spec = importlib.util.spec_from_file_location("cd45_rr_ro", P_RR45)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cd45_rr_ro"] = mod
        spec.loader.exec_module(mod)
        sys.argv = argv
        if sha16(mod.CD43) != PINNED[P_CD43]:
            raise SystemExit("REFUSED: cd45_rr.py loaded another cd43_common.py")
        _RR = mod
    return _RR


def _idx(mask) -> list:
    return [int(i) for i in range(len(mask)) if bool(mask[i])]


def _design(d: dict, geoms: list) -> list:
    ""
    return [geoms.index(d["geom"]), int(d["C"]), int(d["F"]), int(d["Q"]), bool(d["fits"]), round(float(d["E"]), 6),
            round(float(d["area"]), 6)]


def masks():
    ""
    RR = estimators()
    K = RR.K
    PV = K.PV
    cells_checked()
    K.load_cache(with_eighth=True)
    T = PV.techs()
    labels = list(K.LABELS)
    geoms = list(K.GNAMES)
    out = OrderedDict(labels=labels, geoms=geoms, priors=OrderedDict((sc, [float(v) for v in K.PRIOR[sc]])
                                                                       for sc in K.SCEN),
                      s_au=[float(s) for s in K.S_AU], f_band=[float(f) for f in K.F_BAND],
                      area_models=list(AREA_MODELS), port_band=list(PORT_BAND), declared_read=list(DECLARED_READ),
                      s_au_convention=S_AU_CONVENTION, cells=OrderedDict(), rr=OrderedDict())
    cr = PV.c_ref(TECH)
    for lname, caps in (("declared", None), ("check", K.CAPS8)):
        D = K.energy_designs(T, TECH, caps=caps)
        if lname == "declared":
            for cname, (tn, Kk, am, beta) in CELLS.items():
                c = OrderedDict(spec=[tn, Kk, am, beta], by_s_au=OrderedDict())
                for s in K.S_AU:
                    rows = []
                    for f in K.F_BAND:
                        for a, area_model in enumerate(AREA_MODELS):
                            Ds = K.inflate(K.scaled(D, f, area_model), s)
                            pool = [d for d in Ds if d["K"] == Kk and d["area"] <= am * K.A_L + 1e-9]
                            adm = [d for d in pool if d["E"] <= beta * cr]
                            for ap in PORT_BAND:
                                P = K.energy_cell(Ds, T, tn, Kk, am, beta, f, area_model, ap=ap, s_au=s)
                                J, FO, RB = _idx(P["J"]["set"]), _idx(P["FO"]["set"]), _idx(P["RB"]["set"])
                                if sorted({labels.index(d["label"]) for d in adm}) != J:
                                    raise SystemExit(f"REFUSED: {cname} J set differs from the design rows")
                                if sorted({labels.index(d["label"]) for d in adm if d["fits"]}) != FO:
                                    raise SystemExit(f"REFUSED: {cname} FO set differs from the design rows")
                                dj, dfo = OrderedDict(), OrderedDict()
                                for i in J:
                                    ds = [d for d in adm if d["label"] == labels[i]]
                                    dj[str(i)] = _design(min(ds, key=lambda d: (d["E"], d["area"])), geoms)
                                    dsf = [d for d in ds if d["fits"]]
                                    if dsf:
                                        dfo[str(i)] = _design(min(dsf, key=lambda d: (d["E"], d["area"])), geoms)
                                rows.append(OrderedDict(combo=[float(f), a, float(ap)], J=J, FO=FO, RB=RB,
                                                        design_J=dj, design_FO=dfo))
                    c["by_s_au"][skey(s)] = rows
                out["cells"][cname] = c
        base = {f: K.scaled(D, f, "fakeram") for f in K.F_BAND}
        Dsf = {(s, f): K.inflate(base[f], s) for s in K.S_AU for f in K.F_BAND}
        rr = OrderedDict()
        for am in K.AREA_MULT:
            for b in PV.BETAS:
                vals = []
                for s in K.S_AU:
                    for f in K.F_BAND:
                        P = K.energy_cell(Dsf[(s, f)], T, TECH, 1, am, b, f, "fakeram", s_au=s, caps=caps)
                        vals.append(OrderedDict(s_au=float(s), f=float(f), J=_idx(P["J"]["set"]),
                                                RB=_idx(P["RB"]["set"]), FO=_idx(P["FO"]["set"])))
                rr[f"{am!r}|{b!r}"] = OrderedDict(cell=[float(am), float(b)], values=vals)
        out["rr"][lname] = rr
    return out


def table_vector(table: dict, labels: list):
    ""
    if table["kind"] == "prior":
        acc = [float(table["acc_pp"][lab]) for lab in labels]
        return acc, None, None
    if table["kind"] != "measured":
        raise ValueError(table["kind"])
    total = int(table["total"])
    counts = [table["correct"].get(lab) for lab in labels]
    for lab, c in zip(labels, counts):
        if c is not None and (int(c) != c or not 0 <= int(c) <= total):
            raise ValueError(f"{lab}: correct count {c} is not an integer in 0..{total}")
    counts = [None if c is None else int(c) for c in counts]
    acc = [None if c is None else 100.0 * c / total for c in counts]
    return acc, counts, total


def pick(members: list, acc: list):
    ""
    if not members:
        return None
    missing = [i for i in members if acc[i] is None]
    if missing:
        return ("missing", missing)
    best = members[0]
    for i in members[1:]:
        if acc[i] > acc[best]:
            best = i
    return best


def _gap(j, f, acc, counts, total):
    if counts is not None:
        return round((counts[j] - counts[f]) * 100.0 / total, 4), counts[j] - counts[f]
    return round(float(acc[j] - acc[f]), 4), None


def read_cells(masks_: dict, table: dict, exclude=()) -> OrderedDict:
    labels = masks_["labels"]
    geoms = masks_["geoms"]
    acc, counts, total = table_vector(table, labels)
    unknown = sorted(set(exclude) - set(labels))
    if unknown:
        raise ValueError(f"exclude names members outside the family: {unknown}")
    gone = {labels.index(x) for x in exclude}

    def adm(row, key):
        return [i for i in row[key] if i not in gone] if gone else row[key]

    def lab(i):
        return labels[i]

    def design(row, key, i):
        d = row[key].get(str(i))
        if d is None:
            return None
        return OrderedDict(geometry=geoms[d[0]], C=d[1], F=d[2], Q=d[3], resident=d[4], E_over_c_ref_units=d[5],
                           area_mm2=d[6])

    out = OrderedDict()
    for cname, c in masks_["cells"].items():
        per_s = OrderedDict()
        missing_needed: set = set()
        all_gaps = []
        band_ok = True
        for s, rows in c["by_s_au"].items():
            gaps, per, ok, need_s = [], [], True, set()
            for row in rows:
                j, f = pick(adm(row, "J"), acc), pick(adm(row, "FO"), acc)
                det = not (isinstance(j, tuple) or isinstance(f, tuple))
                for p in (j, f):
                    if isinstance(p, tuple):
                        need_s.update(p[1])
                rec = OrderedDict(combo=row["combo"],
                                  J_x10=(lab(j) if isinstance(j, int) else ("not determinable" if j else None)),
                                  FO_x10=(lab(f) if isinstance(f, int) else ("not determinable" if f else None)))
                if det and j is not None and f is not None:
                    g, gi = _gap(j, f, acc, counts, total)
                    rec["gap_x10"] = g
                    if gi is not None:
                        rec["gap_images"] = gi
                    rec["J_resident"] = j in row["FO"]
                    gaps.append((g, gi, row["combo"]))
                else:
                    rec["gap_x10"] = None
                    ok = ok and det
                per.append(rec)
            band_ok = band_ok and ok
            missing_needed |= need_s
            all_gaps += [(g, gi, s, cb) for g, gi, cb in gaps]
            per_s[s] = OrderedDict(
                determinable=ok,
                missing_needed=[lab(i) for i in sorted(need_s)],
                robust_gap_x10=(min(g for g, _, _ in gaps) if (gaps and ok) else None),
                max_gap_x10=(max(g for g, _, _ in gaps) if (gaps and ok) else None),
                J_x10=sorted({p["J_x10"] or "none" for p in per}), FO_x10=sorted({p["FO_x10"] or "none" for p in per}),
                J_resident_combos=sum(1 for p in per if p.get("J_resident")), n_combos=len(per), per_combo=per)
        conv = per_s[skey(masks_["s_au_convention"])]
        band = None
        if band_ok and all_gaps:
            g, gi, s, cb = min(all_gaps, key=lambda x: (x[0], x[2], x[3]))
            band = OrderedDict(gap=g, gap_images=gi, at_s_au=s, at_combo=cb)
        dr_rows = c["by_s_au"][skey(masks_["s_au_convention"])]
        dr = [r for r in dr_rows if tuple(r["combo"]) == tuple(masks_["declared_read"])][0]
        j, f = pick(adm(dr, "J"), acc), pick(adm(dr, "FO"), acc)
        decl = OrderedDict(s_au=masks_["s_au_convention"], combo=dr["combo"])
        if isinstance(j, int) and isinstance(f, int):
            g, gi = _gap(j, f, acc, counts, total)
            decl.update(determinable=True, J=lab(j), FO=lab(f), gap=g, gap_images=gi,
                        J_acc_pp=round(acc[j], 4), FO_acc_pp=round(acc[f], 4),
                        J_correct=None if counts is None else counts[j], FO_correct=None if counts is None else counts[f],
                        J_resident=j in dr["FO"], J_design=design(dr, "design_J", j), FO_design=design(dr, "design_FO", f),
                        ties_J=[lab(i) for i in adm(dr, "J") if i != j and acc[i] == acc[j]],
                        ties_FO=[lab(i) for i in adm(dr, "FO") if i != f and acc[i] == acc[f]])
        else:
            decl.update(determinable=bool(j is not None and f is not None and not isinstance(j, tuple)
                                          and not isinstance(f, tuple)),
                        J=(lab(j) if isinstance(j, int) else ("not determinable" if j else None)),
                        FO=(lab(f) if isinstance(f, int) else ("not determinable" if f else None)), gap=None)
        out[cname] = OrderedDict(
            spec=c["spec"], determinable=band_ok,
            missing_needed=[lab(i) for i in sorted(missing_needed)],
            robust_gap_convention=conv["robust_gap_x10"] if conv["determinable"] else None,
            robust_gap_band=band, declared_read=decl, by_s_au=per_s)
    return out


def read_rr(masks_: dict, tables: list, names: list, ladder: str) -> OrderedDict:
    ""
    RR = estimators()
    labels = masks_["labels"]
    vecs = [table_vector(t, labels) for t in tables]
    per = OrderedDict()
    rec = OrderedDict((c, OrderedDict((n, []) for n in names)) for c in RR.CELLS)
    undeterminable = OrderedDict()
    for key, cell in masks_["rr"][ladder].items():
        am, b = cell["cell"]
        entry = OrderedDict((n, []) for n in names)
        for v in cell["values"]:
            for n, (acc, counts, total) in zip(names, vecs):
                j, rb, fo = pick(v["J"], acc), pick(v["RB"], acc), pick(v["FO"], acc)
                if any(isinstance(p, tuple) for p in (j, rb)):
                    undeterminable.setdefault(f"{am:g} A_L, {b:g} c_ref", set()).add(n)
                    entry[n].append((v["s_au"], v["f"], "not determinable", "not determinable", None))
                    continue
                j = -1 if j is None else j
                rb = -1 if rb is None else rb
                fo = -1 if (fo is None or isinstance(fo, tuple)) else fo
                ok = j >= 0 and rb >= 0 and rb != j
                reg = None
                if ok:
                    reg = (round((counts[j] - counts[rb]) * 100.0 / total, 4) if counts is not None
                           else round(float(acc[j] - acc[rb]), 4))
                entry[n].append((v["s_au"], v["f"], labels[j] if j >= 0 else "none",
                                 labels[rb] if rb >= 0 else "none", reg))
                for cn, cv in RR.CELLS.items():
                    if (am, b) == cv:
                        why = ("qualifies" if ok else ("no admissible J or RB member" if (j < 0 or rb < 0)
                                                       else "RB* is J*'s member"))
                        rec[cn][n].append(OrderedDict(
                            s_au=v["s_au"], f=v["f"], J=labels[j] if j >= 0 else "none",
                            RB=labels[rb] if rb >= 0 else "none", FO=labels[fo] if fo >= 0 else "none",
                            regret=reg, reason=why, n_J=len(v["J"]), n_RB=len(v["RB"])))
        per[(am, b)] = entry
    res = OrderedDict()
    if len(names) == 2:
        both = []
        for c, cell in per.items():
            vals = cell[names[0]] + cell[names[1]]
            if all(v[4] is not None for v in vals):
                both.append((c, min(v[4] for v in vals)))
        both = RR.rank(both)
        res["both"] = [[list(c), m] for c, m in both]
    for n in names:
        q = []
        for c, cell in per.items():
            vals = cell[n]
            if all(v[4] is not None for v in vals):
                q.append((c, min(v[4] for v in vals)))
        q = RR.rank(q)
        top = q[0] if q else None
        res[n] = OrderedDict(
            qualifying=[[list(c), m, round(RR.power_sup(m, 30, 0.05 / 4), 3), round(RR.power_sup(m, 30, 0.05 / 3), 3)]
                        for c, m in q],
            declared=list(top[0]) if top else None,
            declared_min_regret=top[1] if top else None,
            power_a4=round(RR.power_sup(top[1], 30, 0.05 / 4), 3) if top else None,
            power_a3=round(RR.power_sup(top[1], 30, 0.05 / 3), 3) if top else None,
            detail={("%g A_L, %g c_ref" % c): per[c][n] for c, _ in q})
        und = sorted(k for k, v in undeterminable.items() if n in v)
        if und:
            res[n]["not_determinable_cells"] = und
            res[n]["declared"] = "not determinable"
    res["cells"] = rec
    return res


def read(masks_: dict, table: dict) -> OrderedDict:
    out = OrderedDict(table=OrderedDict((k, table[k]) for k in ("kind", "label") if k in table),
                      cells=read_cells(masks_, table), rule_rr=OrderedDict())
    for ladder in ("declared", "check"):
        out["rule_rr"][ladder] = read_rr(masks_, [table], ["table"], ladder)
    return out


def prior_tables(masks_: dict) -> OrderedDict:
    return OrderedDict((sc, {"kind": "prior", "label": sc, "acc_pp": dict(zip(masks_["labels"], v))})
                       for sc, v in masks_["priors"].items())


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def regress(masks_: dict) -> OrderedDict:
    pins = check_pins()
    rec_fine = json.load(open(REC_FINE))["cells"]
    rec_pool = json.load(open(REC_POOL))["cells"]
    rec_rr = json.load(open(REC_RR))
    tabs = prior_tables(masks_)
    reads = OrderedDict((sc, read_cells(masks_, t)) for sc, t in tabs.items())
    records = OrderedDict()
    fields = ("robust_gap_x10", "max_gap_x10", "J_x10", "FO_x10", "n_combos")
    for cname in CELLS:
        for s in (1.0, 1.13):
            for sc in tabs:
                key = f"{cname} | s_AU {s:.3f} | {sc}"
                mine = reads[sc][cname]["by_s_au"][skey(s)]
                m = {k: mine[k] for k in fields}
                m["per_combo"] = [{k: p[k] for k in ("combo", "J_x10", "FO_x10", "gap_x10")} for p in mine["per_combo"]]
                r = {k: rec_fine[key][k] for k in fields}
                r["per_combo"] = [{k: p[k] for k in ("combo", "J_x10", "FO_x10", "gap_x10")}
                                  for p in rec_fine[key]["per_combo"]]
                records[f"v45 fine.json: {key}"] = OrderedDict(equal=canon(m) == canon(r), mine=m["robust_gap_x10"],
                                                               recorded=r["robust_gap_x10"])
    for cname in ("H10.1", "R1'", "K64"):
        for s in masks_["s_au"]:
            for sc in tabs:
                key = f"X10-TA|regression flat +0|{s:g}|{sc[:3]}"
                mine = reads[sc][cname]["by_s_au"][skey(s)]
                m = dict(min=mine["robust_gap_x10"], max=mine["max_gap_x10"], J=mine["J_x10"], FO=mine["FO_x10"],
                         n=mine["n_combos"])
                r = {k: rec_pool[cname][key][k] for k in ("min", "max", "J", "FO", "n")}
                records[f"v43 pool_plan.json: {cname} {key}"] = OrderedDict(equal=canon(m) == canon(r), mine=m["min"],
                                                                            recorded=r["min"])
    t_lea, t_fix = tabs["learned step (LSQ)"], tabs["fixed step (STE)"]
    for ladder in ("declared", "check"):
        mine = read_rr(masks_, [t_lea, t_fix], ["lea", "fix"], ladder)
        for part in ("both", "lea", "fix", "cells"):
            m = json.loads(json.dumps(mine[part]))
            r = rec_rr[ladder][part]
            if part == "both":
                summary, rsummary = (m[0] if m else None), (r[0] if r else None)
            elif part == "cells":
                summary = rsummary = "R1' and R1L per (s_AU, f) records"
            else:
                summary = [m["declared"], m["declared_min_regret"]]
                rsummary = [r["declared"], r["declared_min_regret"]]
            records[f"v45 rr.json: {ladder} ladder, {part}"] = OrderedDict(equal=canon(m) == canon(r),
                                                                           mine=summary, recorded=rsummary)
    headline = OrderedDict()
    for cname in CELLS:
        headline[cname] = OrderedDict((sc, reads[sc][cname]["robust_gap_convention"]) for sc in tabs)
    n_eq = sum(1 for v in records.values() if v["equal"])
    return OrderedDict(pins=pins, records=records, equal=n_eq, of=len(records),
                       robust_gap_at_the_convention=headline,
                       robust_gap_band_minimum=OrderedDict(
                           (c, OrderedDict((sc, reads[sc][c]["robust_gap_band"]["gap"]) for sc in tabs)) for c in CELLS),
                       declared_read_gap=OrderedDict(
                           (c, OrderedDict((sc, reads[sc][c]["declared_read"]["gap"]) for sc in tabs)) for c in CELLS))


def fake_tables(masks_: dict) -> OrderedDict:
    ""
    labels = masks_["labels"]
    prior = masks_["priors"]["fixed step (STE)"]
    a = {"kind": "measured", "label": "fixture (a): round(50 x fixed-step prior)", "total": 5000,
         "correct": {lab: int(math.floor(50 * p + 0.5)) for lab, p in zip(labels, prior)}}
    ra = read_cells(masks_, a)
    gone = ra["H10.1"]["declared_read"]["J"]
    b = {"kind": "measured", "label": f"fixture (b): fixture (a) with {gone} missing", "total": 5000,
         "correct": {lab: (None if lab == gone else c) for lab, c in a["correct"].items()}}
    return OrderedDict(a=a, b=b)


def fixture(masks_: dict) -> OrderedDict:
    out = OrderedDict()
    for k, t in fake_tables(masks_).items():
        r = read(masks_, t)
        cells = OrderedDict()
        for c, v in r["cells"].items():
            cells[c] = OrderedDict(
                determinable=v["determinable"], missing_needed=v["missing_needed"],
                robust_gap_convention=v["robust_gap_convention"], robust_gap_band=v["robust_gap_band"],
                declared_read=OrderedDict((x, v["declared_read"].get(x))
                                          for x in ("J", "FO", "gap", "gap_images", "J_resident")))
        rr = OrderedDict()
        for ld in ("declared", "check"):
            tab = r["rule_rr"][ld]["table"]
            rr[ld] = OrderedDict(declared=tab["declared"], declared_min_regret=tab["declared_min_regret"],
                                 not_determinable_cells=tab.get("not_determinable_cells", []))
        out[k] = OrderedDict(label=t["label"], missing=[lab for lab, c in t["correct"].items() if c is None],
                             cells=cells, rule_rr=rr)
    return out


def dump(obj, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=0, sort_keys=True)
        fh.write("\n")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    if mode == "masks":
        print(f"== price_measured.py {sha16(os.path.abspath(__file__))} masks; pins:")
        for k, v in check_pins().items():
            print(f"   {k} {v}")
        m = masks()
        dump(m, argv[1])
        print(f"   members {len(m['labels'])}; cells {list(m['cells'])}; rule RR cells "
              f"{ {k: len(v) for k, v in m['rr'].items()} }")
        print(f"== output {os.path.basename(argv[1])} sha256 {sha16(argv[1])}")
        return 0
    if mode == "read":
        m = json.load(open(argv[1]), object_pairs_hook=OrderedDict)
        t = json.load(open(argv[2]))
        dump(read(m, t), argv[3])
        print(f"== read {os.path.basename(argv[2])} -> {os.path.basename(argv[3])} sha256 {sha16(argv[3])}")
        return 0
    if mode == "regress":
        m = json.load(open(argv[1]), object_pairs_hook=OrderedDict)
        r = regress(m)
        dump(r, argv[2])
        print(f"== price_measured.py {sha16(os.path.abspath(__file__))} regress on {os.path.basename(argv[1])} "
              f"{sha16(argv[1])}")
        for k, v in r["records"].items():
            print(f"   {'equal' if v['equal'] else 'DIFFERENT'}  {k}  (mine {v['mine']}, recorded {v['recorded']})")
        print(f"== {r['equal']} of {r['of']} recorded records equal byte for byte")
        for c, v in r["robust_gap_at_the_convention"].items():
            print(f"   {c}: robust gap at the convention {dict(v)}; band minimum {dict(r['robust_gap_band_minimum'][c])}; "
                  f"declared read {dict(r['declared_read_gap'][c])}")
        print(f"== output {os.path.basename(argv[2])} sha256 {sha16(argv[2])}")
        return 0 if r["equal"] == r["of"] else 1
    if mode == "fixture":
        m = json.load(open(argv[1]), object_pairs_hook=OrderedDict)
        f = fixture(m)
        dump(f, argv[2])
        for k, v in f.items():
            print(f"== {v['label']}")
            for c, d in v["cells"].items():
                print(f"   {c}: determinable {d['determinable']}, band minimum {d['robust_gap_band']}, convention "
                      f"{d['robust_gap_convention']}, declared read {dict(d['declared_read'])}; needs missing "
                      f"{d['missing_needed']}")
            print(f"   rule RR: {dict(v['rule_rr'])}")
        print(f"== output {os.path.basename(argv[2])} sha256 {sha16(argv[2])}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

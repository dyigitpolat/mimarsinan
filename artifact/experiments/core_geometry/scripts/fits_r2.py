""
import csv
import json
import math
import os
import re
import sys

import numpy as np

RA = "records/research_artifacts"
PD = f"{RA}/06_build/pd_probe/results"
G = f"{RA}/13_geometry_study"

POINTS = {
    "t0_ng45_64x16x4": (f"{PD}/t0_ng45_64x16x4", "nangate45", 2.0, "pd_probe, halo 5"),
    "g2_ng45_128x32x4": (f"{PD}/g2_ng45_128x32x4", "nangate45", 2.0, "pd_probe, halo 5"),
    "g6_ng45_128x64x4": (f"{PD}/g6_ng45_128x64x4", "nangate45", 2.0, "pd_probe g6c, halo 20 and fastroute 0.10"),
    "g3_ng45_256x64x4": (f"{PD}/g3_ng45_256x64x4", "nangate45", 2.0, "pd_probe, halo 5"),
    "g3r2_ng45_256x64x4_rpc2": (f"{PD}/g3r2_ng45_256x64x4_rpc2", "nangate45", 2.0, "pd_probe, halo 5"),
    "x1_ng45_512x64x4": (f"{G}/pd_points/x1_ng45_512x64x4", "nangate45", 2.0, "this study, halo 20"),
    "x2_ng45_128x256x4": (f"{G}/pd_points/x2_ng45_128x256x4", "nangate45", 2.0, "this study, halo 20"),
    "x3_ng45_128x64x1": (f"{G}/pd_points/x3_ng45_128x64x1", "nangate45", 2.0, "this study, halo 20"),
    "x4_ng45_128x64x2": (f"{G}/pd_points/x4_ng45_128x64x2", "nangate45", 2.0, "this study, halo 20"),
    "x5_ng45_128x64x8": (f"{G}/pd_points/x5_ng45_128x64x8", "nangate45", 2.0, "this study, halo 20"),
    "x6_ng45_128x64x1_rpc2": (f"{G}/pd_points/x6_ng45_128x64x1_rpc2", "nangate45", 2.0,
                              "this study, revision 2, halo 20"),
    "x7_ng45_128x64x4_rpc2": (f"{G}/pd_points/x7_ng45_128x64x4_rpc2", "nangate45", 2.0,
                              "this study, revision 2, halo 20"),
}
STUDY_A_BIT = 1.0373001098632812
MACRO_VIEW = {(64, 64): "fakeram45_64x64", (128, 256): "fakeram45_128x256", (128, 32): "fakeram45_128x32"}


def params(d):
    for ln in open(os.path.join(d, "config.mk")):
        if ln.startswith("#"):
            continue
        m = re.search(r"VERILOG_TOP_PARAMS\s*=\s*(.*)", ln)
        if m:
            t = m.group(1).split()
            return {t[i]: int(t[i + 1]) for i in range(0, len(t), 2)}
    raise SystemExit(f"no VERILOG_TOP_PARAMS in {d}/config.mk")


def macro_group_w(d):
    ""
    txt = open(os.path.join(d, "6_finish.rpt")).read()
    i = txt.index("report_power")
    m = re.search(r"\nMacro\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", txt[i:])
    return [float(m.group(k)) for k in range(1, 5)]


def load():
    rows, missing = {}, []
    for name, (d, plat, tns, note) in POINTS.items():
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            missing.append(name)
            continue
        s = json.load(open(f))
        if s.get("route_drc_errors") not in (0, None) or not s.get("instance_area_um2"):
            missing.append(f"{name} (drc {s.get('route_drc_errors')}, no area)")
            continue
        p = params(d)
        r, c, w, rpc, md, mw = p["R"], p["C"], p["W"], p["RPC"], p["MD"], p["MW"]
        nd = math.ceil((r // rpc) / md)
        nwm = (c * w) // mw
        mg = macro_group_w(d)
        rows[name] = dict(platform=plat, R=r, C=c, W=w, RPC=rpc, MD=md, MW=mw, ND=nd, NWM=nwm,
                          macros=nd * nwm * rpc, macro_view=MACRO_VIEW.get((md, mw), f"fakeram45_{md}x{mw}"),
                          target_ns=tns, note=note, bits=r * c * w, macro_w=mg[3],
                          **{k: s[k] for k in ("instance_area_um2", "macro_area_um2", "stdcell_area_um2",
                                               "seq_area_um2", "comb_area_um2", "fmax_hz", "power_total_w",
                                               "power_leakage_w", "total_elapsed_s", "route_drc_errors")})
    return rows, missing


def ols(X, y, names):
    X, y = np.asarray(X, float), np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coef
    loo = []
    for i in range(len(y)):
        keep = [j for j in range(len(y)) if j != i]
        if len(keep) < X.shape[1]:
            loo.append(None)
            continue
        ci, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        loo.append(float((X[i] @ ci - y[i]) / y[i]))
    rel = (fit - y) / y
    return {"coef": dict(zip(names, (float(v) for v in coef))), "n": int(len(y)), "params": int(X.shape[1]),
            "rel_residuals": [float(v) for v in rel], "max_abs_rel_residual": float(np.max(np.abs(rel))),
            "loo_rel_errors": loo,
            "max_abs_loo_rel_error": float(max(abs(v) for v in loo if v is not None)) if any(
                v is not None for v in loo) else None}


def main():
    rows, missing = load()
    out = {"exploratory": "EXPLORATORY: per-geometry fits of the core geometry study, plan revision 2 "
                          "(RA/13_geometry_study); not a record", "points": rows, "missing_points": missing}
    rate = {n: (1e9 / r["target_ns"]) * r["C"] * r["RPC"] for n, r in rows.items()}
    pj = {n: r["power_total_w"] / rate[n] * 1e12 for n, r in rows.items()}

    idle_w = {n: r["macro_w"] * (r["ND"] - 1) / r["ND"] for n, r in rows.items()}
    pj_dc = {n: (r["power_total_w"] - idle_w[n]) / rate[n] * 1e12 for n, r in rows.items()}
    pj_logic = {n: (r["power_total_w"] - r["macro_w"]) / rate[n] * 1e12 for n, r in rows.items()}
    pj_macro = {n: r["macro_w"] / rate[n] * 1e12 for n, r in rows.items()}
    out["per_mac_pj"] = {"measured": pj, "idle_depth_macros_removed": pj_dc, "logic_only": pj_logic,
                         "macro_only": pj_macro, "idle_depth_macro_w": idle_w,
                         "basis": "power at default activity / (500 MHz x C x RPC); idle depth macros: the Macro row x "
                                  "(ND - 1) / ND (cim_tile enables one depth macro per bank per cycle, ce = sel && "
                                  "run)"}

    views = []
    with open(f"{G}/fits/fakeram45_views.csv") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        views.append({k: (float(v) if k not in ("view", "lef_sha256") else v) for k, v in r.items()})
    per_bit = [v["um2_per_bit"] for v in views]
    out["F1_macro_area_per_bit"] = {
        "routed_tiles": {n: r["macro_area_um2"] / r["bits"] for n, r in rows.items()},
        "declared_rule": {"nominal_um2_per_bit": STUDY_A_BIT, "low_um2_per_bit": min(per_bit),
                          "high_um2_per_bit": max(per_bit),
                          "basis": "the study's macro (fakeram45_128x256, g3) at every geometry; the band ends are "
                                   "the densest and the least dense fakeram45 views"}}

    ng = {n: r for n, r in rows.items() if r["platform"] == "nangate45" and r["RPC"] == 1}
    X = [[1.0, r["R"], r["C"], r["C"] * r["W"]] for r in ng.values()]
    f2 = ols(X, [r["stdcell_area_um2"] for r in ng.values()],
             ["k0_um2", "kR_um2_per_row", "kC0_um2_per_col", "kCW_um2_per_col_per_bit"])
    f2["points"] = list(ng)
    k = f2["coef"]

    def logic(R, C, W):
        return k["k0_um2"] + k["kR_um2_per_row"] * R + C * (k["kC0_um2_per_col"] + k["kCW_um2_per_col_per_bit"] * W)

    held = json.load(open(f"{G}/fits/heldout_predictions.json"))
    f2["heldout_revision_1"] = {"written_at": held["written_at"], "scores": {}}
    for law_key, names in (("rows_columns_law:stdcell_area_um2", ("x1_ng45_512x64x4", "x2_ng45_128x256x4")),
                           ("cell_bits_law:stdcell_area_um2", ("x5_ng45_128x64x8",)),
                           ("rows_columns_law:power_total_w", ("x1_ng45_512x64x4", "x2_ng45_128x256x4")),
                           ("cell_bits_law:power_total_w", ("x5_ng45_128x64x8",))):
        for n in names:
            key = "stdcell_area_um2" if "stdcell" in law_key else "power_total_w"
            p_, m_ = held["predictions"][law_key][n], rows[n][key]
            f2["heldout_revision_1"]["scores"][f"{law_key}:{n}"] = {"predicted": p_, "measured": m_,
                                                                     "rel_error": (p_ - m_) / m_}
    out["F2_logic_area"] = f2

    tiles = {}
    for n, r in ng.items():
        mod = r["bits"] * STUDY_A_BIT + logic(r["R"], r["C"], r["W"])
        tiles[n] = {"modeled_um2": mod, "routed_instance_um2": r["instance_area_um2"],
                    "rel_error": mod / r["instance_area_um2"] - 1, "macro_view": r["macro_view"],
                    "routed_um2_per_bit": r["macro_area_um2"] / r["bits"],
                    "in_sample": "logic in-sample (an F2 fit point); macro at the declared view"
                    if r["macro_view"] == "fakeram45_128x256" else
                    "logic in-sample (an F2 fit point); macro at another view than the declared one"}
    out["G_C5_modeled_tiles"] = tiles

    pairs = {"g3r2_minus_g3": ("g3r2_ng45_256x64x4_rpc2", "g3_ng45_256x64x4"),
             "x6_minus_x3": ("x6_ng45_128x64x1_rpc2", "x3_ng45_128x64x1"),
             "x7_minus_g6": ("x7_ng45_128x64x4_rpc2", "g6_ng45_128x64x4")}
    inc = {}
    for key, (a, b) in pairs.items():
        if a in rows and b in rows:
            ra, rb = rows[a], rows[b]
            inc[key] = {"s": ra["W"], "R": ra["R"], "C": ra["C"],
                        "um2_per_col_per_extra_row_slot": (ra["stdcell_area_um2"] - rb["stdcell_area_um2"])
                        / (ra["C"] * (ra["RPC"] - rb["RPC"])),
                        "seq_um2_per_col": (ra["seq_area_um2"] - rb["seq_area_um2"]) / ra["C"],
                        "comb_um2_per_col": (ra["comb_area_um2"] - rb["comb_area_um2"]) / ra["C"]}
    f_inc = {"points": inc}
    if len(inc) == 3:
        pts = list(inc.values())
        law = ols([[1.0, p["s"]] for p in pts], [p["um2_per_col_per_extra_row_slot"] for p in pts],
                  ["i0_um2", "i1_um2_per_bit"])
        i0, i1 = law["coef"]["i0_um2"], law["coef"]["i1_um2_per_bit"]
        law["i_of_s"] = {str(s): i0 + i1 * s for s in (1, 2, 3, 4)}
        law["positive_on_1_to_4"] = bool(all(i0 + i1 * s > 0 for s in (1, 2, 3, 4)))
        law["points_order"] = list(inc)
        f_inc["cell_bits_line"] = law
        s4 = [inc["g3r2_minus_g3"]["um2_per_col_per_extra_row_slot"],
              inc["x7_minus_g6"]["um2_per_col_per_extra_row_slot"]]
        f_inc["s4_points_spread_of_mean"] = abs(s4[0] - s4[1]) / (sum(s4) / 2)
        f_inc["R_dependence_named"] = bool(f_inc["s4_points_spread_of_mean"] > 0.15)
    f_inc["ruled_increment_um2"] = inc.get("g3r2_minus_g3", {}).get("um2_per_col_per_extra_row_slot")
    out["G_C1_row_slot_increment"] = f_inc

    hp = f"{G}/fits/heldout_predictions_rpc2.json"
    if os.path.exists(hp):
        h2 = json.load(open(hp))
        sc = {}
        for n, pr in h2["predictions"].items():
            if n in rows:
                for key in ("stdcell_area_um2", "power_total_w"):
                    sc[f"{key}:{n}"] = {"predicted": pr[key], "measured": rows[n][key],
                                        "rel_error": (pr[key] - rows[n][key]) / rows[n][key]}
                sc[f"pj_per_mac:{n}"] = {"predicted": pr["pj_per_mac_default_activity"], "measured": pj[n],
                                         "rel_error": (pr["pj_per_mac_default_activity"] - pj[n]) / pj[n]}
        out["heldout_rpc2"] = {"written_at": h2["written_at"], "scores": sc}

    f3 = {}
    wser = [n for n, r in ng.items() if (r["R"], r["C"]) == (128, 64)]
    for tag, src in (("measured_revision_1", pj), ("depth_corrected_revision_2", pj_dc)):
        fw = ols([[1.0, rows[n]["W"]] for n in wser], [src[n] for n in wser], ["a_pj", "b_pj_per_bit"])
        a, b = fw["coef"]["a_pj"], fw["coef"]["b_pj_per_bit"]
        fw["points"] = wser
        fw["rho_W"] = {str(s): (a + b * s) / (a + 4 * b) for s in (1, 2, 3, 4, 8)}
        fw["per_4bit_weight_x_slot"] = {str(s): math.ceil(4 / s) * (a + b * s) / (a + 4 * b) for s in (1, 2, 3, 4)}
        f3[f"cell_bits_law_{tag}"] = fw
    f3["adopted"] = "cell_bits_law_depth_corrected_revision_2 (G-C3); no R or C dependence in the main constants"
    f3["operand_bits_rule_W_over_4"] = {str(s): s / 4 for s in (1, 2, 3, 4, 8)}

    rser = [n for n in ("g6_ng45_128x64x4", "g3_ng45_256x64x4", "x1_ng45_512x64x4") if n in rows]
    line = ols([[1.0, rows[n]["R"]] for n in rser], [pj_dc[n] for n in rser], ["e0_pj", "eR_pj_per_row"])
    e0, eR = line["coef"]["e0_pj"], line["coef"]["eR_pj_per_row"]
    line["points"] = rser
    line["rho_R_vs_256"] = {str(R): (e0 + eR * R) / (e0 + eR * 256) for R in (64, 128, 256, 512)}
    line["measured_ratio_vs_g6"] = {n: pj[n] / pj["g6_ng45_128x64x4"] for n in rser}
    line["depth_corrected_ratio_vs_g6"] = {n: pj_dc[n] / pj_dc["g6_ng45_128x64x4"] for n in rser}
    line["positive_on_64_to_512"] = bool(all(e0 + eR * R > 0 for R in (64, 128, 256, 512)))
    f3["E_RC_rows_line_depth_corrected"] = line

    if "x6_ng45_128x64x1_rpc2" in rows and "x7_ng45_128x64x4_rpc2" in rows:
        p1, p4 = pj["x6_ng45_128x64x1_rpc2"], pj["x7_ng45_128x64x4_rpc2"]
        b2 = (p4 - p1) / 3.0
        a2 = p1 - b2
        rp2 = {"a2_pj": a2, "b2_pj_per_bit": b2,
               "rho_W2": {str(s): (a2 + b2 * s) / (a2 + 4 * b2) for s in (1, 2, 3, 4)},
               "positive_on_1_to_4": bool(all(a2 + b2 * s > 0 for s in (1, 2, 3, 4))),
               "per_4bit_weight_x_slot": {str(s): math.ceil(4 / s) * (a2 + b2 * s) / (a2 + 4 * b2)
                                          for s in (1, 2, 3, 4)},
               "rpc2_over_rpc1": {
                   "s1 (x6 / x3)": {"measured": p1 / pj["x3_ng45_128x64x1"],
                                    "depth_corrected": pj_dc["x6_ng45_128x64x1_rpc2"] / pj_dc["x3_ng45_128x64x1"],
                                    "logic_only": pj_logic["x6_ng45_128x64x1_rpc2"] / pj_logic["x3_ng45_128x64x1"],
                                    "same_macro_view": True},
                   "s4 (x7 / g6)": {"measured": p4 / pj["g6_ng45_128x64x4"],
                                    "depth_corrected": pj_dc["x7_ng45_128x64x4_rpc2"] / pj_dc["g6_ng45_128x64x4"],
                                    "logic_only": pj_logic["x7_ng45_128x64x4_rpc2"] / pj_logic["g6_ng45_128x64x4"],
                                    "same_macro_view": False},
                   "s4 at 256 rows (g3r2 / g3)": {
                       "measured": pj["g3r2_ng45_256x64x4_rpc2"] / pj["g3_ng45_256x64x4"],
                       "depth_corrected": pj_dc["g3r2_ng45_256x64x4_rpc2"] / pj_dc["g3_ng45_256x64x4"],
                       "logic_only": pj_logic["g3r2_ng45_256x64x4_rpc2"] / pj_logic["g3_ng45_256x64x4"],
                       "same_macro_view": True}},
               "logic_only_ratio_s1_over_s4": {"rpc1 (x3 / g6)": pj_logic["x3_ng45_128x64x1"]
                                               / pj_logic["g6_ng45_128x64x4"],
                                               "rpc2 (x6 / x7)": pj_logic["x6_ng45_128x64x1_rpc2"]
                                               / pj_logic["x7_ng45_128x64x4_rpc2"]}}
        f3["rpc2_cell_bits_two_point_law"] = rp2
    out["F3_energy_per_mac"] = f3

    lk = {n: r["power_leakage_w"] * 1e3 / (r["instance_area_um2"] * 1e-6) for n, r in ng.items()}
    v = np.array(list(lk.values()))
    out["F4_leakage_mw_per_mm2"] = {"per_point": lk, "mean": float(v.mean()), "min": float(v.min()),
                                    "max": float(v.max()), "cv": float(v.std(ddof=1) / v.mean()),
                                    "vs_g3": {n: lk[n] / lk["g3_ng45_256x64x4"] - 1 for n in lk},
                                    "range_vs_g3": [min(lk[n] / lk["g3_ng45_256x64x4"] - 1 for n in lk),
                                                    max(lk[n] / lk["g3_ng45_256x64x4"] - 1 for n in lk)]}
    out["F5_fmax_mhz"] = {n: r["fmax_hz"] / 1e6 for n, r in rows.items()}
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, indent=1)
    show = {k: out[k] for k in ("missing_points", "G_C5_modeled_tiles", "G_C1_row_slot_increment", "per_mac_pj",
                                "F4_leakage_mw_per_mm2")}
    show["heldout_rpc2"] = out.get("heldout_rpc2")
    show["F2"] = {k: f2[k] for k in ("coef", "max_abs_rel_residual", "max_abs_loo_rel_error")}
    show["F3"] = {k: (v if not isinstance(v, dict) or "coef" not in v else
                      {kk: v[kk] for kk in v if kk not in ("loo_rel_errors",)}) for k, v in f3.items()}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()

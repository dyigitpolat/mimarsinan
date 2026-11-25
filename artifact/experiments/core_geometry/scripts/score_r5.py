""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys

RA = "records/research_artifacts"
G = f"{RA}/13_geometry_study"
OUT = os.environ.get("SCORE_OUT", f"{G}/pd_points_r5")
LABEL = ("CONFIRMATORY (pre-registered, EXPLORATORY study): {what} of the confirmatory run (plan_r5.json, "
         "RA/13_geometry_study); not a record")
BLOCK_LABEL = "CONFIRMATORY (pre-registered, EXPLORATORY study): a routed block"
PINS = {f"{G}/plan_r5.json": "4dbdd6bc2185b5065f4f1b232766806be5ac5df0a272aa44d012bcf57e0a82eb",
        f"{G}/fits/heldout_predictions_r4.json": "9e7b5efbc29c22b8f20b8e0880777b27b02cd143d9f7c2ae355fb0181f0f91e3",
        f"{G}/fits/heldout_predictions_r5.json": "d668d149424b79e285fe4d907b208c30d1d76d10d21d19cf734d830a1e867fa3"}
A_BIT = 1.0373001098632812
F_HZ = 5e8
STUDY_CLOCK_MHZ = 91.0
LOW_END_CLOCK_MHZ = 1500.0
TOL = {"area": 0.025, "energy": 0.025, "leakage": 0.20, "fmax_beside": 0.15}
ANCHOR = "h_ng45_384x192x4"
ARMS = ("B5u", "pipe+P+B", "pipe+bufS+P+B")

CAP_MM2 = {"H10.1": 0.5 * 66.528, "R1L": 0.25 * 66.528}
LANE_DR_UM2 = (1.13 - 1.0) * 0.1155 * 1e6
COLUMNS_PER_LANE = 64
CORES_PER_NOC_TILE = 4
BUFFER_UM2 = 32768 * 8.29840087890625
CAP_TOP_UM2 = CAP_MM2["H10.1"] * 1e6
CAP_TOL_UM2 = 1e-3
MAIN_JS = tuple(range(9))
RP_JS = (0, 2, 4, 6, 8)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def pins():
    for p, want in PINS.items():
        if sha(p) != want:
            raise SystemExit(f"REFUSED: INPUT_DIGEST {p}")


def params(config_text):
    for ln in config_text.splitlines():
        if ln.startswith("#"):
            continue
        m = re.search(r"VERILOG_TOP_PARAMS\s*=\s*(.*)", ln)
        if m:
            t = m.group(1).split()
            return {t[i]: int(t[i + 1]) for i in range(0, len(t), 2)}
    raise SystemExit("REFUSED: CONFIG: no VERILOG_TOP_PARAMS")


def power_groups(rpt_text):
    ""
    i = rpt_text.find("report_power")
    if i < 0:
        return None
    out = {}
    for name in ("Sequential", "Combinational", "Clock", "Macro", "Pad", "Total"):
        m = re.search(r"\n%s\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)" % name, rpt_text[i:])
        if m:
            out[name] = [float(m.group(k)) for k in range(1, 5)]
    return out if "Macro" in out and "Total" in out else None


def per_mac_pj(summary, groups, prm):
    ""
    nd = math.ceil((prm["R"] // prm["RPC"]) / prm["MD"])
    idle_w = groups["Macro"][3] * (nd - 1) / nd
    return (summary["power_total_w"] - idle_w) / (F_HZ * prm["C"] * prm["RPC"]) * 1e12, nd, idle_w


def judge_result(outcome, summary, groups):
    ""
    why = []
    if not outcome.get("run_pd_out", "").startswith("rc=0 "):
        why.append(f"run_pd.sh printed {outcome.get('run_pd_out')!r}")
    if outcome.get("killed_by_bound") not in ("0", 0):
        why.append("killed by the chain's bound")
    if summary is None:
        why.append("summary.json does not read")
    elif summary.get("route_drc_errors") != 0:
        why.append(f"route_drc_errors {summary.get('route_drc_errors')}")
    if groups is None:
        why.append("6_finish.rpt holds no report_power table")
    return (not why), why


def read_tile(D):
    out = {}
    try:
        for ln in open(f"{D}/chain_outcome.txt").read().splitlines()[1:]:
            k, _, v = ln.partition(" ")
            out[k] = v
    except OSError:
        pass
    try:
        summary = json.load(open(f"{D}/summary.json"))
    except (OSError, ValueError):
        summary = None
    try:
        groups = power_groups(open(f"{D}/6_finish.rpt").read())
    except OSError:
        groups = None
    prm = params(open(f"{D}/config.mk").read())
    return out, summary, groups, prm


def score(name, D, pred4, pred5, anchor):
    ""
    fallback = name.endswith("_fb")
    gname = name[len("h_ng45_"):].replace("_fb", "")
    pr = pred4["tiles"][f"h_ng45_{gname}"]
    outcome, summary, groups, prm = read_tile(D)
    R, C, s = prm["R"], prm["C"], prm["W"]
    if f"{R}x{C}x{s}" != gname:
        raise SystemExit(f"REFUSED: TILE: {name}'s config is {R}x{C}x{s}")
    ok, why = judge_result(outcome, summary, groups)
    nd = math.ceil((R // prm["RPC"]) / prm["MD"])
    idle_rows = nd * prm["MD"] - R
    rec = {"confirmatory": LABEL.format(what="a routed tile's score"), "tile": name, "geometry": gname,
           "fallback": fallback, "view": f"fakeram45_{prm['MD']}x{prm['MW']}", "MD": prm["MD"], "MW": prm["MW"],
           "ND": nd, "idle_rows": idle_rows, "has_result": ok, "no_result_because": why,
           "chain_outcome": outcome}
    if fallback:
        fb = pred5["fallback_views"].get(gname) or pred5["fallback_views"].get(f"h_ng45_{gname}")
        rec["declared_fallback_view"] = fb
    if not ok:
        rec.update(measured={"area": False, "timing": False, "energy_leakage": False}, meets=None)
        return rec

    a_rt = R * C * s * A_BIT + summary["stdcell_area_um2"]
    a_fit = pr["tile_area_um2_at_declared_density_A_fit"]
    area_err = a_fit / a_rt - 1
    rec["area"] = {"A_rt_um2": a_rt, "A_fit_um2": a_fit, "rel_error_A_fit_over_A_rt": area_err,
                   "holds": abs(area_err) <= TOL["area"], "stdcell_area_um2": summary["stdcell_area_um2"],
                   "logic_um2_F2": pr["logic_um2_F2"],
                   "logic_rel_error_beside": pr["logic_um2_F2"] / summary["stdcell_area_um2"] - 1,
                   "macro_area_um2_routed_beside": summary.get("macro_area_um2"),
                   "instance_area_um2_routed_beside": summary.get("instance_area_um2")}
    fmax_mhz = (summary.get("fmax_hz") or 0.0) / 1e6
    ws = summary.get("setup_ws")
    closes_2ns = ws is not None and ws >= 0
    study = fmax_mhz >= STUDY_CLOCK_MHZ
    rec["timing"] = {"setup_ws_ns_at_2ns": ws, "fmax_mhz": fmax_mhz, "closes_2ns": closes_2ns,
                     "closes_study_clock_91MHz": study, "closes_low_end_1500MHz": fmax_mhz >= LOW_END_CLOCK_MHZ,
                     "holds": closes_2ns and study, "fmax_F5r4_mhz": pr["fmax_mhz_F5r4"],
                     "fmax_rel_error_beside": pr["fmax_mhz_F5r4"] / fmax_mhz - 1 if fmax_mhz else None,
                     "fmax_beside_within_0.15": (abs(pr["fmax_mhz_F5r4"] / fmax_mhz - 1) <= TOL["fmax_beside"])
                     if fmax_mhz else None}
    e_rt, nd_, idle_w = per_mac_pj(summary, groups, prm)
    l_rt = summary["power_leakage_w"] * 1e3
    rec["energy_leakage_raw"] = {"E_rt_pj_per_mac": e_rt, "E_fit_pj_per_mac": pr["per_mac_pj_F3_E_fit"],
                                 "idle_depth_macro_w": idle_w, "macro_group_w": groups["Macro"],
                                 "L_rt_mw": l_rt, "L_fit_mw": pr["leakage_mw_L_fit"],
                                 "power_total_w": summary["power_total_w"]}
    scored_e = idle_rows == 0
    anchor_ok = anchor is not None and anchor.get("has_result") and anchor.get("idle_rows") == 0 \
        and not anchor.get("fallback")
    if name == ANCHOR:
        anchor_ok = True
        anchor = rec
    el = {"scored": scored_e and anchor_ok,
          "why_not": (None if scored_e and anchor_ok else
                      ("idle rows: printed beside only" if not scored_e else "the anchor's primary has no result"))}
    if anchor_ok:
        ar = anchor["energy_leakage_raw"]
        q_e = (e_rt / pr["per_mac_pj_F3_E_fit"]) / (ar["E_rt_pj_per_mac"] / ar["E_fit_pj_per_mac"])
        q_l = (l_rt / pr["leakage_mw_L_fit"]) / (ar["L_rt_mw"] / ar["L_fit_mw"])
        el.update(q_E=q_e, q_L=q_l, energy_holds=abs(q_e - 1) <= TOL["energy"],
                  leakage_holds=abs(q_l - 1) <= TOL["leakage"],
                  absolute_E_rel_error_beside=pr["per_mac_pj_F3_E_fit"] / e_rt - 1,
                  absolute_L_rel_error_beside=pr["leakage_mw_L_fit"] / l_rt - 1)
    rec["energy_leakage"] = el
    rec["measured"] = {"area": True, "timing": True, "study_clock_closed": study,
                       "low_end_closed": fmax_mhz >= LOW_END_CLOCK_MHZ, "energy_leakage": el["scored"]}
    meets = rec["area"]["holds"] and rec["timing"]["holds"]
    if el["scored"]:
        meets = meets and el["energy_holds"] and el["leakage_holds"]
    rec["meets"] = meets
    return rec


def lanes_of(c):
    return max(1, math.ceil(c / COLUMNS_PER_LANE))


def area_um2(n, core, noc, buffered):
    return n * core + math.ceil(n / CORES_PER_NOC_TILE) * noc + (BUFFER_UM2 if buffered else 0.0)


def rung_brute(lim, core, noc):
    n = 0
    while area_um2(n + 1, core, noc, True) <= lim:
        n += 1
    return n


def rung_set(core, noc, js):
    by_j = {j: rung_brute(CAP_TOP_UM2 * 2 ** (-j / 4), core, noc) for j in js}
    counts = []
    for j in js:
        if by_j[j] >= 1 and by_j[j] not in counts:
            counts.append(by_j[j])
    return by_j, counts


def routed_row(row, tile_um2, k0, router_nom):
    ""
    r, c, s = row["R"], row["C_cells"], row["s"]
    lanes = lanes_of(c)
    core = tile_um2 - k0
    noc = row["noc_tile_um2"]
    core_main = tile_um2 + router_nom / CORES_PER_NOC_TILE + lanes * LANE_DR_UM2
    add_rp, add_rp_s = row["A_rp_core_add_um2"], row["A_rp_s_core_add_um2"]
    main_by_j, lad_main = rung_set(core, noc, MAIN_JS)
    rps_by_j, lad_rp_s = rung_set(core + add_rp_s, noc, RP_JS)
    rp_by_j, lad_rp = rung_set(core + add_rp, noc, RP_JS)
    union = sorted(set(lad_main) | set(lad_rp_s) | set(lad_rp), reverse=True)
    planes = {}
    for nm, cr, lad in (("main", core, lad_main), ("A_rp_s", core + add_rp_s, lad_rp_s), ("A_rp", core + add_rp, lad_rp)):
        planes[nm] = {}
        for kind, buffered in (("unbuffered", False), ("buffered", True)):
            planes[nm][kind] = {}
            for cap_name, cap_mm2 in (("33.264", CAP_MM2["H10.1"]), ("16.632", CAP_MM2["R1L"])):
                planes[nm][kind][cap_name] = len([n for n in lad if area_um2(n, cr, noc, buffered)
                                                  <= cap_mm2 * 1e6 + CAP_TOL_UM2])
    out = dict(row)
    out.update({"modeled_tile_um2": tile_um2, "modeled_core_design_um2": core_main, "core_per_core_um2": core,
                "A_rp_over_main_core": (core_main + add_rp) / core_main,
                "A_rp_s_over_main_core": (core_main + add_rp_s) / core_main,
                "ladder_main_by_j": [main_by_j[j] for j in MAIN_JS], "ladder_main": lad_main,
                "ladder_A_rp_s_by_j": {str(j): rps_by_j[j] for j in RP_JS}, "ladder_A_rp_s": lad_rp_s,
                "ladder_A_rp_by_j": {str(j): rp_by_j[j] for j in RP_JS}, "ladder_A_rp": lad_rp,
                "ladder_counts": union, "planes_rungs_inside_cap": planes})
    return out


def routed_classes(classes, stdcell_um2, k0, q_e, q_l, union):
    out = []
    for c in classes:
        k = json.loads(json.dumps(c))
        cs = k["constants"]
        f = c["idle_column_factor"] or 1.0
        cs["area_per_row_driver_um2"] = 0.0
        cs["area_per_neuron_logic_um2"] = (stdcell_um2 - k0) / c["C_cells"] * f
        if q_e is not None:
            for key in ("e_mac_pj_per_weight_mac", "e_mac_pj_operand_bits_rule", "e_mac_pj_rpc2_two_point_law"):
                cs[key] = cs[key] * q_e
            cs["p_static_per_core_uw"] = [v * q_l for v in cs["p_static_per_core_uw"]]
        k["ladder_counts"] = union
        out.append(k)
    return out


def write_block(plan, sc, out_dir):
    g = sc["geometry"]
    R, C, s = (int(x) for x in g.split("x"))
    row = next(x for x in plan["grid"]["geometry_table"] if (x["R"], x["C_cells"], x["s"]) == (R, C, s))
    classes = [x for x in plan["grid"]["classes"] if (x["R"], x["C_cells"], x["s"]) == (R, C, s)]
    k0 = plan["fits_summary"]["F2"]["k0_um2"]
    routers = {tuple(x["constants"]["area_per_router_um2"]) for x in plan["grid"]["classes"]}
    router_nom = list(routers)[0][1]
    stdcell = sc["area"]["stdcell_area_um2"]
    tile = R * C * s * A_BIT + stdcell
    el = sc["energy_leakage"]
    q_e = el.get("q_E") if el.get("scored") else None
    q_l = el.get("q_L") if el.get("scored") else None
    grow = routed_row(row, tile, k0, router_nom)
    block = {"confirmatory": BLOCK_LABEL + " of the confirmatory run (plan_r5.json section 11.7, RD-6), not a record",
             "R": R, "C_cells": C, "s": s, "tile": sc["tile"], "score_sha256": sc["_score_sha256"],
             "rule": "area_per_row_driver 0; area_per_neuron_logic (routed stdcell - k0) / C x the idle-column factor; "
                     "e_mac forms x q_E and p_static_per_core x q_L only where the geometry is measured for energy and "
                     "leakage; every other constant unchanged; the geometry row rebuilt on the routed core area "
                     "(R C s x 1.0373 + routed stdcell - k0), its three rung sets, planes and union ladder",
             "routed_stdcell_um2": stdcell, "routed_tile_um2_at_declared_density": tile, "k0_um2": k0,
             "q_E": q_e, "q_L": q_l, "geometry_row": grow,
             "classes": routed_classes(classes, stdcell, k0, q_e, q_l, grow["ladder_counts"]),
             "plan_r5_sha256": PINS[f"{G}/plan_r5.json"]}
    os.makedirs(f"{out_dir}/blocks", exist_ok=True)
    path = f"{out_dir}/blocks/{g}.json"
    with open(path, "x") as fh:
        json.dump(block, fh, indent=1)
    dg = sha(path)
    with open(f"{out_dir}/blocks/{g}.sha256", "x") as fh:
        fh.write(f"{dg}  {g}.json\n")
    lines = []
    for c in classes:
        if c["pruned"]:
            continue
        k = c["cells_per_weight"]
        for arch in c["architectures"]:
            for arm in ARMS:
                lines.append(f"{R},{C},{s},{k} {arch} {arm} {arch}__{R}x{C}x{s}k{k}__{arm} {path} {dg}")
    with open(f"{out_dir}/blocks/{g}.queue.txt", "x") as fh:
        fh.write(LABEL.format(what=f"the routed part queue of {g}") + "\n" + "\n".join(lines) + "\n")
    return path, dg, len(lines)


def main_score(name):
    pins()
    plan = json.load(open(f"{G}/plan_r5.json"))
    pred4 = json.load(open(f"{G}/fits/heldout_predictions_r4.json"))
    pred5 = json.load(open(f"{G}/fits/heldout_predictions_r5.json"))
    D = f"{OUT}/{name}"
    anchor = None
    if name != ANCHOR:
        try:
            anchor = json.load(open(f"{OUT}/{ANCHOR}/score.json"))
        except (OSError, ValueError):
            anchor = None
    sc = score(name, D, pred4, pred5, anchor)
    path = f"{D}/score.json"
    with open(path, "x") as fh:
        json.dump(sc, fh, indent=1)
    sc["_score_sha256"] = sha(path)
    msg = (f"{name}: result {sc['has_result']}" + (f" ({'; '.join(sc['no_result_because'])})" if not sc["has_result"]
                                                   else f", meets {sc['meets']}"))
    if sc["has_result"] and not sc["meets"]:
        g = sc["geometry"]
        if os.path.exists(f"{OUT}/blocks/{g}.json"):
            msg += f"; a block of {g} exists already (written once)"
        else:
            bp, dg, n = write_block(plan, sc, OUT)
            msg += f"; routed block {bp} {dg} with {n} routed parts"
    print(msg)
    return 0


def selftest():
    pins()
    plan = json.load(open(f"{G}/plan_r5.json"))
    fails = []

    def check(name, ok, detail=""):
        if not ok:
            fails.append(f"{name}: {detail}")


    D = f"{G}/pd_points/x1_ng45_512x64x4"
    fits = json.load(open(f"{G}/fits/fits_r2.json"))
    summ = json.load(open(f"{D}/summary.json"))
    groups = power_groups(open(f"{D}/6_finish.rpt").read())
    prm = params(open(f"{D}/config.mk").read())
    e, nd, _ = per_mac_pj(summ, groups, prm)
    want = fits["per_mac_pj"]["idle_depth_macros_removed"]["x1_ng45_512x64x4"]
    check("E_rt_x1", abs(e - want) <= 1e-12 * want and nd == 4, (e, want, nd))
    check("macro_row_x1", groups["Macro"][3] == fits["points"]["x1_ng45_512x64x4"]["macro_w"], groups["Macro"])

    lab = "# CONFIRMATORY label\n" + open(f"{D}/6_finish.rpt").read()
    check("labelled_rpt", power_groups(lab) == groups)

    check("result_ok", judge_result({"run_pd_out": "rc=0 wall_s=10", "killed_by_bound": "0"}, summ, groups)[0])
    check("result_drc", not judge_result({"run_pd_out": "rc=0 wall_s=10", "killed_by_bound": "0"},
                                         dict(summ, route_drc_errors=3), groups)[0])
    check("result_rc", not judge_result({"run_pd_out": "rc=124 wall_s=3600", "killed_by_bound": "0"}, summ, groups)[0])
    check("result_killed", not judge_result({"run_pd_out": "rc=0 wall_s=10", "killed_by_bound": "1"}, summ, groups)[0])
    check("result_norpt", not judge_result({"run_pd_out": "rc=0 wall_s=10", "killed_by_bound": "0"}, summ, None)[0])

    k0 = plan["fits_summary"]["F2"]["k0_um2"]
    router_nom = list({tuple(x["constants"]["area_per_router_um2"]) for x in plan["grid"]["classes"]})[0][1]
    n_rows = 0
    for row in plan["grid"]["geometry_table"]:
        rb = routed_row(row, row["modeled_tile_um2"], k0, router_nom)
        for key, v in row.items():
            w = rb[key]
            ok = (abs(v - w) <= 1e-9 * max(1.0, abs(v))) if isinstance(v, float) else (v == w)
            check(f"row_{row['R']}x{row['C_cells']}x{row['s']}.{key}", ok, (v, w))
        n_rows += 1

    n_cls = 0
    for c in plan["grid"]["classes"]:
        row = next(x for x in plan["grid"]["geometry_table"]
                   if (x["R"], x["C_cells"], x["s"]) == (c["R"], c["C_cells"], c["s"]))
        logic = row["modeled_tile_um2"] - c["R"] * c["C_cells"] * c["s"] * A_BIT
        rc = routed_classes([c], logic, k0, None, None, row["ladder_counts"])[0]["constants"]
        f = c["idle_column_factor"] or 1.0
        core_cls = (c["R"] * c["C_cells"] * rc["area_per_cell_um2"][1] / f + c["R"] * rc["area_per_row_driver_um2"]
                    + c["C_cells"] * rc["area_per_neuron_logic_um2"] / f)
        check(f"class_core_{c['R']}x{c['C_cells']}x{c['s']}k{c['cells_per_weight']}",
              abs(core_cls - row["core_per_core_um2"]) <= 1e-9 * row["core_per_core_um2"], (core_cls, row["core_per_core_um2"]))
        orig = c["constants"]
        core_orig = (c["R"] * c["C_cells"] * orig["area_per_cell_um2"][1] / f + c["R"] * orig["area_per_row_driver_um2"]
                     + c["C_cells"] * orig["area_per_neuron_logic_um2"] / f)
        check(f"class_core_orig_{c['R']}x{c['C_cells']}x{c['s']}k{c['cells_per_weight']}",
              abs(core_orig - row["core_per_core_um2"]) <= 1e-9 * row["core_per_core_um2"], (core_orig, row["core_per_core_um2"]))
        n_cls += 1

    pred4 = json.load(open(f"{G}/fits/heldout_predictions_r4.json"))
    check("anchor_in_predictions", ANCHOR in pred4["tiles"] and pred4["tiles"][ANCHOR]["idle_rows"] == 0)
    res = {"confirmatory": LABEL.format(what="score_r5.py's selftest"), "ok": not fails, "failures": fails,
           "rows_rebuilt": n_rows, "classes_rebuilt": n_cls, "E_rt_x1": e}
    print(json.dumps(res, indent=1))
    return 0 if not fails else 3


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        raise SystemExit(main_score(sys.argv[2]))
    if sys.argv[1:2] == ["selftest"]:
        raise SystemExit(selftest())
    raise SystemExit(__doc__)

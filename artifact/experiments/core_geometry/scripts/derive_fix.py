""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

LABEL = "CONFIRMATORY (pre-registered, EXPLORATORY study)"
PIN_SUMS = "a4c9296b03c87b8fcfd86c4ab6548fba38a3c527d4a91bf1c3be236c8cbf12c3"
PIN_PLAN = "4dbdd6bc2185b5065f4f1b232766806be5ac5df0a272aa44d012bcf57e0a82eb"
PIN_ANALYSIS = "7da54e224bb8fc4182bfec614fd563e197611dc116b5998db2215d073ae0b6cd"
PIN_SWAP = "019964d23239758668e8bf3892346eebc7a82677a091991a6a4a10dc3b3ba5c3"
PIN_CSV = "3a8cdfd5a5ecc8379fa8d2559ef3cf13877447f02d5d55b3d456e4c5dd966008"
FLOOR, FIT, Q, NDRAWS = 0.05, 0.025, 0.10, 4000
VARIANT_NAME = {0: "main edge-host variant", 1: "low edge-host variant", 2: "high edge-host variant"}
END_NAME = {0: "low chip end (1.5 GHz)", 1: "nominal chip end", 2: "high chip end"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def refuse(msg: str):
    raise SystemExit(f"REFUSED: {msg}")


def geom(design: str) -> str:
    return design.split("@")[1].split("/")[0]


def bh_step_up(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        adj[i] = min(run, 1.0)
    return adj


def reading_name(r: int) -> str:
    v, e = divmod(r, 3)
    return f"r {r}: {VARIANT_NAME[v]}, {END_NAME[e]}"


def pct(x, nd=2):
    return None if x is None else round(100 * x, nd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ca", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ca, out = Path(a.ca), Path(a.out)
    if out.exists():
        refuse(f"OUT_EXISTS: {out}")

    if sha(ca / "SHA256SUMS") != PIN_SUMS:
        refuse("SUMS_DIGEST")
    sums = {}
    for line in (ca / "SHA256SUMS").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        h, p = line.split(None, 1)
        sums[p.strip()] = h
    for name, pin in (("analysis.json", PIN_ANALYSIS), ("beside_reference_swap.json", PIN_SWAP),
                      ("tests.csv", PIN_CSV)):
        h = sha(ca / name)
        if h != pin or sums.get(name) != pin:
            refuse(f"INPUT_DIGEST: {name}")
    if sha(Path(a.plan)) != PIN_PLAN:
        refuse("INPUT_DIGEST: plan_r5.json")
    A = json.loads((ca / "analysis.json").read_text())
    SW = json.loads((ca / "beside_reference_swap.json").read_text())
    P = json.loads(Path(a.plan).read_text())
    if not A["confirmatory"].startswith(LABEL) or not SW["confirmatory"].startswith(LABEL):
        refuse("LABEL")
    if A["input_digests"].get(str(Path(a.plan).resolve())) not in (None, PIN_PLAN):
        refuse("PLAN_PIN_IN_ANALYSIS")
    tests = {t["id"]: t for t in A["tests"]}
    if len(tests) != 36:
        refuse("NOT_36")

    rows = list(csv.DictReader(io.StringIO("\n".join(
        ln for ln in (ca / "tests.csv").read_text().splitlines() if not ln.startswith("#")))))
    csv_diff = 0
    for r in rows:
        t = tests[r["id"]]
        for k in ("T", "T_closed_clock", "T_two_sided", "nominal_ratio_S_over_J", "saving_on_separate_base", "p",
                  "bh_pass2"):
            x, y = r[k], t[k]
            if (x in ("", "None")) != (y is None):
                csv_diff += 1
            elif y is not None and abs(float(x) - y) > 1e-12:
                csv_diff += 1
        if r["decision"] != t["decision"]:
            csv_diff += 1

    hog = P["declared"]["held_out_grid"]
    n_grid = len(hog["rows"]) * len(hog["cell_columns"]) * len(hog["cell_bits"])
    two_sided_width = (1 + FIT) / (1 - FIT) - 1

    ruled = {}
    for i, t in tests.items():
        if t["decision"] == "CONFIRMED":
            ruled[i] = "CONFIRMED"
        elif t["decision"] == "UNCONFIRMABLE":
            ruled[i] = "UNCONFIRMABLE"
        elif not t["determinable"]:
            ruled[i] = "NOT DETERMINABLE"
        else:
            if t["beyond"] or (t["T"] is not None and t["T"] > 1e-9):
                refuse(f"NOT_CONFIRMED_BEYOND: {i}")
            ruled[i] = "NOT CONFIRMED (not beyond the noise)"
    counts = {}
    for i, lab in ruled.items():
        counts.setdefault(lab, []).append(i)

    p_obs = 1 / (NDRAWS + 1)
    bh1 = bh_step_up([tests[i]["p"] for i in sorted(tests)])
    bh2 = bh_step_up([tests[i]["p_pass2"] for i in sorted(tests)])
    bh_diff = sum(abs(x - tests[i]["bh_pass1"]) > 1e-12 for x, i in zip(bh1, sorted(tests))) + \
        sum(abs(x - tests[i]["bh_pass2"]) > 1e-12 for x, i in zip(bh2, sorted(tests)))
    conf = sorted(i for i, lab in ruled.items() if lab == "CONFIRMED")
    for i in conf:
        t = tests[i]
        if abs(t["p_pass2"] - p_obs) > 1e-15 or not t["T"] >= FLOOR:
            refuse(f"CONFIRMED_FORM: {i}")

    six = {}
    for i in conf:
        t = tests[i]
        fr = t["four_readings"]["readings"]
        sw = SW["tests"][i]
        main_T, trs_T, eops_T = fr["main"]["T"], fr["T_rs"]["T"], fr["E_ops"]["T"]
        arps2_T = fr["A_rp_s"]["T"]
        arps_decl_det = sw["A_rp_s | R18 w1 b4444"]["determinable"]
        if abs(main_T - t["T"]) > 1e-12:
            refuse(f"MAIN_NE_T: {i}")
        if abs(sw["A_rp_s | R18 w1/2 b4444"]["T"] - arps2_T) > 1e-12 or abs(sw["T_rs | R18 w1 b4444"]["T"] - trs_T) > 1e-12:
            refuse(f"SWAP_NE_FOUR: {i}")
        four = {"main": main_T, "A_rp_s": arps2_T, "T_rs": trs_T, "E_ops": eops_T}
        smallest = min(four.values())
        set_by = min(four, key=lambda k: four[k])
        cell_bits = int(geom(t["set_at"]["J"]).split("x")[2])
        saving = 1 - 1 / (1 + t["T"])
        six[i] = {
            "setting": t["setting"], "cell": t["cell"], "cell_bits_of_J": cell_bits,
            "target_member": t["target_member"], "target_images": t["target_images"],
            "T_pct": pct(t["T"]), "T_closed_clock_pct": pct(t["T_closed_clock"]),
            "T_two_sided_pct": pct(t["T_two_sided"]), "T_rs_pct": pct(trs_T), "E_ops_pct": pct(eops_T),
            "A_rp_s_second_ref_pct": pct(arps2_T), "A_rp_s_declared_ref_determinable": arps_decl_det,
            "T_rs_two_sided_pct": pct(fr["T_rs"].get("T_two_sided")),
            "smallest_over_four_pct": pct(smallest), "smallest_set_by": set_by,
            "by_floor_under_main_T_rs_E_ops": min(main_T, trs_T, eops_T) >= FLOOR,
            "by_floor_under_main_T_rs_two_sided": min(main_T, trs_T, t["T_two_sided"]) >= FLOOR,
            "holds_under_all_four_by_floor": smallest >= FLOOR and arps_decl_det,
            "nominal_ratio": round(t["nominal_ratio_S_over_J"], 3),
            "saving_on_separate_base_pct": pct(saving),
            "saving_recomputed_equals_stored": abs(saving - t["saving_on_separate_base"]) < 1e-12,
            "set_at": reading_name(t["set_at"]["r"]),
            "J": t["set_at"]["J"], "S": t["set_at"]["S"], "band_at_set_reading": sorted(t["band_at_set_reading"]),
            "T_rs_S": fr["T_rs"]["set_at"]["S"], "T_rs_band": sorted(fr["T_rs"]["band_at_set_reading"]),
            "cost_geometries_all_routed_and_meet": all(
                st["has_result"] and st["meets"] and st["measured"]["study_clock_closed"]
                for st in t["cost_geometry_states"].values()),
            "p": t["p_pass2"], "bh_pass2": t["bh_pass2"],
        }
    robust = sorted(i for i, s in six.items() if s["by_floor_under_main_T_rs_two_sided"]
                    and s["by_floor_under_main_T_rs_E_ops"])
    rest = sorted(set(six) - set(robust))

    swap = {}
    for i in conf:
        sw = SW["tests"][i]
        jg = geom(tests[i]["set_at"]["J"])
        rec = {}
        for key in ("main | R18 w1/2 b4444", "A_rp_s | R18 w1/2 b4444", "T_rs | R18 w1/2 b4444",
                    "E_ops | R18 w1/2 b4444"):
            o = sw[key]
            rec[key] = {"T_pct": pct(o["T"]), "band": sorted(o["band_at_set_reading"] or []),
                        "J_geometry_in_band": jg in (o["band_at_set_reading"] or []),
                        "S_geometry": o["set_at"]["S_geometry"] if o["set_at"] else None}
        swap[i] = rec

    energy = {}
    s4_band_union = set()
    for i, t in tests.items():
        if t["metric"] != "energy":
            continue
        sa = t["set_at"]
        band = sorted(t["band_at_set_reading"])
        energy[i] = {"setting": t["setting"], "cell": t["cell"], "target_images": t["target_images"],
                     "T_pct": pct(t["T"]), "T_own_leakage_pct": pct(t["T_own_leakage"]),
                     "T_declared_G_ref_pct": pct(t["T_declared_G_ref"]), "set_at": reading_name(sa["r"]),
                     "J": sa["J"], "S": sa["S"], "one_design": sa["J"] == sa["S"],
                     "same_geometry": geom(sa["J"]) == sa["S_geometry"],
                     "J_geometry_in_band": geom(sa["J"]) in band, "band_size_at_set_reading": len(band),
                     "band_sizes": t["band_sizes"], "main_nominal_r1": {"J": t["J_main_nominal"],
                                                                       "S": t["S_main_nominal"]},
                     "ceiling_pct": pct(t["ceiling"])}
        for g in band:
            if g.endswith("x4"):
                s4_band_union.add(g)

    priced, refused = set(), {}
    for line in Path(a.census).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        n, R, C, s, outcome, rname = line.split()
        g = f"{R}x{C}x{s}"
        if outcome == "priced":
            priced.add(g)
        else:
            refused.setdefault(g, set()).add(rname)
    ref_s4 = sorted(g for g in priced if g.endswith("x4"))
    ref_s2 = sorted(g for g in priced if g.endswith("x2"))
    ref_s2_sub384_refused = sorted(g for g in refused if g.endswith("x2") and int(g.split("x")[0]) < 384)
    ref_s2_sub384_priced = sorted(g for g in priced if g.endswith("x2") and int(g.split("x")[0]) < 384)
    ref_s13 = sorted(g for g in priced if g.endswith("x3") or g.endswith("x1"))

    area = {i: {"T_pct": pct(t["T"]), "J": t["set_at"]["J"] if t["set_at"] else None,
                "one_design": bool(t["set_at"]) and t["set_at"]["J"] == t["set_at"]["S"]}
            for i, t in tests.items() if t["metric"] == "area" and t["determinable"]}
    nd = {i: t["reason_not_determinable"] for i, t in tests.items() if not t["determinable"]}
    nb_time = {i: {"T_pct": pct(t["T"]), "J": t["set_at"]["J"], "one_design": t["set_at"]["J"] == t["set_at"]["S"],
                   "T_declared_G_ref_pct": pct(t["T_declared_G_ref"])}
               for i, t in tests.items() if ruled[i].startswith("NOT CONFIRMED") and t["metric"] == "time"}

    rt = A["routed_tiles"]
    with_result = [r for r in rt if r["has_result"]]
    geoms_with_result = sorted({r["geometry"] for r in with_result})
    tiles = {"attempts": len(rt), "with_result": len(with_result), "geometries_with_result": len(geoms_with_result),
             "close_91MHz": sum(bool(r["closes_91MHz"]) for r in with_result),
             "close_2ns": sum(bool(r["closes_2ns"]) for r in with_result),
             "close_1p5GHz": sum(bool(r["closes_1p5GHz"]) for r in with_result),
             "fmax_mhz_range": [min(r["fmax_mhz"] for r in with_result), max(r["fmax_mhz"] for r in with_result)],
             "meet": sum(bool(r["meets"]) for r in with_result),
             "no_result": sorted(r["tile"] for r in rt if not r["has_result"])}

    def rng(key, ids):
        v = [six[i][key] for i in ids]
        return [min(v), max(v)]
    derived = {
        "confirmatory": f"{LABEL}: the fix pass's re-derived numbers (G/confirm_analysis_fix.md); not a record",
        "inputs": {"SHA256SUMS": PIN_SUMS, "analysis.json": PIN_ANALYSIS, "beside_reference_swap.json": PIN_SWAP,
                   "tests.csv": PIN_CSV, "plan_r5.json": PIN_PLAN,
                   "census": sha(Path(a.census))},
        "tests_csv_vs_analysis_differences": csv_diff,
        "bh_recomputed_differences": bh_diff,
        "p_observed": p_obs, "bh_six": bh2[sorted(tests).index(conf[0])],
        "held_out_geometries": n_grid, "band_one_sided_pct": 100 * FIT,
        "band_two_sided_pct": round(100 * two_sided_width, 2),
        "labels_as_ruled": {k: sorted(v) for k, v in counts.items()},
        "label_counts": {k: len(v) for k, v in counts.items()},
        "confirmed_target_members": sorted({six[i]["target_member"] for i in six}),
        "confirmed_target_images": sorted({six[i]["target_images"] for i in six}),
        "target_share_pct": round(100 * 2957 / 5000, 1),
        "six": six, "robust_core": robust, "four_bit_rest": rest,
        "ranges": {"T_pct": rng("T_pct", conf), "T_closed_clock_pct": rng("T_closed_clock_pct", conf),
                   "saving_pct": rng("saving_on_separate_base_pct", conf), "ratio": rng("nominal_ratio", conf),
                   "rest_T_rs_pct": rng("T_rs_pct", rest), "rest_two_sided_pct": rng("T_two_sided_pct", rest),
                   "rest_T_rs_or_two_sided_pct": [min(rng("T_rs_pct", rest)[0], rng("T_two_sided_pct", rest)[0]),
                                                  max(rng("T_rs_pct", rest)[1], rng("T_two_sided_pct", rest)[1])],
                   "robust_main_pct": [six[i]["T_pct"] for i in robust],
                   "robust_T_rs_pct": [six[i]["T_rs_pct"] for i in robust],
                   "robust_two_sided_pct": [six[i]["T_two_sided_pct"] for i in robust],
                   "robust_E_ops_pct": [six[i]["E_ops_pct"] for i in robust],
                   "smallest_over_four_pct": sorted({six[i]["smallest_over_four_pct"] for i in conf})},
        "none_holds_under_all_four": not any(six[i]["holds_under_all_four_by_floor"] for i in conf)
        and all(six[i]["A_rp_s_second_ref_pct"] == 0 and not six[i]["A_rp_s_declared_ref_determinable"] for i in conf),
        "swap": swap, "energy": energy, "energy_s4_band_union": sorted(s4_band_union),
        "census": {"priced_s4": ref_s4, "priced_s2": ref_s2, "priced_s3_or_s1": ref_s13,
                   "s2_below_384_rows_refused": ref_s2_sub384_refused,
                   "s2_below_384_rows_priced": ref_s2_sub384_priced,
                   "refusal_names": sorted({n for v in refused.values() for n in v})},
        "area": area, "not_determinable": nd, "not_beyond_time": nb_time, "routed_tiles": tiles,
        "chance_count": A["chance_count_O_1"],
    }
    out.write_text(json.dumps(derived, indent=1) + "\n")
    print(json.dumps({"csv_diff": csv_diff, "bh_diff": bh_diff, "labels": derived["label_counts"], "robust": robust,
                      "rest": rest, "ranges": derived["ranges"], "none_all_four": derived["none_holds_under_all_four"],
                      "grid": n_grid, "tiles": tiles, "census": derived["census"],
                      "s4_band_union": derived["energy_s4_band_union"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

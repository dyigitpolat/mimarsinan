""
import collections
import json
import os
import sys

OUT = sys.argv[1]
INPUTS, PU, PR, WU, WR, V5B, CA = (json.load(open(p)) for p in sys.argv[2:9])
REL = 1e-12


def close(a, b, rel=REL):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def flag_value(row, kind):
    for f in row.get("flags") or ():
        if f["kind"] == kind:
            return f["value"]
    return None


def price_of(row):
    return {"nominal": row.get("nominal"), "low": row.get("low"), "high": row.get("high"),
            "refusal": row.get("refusal"),
            "flags": [{"kind": f["kind"], "value": f["value"]} for f in row.get("flags") or ()],
            "readings": {k: flag_value(row, k) for k in ("t1_reading", "fixed_outside_reading")
                         if flag_value(row, k) is not None},
            "stand_in_constants": row.get("stand_in_constants") or []}


def record_rows(name):
    unit, r5 = PU[name], PR[name]
    assert [r["row"] for r in unit] == [r["row"] for r in r5]
    srcs = {i["path"]: i for i in INPUTS[name]}
    rows = []
    for u, r in zip(unit, r5):
        src = srcs[u["source"]]
        row = {"row": u["row"], "status": u["status"], "description": u["description"],
               "source": u["source"], "source_sha256": u["source_sha256"],
               "pointers": u["pointers"], "read": u["read"], "core_types": u["core_types"], "census": u["census"],
               "census_withheld": u["census_withheld"], "readout_bits": u["readout_bits"], "geometry": u["geometry"],
               "unit": price_of(u), "r5": price_of(r),
               "unit_over_r5": None if not (u.get("nominal") and r.get("nominal")) else u["nominal"] / r["nominal"],
               "per_core_unit_mm2": None if u.get("nominal") is None else u["nominal"] / u["census"]["cores_physical"]}
        if "tr_path" in src:
            row["tr_twin"] = {"path": src["tr_path"], "sha256": src["tr_sha256"], "equal": src["tr_equal"]}
        if "per_type_reading" in u:
            row["per_type_reading"] = u["per_type_reading"]
        rows.append(row)
    checks = {"rows": len(rows), "unit_priced": sum(r["unit"]["nominal"] is not None for r in rows),
              "r5_priced": sum(r["r5"]["nominal"] is not None for r in rows),
              "descriptions": dict(collections.Counter(r["description"] for r in rows)),
              "weight_bits_withheld": [r["row"] for r in rows if r["census_withheld"]],
              "twins_equal": sum(1 for r in rows if (r.get("tr_twin") or {}).get("equal")) if name == "routeb" else None}
    return rows, checks


def exploration():
    unit, r5 = PU["explore"], PR["explore"]
    rows = []
    for u, r in zip(unit, r5):
        assert all(u[k] == r[k] for k in ("target", "network", "budget", "cores", "weight_bits", "points"))
        rows.append({k: u[k] for k in ("target", "network", "budget", "cores", "passes", "weight_bits", "softcores",
                                       "points")} | {"census": u["census"], "core": u["core"],
                                                     "unit": price_of(u), "r5": price_of(r)})
    pts = lambda pred: sum(x["points"] for x in rows if pred(x))
    kinds = lambda x: {f["kind"] for f in x["unit"]["flags"]}
    counts = {"configurations": len(rows), "points": pts(lambda x: True),
              "unit_refused_points": pts(lambda x: x["unit"]["nominal"] is None),
              "r5_refused_points": pts(lambda x: x["r5"]["nominal"] is None),
              "r5_refused_configurations": sum(1 for x in rows if x["r5"]["nominal"] is None),
              "beyond_published_points": pts(lambda x: "beyond_published" in kinds(x)),
              "below_published_points": pts(lambda x: "below_published" in kinds(x)),
              "anchor_identity_points": pts(lambda x: "anchor_identity" in kinds(x)),
              "off_word_width_points": pts(lambda x: "off_word_width" in kinds(x)),
              "weight_bits_not_stated_points": pts(lambda x: "weight_bits_not_stated" in kinds(x)),
              "complete_terms_points": pts(lambda x: "complete_terms" in kinds(x)),
              "per_target_points": dict(collections.Counter({x["target"]: 0 for x in rows}))}
    per = collections.Counter()
    for x in rows:
        per[x["target"]] += x["points"]
    counts["per_target_points"] = dict(sorted(per.items()))

    v5 = collections.defaultdict(list)
    for x in V5B["regime"]:
        v5[(x["target"], x["cores"], x["weight_bits"])].append(x)
    agree = differ = 0
    worst = 0.0
    for x in rows:
        cands = v5.get((x["target"], x["cores"], x["weight_bits"]))
        u = x["unit"]
        ok = False
        for c in cands or ():
            if "nominal" in c and all(close(u[k], c[k], 1e-12) for k in ("nominal", "low", "high")):
                ok = True
                break
            if "nominal" in c:
                worst = max(worst, max(abs(u[k] - c[k]) / max(1.0, abs(c[k])) for k in ("nominal", "low", "high")))
        agree += ok
        differ += not ok

    def key(r):
        try:
            budget = str(float(r["budget"]))
        except (TypeError, ValueError):
            budget = str(r["budget"])
        return (r["target"], r["network"], budget, r["cores"], r["passes"], r["weight_bits"],
                r["softcores"])
    present = {key(r): r for r in CA["regime"]["rows"]}
    r5_agree = r5_refused_agree = r5_differ = 0
    for x in rows:
        p = present[key(x)]
        if p["present_mm2"] is None:
            r5_refused_agree += x["r5"]["nominal"] is None
            r5_differ += x["r5"]["nominal"] is not None
        else:
            ok = x["r5"]["nominal"] is not None and close(x["r5"]["nominal"], p["present_mm2"], 1e-9)
            r5_agree += ok
            r5_differ += not ok
    complete = [x for x in rows if "complete_terms" in kinds(x)]
    r1_equal = sum(1 for x in complete if all(close(x["unit"][k], x["r5"][k], 0.0) for k in ("nominal", "low",
                                                                                               "high")))
    checks = {"counts": counts,
              "against_build_check_v5_built_b6": {"rows_equal_1e-12": agree, "rows_differ": differ,
                                                  "largest_relative_difference": worst},
              "r5_against_config_area_present": {"priced_equal_1e-9": r5_agree, "refused_both": r5_refused_agree,
                                                 "differ": r5_differ},
              "complete_terms_r1": {"configurations": len(complete), "points": sum(x["points"] for x in complete),
                                    "unit_equal_r5_bitwise": r1_equal}}
    return rows, checks


def w48_rr():
    u, r = WU, WR
    assert len(u["readings"]) == len(r["readings"])
    rows = []
    for a, b in zip(u["readings"], r["readings"]):
        assert all(a[k] == b[k] for k in ("cell", "i", "side", "variant", "K", "row"))
        rows.append({k: a[k] for k in ("cell", "i", "side", "variant", "K", "arm", "member", "row", "chip",
                                       "study_chip", "recorded")} |
                    {"unit": a["repriced"], "unit_equal_record": a["equal"], "unit_refusal": a["refusal"],
                     "r5": b["repriced"], "r5_equal_record": b["equal"],
                     "tile_recorded": a["tile_recorded"], "tile_unit": a["tile_repriced"],
                     "tile_unit_equal": a["tile_equal"], "tile_r5_equal": b["tile_equal"],
                     "unit_over_record": None if not (a["repriced"] and a["recorded"]) else
                     a["repriced"][1] / a["recorded"][1]})
    study = [x for x in rows if x["study_chip"]]
    generic = [x for x in rows if not x["study_chip"]]
    picks = []
    for a, b in zip(u["rr_picks"], r["rr_picks"]):
        picks.append(a | {"r5_repriced": b.get("repriced"), "r5_equal": b.get("equal")})
    checks = {"readings": len(rows), "study_chip_readings": len(study),
              "study_unit_equal_record": sum(x["unit_equal_record"] for x in study),
              "study_r5_equal_record": sum(x["r5_equal_record"] for x in study),
              "study_tile_unit_equal": sum(1 for x in study if x["tile_unit_equal"]),
              "study_tile_checked": sum(1 for x in study if x["tile_unit_equal"] is not None),
              "generic_readings": len(generic),
              "generic_r5_equal_record": sum(x["r5_equal_record"] for x in generic),
              "generic_unit_equal_record": sum(x["unit_equal_record"] for x in generic),
              "generic_unit_over_record": sorted({round(x["unit_over_record"], 6) for x in generic
                                                  if x["unit_over_record"]}),
              "planes_in_record": u["planes"], "rows_digest_ok": [u["summary"]["rows_digest_ok"],
                                                                  r["summary"]["rows_digest_ok"]],
              "unique_pricings": u["summary"]["unique_pricings"],
              "rr_picks": len(picks), "rr_repriced": sum(1 for p in picks if p.get("repriced") is not None),
              "rr_unit_equal": sum(1 for p in picks if p.get("equal")),
              "rr_r5_equal": sum(1 for p in picks if p.get("r5_equal")),
              "w48_sources": {k: u[k] for k in ("w48_pairs_json_sha256", "w48_pairs_py_sha256", "record_rows_sha256",
                                                "rr_classes_sha256")}}
    return {"readings": rows, "rr_picks": picks}, checks


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {"pass": "U-AREA W9, the pricing-only pass: each record's census by its JSON pointers, priced by the "
                    "unit's description by name (UB6 uarea.patch 2923eaa5, kept tree applied digest 91b15573), "
                    "never the sealed physics; R5's price beside (my byte copy of R5 e5f6bb37); lanes out",
            "unit_tree_digest": "91b15573d97a091f3ba48cacb8e72404f42b59532776be5cacb14454cd9761e1"}
    summary = {}
    for name, fname in (("routeb", "routeb_rows.json"), ("ladder", "ladder_rows.json")):
        rows, checks = record_rows(name)
        extra = {"ladder_rows_txt": INPUTS["ladder_rows_txt"],
                 "ladder_smoke_not_priced": INPUTS["ladder_smoke_not_priced"]} if name == "ladder" else {}
        json.dump({"format": f"uarea-w9-{name}/1", **meta, **extra, "checks": checks, "rows": rows},
                  open(os.path.join(OUT, fname), "w"), indent=1)
        summary[name] = checks
    rows, checks = exploration()
    json.dump({"format": "uarea-w9-exploration/1", **meta,
               "source": "FS/cl5_exploration_t1_20261110T065255 regime_map.json ff6fce99 (s01's configurations); "
                         "cores per tile of config_area.json r2 laws", "cpt": PU["explore_cpt"], "checks": checks,
               "rows": rows}, open(os.path.join(OUT, "exploration.json"), "w"), indent=1)
    summary["exploration"] = checks
    body, checks = w48_rr()
    json.dump({"format": "uarea-w9-w48-rr/1", **meta, "checks": checks, **body},
              open(os.path.join(OUT, "w48_rr_study_chips.json"), "w"), indent=1)
    summary["w48_rr"] = checks
    summary["identities"] = {k: {x: v.get(x) for x in ("nominal", "low", "high")} for k, v in PU["identities"].items()}
    json.dump(summary, open(os.path.join(OUT, "side_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()

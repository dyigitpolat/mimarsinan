""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys

FORMS = ("P-A", "P-B")
NAMED = ("sample", "core", "capture_events", "device_cycles", "stimulus_words")


class Refusal(Exception):
    pass


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path, digest):
    if sha256(path) != digest:
        raise Refusal(f"{path} does not hash to {digest}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def tsv_named(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in NAMED if c not in (reader.fieldnames or [])]
        if missing:
            raise Refusal(f"{path} lacks {missing}")
        return [{c: int(row[c]) for c in NAMED} for row in reader]


def report_named(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    certs = doc.get("certificates") or []
    placement = doc.get("placement") or {}
    return {"passed": bool(doc.get("passed")) and all(bool(c.get("passed")) for c in certs),
            "fabric_cores": placement.get("fabric_cores"),
            "programming_runs": {str(int(c["core"])): int(c["programming_run"]["device_cycles"])
                                 for c in (doc.get("per_core") or []) if c.get("programming_run")}}


def spread(values):
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values), "max": values[-1]}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def read_rows(lock, readings):
    pattern = re.compile(lock["campaign_pattern"])
    rows, programming = [], {}
    for key, job in sorted(readings["jobs"].items()):
        real = lock["realizations"][key]
        by_pass = {(p["sample"], p["core"]): p for p in real["passes"]}
        folder = job["folder"]
        for name in sorted(os.listdir(folder)):
            m = pattern.match(name)
            if not m or not os.path.isdir(os.path.join(folder, name)):
                continue
            nc, arm, rnd = int(m.group("nc")), m.group("arm"), int(m.group("round"))
            rep = report_named(os.path.join(folder, name, "deployment_report.json"))
            if not rep["passed"]:
                raise Refusal(f"{key} {name}: not passed whole; the readings read every campaign as passed")
            for core, cycles in rep["programming_runs"].items():
                programming.setdefault((key, int(core)), set()).add(cycles)
            for r in tsv_named(os.path.join(folder, name, "deployment_samples.tsv")):
                p = by_pass[(r["sample"], r["core"])]
                if r["stimulus_words"] != p["stimulus_words"]:
                    raise Refusal(f"{key} {name}: census mismatch at sample {r['sample']} core {r['core']}")
                pred = p["pred"][arm]
                rows.append({"real": key, "nc": nc, "arm": arm, "round": rnd, "sample": r["sample"], "core": r["core"],
                             "pass_index": p["pass_index"], "words": r["stimulus_words"],
                             "capture_events": r["capture_events"], "measured": r["device_cycles"],
                             "pred": pred, "err": {f: (pred[f] - r["device_cycles"]) / r["device_cycles"] for f in FORMS}})
    return rows, {k: sorted(v) for k, v in programming.items()}


def cell_summary(rows):
    out = {"n": len(rows), "distinct_sample_pass": len({(r["sample"], r["pass_index"]) for r in rows})}
    for f in FORMS:
        errs = [r["err"][f] for r in rows]
        out[f] = dict(spread(errs), share_above_zero=sum(e > 0 for e in errs) / len(errs))
    return out


def same(a, b):
    return all(a[f][s] == b[f][s] for f in FORMS for s in ("n", "min", "median", "max"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--lock-sha256", required=True)
    ap.add_argument("--readings", required=True)
    ap.add_argument("--readings-sha256", required=True)
    ap.add_argument("--model", required=True, help="the locked time model; checked against the lock's model digest")
    ap.add_argument("--out", required=True)
    o = ap.parse_args(argv)
    try:
        lock = load(o.lock, o.lock_sha256)
        readings = load(o.readings, o.readings_sha256)
        model = load(o.model, lock["model"]["sha256"])
        rows, programming = read_rows(lock, readings)
    except Refusal as exc:
        print(f"REFUSED: {exc}")
        return 2
    grammar = {name: float(entry["value"]) for name, entry in model["grammar"].items()}
    out = {"schema": "ufidfpga_adoption_r10_e2/1",
           "lock": {"path": os.path.abspath(o.lock), "sha256": o.lock_sha256},
           "readings": {"path": os.path.abspath(o.readings), "sha256": o.readings_sha256},
           "realizations": {}}
    for key in sorted({r["real"] for r in rows}):
        real = lock["realizations"][key]
        coeff = {name: t["cycles"] for name, t in real["coefficients"].items()}
        cps = int(real["cycles_per_sample"])
        rr = [r for r in rows if r["real"] == key]
        res = {"coefficients": coeff, "cycles_per_sample": cps, "pass_order": real["pass_order"],
               "cores": real["cores"], "programming_runs_measured": {str(c): v for (k, c), v in programming.items() if k == key},
               "cells": {}, "checks": {}, "per_pass": {}, "largest": {}, "smallest": {}, "spearman_err_words": {}, "spearman_over_pb": {}, "tables": {},
               "as_run_split": {}, "fits": {}}

        groups = {}
        for r in rr:
            for scope in (str(r["nc"]), "all"):
                groups.setdefault(f"nc{scope}_{r['arm']}_pass{r['pass_index']}", []).append(r)
        for cid, g in sorted(groups.items()):
            mine = cell_summary(g)
            theirs = readings["realizations"][key]["cells"][cid]
            if not same(mine, theirs):
                print(f"REFUSED: {key} {cid} differs from the readings' cell")
                return 2
            res["cells"][cid] = mine
            res["checks"][cid] = "equal to readings.json"
        for arm in ("as_run", "resident"):
            ga = [r for r in rr if r["arm"] == arm]
            res["cells"][f"all_{arm}_all_passes"] = cell_summary(ga)

        distinct = {}
        for r in rr:
            distinct.setdefault((r["arm"], r["pass_index"], r["sample"]), set()).add(r["measured"])
        if any(len(v) != 1 for v in distinct.values()):
            print(f"REFUSED: {key}: a (arm, pass, sample) has more than one measured value")
            return 2
        res["checks"]["one_measured_value_per_arm_pass_sample"] = len(distinct)
        one = {}
        for r in rr:
            one.setdefault((r["arm"], r["pass_index"], r["sample"]), r)
        fetch_words = cps * (grammar["tag_words_per_cycle"] + grammar["wait_words_per_cycle"])
        for arm in ("as_run", "resident"):
            for index in sorted({k[1] for k in one}):
                ps = [one[k] for k in sorted(one) if k[0] == arm and k[1] == index]
                core = ps[0]["core"]
                cplan = real["cores"][str(core)]
                barrier = coeff["barrier"] * cplan["barrier_cycles"] * cps
                table = []
                for r in ps:
                    nondelivery = r["pred"]["P-A"] - r["pred"]["P-B"]
                    load_price = r["pred"]["P-A"] - one[("resident", index, r["sample"])]["pred"]["P-A"]
                    table.append({"sample": r["sample"], "words": r["words"], "capture_events": r["capture_events"],
                                  "measured": r["measured"], "pred_P-A": r["pred"]["P-A"], "pred_P-B": r["pred"]["P-B"],
                                  "err_P-A": r["err"]["P-A"], "err_P-B": r["err"]["P-B"],
                                  "barrier_term": barrier, "load_term": load_price,
                                  "event_term_P-A": r["pred"]["P-A"] - barrier - load_price,
                                  "nondelivery_price": nondelivery,
                                  "measured_over_P-B": r["measured"] - r["pred"]["P-B"],
                                  "card_share_of_nondelivery_price": (r["measured"] - r["pred"]["P-B"]) / nondelivery})
                tag = f"{arm}_pass{index}"
                res["per_pass"][tag] = {
                    "core": core, "clear_prefix_words": cplan["clear_prefix_words"],
                    "declared_program_words": cplan["program_words"], "barrier_cycles": cplan["barrier_cycles"],
                    "programming_run_measured": programming.get((key, core)),
                    "distinct_passes": len(table),
                    "words": spread([t["words"] for t in table]),
                    "measured": spread([t["measured"] for t in table]),
                    "err_P-A": dict(spread([t["err_P-A"] for t in table]), share_above_zero=sum(t["err_P-A"] > 0 for t in table) / len(table)),
                    "err_P-B": dict(spread([t["err_P-B"] for t in table]), share_above_zero=sum(t["err_P-B"] > 0 for t in table) / len(table)),
                    "barrier_term": barrier, "load_term": spread([t["load_term"] for t in table]),
                    "event_term_P-A": spread([t["event_term_P-A"] for t in table]),
                    "nondelivery_price": spread([t["nondelivery_price"] for t in table]),
                    "measured_over_P-B": spread([t["measured_over_P-B"] for t in table]),
                    "card_share_of_nondelivery_price": spread([t["card_share_of_nondelivery_price"] for t in table]),
                    "nondelivery_words": cplan["clear_prefix_words"] + fetch_words + grammar["end_words"],
                }
                res["spearman_err_words"][tag] = spearman([t["err_P-A"] for t in table], [t["words"] for t in table])
                res["spearman_over_pb"][tag] = {"words": spearman([t["measured_over_P-B"] for t in table], [t["words"] for t in table]),
                                                "capture_events": spearman([t["measured_over_P-B"] for t in table], [t["capture_events"] for t in table])}
                res["tables"][tag] = table
                ordered = sorted(table, key=lambda t: t["err_P-A"], reverse=True)
                res["largest"][tag] = ordered[:5]
                res["smallest"][tag] = ordered[-3:]
                fit = readings["realizations"][key]["cells"][f"nc1_{arm}_pass{index}"]["fit"]
                price = coeff["event"] / grammar["words_per_event"]
                res["fits"][tag] = {"slope_per_word": fit["slope"], "intercept": fit["intercept"], "r2": fit["r2"],
                                    "P-A_price_per_word": price,
                                    "slope_over_price": fit["slope"] / price - 1.0,

                                    "words_at_barrier_only": ((barrier - fit["intercept"]) / fit["slope"]) if arm == "resident" else None}

        for index in sorted({k[1] for k in one}):
            split = []
            for k in sorted(one):
                if k[0] != "as_run" or k[1] != index:
                    continue
                a, rres = one[k], one[("resident", index, k[2])]
                streamed = a["measured"] - rres["measured"]
                load_price = a["pred"]["P-A"] - rres["pred"]["P-A"]
                split.append({"streamed": streamed, "load_shortfall": load_price - streamed,
                              "resident_over": rres["pred"]["P-A"] - rres["measured"],
                              "check": abs((load_price - streamed) + (rres["pred"]["P-A"] - rres["measured"]) - (a["pred"]["P-A"] - a["measured"]))})
            res["as_run_split"][f"pass{index}"] = {
                "streamed_measured": spread([s["streamed"] for s in split]),
                "load_shortfall": spread([s["load_shortfall"] for s in split]),
                "resident_over": spread([s["resident_over"] for s in split]),
                "max_identity_residual": max(s["check"] for s in split)}
        out["realizations"][key] = res

    for key, res in out["realizations"].items():
        real = lock["realizations"][key]
        c0 = sorted(real["programming_run"])[0]
        end_words = (real["programming_run"][c0]["P-A"] - real["programming_run"][c0]["P-B"]) / (res["coefficients"]["event"] / grammar["words_per_event"])
        res["checks"]["end_words_from_lock"] = end_words
        res["checks"]["end_words_model"] = grammar["end_words"]
    with open(o.out, "x", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")

    for key, res in out["realizations"].items():
        print(f"== {key}: coefficients {res['coefficients']}, cycles per sample {res['cycles_per_sample']}, pass order {res['pass_order']}")
        print(f"   checks: {sum(v == 'equal to readings.json' for v in res['checks'].values())} cells equal to readings.json; "
              f"one measured value per (arm, pass, sample): {res['checks']['one_measured_value_per_arm_pass_sample']} keys; "
              f"END words from the lock {res['checks']['end_words_from_lock']:.6g}")
        for cid in sorted(c for c in res["cells"] if c.startswith("nc")):
            c = res["cells"][cid]
            print(f"   cell {cid}: n {c['n']} ({c['distinct_sample_pass']} distinct)"
                  + "".join(f"; {f} min {c[f]['min']:+.4%} median {c[f]['median']:+.4%} max {c[f]['max']:+.4%} share>0 {c[f]['share_above_zero']:.4f}" for f in FORMS))
        for cid in ("all_as_run_all_passes", "all_resident_all_passes"):
            c = res["cells"][cid]
            print(f"   pooled {cid}: n {c['n']}"
                  + "".join(f"; {f} min {c[f]['min']:+.4%} median {c[f]['median']:+.4%} max {c[f]['max']:+.4%} share>0 {c[f]['share_above_zero']:.4f}" for f in FORMS))
        for tag, p in sorted(res["per_pass"].items()):
            print(f"   pass {tag}: core {p['core']}, clear prefix {p['clear_prefix_words']} words, non-delivery words {p['nondelivery_words']}, "
                  f"declared program {p['declared_program_words']} words, programming run measured {p['programming_run_measured']}, "
                  f"barrier term {p['barrier_term']:.0f}; words {p['words']['min']}..{p['words']['median']}..{p['words']['max']}; "
                  f"measured {p['measured']['min']}..{p['measured']['median']}..{p['measured']['max']}; "
                  f"non-delivery price {p['nondelivery_price']['min']:.0f}..{p['nondelivery_price']['max']:.0f}; "
                  f"measured over P-B {p['measured_over_P-B']['min']:.0f}..{p['measured_over_P-B']['median']:.0f}..{p['measured_over_P-B']['max']:.0f}; "
                  f"card share of that price {p['card_share_of_nondelivery_price']['min']:.3f}..{p['card_share_of_nondelivery_price']['median']:.3f}..{p['card_share_of_nondelivery_price']['max']:.3f}; "
                  f"Spearman(err P-A, words) {res['spearman_err_words'][tag]:+.3f}; Spearman(measured over P-B, words) {res['spearman_over_pb'][tag]['words']:+.3f}, (.., capture events) {res['spearman_over_pb'][tag]['capture_events']:+.3f}")
            f = res["fits"][tag]
            print(f"      readings' fit: slope {f['slope_per_word']:.4f} per word against P-A's {f['P-A_price_per_word']:.4f} ({f['slope_over_price']:+.2%}), "
                  f"intercept {f['intercept']:.0f}"
                  + (f", words at which the readings' line meets the barrier term {f['words_at_barrier_only']:.0f}"
                     if f["words_at_barrier_only"] is not None else ""))
            for t in res["largest"][tag][:3]:
                print(f"      largest: sample {t['sample']} words {t['words']} capture events {t['capture_events']} measured {t['measured']} "
                      f"P-A {t['pred_P-A']:.0f} ({t['err_P-A']:+.2%}; event term {t['event_term_P-A']:.0f}, non-delivery price {t['nondelivery_price']:.0f}) "
                      f"P-B {t['pred_P-B']:.0f} ({t['err_P-B']:+.2%})")
            t = res["smallest"][tag][-1]
            print(f"      smallest: sample {t['sample']} words {t['words']} capture events {t['capture_events']} measured {t['measured']} "
                  f"P-A {t['pred_P-A']:.0f} ({t['err_P-A']:+.2%}) P-B {t['pred_P-B']:.0f} ({t['err_P-B']:+.2%})")
        for tag, s in sorted(res["as_run_split"].items()):
            print(f"   as run minus resident, {tag}: streamed measured {s['streamed_measured']['min']}..{s['streamed_measured']['max']}; "
                  f"load term short by {s['load_shortfall']['min']:.0f}..{s['load_shortfall']['max']:.0f}; "
                  f"resident P-A over by {s['resident_over']['min']:.0f}..{s['resident_over']['max']:.0f}; identity residual {s['max_identity_residual']:.3g}")
    print(f"wrote {o.out} sha256 {sha256(o.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

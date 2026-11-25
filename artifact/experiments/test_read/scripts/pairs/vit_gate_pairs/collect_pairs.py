""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairs_common as pc
import pairs_stats as st

N = 5000
BITS_PRIMARY = 8
BITS_SECONDARY = (6, 4)
PRIMARY = ("softmax", "B", BITS_PRIMARY)
PRINTED = [("softmax", "B", b, "secondary_bits") for b in BITS_SECONDARY] + [("softmax", "A", BITS_PRIMARY, "stage_a")] \
    + [("relu", "A", b, "relu_stage_a") for b in (8, 6, 4)]
TRAINED = ("F", "QF-T", "QF-O8", "QF-O6", "QF-O4", "F-R")
STAGE_A = {"A": "F", "A-R": "F-R"}
FORM_JOBS = {"softmax": {"float": "F", "ptq": "A", "twin": "QF-T"}, "relu": {"float": "F-R", "ptq": "A-R"}}


def RS():
    return pc.load_locked("readings_seeds.py")


def fpp(images: float, total: int = N) -> str:
    return f"{100.0 * images / total:+.2f}"


def fp(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def dnums(ds: list) -> dict:
    return {f"d{k + 1}": f"{int(x):+d}" for k, x in enumerate(ds)}


def series_name(form: str, stage: str, bits: int) -> str:
    return f"{form} stage {stage} {bits} bits"


def entries_e01(readings: OrderedDict) -> OrderedDict:
    ""
    rs = RS()
    out = OrderedDict()
    for form, stage, bits in [PRIMARY] + [(f, s, b) for f, s, b, _ in PRINTED]:
        ents, reasons = OrderedDict(), []
        for lab, r in readings.items():
            e, why = (rs.stage_b_entry(r, form, "fallback", bits) if stage == "B" else rs.stage_a_entry(r, form, bits))
            ents[lab] = e
            if e is None:
                reasons.append(f"{lab}: {why}")
        out[(form, stage, bits)] = (ents, reasons)
    return out


def test_entry(twin: int, operand: int, flt: int, total: int, twin_line_images: int) -> dict:
    ""
    return {"delta_images": int(operand) - int(twin), "operand_correct": int(operand), "twin_correct": int(twin),
            "twin_valid": int(twin) - int(flt) >= -int(twin_line_images), "se_pp": None, "float_correct": int(flt)}


def entries_test(counts: OrderedDict, total: int, twin_line_images: int) -> OrderedDict:
    ""
    out = OrderedDict()
    for form, stage, bits in [PRIMARY] + [(f, s, b) for f, s, b, _ in PRINTED]:
        ents = OrderedDict()
        for lab, c in counts.items():
            if stage == "B":
                ents[lab] = test_entry(c["QF-T"], c[f"QF-O{bits}"], c["F"], total, twin_line_images)
            else:
                a = c["A" if form == "softmax" else "A-R"]
                flt = c["F" if form == "softmax" else "F-R"]
                ents[lab] = test_entry(a["twin"], a[f"operand_{bits}"], flt, total, twin_line_images)
        out[(form, stage, bits)] = (ents, [])
    return out


def pending_reason(states: OrderedDict, jobs) -> str | None:
    ""
    waits = [f"{lab}: {j} {sts.get(j) or 'not started'}" for lab, sts in states.items() for j in jobs
             if not pc.ended(sts.get(j))]
    return "; ".join(waits) if waits else None


def read_vit(series: OrderedDict, s0: dict | None, rp: dict, states: OrderedDict | None = None, total: int = N,
             read: str = "e01") -> OrderedDict:
    ""
    R = rp["readings"]
    S, C = R["sentences"], R["classes"]
    th = R["thresholds"]
    conf = th["conf"]
    margin = int(round(th["margin_pp"] * total / 100))
    words = dict(read=R["read_words"][read], total=f"{total:,}")
    out = OrderedDict(read=read, total=total, margin_images=margin)
    ents, reasons = series[PRIMARY]
    n_seeds = len(ents)
    if n_seeds != th["n_seeds"]:
        raise SystemExit(f"REFUSED: {n_seeds} seeds read, {th['n_seeds']} declared")
    prim = OrderedDict(form="softmax", stage="B", recipe="fallback", bits=BITS_PRIMARY, reasons=reasons, entries=ents)
    pend = pending_reason(states, pc.READING_JOBS) if states is not None else None
    failed = [f"{lab}: {j} {sts.get(j)}" for lab, sts in (states or {}).items() for j in pc.READING_JOBS
              if str(sts.get(j) or "").startswith(pc.TERMINAL_FAILURE_PREFIX)]
    if failed:
        prim.update(cls=C["not_determinable"], sentence=S["not_determinable"].format(reason="; ".join(failed)))
    elif pend:
        prim.update(cls=C["pending"], sentence=S["pending"].format(reason=pend))
    elif reasons:
        prim.update(cls=C["not_determinable"], sentence=S["not_determinable"].format(reason="; ".join(reasons)))
    else:
        d = [int(e["delta_images"]) for e in ents.values()]
        a = st.paired(d, total, conf, mu0_images=margin)
        degraded = [lab for lab, e in ents.items() if not e["twin_valid"]]
        X = 100.0 * sum(int(e["operand_correct"]) for e in ents.values()) / (len(d) * total)
        Y = 100.0 * sum(int(e["twin_correct"]) for e in ents.values()) / (len(d) * total)
        nums = dict(words, mean=fpp(a["mean_images"], total), lower95=fpp(a["lower_images"], total), p=fp(a["p"]),
                    X=f"{X:.2f}", Y=f"{Y:.2f}", **dnums(d))
        prim.update(d_images=d, mean_pp=round(a["mean_pp"], 4), sd_pp=round(a["sd_pp"], 4),
                    lower95_pp=round(a["lower_pp"], 4), p_one_sided=a["p"], t=a["t_stat"],
                    operand_mean_pct=round(X, 4), twin_mean_pct=round(Y, 4), degraded=degraded)
        if degraded:
            prim.update(cls=C["not_determinable_degraded"], sentence=S["nd_degraded"].format(
                reason=f"the QAT twin under the fallback recipe lies more than 1.0 pp under the float job at "
                       f"{', '.join(degraded)}", **nums))
        elif a["lower_images"] >= margin:
            prim.update(cls=C["holds"], sentence=S["holds"].format(**nums))
        else:
            prim.update(cls=C["not_shown"], sentence=S["not_shown"].format(**nums))
    out["primary"] = prim

    if s0 is not None:
        e0 = s0["entry"]
        out["s0"] = OrderedDict(entry=e0, sentence=S["s0_beside"].format(
            op0=e0["operand_correct"], tw0=e0["twin_correct"], d0=f"{int(e0['delta_images']):+d}",
            d0pp=fpp(int(e0["delta_images"]), total), **words))
    else:
        out["s0"] = OrderedDict(entry=None, sentence=S["s0_missing"].format(**words))
    if "d_images" in prim and s0 is not None:
        d6 = [int(s0["entry"]["delta_images"])] + prim["d_images"]
        a6 = st.paired(d6, total, conf, mu0_images=margin)
        out["with_s0"] = OrderedDict(d_images=d6, mean_pp=round(a6["mean_pp"], 4), lower95_pp=round(a6["lower_pp"], 4),
                                     p_one_sided=a6["p"], df=a6["df"])
        out["with_s0"]["sentence"] = S["with_s0"].format(
            mean=fpp(a6["mean_images"], total), lower95=fpp(a6["lower_images"], total), p=fp(a6["p"]),
            d0=f"{d6[0]:+d}", **dnums(d6[1:]), **words)
    else:
        out["with_s0"] = None

    printed = OrderedDict()
    for form, stage, b, key in PRINTED:
        es, why = series[(form, stage, b)]
        name = series_name(form, stage, b)
        if why:
            printed[name] = OrderedDict(sentence=S["printed_nd"].format(series=R["series_words"][key], bits=b,
                                                                        reason="; ".join(why)))
            continue
        d = [int(e["delta_images"]) for e in es.values()]
        a = st.paired(d, total, conf, mu0_images=margin)
        side = "at or above the -1.0 pp margin" if a["lower_images"] >= margin else "below the -1.0 pp margin"
        nvalid = sum(1 for e in es.values() if e["twin_valid"])
        printed[name] = OrderedDict(d_images=d, mean_pp=round(a["mean_pp"], 4), lower95_pp=round(a["lower_pp"], 4),
                                    p_one_sided=a["p"], twins_valid=nvalid,
                                    sentence=S["printed"].format(series=R["series_words"][key], bits=b,
                                                                 mean=fpp(a["mean_images"], total),
                                                                 lower95=fpp(a["lower_images"], total), side=side,
                                                                 nvalid=nvalid, n=len(d), **dnums(d), **words))
    out["printed_series"] = printed

    if "d_images" in prim:
        fl = [100.0 * e["float_correct"] / total for e in ents.values()]
        tw = [100.0 * e["twin_correct"] / total for e in ents.values()]
        op = [100.0 * e["operand_correct"] / total for e in ents.values()]

        def sd(xs):
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        se = [e.get("se_pp") for e in ents.values()]
        out["spread"] = OrderedDict(float_pct=fl, twin_pct=tw, operand_pct=op, float_sd_pp=sd(fl), twin_sd_pp=sd(tw),
                                    operand_sd_pp=sd(op), delta_sd_pp=prim["sd_pp"], paired_se_pp=se)
        out["spread"]["sentence"] = S["spread"].format(
            float_list=", ".join(f"{x:.2f}" for x in fl), float_sd=f"{sd(fl):.2f}",
            twin_list=", ".join(f"{x:.2f}" for x in tw), twin_sd=f"{sd(tw):.2f}",
            op_list=", ".join(f"{x:.2f}" for x in op), op_sd=f"{sd(op):.2f}", delta_sd=f"{prim['sd_pp']:.2f}",
            se_list=", ".join("-" if x is None else f"{x:.2f}" for x in se), **words)
    else:
        out["spread"] = None
    out["printed"] = R["printed"]
    return out


def s0_entry(reading: dict) -> dict:
    ""
    e, why = RS().stage_b_entry(reading, "softmax", "fallback", BITS_PRIMARY)
    if e is None:
        raise SystemExit(f"REFUSED: seed 0's 8-bit fallback pair is not readable: {why}")
    row = pc.read_json(pc.S0_RESULT_JSON)["readings"]["forms"]["softmax"]["per_bits"][str(BITS_PRIMARY)]
    sb = row["stage_b"]
    if (row["stage_b_recipe"], sb["job"], sb["twin_job"], sb["operand_correct"], sb["twin_correct"]) != (
            "fallback", e["operand_job"], e["twin_job"], e["operand_correct"], e["twin_correct"]):
        raise SystemExit("REFUSED: seed 0's recomputed 8-bit fallback pair differs from vit_gate/gate_result.json")
    return {"entry": e}


def w06_rows() -> list:
    ""
    rows = []
    for lab in pc.NEW_SEEDS:
        h = pc.harness_dir(lab)
        for j in pc.JOBS:
            p = h / "runs" / j / "result.json"
            rec = pc.read_json(p) if p.exists() else None
            if not rec or rec.get("status") != "ok" or rec.get("smoke"):
                raise SystemExit(f"REFUSED: {lab}/{j} has no ok record: the W06 rows need every declared job ended ok")
            row = OrderedDict(track="vit_gate_pairs", seed=lab, seed_value=pc.SEEDS[lab], job=j, kind=rec.get("kind"),
                              record_sha256=pc.sha256_file(p), e01_counts=rec.get("counts"))
            if j in TRAINED:
                ck = h / "runs" / j / "final.pt"
                if not ck.exists() or pc.sha256_file(ck) != rec.get("final_sha256"):
                    raise SystemExit(f"REFUSED: {lab}/{j}: final.pt is missing or differs from its record")
                row.update(checkpoint=f"{lab}/runs/{j}/final.pt", checkpoint_sha256=rec["final_sha256"])
            else:
                fi = rec.get("float_input") or {}
                row.update(checkpoint=f"{lab}/runs/{STAGE_A[j]}/final.pt", checkpoint_sha256=fi.get("sha256"),
                           calibration_sha256=rec.get("calibration_sha256"),
                           twin_steps_sha256=rec.get("twin_steps_sha256"))
            rows.append(row)
    return rows


def test_counts(test: dict, rows: list) -> tuple:
    ""
    if test.get("total") != pc.TEST_TOTAL or test.get("read") != "cifar100-test":
        raise SystemExit(f"REFUSED: the test-read file must hold read 'cifar100-test' and total {pc.TEST_TOTAL}")
    got = {(r.get("seed"), r.get("job")): r for r in test.get("rows", []) if r.get("track") == "vit_gate_pairs"}
    out = OrderedDict((lab, OrderedDict()) for lab in pc.NEW_SEEDS)

    def cnt(v, where):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= pc.TEST_TOTAL:
            raise SystemExit(f"REFUSED: {where}: a test count must be an integer in 0..{pc.TEST_TOTAL}")
        return v
    for w in rows:
        r = got.get((w["seed"], w["job"]))
        if r is None:
            raise SystemExit(f"REFUSED: the test read holds no row for {w['seed']}/{w['job']}")
        for k in ("checkpoint_sha256", "calibration_sha256", "twin_steps_sha256"):
            if k in w and r.get(k) != w[k]:
                raise SystemExit(f"REFUSED: {w['seed']}/{w['job']}: the test read's {k} differs from w06_rows.json")
        if w["job"] in STAGE_A:
            c = r.get("counts") or {}
            out[w["seed"]][w["job"]] = {k: cnt(c.get(k), f"{w['seed']}/{w['job']}/{k}")
                                        for k in ("twin", "operand_8", "operand_6", "operand_4")}
        else:
            out[w["seed"]][w["job"]] = cnt(r.get("correct"), f"{w['seed']}/{w['job']}")
    s0 = None
    s0r = {r.get("job"): r.get("correct") for r in test.get("s0_rows", []) if r.get("track", "vit_gate") == "vit_gate"}
    if all(isinstance(s0r.get(j), int) for j in ("F", "QF-T", "QF-O8")):
        line = int(round(1.0 * pc.TEST_TOTAL / 100))
        s0 = {"entry": test_entry(s0r["QF-T"], s0r["QF-O8"], s0r["F"], pc.TEST_TOTAL, line)}
    return out, s0


def render(res: OrderedDict, rp: dict, header: str, beside: str | None = None) -> str:
    p = res["primary"]
    total = res["total"]
    L = ["# ViT Non-Inferiority over Seeds s1 to s5: Result", "", header, "",
         f"## Primary Reading (softmax block, 8 payload bits, stage B, fallback recipe; {rp['readings']['read_words'][res['read']]})",
         "", f"| class | delta at s1, s2, s3, s4, s5 (images of {total:,}) | mean (pp) | SD (pp) | one-sided 95% lower (pp) | "
             "margin | p (H0: mean at most -1.0 pp) | seed 0 delta (images) |", "|---|---|---|---|---|---|---|---|"]
    d = ", ".join(f"{x:+d}" for x in p.get("d_images", [])) or "-"
    g = lambda k, f: f.format(p[k]) if p.get(k) is not None else "-"
    e0 = res["s0"]["entry"]
    L.append(f"| {p['cls']} | {d} | {g('mean_pp', '{:+.2f}')} | {g('sd_pp', '{:.2f}')} | {g('lower95_pp', '{:+.2f}')} | "
             f"-1.00 | {fp(p['p_one_sided']) if p.get('p_one_sided') is not None else '-'} | "
             f"{'-' if e0 is None else format(int(e0['delta_images']), '+d')} |")
    L += ["", f"- {p['sentence']}"]
    if beside:
        L.append(f"- {beside}")
    L += ["", "## Beside the Reading (printed, never readings)", "", f"- {res['s0']['sentence']}"]
    if res.get("with_s0"):
        L.append(f"- {res['with_s0']['sentence']}")
    L += [f"- {v['sentence']}" for v in res["printed_series"].values()]
    if res.get("spread"):
        L.append(f"- {res['spread']['sentence']}")
    L += ["", "## Printed Lines (verbatim, rules_pairs.md section 7)", ""] + [f"- {v}" for v in res["printed"].values()]
    return "\n".join(L) + "\n"


def e01_readings(lock: dict) -> tuple:
    import build_harness

    for lab in pc.NEW_SEEDS:
        problems = build_harness.verify(lab)
        if problems:
            raise SystemExit(f"REFUSED: {lab} does not verify: {'; '.join(problems)}")
    r0 = pc.probe(pc.S0, "reading")
    s0 = s0_entry(r0)
    readings, states = OrderedDict(), OrderedDict()
    for lab in pc.NEW_SEEDS:
        h = pc.harness_dir(lab)
        r = pc.probe(h, "reading")
        for j, rec in r["records"].items():
            if rec.get("status") == "ok":
                if rec.get("seed") != pc.SEEDS[lab]:
                    raise SystemExit(f"REFUSED: {lab}/{j} carries seed {rec.get('seed')}, declared {pc.SEEDS[lab]}")
                if rec.get("smoke"):
                    raise SystemExit(f"REFUSED: {lab}/{j} is a smoke record")
                if float(rec.get("started_unix") or 0.0) < float(lock["locked_at_unix"]):
                    raise SystemExit(f"REFUSED: {lab}/{j} started before the pairs lock")
                if j not in pc.JOBS:
                    raise SystemExit(f"REFUSED: {lab}/{j} is a record of an undeclared job")
        readings[lab] = r
        states[lab] = pc.probe(h, "state")["statuses"]
    return readings, states, s0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--w06-rows", action="store_true")
    ap.add_argument("--test-read")
    args = ap.parse_args(argv)
    rp = pc.read_json(pc.RULES_JSON, ordered=True)
    pc.require_locked_sources()
    if args.dry:
        r0 = pc.probe(pc.S0, "dry")
        s0 = s0_entry(r0)
        readings = OrderedDict((lab, r0) for lab in pc.NEW_SEEDS)
        states = OrderedDict((lab, {j: "ok" for j in pc.READING_JOBS}) for lab in pc.NEW_SEEDS)
        res = read_vit(entries_e01(readings), s0, rp, states)
        d = pc.RUNS / "_dry"
        d.mkdir(parents=True, exist_ok=True)
        pc.atomic_write_json(d / "pairs_result_dry.json", res)
        (d / "pairs_result_dry.md").write_text(render(res, rp, "DRY: seed 0's records stand in for every seed; not a "
                                                               "result."))
        print(json.dumps({"class": res["primary"]["cls"], "d_images": res["primary"].get("d_images")}))
        return 0
    import make_pairs_lock

    lock = make_pairs_lock.verify()
    if args.w06_rows:
        rows = w06_rows()
        out = OrderedDict(label=("the rows of vit_gate_pairs the W06 test-read lock must list (rules_pairs.md section "
                                 "7): 8 jobs at each of s1 to s5"), written_at=pc.now_stamp(),
                          pairs_lock_sha256=pc.sha256_file(pc.LOCK_JSON), n_rows=len(rows), rows=rows)
        pc.atomic_write_json(pc.W06_ROWS_JSON, out, indent=1)
        print(f"w06_rows.json: {len(rows)} rows (sha256 {pc.sha256_file(pc.W06_ROWS_JSON)})")
        return 0
    readings, states, s0 = e01_readings(lock)
    dev = read_vit(entries_e01(readings), s0, rp, states)
    if args.test_read:
        w = pc.read_json(pc.W06_ROWS_JSON)
        if not w:
            raise SystemExit("REFUSED: w06_rows.json is missing (bash run_pairs.sh w06-rows, before the W06 lock)")
        if json.loads(json.dumps(w06_rows())) != w["rows"]:
            raise SystemExit("REFUSED: w06_rows.json differs from the rows the records give now")
        test = pc.read_json(Path(args.test_read))
        if not test:
            raise SystemExit(f"REFUSED: {args.test_read} is not a readable test-read file")
        counts, ts0 = test_counts(test, w["rows"])
        line = int(round(rp["readings"]["thresholds"]["twin_validity_pp"] * pc.TEST_TOTAL / 100))
        rec = read_vit(entries_test(counts, pc.TEST_TOTAL, line), ts0, rp, None, pc.TEST_TOTAL, "test")
        S = rp["readings"]["sentences"]
        beside = S["development_beside"].format(
            dev_class=dev["primary"]["cls"],
            mean="-" if dev["primary"].get("mean_pp") is None else f"{dev['primary']['mean_pp']:+.2f}",
            lower95="-" if dev["primary"].get("lower95_pp") is None else f"{dev['primary']['lower95_pp']:+.2f}")
        rec.update(collected_at=pc.now_stamp(), pairs_lock_sha256=pc.sha256_file(pc.LOCK_JSON),
                   w06_rows_sha256=pc.sha256_file(pc.W06_ROWS_JSON),
                   test_read_sha256=pc.sha256_file(Path(args.test_read)), test_lock_sha256=test.get("test_lock_sha256"),
                   development=dev, development_beside=beside)
        pc.atomic_write_json(pc.HERE / "pairs_result_record.json", rec)
        header = (f"Collected {rec['collected_at']}: the class of record on the W06 test read (test-read file sha256 "
                  f"{rec['test_read_sha256'][:16]}), under pairs_lock.json (sha256 {rec['pairs_lock_sha256'][:16]}, "
                  f"locked {lock['locked_at']}); seeds s1 to s5; the E01 reading (development numbers) printed beside; "
                  "seed 0 printed beside, in no reading.")
        (pc.HERE / "pairs_result_record.md").write_text(render(rec, rp, header, beside))
        print(json.dumps({"class_of_record": rec["primary"]["cls"], "development": dev["primary"]["cls"]}))
        return 0
    dev["collected_at"] = pc.now_stamp()
    dev["pairs_lock_sha256"] = pc.sha256_file(pc.LOCK_JSON)
    dev["harness_locks"] = OrderedDict((lab, readings[lab]["lock_sha256"]) for lab in pc.NEW_SEEDS)
    dev["calibration_sha256"] = OrderedDict((lab, readings[lab]["series"]["softmax"].get("calibration_sha256"))
                                            for lab in pc.NEW_SEEDS)
    dev["class_of_record"] = rp["readings"]["sentences"]["record_pending"]
    pc.atomic_write_json(pc.HERE / "pairs_result.json", dev)
    header = (f"Collected {dev['collected_at']} under pairs_lock.json (sha256 {dev['pairs_lock_sha256'][:16]}, locked "
              f"{lock['locked_at']}); seeds s1 to s5; counts MEASURED on the 5,000 E01 images: development numbers. "
              + rp["readings"]["sentences"]["record_pending"])
    (pc.HERE / "pairs_result.md").write_text(render(dev, rp, header))
    print(json.dumps({"development_class": dev["primary"]["cls"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

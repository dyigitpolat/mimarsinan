""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairs_common as pc
import pairs_stats as st

INCOMPLETE = ("refused", "crashed", "teacher_unavailable", "float_unavailable")
FAILED = "failed_"
READS = ("test", "e01")
CHECKPOINT = {"float": "float_final.pt", "qat": "qat_final.pt"}


def spp(images: float, total: int) -> str:
    ""
    return f"{100.0 * images / total:+.2f}"


def fp(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def ftotal(total: int) -> str:
    return f"{total:,}"


def lines_for(th: dict, total: int, n: int) -> OrderedDict:
    ""
    per = OrderedDict((k, int(round(th[k] * total / 100))) for k in ("floor_pp", "rule_pp"))
    return OrderedDict(floor_per_seed=per["floor_pp"], rule_per_seed=per["rule_pp"],
                       floor_sum=n * per["floor_pp"], rule_sum=n * per["rule_pp"])


def parse_started(rec: dict) -> float | None:
    s = rec.get("started_at")
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None


def checked_count(job: str, rec: dict, key: str) -> int:
    c, total = rec.get(key), rec.get("val_total")
    if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= pc.TOTAL or total != pc.TOTAL:
        raise SystemExit(f"REFUSED: {job}: a counted record needs an integer {key} in 0..{pc.TOTAL} and val_total "
                         f"{pc.TOTAL}; got {c!r} and {total!r}")
    return c


def instance_view(tag: str, inst: Path, lock_unix: float) -> OrderedDict:
    ""
    rules = pc.read_json(inst / "headline_rules.json")
    lock = pc.read_json(inst / "headline_lock.json")
    if not rules or not lock:
        raise SystemExit(f"REFUSED: {tag} has no rules or no instance lock")
    seed = pc.SEEDS[tag]
    if rules["recipe"]["seed"] != seed:
        raise SystemExit(f"REFUSED: {tag}/headline_rules.json carries seed {rules['recipe']['seed']}, declared {seed}")
    digests = lock["digests"]
    jobs = {j["id"]: j for j in rules["jobs"]}
    recs = OrderedDict()
    for jid in pc.JOBS:
        p = inst / "runs" / jid / "result.json"
        recs[jid] = pc.read_json(p) if p.exists() else None
    out = OrderedDict()
    for jid in pc.JOBS:
        r = recs[jid]
        row = OrderedDict(state="pending", status=None, count=None, float_count=None, problem=None)
        if r is None:
            out[jid] = row
            continue
        status = r.get("status")
        row["status"] = status
        d = r.get("digests")
        if d is None and status in INCOMPLETE:
            out[jid] = row
            continue
        if d != digests:
            raise SystemExit(f"REFUSED: {tag}/{jid}: a record under other digests than the instance lock")
        if r.get("smoke"):
            raise SystemExit(f"REFUSED: {tag}/{jid}: a smoke record in the run directory")
        t0 = parse_started(r)
        if t0 is None or t0 < math.floor(lock_unix):
            raise SystemExit(f"REFUSED: {tag}/{jid}: a record that started before the pairs lock ({r.get('started_at')})")
        if status is not None and status.startswith(FAILED):
            row.update(state="failed", problem=f"{jid} ended {status}")
        elif status == "ok":
            if (r.get("recipe") or {}).get("seed") != seed:
                raise SystemExit(f"REFUSED: {tag}/{jid}: the record's recipe seed is not {seed}")
            if jobs[jid]["kind"] == "float":
                row.update(state="ok", count=checked_count(jid, r, "float_val_correct"))
            else:
                fj = jobs[jid]["depends_on"][0]
                src = r.get("float_record") or {}
                phase = pc.read_json(inst / "runs" / fj / "float_phase.json") or {}
                if src.get("job") != fj or src.get("float_final_sha256") != phase.get("float_final_sha256"):
                    raise SystemExit(f"REFUSED: {tag}/{jid}: the QAT record did not read the float of {fj}")
                row.update(state="ok", count=checked_count(jid, r, "qat_val_correct"),
                           float_count=checked_count(jid, r, "float_val_correct"))
        out[jid] = row
    for jid, j in jobs.items():
        if j["kind"] == "qat" and out[j["depends_on"][0]]["state"] == "failed" and out[jid]["state"] != "ok":
            out[jid].update(state="failed", problem=f"{jid} blocked: float job {j['depends_on'][0]} failed")
    return out


def pair_series(views: OrderedDict, cell: str) -> OrderedDict:
    p = pc.PAIRS[cell]
    jj, fj = pc.MEMBER_JOBS[p["J"]], pc.MEMBER_JOBS[p["FO"]]
    out = OrderedDict(d=OrderedDict(), state="ok", reasons=[])
    for tag, v in views.items():
        a, b = v[jj], v[fj]
        for job, row in ((jj, a), (fj, b)):
            if row["state"] == "failed":
                out["state"] = "failed"
                out["reasons"].append(f"{tag}: {row['problem']}")
            elif row["state"] == "pending" and out["state"] != "failed":
                out["state"] = "pending"
                out["reasons"].append(f"{tag}: {job} has not ended")
        out["d"][tag] = (a["count"] - b["count"]) if a["state"] == b["state"] == "ok" else None
    return out


def fill(template: str, **kw) -> str:
    return template.format(**kw)


def pair_words(cell: str) -> dict:
    p = pc.PAIRS[cell]
    return dict(cell=cell, J=p["J"], FO=p["FO"], J_passes=p["J_passes"], J_geometry=p["J_geometry"])


def dnums(ds: list) -> dict:
    return {f"d{k + 1}": f"{int(x):+d}" for k, x in enumerate(ds)}


def read_pairs(views: OrderedDict, s0: OrderedDict | None, rp: dict, total: int = pc.TOTAL,
               read: str = "e01") -> OrderedDict:
    ""
    if read not in READS:
        raise SystemExit(f"REFUSED: read {read!r}")
    R = rp["readings"]
    th = R["thresholds"]
    S = R["sentences"]
    alpha, conf, sconf = th["family_alpha"], th["conf"], th["simultaneous_conf"]
    tags = list(views)
    n = len(tags)
    if n != th["n_seeds"]:
        raise SystemExit(f"REFUSED: {n} seeds read, {th['n_seeds']} declared")
    L = lines_for(th, total, n)
    words = dict(read=R["read_words"][read], total=ftotal(total))
    series = OrderedDict((c, pair_series(views, c)) for c in pc.PAIRS)
    stats = OrderedDict()
    for c, s in series.items():
        if s["state"] == "ok":
            ds = [s["d"][t] for t in tags]
            stats[c] = OrderedDict(p95=st.paired(ds, total, conf), p975=st.paired(ds, total, sconf))
    pv = OrderedDict((c, stats[c]["p95"]["p"] if c in stats else 1.0) for c in pc.PRIMARY)
    all_settled = all(series[c]["state"] != "pending" for c in pc.PRIMARY)
    holm = st.holm(pv, alpha) if all_settled else None
    out = OrderedDict(read=read, total=total, n_seeds=n, lines=L, cells=OrderedDict(), holm=holm)
    for c in pc.PAIRS:
        s = series[c]
        w = dict(pair_words(c), **words)
        row = OrderedDict(role=pc.PAIRS[c]["role"], J=pc.PAIRS[c]["J"], FO=pc.PAIRS[c]["FO"],
                          d_images=s["d"], state=s["state"], reasons=s["reasons"], materiality=None)
        s0d = None if s0 is None else s0[pc.PAIRS[c]["J"]] - s0[pc.PAIRS[c]["FO"]]
        row["s0_d_images"] = s0d
        nums = None
        if c in stats:
            a, b = stats[c]["p95"], stats[c]["p975"]
            row.update(mean_pp=round(a["mean_pp"], 4), sd_pp=round(a["sd_pp"], 4), se_pp=round(a["se_pp"], 4),
                       lower95_pp=round(a["lower_pp"], 4), lower975_pp=round(b["lower_pp"], 4), t=a["t_stat"],
                       p_one_sided=a["p"], sum_images=a["sum_images"])
            ds = [s["d"][t] for t in tags]
            nums = dict(w, mean=spp(a["mean_images"], total), lower95=spp(a["lower_images"], total),
                        lower975=spp(b["lower_images"], total), p=fp(a["p"]), **dnums(ds))
        if c in pc.PRIMARY:
            if s["state"] == "pending":
                row["class"] = R["classes"]["pending"]
                row["sentence"] = fill(S["pending"], cell=c, reason="; ".join(s["reasons"]))
            elif s["state"] == "failed":
                row["class"] = R["classes"]["not_determinable"]
                row["sentence"] = fill(S["not_determinable"], cell=c, reason="; ".join(s["reasons"]))
            elif holm is None:
                row["class"] = R["classes"]["pending"]
                row["sentence"] = fill(S["pending"], cell=c, reason="the other primary pair is pending (Holm needs both)")
            else:
                h = holm[c]
                row.update(p_holm=h["p_holm"], holm_rank=h["rank"], holm_level=h["level"], holm_rejected=h["rejected"])
                nums["p_holm"] = fp(h["p_holm"])
                holds = h["rejected"] and row["sum_images"] >= L["floor_sum"]
                row["class"] = R["classes"]["holds" if holds else "not_shown"]
                if holds:
                    material = row["sum_images"] >= L["rule_sum"]
                    row["materiality"] = R["materiality"]["material" if material else "below_rule"]
                    key = "holds_material" if material else "holds_below_rule"
                else:
                    key = "not_shown_test" if not h["rejected"] else "not_shown_mean"
                row["sentence_key"] = key
                row["sentence"] = fill(S[key], **nums)
        else:
            row["class"] = None
            row["sentence"] = (fill(S["secondary"], **nums) if c in stats else
                               fill(S["secondary_nd"], cell=c, reason="; ".join(s["reasons"]) or "no counts"))
        if s0 is not None:
            row["s0_sentence"] = fill(S["s0_beside"], **w, J0=s0[pc.PAIRS[c]["J"]], FO0=s0[pc.PAIRS[c]["FO"]],
                                      d0=f"{s0d:+d}", d0pp=spp(s0d, total), s0_reading=rp["pairs"][c]["s0_reading"])
        else:
            row["s0_sentence"] = fill(S["s0_missing"], cell=c, **words)
        out["cells"][c] = row

    k = series["K64"]
    if k["state"] == "ok" and s0 is not None:
        ds = [out["cells"]["K64"]["s0_d_images"]] + [k["d"][t] for t in tags]
        a = st.paired(ds, total, conf)
        out["k64_with_s0"] = OrderedDict(d_images=ds, mean_pp=round(a["mean_pp"], 4), lower95_pp=round(a["lower_pp"], 4),
                                         p_one_sided=a["p"], df=a["df"])
        out["k64_with_s0"]["sentence"] = fill(
            S["k64_with_s0"], mean=spp(a["mean_images"], total), d0=f"{ds[0]:+d}", lower95=spp(a["lower_images"], total),
            p=fp(a["p"]), **words, **dnums(ds[1:]))
    else:
        out["k64_with_s0"] = None

    mem = OrderedDict()
    for label, job in pc.MEMBER_JOBS.items():
        xs = [views[t][job]["count"] for t in tags if views[t][job]["state"] == "ok"]
        if len(xs) == n:
            mem[label] = xs
    pooled = st.pooled_sd(mem)
    diag = OrderedDict(pooled_sd_pp=(round(100.0 * pooled["sd_images"] / total, 4) if pooled["sd_images"] else None),
                       pooled_df=pooled["df"], pooled_members=pooled["members"], per_pair=OrderedDict())
    for c in pc.PAIRS:
        if c in stats and pooled["sd_images"]:
            a = stats[c]["p95"]
            pl = st.pooled_lower(a["mean_images"], a["n"], pooled, conf)
            diag["per_pair"][c] = OrderedDict(
                pooled_lower95_pp=round(100.0 * pl["lower_images"] / total, 4),
                paired_sd_pp=round(a["sd_pp"], 4), sqrt2_pooled_sd_pp=round(100.0 * pl["sd_diff_images"] / total, 4))
            diag["per_pair"][c]["sentence"] = fill(
                S["pooled_beside"], cell=c, pooled_sd=f"{diag['pooled_sd_pp']:.2f}", pooled_df=pooled["df"],
                pooled_lower=spp(pl["lower_images"], total), paired_sd=f"{a['sd_pp']:.2f}",
                sqrt2_pooled=f"{100.0 * pl['sd_diff_images'] / total:.2f}", **words)
    diag["members"] = OrderedDict((t, OrderedDict((j, views[t][j]["count"]) for j in pc.JOBS)) for t in tags)
    diag["r18_w1_float"] = OrderedDict((t, views[t]["F-R18-w100"]["count"]) for t in tags)
    diag["r1l_note"] = S["r1l_note"]
    out["diagnostics"] = diag
    out["printed"] = R["printed"]
    return out


def development_lines(record: OrderedDict, dev: OrderedDict, rp: dict) -> OrderedDict:
    ""
    S = rp["readings"]["sentences"]
    out = OrderedDict()
    for c in pc.PRIMARY:
        r = dev["cells"][c]
        out[c] = fill(S["development_beside"], cell=c, dev_class=r["class"],
                      mean="-" if r.get("mean_pp") is None else f"{r['mean_pp']:+.2f}",
                      lower95="-" if r.get("lower95_pp") is None else f"{r['lower95_pp']:+.2f}",
                      p_holm="-" if r.get("p_holm") is None else fp(r["p_holm"]))
    return out


def s0_counts() -> OrderedDict:
    if pc.sha256_file(pc.S0 / "headline_result.json") != pc.S0_RESULT_JSON_SHA256:
        raise SystemExit("REFUSED: headline_v46/headline_result.json differs from its pin")
    res = pc.s0_result()
    return OrderedDict((label, res["members"][label]["correct"]) for label in pc.MEMBER_JOBS)


def w06_rows(lock_unix: float) -> list:
    ""
    rows = []
    for t in pc.SEEDS:
        inst = pc.instance_dir(t)
        view = instance_view(t, inst, lock_unix)
        rules = {j["id"]: j for j in pc.read_json(inst / "headline_rules.json")["jobs"]}
        for jid in pc.JOBS:
            if view[jid]["state"] != "ok":
                raise SystemExit(f"REFUSED: {t}/{jid} is {view[jid]['state']} ({view[jid]['status']}): the W06 rows "
                                 "need every record of the pairs ended ok")
            kind = rules[jid]["kind"]
            ck = inst / "runs" / jid / CHECKPOINT[kind]
            if not ck.exists():
                raise SystemExit(f"REFUSED: {t}/{jid}: {ck.name} is missing")
            rec = pc.read_json(inst / "runs" / jid / "result.json")
            sha = pc.sha256_file(ck)
            row = OrderedDict(track="headline_v46_pairs", seed=t, recipe_seed=pc.SEEDS[t], job=jid, kind=kind,
                              member=rules[jid].get("member"), checkpoint=f"{t}/runs/{jid}/{ck.name}",
                              checkpoint_sha256=sha, record_sha256=pc.sha256_file(inst / "runs" / jid / "result.json"),
                              e01_correct=view[jid]["count"])
            if kind == "float" and rec.get("float_final_sha256") != sha:
                raise SystemExit(f"REFUSED: {t}/{jid}: float_final.pt differs from its record's float_final_sha256")
            if kind == "qat":
                row["float_checkpoint_sha256"] = rec.get("float_checkpoint_sha256")
            rows.append(row)
    return rows


def test_views(test: dict, rows: list) -> tuple:
    ""
    if test.get("total") != pc.TEST_TOTAL or test.get("read") != "cifar100-test":
        raise SystemExit(f"REFUSED: the test-read file must hold read 'cifar100-test' and total {pc.TEST_TOTAL}")
    got = {(r.get("seed"), r.get("job")): r for r in test.get("rows", []) if r.get("track") == "headline_v46_pairs"}
    views = OrderedDict((t, OrderedDict()) for t in pc.SEEDS)
    for w in rows:
        r = got.get((w["seed"], w["job"]))
        if r is None:
            raise SystemExit(f"REFUSED: the test read holds no row for {w['seed']}/{w['job']}")
        if r.get("checkpoint_sha256") != w["checkpoint_sha256"]:
            raise SystemExit(f"REFUSED: {w['seed']}/{w['job']}: the test read's checkpoint differs from w06_rows.json")
        c = r.get("correct")
        if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= pc.TEST_TOTAL:
            raise SystemExit(f"REFUSED: {w['seed']}/{w['job']}: a test count must be an integer in 0..{pc.TEST_TOTAL}")
        views[w["seed"]][w["job"]] = OrderedDict(state="ok", count=c, float_count=None, status="ok", problem=None)
    s0 = None
    s0rows = {r.get("member"): r.get("correct") for r in test.get("s0_rows", []) if r.get("track", "headline_v46") == "headline_v46"}
    if s0rows and all(isinstance(s0rows.get(m), int) for m in pc.MEMBER_JOBS):
        s0 = OrderedDict((m, s0rows[m]) for m in pc.MEMBER_JOBS)
    return views, s0


def render(result: OrderedDict, rp: dict, header: str, beside: OrderedDict | None = None) -> str:
    total = result["total"]
    tags = list(result["diagnostics"]["members"])
    dh = ", ".join(tags)
    L = ["# Fixed-Pair Confirmation of the Headline: Result", "", header, ""]
    L += [f"## Primary Readings (Holm over H10.1 and K64, family alpha 0.05; {rp['readings']['read_words'][result['read']]})",
          "", f"| pair | class | materiality | d at {dh} (images of {ftotal(total)}) | mean (pp) | SD (pp) | "
              "one-sided 95% lower (pp) | simultaneous 97.5% lower (pp) | p one-sided | p Holm | seed 0 d (images, selected) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cellrow(c, r):
        d = ", ".join("-" if v is None else f"{v:+d}" for v in r["d_images"].values())
        g = lambda k, f: (f.format(r[k]) if r.get(k) is not None else "-")
        s0d = "-" if r.get("s0_d_images") is None else f"{r['s0_d_images']:+d}"
        return (f"| {c} | {r['class'] or 'printed'} | {r.get('materiality') or '-'} | {d} | {g('mean_pp', '{:+.2f}')} | "
                f"{g('sd_pp', '{:.2f}')} | {g('lower95_pp', '{:+.2f}')} | {g('lower975_pp', '{:+.2f}')} | "
                f"{fp(r['p_one_sided']) if r.get('p_one_sided') is not None else '-'} | "
                f"{fp(r['p_holm']) if r.get('p_holm') is not None else '-'} | {s0d} |")
    for c in pc.PRIMARY:
        L.append(cellrow(c, result["cells"][c]))
    L += [""] + [f"- {result['cells'][c]['sentence']}" for c in pc.PRIMARY]
    if beside:
        L += [f"- {v}" for v in beside.values()]
    L += ["", "## Secondary Pairs (outside the Holm family; printed, not readings)", "",
          f"| pair | class | materiality | d at {dh} (images of {ftotal(total)}) | mean (pp) | SD (pp) | "
          "one-sided 95% lower (pp) | simultaneous 97.5% lower (pp) | p one-sided | p Holm | seed 0 d (images, selected) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in pc.SECONDARY:
        L.append(cellrow(c, result["cells"][c]))
    L += [""] + [f"- {result['cells'][c]['sentence']}" for c in pc.SECONDARY]
    L += [f"- {result['diagnostics']['r1l_note']}", ""]
    L += ["## Beside the Readings (printed, never readings)", ""]
    if result.get("k64_with_s0"):
        L.append(f"- {result['k64_with_s0']['sentence']}")
    L += [f"- {result['cells'][c]['s0_sentence']}" for c in pc.PAIRS]
    L += [f"- {v['sentence']}" for v in result["diagnostics"]["per_pair"].values()]
    L += ["", "| seed | " + " | ".join(pc.JOBS) + " |", "|---|" + "---|" * len(pc.JOBS)]
    for t, row in result["diagnostics"]["members"].items():
        L.append(f"| {t} | " + " | ".join("-" if v is None else str(v) for v in row.values()) + " |")
    L += ["", f"Counts are correct images of the {ftotal(total)} images of the read (float jobs: the float phase's final "
              "count).", "", "## Printed Lines (verbatim, rules_pairs.md section 7)", ""]
    L += [f"- {v}" for v in result["printed"].values()]
    return "\n".join(L) + "\n"


def instance_verify(inst: Path) -> dict:
    out = subprocess.run([pc.PY, "-s", "-B", str(pc.HERE / "instance_probe.py"), "verify", "--instance", str(inst)],
                         capture_output=True, text=True, timeout=1800,
                         env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
    if out.returncode != 0:
        raise SystemExit(f"REFUSED: {inst.name}: its lock does not verify: {(out.stderr or out.stdout).strip()[-600:]}")
    return json.loads(out.stdout)


def locked_views():
    import make_pairs_lock

    lock = make_pairs_lock.verify()
    pc.require_locked_sources()
    views, verified = OrderedDict(), OrderedDict()
    for t in pc.SEEDS:
        inst = pc.instance_dir(t)
        v = instance_verify(inst)
        verified[t] = OrderedDict(code_sha256=v["code_sha256"], fingerprint=v["fingerprint"],
                                  locked_at=v["lock"]["locked_at"])
        views[t] = instance_view(t, inst, float(lock["locked_at_unix"]))
    return lock, views, verified


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--w06-rows", action="store_true")
    ap.add_argument("--test-read")
    args = ap.parse_args(argv)
    rp = pc.read_json(pc.RULES_JSON, ordered=True)
    s0 = s0_counts()
    if args.dry:
        res = pc.s0_result()
        views = OrderedDict()
        for t in pc.SEEDS:
            views[t] = OrderedDict()
            for j in pc.JOBS:
                if j.startswith("Q-"):
                    label = next(k for k, v in pc.MEMBER_JOBS.items() if v == j)
                    views[t][j] = OrderedDict(state="ok", count=res["members"][label]["correct"], float_count=None)
                else:
                    views[t][j] = OrderedDict(state="ok", count=None, float_count=None)
        result = read_pairs(views, s0, rp)
        d = pc.HERE / "runs_pairs" / "_dry"
        d.mkdir(parents=True, exist_ok=True)
        pc.write_json(d / "pairs_result_dry.json", result)
        (d / "pairs_result_dry.md").write_text(render(result, rp, "DRY: seed 0's counts stand in for every seed; not a "
                                                                   "result."))
        print(json.dumps({c: [r["class"], r.get("sentence_key")] for c, r in result["cells"].items()}))
        return 0
    lock, views, verified = locked_views()
    if args.w06_rows:
        rows = w06_rows(float(lock["locked_at_unix"]))
        out = OrderedDict(label=("the checkpoints of headline_v46_pairs the W06 test-read lock must list "
                                 "(rules_pairs.md section 7): 12 jobs at each of s1 to s5"),
                          written_at=pc.now_stamp(), pairs_lock_sha256=pc.sha256_file(pc.LOCK_JSON),
                          n_rows=len(rows), rows=rows)
        pc.write_json(pc.W06_ROWS_JSON, out)
        print(f"w06_rows.json: {len(rows)} rows (sha256 {pc.sha256_file(pc.W06_ROWS_JSON)})")
        return 0
    dev = read_pairs(views, s0, rp, pc.TOTAL, "e01")
    if args.test_read:
        w = pc.read_json(pc.W06_ROWS_JSON)
        if not w:
            raise SystemExit("REFUSED: w06_rows.json is missing (bash run_pairs.sh w06-rows, before the W06 lock)")
        now_rows = json.loads(json.dumps(w06_rows(float(lock["locked_at_unix"]))))
        if now_rows != w["rows"]:
            raise SystemExit("REFUSED: w06_rows.json differs from the rows the records give now")
        test = pc.read_json(Path(args.test_read))
        if not test:
            raise SystemExit(f"REFUSED: {args.test_read} is not a readable test-read file")
        tviews, ts0 = test_views(test, w["rows"])
        rec = read_pairs(tviews, ts0, rp, pc.TEST_TOTAL, "test")
        rec.update(collected_at=pc.now_stamp(), pairs_lock_sha256=pc.sha256_file(pc.LOCK_JSON),
                   w06_rows_sha256=pc.sha256_file(pc.W06_ROWS_JSON), test_read_sha256=pc.sha256_file(Path(args.test_read)),
                   test_lock_sha256=test.get("test_lock_sha256"), development=dev,
                   development_beside=development_lines(rec, dev, rp), instances=verified)
        pc.write_json(pc.HERE / "pairs_result_record.json", rec)
        header = (f"Collected {rec['collected_at']}: the classes of record on the W06 test read (test-read file sha256 "
                  f"{rec['test_read_sha256'][:16]}, its lock {str(rec['test_lock_sha256'])[:16]}), under pairs_lock.json "
                  f"(sha256 {rec['pairs_lock_sha256'][:16]}, locked {lock['locked_at']}); seeds s1 to s5; the E01 "
                  "readings (development numbers) printed beside; seed 0 printed beside, in no reading.")
        (pc.HERE / "pairs_result_record.md").write_text(render(rec, rp, header, rec["development_beside"]))
        print(json.dumps({c: r["class"] for c, r in rec["cells"].items()}))
        return 0
    dev.update(collected_at=pc.now_stamp(), pairs_lock_sha256=pc.sha256_file(pc.LOCK_JSON), instances=verified,
               views=views, class_of_record=rp["readings"]["sentences"]["record_pending"])
    pc.write_json(pc.HERE / "pairs_result.json", dev)
    header = (f"Collected {dev['collected_at']} under pairs_lock.json (sha256 {dev['pairs_lock_sha256'][:16]}, locked "
              f"{lock['locked_at']}); seeds s1 to s5 (recipe seeds 1 to 5); counts MEASURED on the 5,000 E01 images: "
              "development numbers. " + rp["readings"]["sentences"]["record_pending"])
    (pc.HERE / "pairs_result.md").write_text(render(dev, rp, header))
    print(json.dumps({c: r["class"] for c, r in dev["cells"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

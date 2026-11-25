""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cl6_common as K

RANK_GAP_PP = 2.0
BUDGET_ORDER = ("resident", "half", "eighth")


def load(points_dir: Path) -> list:
    rows = []
    for p in sorted(points_dir.glob("*/*/*/adc*/point.json")):
        member, arm, budget = p.parts[-5], p.parts[-4], p.parts[-3]
        d = json.loads(p.read_text())
        d["budget"] = budget
        d["path"] = str(p.parent)
        rows.append(d)
    return rows


_ENDPOINT = re.compile(r"\[MBH-ENDPOINT\] tuner=(\S+) target=([0-9.]+) entry=([0-9.]+) exit=([0-9.]+) budget=(\d+) "
                       r"steps_used=(\d+) engaged=(\w+) reached=(\w+)")


def training_of(d) -> dict:
    ""
    out = {}
    log = Path(d["path"]) / "point.log"
    if log.exists():
        for m in _ENDPOINT.finditer(log.read_text()):
            out = {"tuner": m.group(1), "target": float(m.group(2)), "entry": float(m.group(3)),
                   "exit": float(m.group(4)), "budget": int(m.group(5)), "steps_used": int(m.group(6)),
                   "engaged": m.group(7) == "True", "reached": m.group(8) == "True"}
    rec = Path(d["path"]) / "deployment_record.json.gz"
    if rec.exists():
        with gzip.open(rec, "rt") as f:
            ledger = json.load(f).get("adaptation_ledger") or {}
        out["record_endpoint_steps"] = ledger.get("endpoint_steps")
        out["record_total_steps"] = ledger.get("total_steps")
    return out


def count_of(d) -> int | None:
    ""
    if d.get("status") != "ok":
        return None
    m = float(d["deployed"]["metric"])
    n = int(d["denominator"])
    c = round(m * n)
    if abs(m * n - c) > 1e-6:
        raise Refusal(f"COUNT_NOT_WHOLE: {d['path']}: metric {m} x {n} = {m * n} is no whole count")
    return c


class Refusal(Exception):
    ""


def run_declarations(root: Path) -> dict:
    ""
    rec_path = root / "run_record.json"
    if not rec_path.is_file():
        raise Refusal(f"RUN_RECORD_ABSENT: {rec_path}")
    rec = json.loads(rec_path.read_text())
    lock_path = Path(rec["lock"])
    sha = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None
    if sha != rec.get("lock_sha256"):
        raise Refusal(f"LOCK_MISMATCH: {lock_path} is {str(sha)[:16]}, not the run record's "
                      f"{str(rec.get('lock_sha256'))[:16]}")
    mnist = json.loads(lock_path.read_text())["tiers"]["mnist"]
    cal, cnt = int(mnist["calibration_split"]["images"]), int(mnist["count_split"]["images"])
    n = cal + cnt
    return {"record": rec, "record_sha256": hashlib.sha256(rec_path.read_bytes()).hexdigest(),
            "label": mnist["count_split"]["label"], "count": cnt,
            "halves": {"calibration": {"n": n, "first": 0, "last": n // 2 - 1, "size": n // 2},
                       "count": {"n": n, "first": n // 2, "last": n - 1, "size": n - n // 2}}}


def guard_reading(d: dict, halves: dict) -> list:
    ""
    log = Path(d["path"]) / "guard.jsonl"
    ev = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.is_file() else []
    if not any(e.get("event") == "installed" for e in ev):
        return [f"GUARD_COUNTS_ABSENT: {d['path']}: no guard log with an install ({log.name})"]
    out = []
    if any(e.get("event") == "refused_read" for e in ev):
        out.append(f"GUARD_REFUSED_READ: {d['path']}: {sum(e.get('event') == 'refused_read' for e in ev)} refused read(s)")
    for role in ("calibration", "count"):
        served = {json.dumps({k: e.get(k) for k in ("n", "first", "last", "size")}, sort_keys=True)
                  for e in ev if e.get("event") == f"{role}_half_served"}
        want = json.dumps(halves[role], sort_keys=True)
        if served - {want} or (d.get("status") == "ok" and served != {want}):
            out.append(f"SPLIT_MISMATCH: {d['path']}: {role} half served {sorted(served)}, declared {want}")
    if not out:
        d["denominator"] = halves["count"]["size"]
    return out


def platform_mismatch(d) -> list:
    ip = d["identity_platform"]
    want = {"cores": d["cores"], "allow_scheduling": d["allow_scheduling"], "max_schedule_passes": 128,
            "readout_law": {**K.READOUT_LAW, "adc_bits": d["adc_bits"]}}
    return [k for k, v in want.items() if ip.get(k) != v]


def main() -> None:
    if not os.environ.get("CL6_OUT"):
        sys.exit("REFUSED: ROOT_UNSET: set CL6_OUT to the run's own root")
    root = Path(os.environ["CL6_OUT"]).resolve()
    try:
        summarize(root, Path(sys.argv[1]) if len(sys.argv) > 1 else root / "points")
    except Refusal as refusal:
        sys.exit(f"REFUSED: {refusal}")


def summarize(root: Path, points_dir: Path) -> None:
    decl = run_declarations(root)
    rows = load(points_dir)
    refusals = [r for d in rows for r in guard_reading(d, decl["halves"])]
    for d in rows:
        if not refusals:
            try:
                count_of(d)
            except Refusal as r:
                refusals.append(str(r))
    if refusals:
        for r in refusals:
            print(f"REFUSED: {r}", file=sys.stderr)
        raise Refusal(f"{refusals[0].split(':')[0]}: {len(refusals)} refusal(s) over {len(rows)} points; nothing written")
    denominator = decl["count"]
    fails, table = [], {}
    for d in rows:
        key = (d["member"], d["arm"], d["budget"], d["adc_bits"])
        table[key] = d
        if d.get("status") != "ok":
            fails.append(f"{d['path']}: {d.get('error_type')}: {d.get('error', '')[:200]}")
            continue
        mm = platform_mismatch(d)
        if mm:
            fails.append(f"{d['path']}: (1) identity.platform differs in {mm}")
        if d["budget"] in ("half", "eighth") and d["pass_count"] == 1:
            fails.append(f"{d['path']}: (2) pass_count 1 at the {d['budget']} budget")
        if not d["atol0_certified"]:
            fails.append(f"{d['path']}: not atol-0 certified: {d['certificates']}")
    members = sorted({k[0] for k in table})
    arms = [a for a in K.ARMS if any(k[1] == a for k in table)]
    out = {"label": decl["label"], "count_split": denominator, "run_record": str(root / "run_record.json"),
           "run_record_sha256": decl["record_sha256"], "lock_sha256": decl["record"]["lock_sha256"],
           "points": len(rows), "src_digests": sorted({d.get("src_digest") for d in rows}), "members": {}}
    samples = certificate_samples(rows)
    md = ["# CL6 Converter-Precision Demonstration: Point Tables", "",
          f"Single seed, MNIST, CPU, under the run record {decl['record_sha256'][:16]} (lock "
          f"{decl['record']['lock_sha256'][:16]}); every accuracy is a count of {denominator:,} images: "
          f"{decl['label']} (MEASURED, toolchain output on this tree).", "",
          "## Certificate Samples per Member", "",
          "| member | edge | samples per point | windows per point | points |", "|---|---|---|---|---|"]
    for (member, edge), v in sorted(samples.items()):
        md.append(f"| {member} | {edge} | {', '.join(map(str, v['samples']))} | "
                  f"{', '.join(map(str, v['windows'])) or '-'} | {v['points']} |")
    md.append("")
    for member in members:
        mout = {}
        for arm in arms:
            grid = {}
            for b in BUDGET_ORDER:
                for adc in K.ADC_SWEEP:
                    d = table.get((member, arm, b, adc))
                    grid[(b, adc)] = None if d is None else count_of(d)
            budget_spread = {}
            for adc in K.ADC_SWEEP:
                vals = [grid[(b, adc)] for b in BUDGET_ORDER]
                certified = all(table.get((member, arm, b, adc), {}).get("atol0_certified") for b in BUDGET_ORDER)
                if None in vals:
                    budget_spread[adc] = None
                    continue
                spread = max(vals) - min(vals)
                budget_spread[adc] = spread
                if certified and spread != 0:
                    fails.append(f"{member}/{arm}/adc{adc}: (3) accuracy spread across budgets {spread} != 0 "
                                 f"under atol-0 certificates ({vals})")
            adc_spread = {}
            for b in BUDGET_ORDER:
                vals = [grid[(b, adc)] for adc in K.ADC_SWEEP]
                adc_spread[b] = None if None in vals else max(vals) - min(vals)
            ref = {b: table.get((member, arm, b, K.AT_LEAST_A)) for b in BUDGET_ORDER}
            a_bits = sorted({r.get("adc_width_max") for r in ref.values() if r and r.get("status") == "ok"})
            passes = {b: {adc: table.get((member, arm, b, adc), {}).get("pass_count") for adc in K.ADC_SWEEP}
                      for b in BUDGET_ORDER}
            reprog = {b: {adc: table.get((member, arm, b, adc), {}).get("reprogram_passes") for adc in K.ADC_SWEEP}
                      for b in BUDGET_ORDER}


            period = {b: {adc: period_census_of(table.get((member, arm, b, adc))) for adc in K.ADC_SWEEP}
                      for b in BUDGET_ORDER}
            ir_digests = {adc: sorted({table.get((member, arm, b, adc), {}).get("ir_function_digest")
                                       for b in BUDGET_ORDER
                                       if table.get((member, arm, b, adc), {}).get("ir_digest_version") == 2}
                                      - {None}) for adc in K.ADC_SWEEP}
            wq = sorted({d.get("wq_model_digest") for k, d in table.items() if k[0] == member and k[1] == arm}
                        - {None})
            tuning = sorted({json.dumps(training_of(d), sort_keys=True)
                             for k, d in table.items() if k[0] == member and k[1] == arm and d.get("status") == "ok"})
            mout[arm] = {"counts": {f"{b}/adc{adc}": v for (b, adc), v in grid.items()},
                         "budget_spread": budget_spread, "adc_spread": adc_spread, "A_bits": a_bits,
                         "pass_count": passes, "reprogram_passes_legacy_field": reprog,
                         "period_census_k1": period, "ir_function_digests": ir_digests,
                         "wq_model_digests": wq, "training_after_start": [json.loads(t) for t in tuning],
                         "cores": {b: next((d["cores"] for k, d in table.items() if k[:3] == (member, arm, b)), None)
                                   for b in BUDGET_ORDER}}
            md += [f"## {member}, {arm} arm", "",
                   f"A (widest accumulator of the program, bits, from the stamped laws at adc_bits 32): {a_bits}; "
                   f"training after ReadoutLawStep: {[json.loads(t) for t in tuning]}", "",
                   "| adc_bits | " + " | ".join(f"{b} (count)" for b in BUDGET_ORDER) +
                   " | spread across budgets | passes (resident / half / eighth) | written passes and bytes per period "
                   "at K = 1 (resident / half / eighth) |",
                   "|---|" + "---|" * (len(BUDGET_ORDER) + 3)]
            for adc in K.ADC_SWEEP:
                label = f"{adc} (at least A)" if adc == K.AT_LEAST_A else str(adc)
                md.append(f"| {label} | " + " | ".join(str(grid[(b, adc)]) for b in BUDGET_ORDER) +
                          f" | {budget_spread[adc]} | " + " / ".join(str(passes[b][adc]) for b in BUDGET_ORDER) + " | "
                          + " / ".join("-" if period[b][adc] is None else
                                       f"{period[b][adc]['written_passes']}, {period[b][adc]['bytes']:,} B"
                                       for b in BUDGET_ORDER) + " |")
            md += ["", "| budget | cores per type | spread across adc_bits (count) |", "|---|---|---|"]
            for b in BUDGET_ORDER:
                md.append(f"| {b} | {mout[arm]['cores'][b] and [c['count'] for c in mout[arm]['cores'][b]]} | "
                          f"{adc_spread[b]} |")
            md.append("")
        out["members"][member] = mout
    ranking = []
    if len(members) == 2:
        a, b = members
        for arm in arms:
            for bud in BUDGET_ORDER:
                ref_a = out["members"][a][arm]["counts"].get(f"{bud}/adc{K.AT_LEAST_A}")
                ref_b = out["members"][b][arm]["counts"].get(f"{bud}/adc{K.AT_LEAST_A}")
                if ref_a is None or ref_b is None:
                    continue
                ref_order = ref_a >= ref_b
                for adc in K.ADC_SWEEP:
                    ca = out["members"][a][arm]["counts"].get(f"{bud}/adc{adc}")
                    cb = out["members"][b][arm]["counts"].get(f"{bud}/adc{adc}")
                    if ca is None or cb is None:
                        continue
                    gap_pp = abs(ca - cb) / denominator * 100
                    changed = (ca >= cb) != ref_order
                    verdict = ("no change" if not changed else
                               "ranking change (gap at least 2 pp)" if gap_pp >= RANK_GAP_PP else
                               "within seed noise at n = 1")
                    ranking.append({"arm": arm, "budget": bud, "adc_bits": adc, a: ca, b: cb,
                                    "gap_pp": round(gap_pp, 2), "verdict": verdict})
        md += ["## Member Ranking against adc_bits", "",
               f"Reference order at adc_bits 32; a change counts at a gap of at least {RANK_GAP_PP} pp, a smaller "
               "one is printed \"within seed noise at n = 1\".", "",
               f"| arm | budget | adc_bits | {a} | {b} | gap (pp) | reading |", "|---|---|---|---|---|---|---|"]
        for r in ranking:
            md.append(f"| {r['arm']} | {r['budget']} | {r['adc_bits']} | {r[a]} | {r[b]} | {r['gap_pp']} | "
                      f"{r['verdict']} |")
        md.append("")
    out["ranking"] = ranking
    out["certificate_samples"] = {f"{m}|{e}": v for (m, e), v in sorted(samples.items())}
    out["invariance"] = invariance(out, members, arms, denominator)
    md += ["## Invariance over Chips (W24)", "", out["invariance"]["statement"], ""]
    out["failures"] = fails
    out["pass"] = not fails
    md += ["## Fail Conditions", "", "PASS" if not fails else "FAIL", ""] + [f"- {f}" for f in fails]
    K.write_json(root / "summary.json", out)
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"points": len(rows), "pass": not fails, "failures": fails[:20]}, indent=1))


def period_census_of(d) -> "dict | None":
    ""
    if not d or d.get("status") != "ok":
        return None
    rec_path = Path(d["path"]) / "deployment_record.json.gz"
    if not rec_path.exists():
        return None
    import tempfile
    from mimarsinan.deployment_record.build.program_census import record_program_passes
    from mimarsinan.deployment_record.build.write_census import core_capacity_of, period_census
    from mimarsinan.deployment_record.schema import load_deployment_record
    with gzip.open(rec_path, "rt") as fh, tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as out:
        out.write(fh.read())
    rec = load_deployment_record(out.name)
    Path(out.name).unlink()
    plat = dict(rec.identity.platform)
    census = period_census(record_program_passes(rec), weight_bits=int(plat.get("weight_bits", 8)),
                           capacity=core_capacity_of(plat)).census
    return {"written_passes": sum(1 for p in census.passes if not p.resident),
            "bytes": int(sum(p.payload_bytes for p in census.passes))}


def certificate_samples(rows) -> dict:
    ""
    out: dict = {}
    for d in rows:
        for c in d.get("certificates") or []:
            v = out.setdefault((d["member"], c.get("edge")), {"samples": set(), "windows": set(), "points": 0})
            v["samples"].add(c.get("samples"))
            if c.get("windows") is not None:
                v["windows"].add(c["windows"])
            v["points"] += 1
    return {k: {"samples": sorted(v["samples"] - {None}), "windows": sorted(v["windows"]), "points": v["points"]}
            for k, v in out.items()}


def invariance(out: dict, members, arms, denominator: int) -> dict:
    ""
    cells = [(m, a, adc, s) for m in members for a in arms
             for adc, s in out["members"][m][a]["budget_spread"].items()]
    zero = [c for c in cells if c[3] == 0]
    spread = {f"{m}/{a}": out["members"][m][a]["adc_spread"] for m in members for a in arms}
    statement = (
        "Under the atol-0 certificate (model = identity on the R edge, identity = packed on the C edge), the deployed "
        "function depends only on the readout law, the operand grids and the weight bits: chips that agree on these "
        "compute one function at every core budget, so a design point needs no adaptation of the trained network "
        "beyond them. Conditions: the certificate holds at every point (both edges, max |delta| 0) and the point "
        "declares the same readout law, adc_bits and weight bits. Evidence: the accuracy spread across the three "
        f"core budgets is exactly 0 in {len(zero)} of {len(cells)} (member, arm, adc_bits) cells (MEASURED, counts of "
        f"{denominator:,} count-half images), against a spread across adc_bits per budget of {spread}.")
    return {"statement": statement, "cells_zero_spread": len(zero), "cells": len(cells),
            "conditions": ["atol-0 certificate on both value edges at every point",
                           "the same readout law, adc_bits and weight bits"], "class": "MEASURED"}


if __name__ == "__main__":
    main()

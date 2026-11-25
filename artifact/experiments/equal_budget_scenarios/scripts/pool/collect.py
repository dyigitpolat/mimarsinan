""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gpu_plan
import price_measured as PM
import reuse
import screen_common as sc

TOTAL = 5000
LOW_COUNT = 250
INCOMPLETE = ("refused", "crashed", "teacher_unavailable", "float_unavailable")


def checked_count(job: str, rec: dict, key: str = "qat_val_correct") -> int:
    ""
    c, total = rec.get(key), rec.get("val_total")
    if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= TOTAL or total != TOTAL:
        raise SystemExit(f"REFUSED: {job}: a counted record needs an integer {key} in 0..{TOTAL} and val_total "
                         f"{TOTAL}; got {c!r} and {total!r}")
    return c


def foreign_record(rec: dict, lock: dict) -> bool:
    ""
    d = rec.get("digests")
    if d is None:
        return rec.get("status") not in INCOMPLETE
    return d != lock


def member_states(rules: dict, lock: dict, decision: dict, runs: Path = sc.RUNS) -> tuple[OrderedDict, list]:
    ""
    statuses, foreign = {}, []
    for j in rules["jobs"]:
        p = runs / j["id"] / "result.json"
        if not p.exists():
            statuses[j["id"]] = None
            continue
        r = json.loads(p.read_text())
        if foreign_record(r, lock):
            foreign.append(j["id"])
            statuses[j["id"]] = None
            continue
        statuses[j["id"]] = r
    qstate = gpu_plan.queue_state(rules, {k: (v or {}).get("status") for k, v in statuses.items()},
                                  set(decision["reused_jobs"]))
    out = OrderedDict()
    reused_vals = decision.get("reused_values", {})
    for m in rules["members"]:
        job = m["job"]
        st = qstate[job]
        row = OrderedDict(job=job, arch=m["arch"], bits="b" + "".join(map(str, m["bits"])), status=st)
        if st == "reused":
            v = reused_vals[job]
            row.update(state="measured", correct=checked_count(job, v), float_correct=v.get("float_val_correct"),
                       source=v["source"])
        elif st == "ok":
            r = statuses[job]
            row.update(state="measured", correct=checked_count(job, r), float_correct=r.get("float_val_correct"),
                       source=f"runs/{job}/result.json")
        elif st.startswith("failed_") or st == "blocked":
            dep = next((d for d in next(j for j in rules["jobs"] if j["id"] == job)["depends_on"]), None)
            why = st if st != "blocked" else f"blocked: float job {dep} ended {statuses[dep]['status']}"
            row.update(state="missing", correct=None, float_correct=None, source=why)
        else:
            row.update(state="pending", correct=None, float_correct=None, source=st)
        if row.get("correct") is not None and row["correct"] < LOW_COUNT:
            row["flag"] = "below 5% (5x chance): a measurement, flagged"
        out[m["label"]] = row
    return out, foreign


def measured_table(states: OrderedDict) -> dict:
    return {"kind": "measured", "label": "MEASURED single seed s0, E01 (5,000 images)", "total": TOTAL,
            "correct": {lab: row["correct"] for lab, row in states.items()}}


def classify(gap_pp, thresholds: dict, verbatim: dict) -> str:
    if gap_pp is None:
        return verbatim["not_determinable"]
    g = round(float(gap_pp), 9)
    if g >= float(thresholds["holds"]):
        return verbatim["holds"]
    if g < float(thresholds["at_risk_below"]):
        return verbatim["at_risk"]
    return verbatim["inconclusive"]


def band_keys(rules: dict) -> tuple[str, str]:
    ""
    return PM.skey(float(rules["pricing"]["convention"])), PM.skey(float(rules["pricing"]["band"][0]))


def gaps_of(cell: dict, conv: str, s1: str) -> OrderedDict:
    ""
    b = cell["robust_gap_band"]
    conv_images = [p["gap_images"] for p in cell["by_s_au"][conv]["per_combo"] if p.get("gap_images") is not None]
    return OrderedDict(
        robust_gap_convention_pp=cell["robust_gap_convention"],
        robust_gap_convention_images=(min(conv_images) if (conv_images and cell["robust_gap_convention"] is not None)
                                      else None),
        band_minimum_pp=b["gap"] if b else None,
        s_au_1_pp=cell["by_s_au"][s1]["robust_gap_x10"],
        declared_read_gap_pp=cell["declared_read"].get("gap"))


def readings(rules: dict, measured: OrderedDict, priors: OrderedDict, states: OrderedDict) -> OrderedDict:
    r = rules["readings"]
    conv, s1 = band_keys(rules)
    out = OrderedDict()
    cls = rules["weight_quantizer_class"]["declared"]
    other = rules["weight_quantizer_class"]["other_class_printed"]
    for cname in r["cells"]:
        c = measured["cells"][cname]
        need = c["by_s_au"][conv]["missing_needed"]
        missing = [lab for lab in need if states[lab]["state"] == "missing"]
        pending = [lab for lab in need if states[lab]["state"] == "pending"]
        g = gaps_of(c, conv, s1)
        if missing:
            reading = r["verbatim"]["not_determinable"]
        elif pending:
            reading = "pending"
        else:
            reading = classify(g["robust_gap_convention_pp"], r["thresholds_pp"], r["verbatim"])
        prior = OrderedDict()
        for name in (cls, other):
            gp = gaps_of(priors[name]["cells"][cname], conv, s1)
            gp["reading"] = classify(gp["robust_gap_convention_pp"], r["thresholds_pp"], r["verbatim"])
            prior[name] = gp
        out[cname] = OrderedDict(reading=reading, quantity=r["quantity"], **g, declared_read=c["declared_read"],
                                 missing_needed=missing, pending_needed=pending,
                                 needed_beyond_the_convention=[lab for lab in c["missing_needed"] if lab not in need],
                                 prior=prior)
    return out


def reading_combo(cell: dict, conv: str):
    ""
    b = cell["by_s_au"][conv]
    g = cell["robust_gap_convention"]
    if g is None:
        return None
    return next(k for k, p in enumerate(b["per_combo"]) if p.get("gap_x10") == g)


def rivals(row: dict, acc: list, labels: list, within_pp: float) -> OrderedDict:
    ""
    out = OrderedDict()
    for key in ("J", "FO"):
        best = PM.pick(row[key], acc)
        near = [labels[i] for i in row[key] if i != best and round(acc[best] - acc[i], 9) <= within_pp]
        out[key] = OrderedDict(pick=labels[best], set_size=len(row[key]), within=len(near), members=near)
    return out


def class_shares(rows: list, acc: list, rules: dict) -> OrderedDict | None:
    ""
    d = rules["readings"]["diagnostics"]["perturbation"]
    th = rules["readings"]["thresholds_pp"]
    rows = [r for r in rows if r["J"] and r["FO"]]
    need = sorted({i for r in rows for i in r["J"] + r["FO"]})
    if not rows or any(acc[i] is None for i in need):
        return None
    base = np.array([np.nan if a is None else float(a) for a in acc])
    rng = np.random.default_rng(int(d["seed"]))
    noisy = base[None, :] + rng.normal(0.0, float(d["sigma_pp"]), size=(int(d["draws"]), len(acc)))
    gap = np.stack([noisy[:, r["J"]].max(axis=1) - noisy[:, r["FO"]].max(axis=1) for r in rows], axis=1).min(axis=1)
    holds = float(np.mean(gap >= float(th["holds"])))
    at_risk = float(np.mean(gap < float(th["at_risk_below"])))
    return OrderedDict(holds=round(holds, 3), inconclusive=round(1.0 - holds - at_risk, 3), at_risk=round(at_risk, 3),
                       sigma_pp=float(d["sigma_pp"]), draws=int(d["draws"]), seed=int(d["seed"]))


def two_bit_members(rules: dict) -> list[str]:
    return [m["label"] for m in rules["members"] if 2 in m["bits"]]


def diagnostics(rules: dict, masks: OrderedDict, tables: OrderedDict, reads: OrderedDict) -> OrderedDict:
    ""
    conv, _ = band_keys(rules)
    within = float(rules["readings"]["diagnostics"]["rivals_within_pp"])
    labels = masks["labels"]
    excl = two_bit_members(rules)
    out = OrderedDict()
    for name, table in tables.items():
        acc, _, _ = PM.table_vector(table, labels)
        no2 = PM.read_cells(masks, table, exclude=excl)
        for cname in rules["readings"]["cells"]:
            cell = reads[name]["cells"][cname]
            k = reading_combo(cell, conv)
            rows = masks["cells"][cname]["by_s_au"][conv]
            e = OrderedDict(table=name)
            if k is None:
                e.update(reading_combo=None, rivals=None)
            else:
                e.update(reading_combo=rows[k]["combo"], rivals=rivals(rows[k], acc, labels, within))
            e["convention_gap_without_2bit_pp"] = no2[cname]["robust_gap_convention"]
            e["class_shares_under_perturbation"] = class_shares(rows, acc, rules)
            out.setdefault(cname, OrderedDict())[name] = e
    return out


def fmt(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def combo_txt(cb) -> str:
    if cb is None:
        return "-"
    f, a, ap = cb
    return f"f {f:g}, {PM.AREA_MODELS[int(a)]}, a_p {ap:g}"


def write_markdown(rules: dict, states: OrderedDict, rd: OrderedDict, diag: OrderedDict, measured: OrderedDict,
                   priors: OrderedDict, decision: dict, foreign: list, lock: dict, stamp: str) -> str:
    cls = rules["weight_quantizer_class"]["declared"]
    other = rules["weight_quantizer_class"]["other_class_printed"]
    d = rules["readings"]["diagnostics"]
    prior_acc = {sc_: dict(zip(priors["_labels"], v)) for sc_, v in priors["_vectors"].items()}
    L = [f"# Headline check result (headline_v46)", "",
         f"Collected {stamp} from records under headline_lock.json (locked {lock.get('_locked_at', '-')}; code "
         f"{lock['code_sha256'][:16]}, rules md {lock['rules_md_sha256'][:16]}). Single seed s0; accuracies are "
         f"MEASURED integer correct counts of the 5,000 E01 images; priors are DECLARED (cd_family.prior); gaps are "
         f"the v4.5 estimators' admissibility read with the table named. Weight quantizer class of this check: {cls} "
         f"(its prior is printed beside every reading; the {other} prior beside it).", "",
         f"**Single-seed caveat.** {rules['readings']['noise_caveat']}.", "",
         "## Declared readings", "",
         f"Rule (verbatim, headline_rules.md section 7): {rules['readings']['rule']}", "",
         f"{rules['readings']['decision'][0].upper()}{rules['readings']['decision'][1:]}.", "",
         f"Reading quantity: {rules['readings']['quantity']}. Beside it, as codesign.md's E313 paragraph prints them: "
         f"the band minimum over s_AU {{1, 1.069, 1.130}} and the s_AU 1 value.", "",
         f"| cell | reading (measured, one seed) | robust gap at the convention (pp) | band minimum (pp) | s_AU 1 (pp) | "
         f"declared read: J* - FO* (pp) | prior {cls}: convention [band minimum; s_AU 1], reading | prior {other}: "
         f"convention [band minimum; s_AU 1], reading |",
         "|---|---|---|---|---|---|---|---|"]
    for cname, v in rd.items():
        dr = v["declared_read"]
        drs = (f"{dr.get('J')} - {dr.get('FO')} = {fmt(dr.get('gap'))}" if dr.get("gap") is not None
               else f"{dr.get('J')} / {dr.get('FO')}")
        pc, po = v["prior"][cls], v["prior"][other]

        def pr(p):
            return (f"{fmt(p['robust_gap_convention_pp'], 4)} [{fmt(p['band_minimum_pp'], 4)}; "
                    f"{fmt(p['s_au_1_pp'], 4)}], {p['reading']}")
        L.append(f"| {cname} | {v['reading']} | {fmt(v['robust_gap_convention_pp'])} | {fmt(v['band_minimum_pp'])} | "
                 f"{fmt(v['s_au_1_pp'])} | {drs} | {pr(pc)} | {pr(po)} |")
    notes = [f"{c}: J* or FO* at the convention needs the missing members {v['missing_needed']}" for c, v in rd.items()
             if v["missing_needed"]]
    notes += [f"{c}: J* or FO* at the convention needs the pending members {v['pending_needed']}" for c, v in rd.items()
              if v["pending_needed"]]
    notes += [f"{c}: the band minimum and s_AU 1 values also need {v['needed_beyond_the_convention']} (printed '-' "
              f"until they have final counts; the reading does not need them)" for c, v in rd.items()
              if v["needed_beyond_the_convention"]]
    L += ["", "Notes: " + ("; ".join(notes) if notes else "none") + ".", ""]
    L += ["## Diagnostics beside the readings", "",
          f"Not readings (headline_rules.md section 7): at the reading's combination (the convention combination with "
          f"the smallest gap), the J members other than J* within {d['rivals_within_pp']:g} pp of J* and the FO "
          f"members other than FO* within {d['rivals_within_pp']:g} pp of FO* (of the set's size); the robust gap at "
          f"the convention with every member that has a 2-bit stage excluded from both sides "
          f"({len(two_bit_members(rules))} members; the 2-bit grid has 3 levels); and the shares of the three classes "
          f"when every member's accuracy is perturbed by a Gaussian of sigma {d['perturbation']['sigma_pp']:g} pp "
          f"({d['perturbation']['draws']:,} draws, seed {d['perturbation']['seed']}; J* and FO* re-picked per draw), "
          f"the table taken as the truth.", "",
          "| cell | table | reading combination | J within 1.0 pp of J* | FO within 1.0 pp of FO* | convention gap "
          "without 2-bit members (pp) | holds / inconclusive / at risk under the perturbation |",
          "|---|---|---|---|---|---|---|"]
    for cname, per in diag.items():
        for name, e in per.items():
            rv = e["rivals"]
            jt = "-" if rv is None else f"{rv['J']['within']} of {rv['J']['set_size'] - 1}"
            ft = "-" if rv is None else f"{rv['FO']['within']} of {rv['FO']['set_size'] - 1}"
            s = e["class_shares_under_perturbation"]
            st = "-" if s is None else f"{s['holds']:.3f} / {s['inconclusive']:.3f} / {s['at_risk']:.3f}"
            L.append(f"| {cname} | {name} | {combo_txt(e['reading_combo'])} | {jt} | {ft} | "
                     f"{fmt(e['convention_gap_without_2bit_pp'], 4)} | {st} |")
    L += ["", "## Measured and prior gaps per cell", "",
          "Per cell and band value: the robust gap over the cell's 8 combinations, J* and FO* (sets over the "
          "combinations), J* resident in how many combinations.", "",
          "| cell | s_AU | table | robust gap (pp) | J* | FO* | J* resident |", "|---|---|---|---|---|---|---|"]
    for cname in rules["readings"]["cells"]:
        for s in measured["cells"][cname]["by_s_au"]:
            for name, t in (("measured", measured), (cls, priors[cls]), (other, priors[other])):
                b = t["cells"][cname]["by_s_au"][s]
                L.append(f"| {cname} | {s} | {name} | {fmt(b['robust_gap_x10'], 4)} | {', '.join(b['J_x10'])} | "
                         f"{', '.join(b['FO_x10'])} | {b['J_resident_combos']} of {b['n_combos']} |")
    L += ["", "## Declared read: picks and residency", "",
          "| cell | table | J* (count) | J* design | FO* (count) | FO* design |", "|---|---|---|---|---|---|"]

    def des(dd):
        if not dd:
            return "-"
        return f"{dd['geometry']} C{dd['C']} F{dd['F']} Q{dd['Q']} {'resident' if dd['resident'] else 'multi-pass'}"
    for cname in rules["readings"]["cells"]:
        for name, t in (("measured", measured), (cls, priors[cls])):
            dr = t["cells"][cname]["declared_read"]
            L.append(f"| {cname} | {name} | {dr.get('J')} ({dr.get('J_correct', '-')}) | {des(dr.get('J_design'))} | "
                     f"{dr.get('FO')} ({dr.get('FO_correct', '-')}) | {des(dr.get('FO_design'))} |")
    L += ["", "## Rule RR on the measured table", "",
          "| ladder | table | declared cell | smallest regret (pp) | qualifying cells | not determinable |",
          "|---|---|---|---|---|---|"]
    for ld in ("declared", "check"):
        for name, t in (("measured", measured), (cls, priors[cls]), (other, priors[other])):
            x = t["rule_rr"][ld]["table"]
            L.append(f"| {ld} | {name} | {x['declared']} | {x['declared_min_regret']} | {len(x['qualifying'])} | "
                     f"{', '.join(x.get('not_determinable_cells', [])) or '-'} |")
    L += ["", "## Per-member accuracies", "",
          "Float: the member's float starting point (correct of 5,000 on E01; R18 the screen's checkpoints, R10 and "
          "R34 this directory's float jobs). Every member distils from the screen's R18 w1 float teacher (75.36%).", "",
          f"| member | job | state | correct of 5,000 | top-1 (%) | float (of 5,000) | prior {cls} (%) | prior {other} "
          f"(%) | source |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, row in states.items():
        c = row["correct"]
        fc = row.get("float_correct")
        L.append(f"| {lab} | {row['job']} | {row['state']}{' (' + row['flag'] + ')' if row.get('flag') else ''} | "
                 f"{'-' if c is None else f'{c:,}'} | {'-' if c is None else f'{100 * c / TOTAL:.2f}'} | "
                 f"{'-' if fc is None else f'{fc:,}'} | {prior_acc[cls][lab]:.2f} | {prior_acc[other][lab]:.2f} | "
                 f"{row['source']} |")
    L += ["", "## Run records", "",
          f"- Reuse decision: {decision['decision']} (reused jobs {decision['reused_jobs']}).",
          f"- Foreign records refused (not under headline_lock.json): {foreign or 'none'}.",
          f"- Member states: " + ", ".join(f"{k} {v}" for k, v in sorted(
              {s: sum(1 for r in states.values() if r['state'] == s) for s in ('measured', 'missing', 'pending')}.items())),
          ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rules-sha256", default=None,
                   help="the launcher's rules hash (default: RULES_SHA256 of run_headline.sh)")
    args = p.parse_args(argv)
    sc.verify_lock(args.rules_sha256 or sc.launcher_rules_sha256())
    rules = sc.load_rules()
    lock = sc.locked_digests()
    masks = json.loads(sc.MASKS_JSON.read_text(), object_pairs_hook=OrderedDict)
    decision = reuse.verified_decision(rules)
    states, foreign = member_states(rules, lock, decision)
    table = measured_table(states)
    tables = OrderedDict([("measured", table)])
    tables.update(PM.prior_tables(masks))
    reads = OrderedDict((k, PM.read(masks, t)) for k, t in tables.items())
    measured = reads["measured"]
    priors = OrderedDict((k, v) for k, v in reads.items() if k != "measured")
    rd = readings(rules, measured, priors, states)
    diag = diagnostics(rules, masks, tables, reads)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    priors_md = OrderedDict(priors)
    priors_md["_labels"] = masks["labels"]
    priors_md["_vectors"] = masks["priors"]
    lock_md = dict(lock, _locked_at=sc.read_lock().get("locked_at"))
    md = write_markdown(rules, states, rd, diag, measured, priors_md, decision, foreign, lock_md, stamp)
    (HERE / "headline_result.md").write_text(md)
    out = OrderedDict(collected_at=stamp, lock=lock, reuse=decision, members=states, measured_table=table,
                      readings=rd, diagnostics=diag, measured=measured, foreign=foreign)
    (HERE / "headline_result.json").write_text(json.dumps(out, indent=1, sort_keys=False))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

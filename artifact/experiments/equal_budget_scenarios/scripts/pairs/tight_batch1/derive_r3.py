""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PT = HERE.parent
RA = PT.parent.parent
RR = RA / "12_tcad_readiness" / "tiers" / "paper_v6" / "fix5" / "study_build" / "reread"
SEEDS = ("s1", "s2", "s3", "s4", "s5")
TEST_TOTAL, E01_TOTAL = 10000, 5000
CONF = 0.95


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = load_module(PT / "pairs_stats.py", "pairs_stats_r3")
mk = load_module(HERE / "make_lock_r3.py", "make_lock_r3")


def read_json(p: Path):
    return json.loads(p.read_text(), object_pairs_hook=OrderedDict)


def spp(images: float, total: int) -> str:
    return f"{100.0 * images / total:+.2f}"


def fp(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def count_table(members: dict, total: int, jobs: tuple) -> OrderedDict:
    ""
    out = OrderedDict()
    for t in SEEDS:
        if t not in members:
            raise SystemExit(f"REFUSED: no counts at {t}")
        out[t] = OrderedDict()
        for j in jobs:
            c = members[t].get(j)
            if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= total:
                raise SystemExit(f"REFUSED: {t}/{j}: a count must be an integer in 0..{total}, got {c!r}")
            out[t][j] = c
    return out


def series(counts: dict, j_job: str, fo_job: str) -> list:
    ""
    return [counts[t][j_job] - counts[t][fo_job] for t in SEEDS]


def contrast(ds: list, total: int) -> OrderedDict:
    ""
    a = st.paired(ds, total, CONF)
    return OrderedDict(d_images=a["d_images"], sum_images=a["sum_images"], n=a["n"], df=a["df"],
                       mean_pp=round(a["mean_pp"], 4), sd_pp=round(a["sd_pp"], 4), se_pp=round(a["se_pp"], 4),
                       lower95_pp=round(a["lower_pp"], 4), t=a["t_stat"], t_quantile=a["t_quantile"],
                       p_one_sided=a["p"], mean_images=a["mean_images"], lower_images=a["lower_images"],
                       sd_images=a["sd_images"])


def check_pairs_of_record(counts: dict, rec: dict, rp2: dict) -> list:
    ""
    out = []
    for c, p in rp2["pairs"].items():
        ds = series(counts, p["J_job"], p["FO_job"])
        got = [rec["cells"][c]["d_images"][t] for t in SEEDS]
        if ds != got:
            raise SystemExit(f"REFUSED: {c}: the counts give {ds}, the record's d_images {got}")
        out.append(f"{c}: {ds} equal the record's d_images")
    return out


def s0_test_counts(rec: dict, labels: list) -> OrderedDict:
    ""
    found = OrderedDict()
    for c, row in rec["cells"].items():
        s = row["s0_sentence"]
        for lab in (row["J"], row["FO"]):
            m = re.search(re.escape(lab) + r" (\d+)", s)
            if not m:
                raise SystemExit(f"REFUSED: {c}: no seed-0 count of {lab} in its sentence")
            v = int(m.group(1))
            if found.get(lab, v) != v:
                raise SystemExit(f"REFUSED: two seed-0 counts of {lab}")
            found[lab] = v
        if found[row["J"]] - found[row["FO"]] != row["s0_d_images"]:
            raise SystemExit(f"REFUSED: {c}: the parsed seed-0 counts do not give its s0_d_images")
    return OrderedDict((lab, found[lab]) for lab in labels)


def fill(template: str, **kw) -> str:
    return template.format(**kw)


def main() -> int:
    if mk.verify() != 0:
        raise SystemExit("REFUSED: lock_r3.json does not verify")
    lock = read_json(HERE / "lock_r3.json")
    rv = read_json(HERE / "rules_pairs_r3.json")
    sel = read_json(HERE / "selections_r3.json")
    rp2 = read_json(PT / "rules_pairs.json")
    rec = read_json(PT / "pairs_result_record.json")
    w06 = read_json(PT / "w06_rows.json")
    fc = rv["fourth_contrast"]
    jj, fj = fc["J_job"], fc["FO_job"]
    checks = []

    if rec.get("read") != "test" or rec.get("total") != TEST_TOTAL or rec.get("n_seeds") != 5:
        raise SystemExit("REFUSED: pairs_result_record.json is not the test read of 10,000 images at five seeds")
    checks.append("pairs_result_record.json: read 'test', total 10,000, five seeds")
    if rec["pairs_lock_sha256"] != lock["files_sha256"]["pairs_lock.json"]:
        raise SystemExit("REFUSED: the record does not pin pairs_lock.json")
    if rec["w06_rows_sha256"] != lock["files_sha256"]["w06_rows.json"]:
        raise SystemExit("REFUSED: the record does not pin w06_rows.json")
    checks.append("pairs_result_record.json pins pairs_lock.json and w06_rows.json at their locked digests")
    listed = {(r["seed"], r["job"]): r for r in w06["rows"]}
    for t in SEEDS:
        for j in (jj, fj):
            if (t, j) not in listed:
                raise SystemExit(f"REFUSED: w06_rows.json lists no {t}/{j}")
    checks.append(f"w06_rows.json lists {jj} and {fj} at each of s1 to s5 ({len(w06['rows'])} rows)")
    all_jobs = tuple(rec["diagnostics"]["members"]["s1"].keys())
    test_counts = count_table(rec["diagnostics"]["members"], TEST_TOTAL, all_jobs)
    checks.append(f"every test count of the 60 rows is an integer in 0 to 10,000 ({len(all_jobs)} jobs at 5 seeds)")
    checks += check_pairs_of_record(test_counts, rec, rp2)

    e01 = OrderedDict((t, OrderedDict((j, listed[(t, j)]["e01_correct"]) for j in all_jobs)) for t in SEEDS)
    e01 = count_table(e01, E01_TOTAL, all_jobs)
    dev_members = rec["development"]["diagnostics"]["members"]
    for t in SEEDS:
        for j in all_jobs:
            if dev_members[t][j] != e01[t][j]:
                raise SystemExit(f"REFUSED: {t}/{j}: w06_rows.json's E01 count differs from the record's development")
    checks.append("w06_rows.json's E01 counts equal pairs_result_record.json development.diagnostics.members")

    ds = series(test_counts, jj, fj)
    test = contrast(ds, TEST_TOTAL)
    de = series(e01, jj, fj)
    dev = contrast(de, E01_TOTAL)
    s0t = s0_test_counts(rec, [fc["J"], fc["FO"]])
    d0 = s0t[fc["J"]] - s0t[fc["FO"]]
    J0e, FO0e = sel["fourth_contrast"]["s0_dev_J"], sel["fourth_contrast"]["s0_dev_FO"]
    d0e = J0e - FO0e
    r1l = sel["cells"]["R1L"]
    S = fc["sentences"]
    read_words = rp2["readings"]["read_words"]
    w = dict(J=fc["J"], FO=fc["FO"], J_passes=fc["J_design"]["passes"], J_geometry=fc["J_design"]["geometry"])
    dn = {f"d{k + 1}": f"{x:+d}" for k, x in enumerate(ds)}
    dne = {f"d{k + 1}": f"{x:+d}" for k, x in enumerate(de)}
    sentences = OrderedDict(
        record=fill(S["record"], **w, read=read_words["test"], total=f"{TEST_TOTAL:,}",
                    mean=spp(test["mean_images"], TEST_TOTAL), sd=f"{test['sd_pp']:.2f}",
                    lower95=spp(test["lower_images"], TEST_TOTAL), p=fp(test["p_one_sided"]), **dn),
        s0_beside=fill(S["s0_beside"], J=fc["J"], FO=fc["FO"], J0=s0t[fc["J"]], FO0=s0t[fc["FO"]],
                       total=f"{TEST_TOTAL:,}", read=read_words["test"], d0=f"{d0:+d}", d0pp=spp(d0, TEST_TOTAL),
                       J0e=f"{J0e:,}", FO0e=f"{FO0e:,}", d0epp=spp(d0e, E01_TOTAL)),
        development_beside=fill(S["development_beside"], mean=spp(dev["mean_images"], E01_TOTAL),
                                sd=f"{dev['sd_pp']:.2f}", lower95=spp(dev["lower_images"], E01_TOTAL),
                                p=fp(dev["p_one_sided"]), **dne),
        cell_gain_note=fill(S["cell_gain_note"], gain=f"{r1l['gain_at_convention_pp']:.2f}",
                            FOfig=r1l["figure_setting"]["single"][1], d0epp=spp(d0e, E01_TOTAL)),
    )

    cell = (fc["paper"]["table_vi_cell"].replace("{mean}", spp(test["mean_images"], TEST_TOTAL))
            .replace("{lower95}", spp(test["lower_images"], TEST_TOTAL)))
    note = fc["paper"]["note"]
    n_cell, n_note = len(cell.split()), len(note.split())
    if n_cell > fc["paper"]["table_vi_cell_words_max"] or n_note > fc["paper"]["note_words_max"]:
        raise SystemExit(f"REFUSED: the paper's line is too long ({n_cell} and {n_note} words)")
    when = subprocess.run(["date", "-Is"], capture_output=True, text=True, check=True).stdout.strip()
    out = OrderedDict(
        label=("the fourth contrast of headline_v46_pairs revision 3: the tight batch-1 cell's selection at the point "
               "of record on the test read of record, descriptive, declared after its members' test reads"),
        derived_at=when, lock_sha256=sha256(HERE / "lock_r3.json"), locked_at=lock["locked_at"],
        locked_at_unix=lock["locked_at_unix"],
        inputs=OrderedDict((k, lock["files_sha256"][k]) for k in ("pairs_result_record.json", "w06_rows.json",
                                                                   "pairs_stats.py", "rules_pairs.json",
                                                                   "r3_nominal/rules_pairs_r3.json",
                                                                   "r3_nominal/selections_r3.json")),
        members=OrderedDict(J=fc["J"], J_job=jj, J_design=fc["J_design"], FO=fc["FO"], FO_job=fj,
                            FO_design=fc["FO_design"]),
        standing=fc["standing"],
        test=OrderedDict(read=read_words["test"], total=TEST_TOTAL,
                         counts=OrderedDict((t, OrderedDict(J=test_counts[t][jj], FO=test_counts[t][fj])) for t in SEEDS),
                         **test),
        seed0=OrderedDict(test_J=s0t[fc["J"]], test_FO=s0t[fc["FO"]], test_d_images=d0, test_d_pp=round(100.0 * d0 / TEST_TOTAL, 4),
                          e01_J=J0e, e01_FO=FO0e, e01_d_images=d0e, e01_d_pp=round(100.0 * d0e / E01_TOTAL, 4),
                          standing="printed beside, in no statistic and in no figure"),
        development=OrderedDict(read=read_words["e01"], total=E01_TOTAL,
                                counts=OrderedDict((t, OrderedDict(J=e01[t][jj], FO=e01[t][fj])) for t in SEEDS),
                                **dev),
        cell_gain=OrderedDict(gain_at_convention_pp=r1l["gain_at_convention_pp"],
                              attained_against=r1l["figure_setting"]["single"][1],
                              declared_read_gap_pp=r1l["declared_read_gap_pp"]),
        sentences=sentences,
        paper=OrderedDict(table_vi_cell=cell, table_vi_cell_words=n_cell, note=note, note_words=n_note),
        checks=checks,
    )
    (HERE / "pairs_r3_record.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Pairs Track Revision 3: the Fourth Contrast on the Test Read of Record", "",
         f"Derived {when} under lock_r3.json (sha256 {out['lock_sha256'][:16]}, locked {lock['locked_at']}), from "
         f"pairs_result_record.json ({out['inputs']['pairs_result_record.json'][:16]}) and w06_rows.json "
         f"({out['inputs']['w06_rows.json'][:16]}) with pairs_stats.py ({out['inputs']['pairs_stats.py'][:16]}). "
         "Descriptive, outside the Holm family and outside the locked secondary pairs, declared after its members' "
         "test reads. Printed, not a reading.", "",
         "## Members", "",
         f"- J: {fc['J']} (job {jj}), {fc['J_design']['geometry']}, {fc['J_design']['tiles']} tiles, "
         f"{fc['J_design']['passes']} passes.",
         f"- FO: {fc['FO']} (job {fj}), {fc['FO_design']['geometry']}, {fc['FO_design']['tiles']} tiles, one pass, "
         "resident.", "",
         "## Per-Seed Counts and Differences", "",
         "| seed | J test | FO test | d test (of 10,000) | J E01 | FO E01 | d E01 (of 5,000) |",
         "|---|---|---|---|---|---|---|"]
    for k, t in enumerate(SEEDS):
        L.append(f"| {t} | {test_counts[t][jj]:,} | {test_counts[t][fj]:,} | {ds[k]:+d} | {e01[t][jj]:,} | "
                 f"{e01[t][fj]:,} | {de[k]:+d} |")
    L += ["", "## Statistic", "",
          "| read | mean (pp) | SD (pp) | one-sided 95% lower (pp) | one-sided p (unadjusted) | sum (images) |",
          "|---|---|---|---|---|---|",
          f"| test, 10,000 images (the read of record) | {spp(test['mean_images'], TEST_TOTAL)} | {test['sd_pp']:.2f} | "
          f"{spp(test['lower_images'], TEST_TOTAL)} | {fp(test['p_one_sided'])} | {test['sum_images']:+,d} |",
          f"| E01, 5,000 images (development, beside) | {spp(dev['mean_images'], E01_TOTAL)} | {dev['sd_pp']:.2f} | "
          f"{spp(dev['lower_images'], E01_TOTAL)} | {fp(dev['p_one_sided'])} | {dev['sum_images']:+,d} |", "",
          "## Sentences", ""]
    L += [f"- {s}" for s in sentences.values()]
    L += ["", "## The Paper's Line", "",
          f"- Table VI, the tight batch-1 column's five-seed gain: `{cell}` ({n_cell} words).",
          f"- Note: \"{note}\" ({n_note} words).", "",
          "## Checks", ""]
    L += [f"- {c}" for c in checks]
    (HERE / "pairs_r3_record.md").write_text("\n".join(L) + "\n")
    print(json.dumps(OrderedDict(d_test=ds, mean_pp=test["mean_pp"], sd_pp=test["sd_pp"], lower95_pp=test["lower95_pp"],
                                 p=test["p_one_sided"], d_e01=de, dev_mean_pp=dev["mean_pp"], s0_test=d0, s0_e01=d0e,
                                 cell=cell)))
    print(f"pairs_r3_record.json {sha256(HERE / 'pairs_r3_record.json')}")
    print(f"pairs_r3_record.md {sha256(HERE / 'pairs_r3_record.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

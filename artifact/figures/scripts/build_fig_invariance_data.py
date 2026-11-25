""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
sys.path.insert(0, str(F3.parent / "figures_v2" / "scripts"))
import _ledger as LG

NAME = "fig_invariance"


MEMBERS = (
    ("C1.T-INT.b", "ResNet-18 w1", "rone",
     ("B0.256x256", "B1.256x256", "B2.256x256", "B5.256x64"), ("B5u.256x64", "B5m.256x64")),
    ("S1.T-INT.b", "DenseNet-BC k12", "densenet",
     ("B0.256x256", "B1.256x256", "B2.256x256"), ("B3.256x256",)),
    ("S4.T-INT.b", "PyramidNet-110", "pyramid",
     ("B0.256x256", "B1.256x256", "B2.256x256"), ("B3.256x256",)),
    ("ViT.T-VAL8-OP.b-op", "ViT-Lite-7/4, operand cores", "vitop",
     ("B0.256x256", "B1.256x256"), ("B3.256x256",)),
)
PLAN_NAME = {"B0": "Resident", "B1": "Default schedule", "B2": "Default schedule", "B3": "Floor plan",
             "B5": "Study pick", "B5u": "Default schedule", "B5m": "One weight copy per bank"}

PLANNED = F3.parent / "records/tiers/paper_v6/fix4/ladder/planned_ladder.json"
PLANNED_SHA256 = "66f10553c7e72de848c89b37d4ec6914d7c396d683d2d3026aba7bccab259c58"

RELEASE_PLANNED, RELEASE_PLANNED_SHA256 = PLANNED, PLANNED_SHA256


TABLE_REL = Path("tables") / "t2" / "ladder_rows.jsonl"
LADDER_RUNS = ()
LADDER_RUN = COLLECTED = COLLECTED_SHA256 = None
COLLECTED_BASIS = ("not a ledger key yet: a row of the collected table of the dense ladder (sha256 pinned in the "
                   "builder) and a field of the row's own record; record_sha256 is the record's sha256 as read")


def row_record(keys, row: str):
    ""
    for key in keys:
        for s in LG.sources(key):
            p = LG.source_path(s)
            if p.name == "result.json" and p.parent.name == row:
                return p, key, s.get("sha256")
    return None, None, None


def comparison(check_key: str, row: str):
    ""
    got = {}
    for s in LG.sources(check_key):
        jp = str(s.get("json_path") or "")
        if jp.startswith(f"K10/{row}/"):
            p = LG.source_path(s)
            doc = LG.load_json(p)
            field = jp.rsplit("/", 1)[1]
            got[field] = LG.json_at(doc, jp)
            got["source"] = f"{LG.rel(p)}#K10/{row}"
    if "outputs_differ" in got and "outputs" in got:
        return got["outputs_differ"], got["outputs"], got["source"]
    return None, None, None


def build_row(member: str, tail: str, count_key: str, check_key: str, check_form: str):
    row = f"{member}.{tail}"
    budget = tail.split(".")[0]
    path, record_key, record_sha = row_record((check_key, count_key), row)
    if path is None:
        return {"row": row, "budget": budget, "status": "pending", "why": "no deployment record among the key's sources"}
    if record_sha and LG.sha256(path) != record_sha:
        raise LG.LedgerError(f"{row}: the record's sha256 is not the one ledger key {record_key} holds for it")
    out, _ = record_fields(row, budget, PLAN_NAME.get(budget, budget), path, record_key, record_sha)
    if budget == "B0":
        out.update(status="reference", differing=None, outputs=None, differing_source=None)
        return out
    differ, outputs, dsrc = comparison(check_key, row)
    if check_form == "pending" or differ is None:
        out.update(status="pending", differing=None, outputs=None, differing_source=None,
                   why="check key pending" if check_form == "pending" else "no comparison of this row in the records")
    else:
        out.update(status="equal" if differ == 0 else "differs", differing=differ, outputs=outputs,
                   differing_source=dsrc)
    return out


def record_fields(row: str, budget: str, plan: str, path: Path, record_key, record_sha):
    ""
    doc = LG.load_json(path)
    src = LG.rel(path)
    dep = doc["deployment"]
    k9 = doc["K9"]
    sched = (doc.get("record") or {}).get("schedule") or {}
    k11 = doc.get("K11") or {}
    passes = sched["pass_count"]
    parts = (sched.get("reprogram_passes") or 0) + (sched.get("reuse_passes") or 0)
    if passes and passes != parts:
        raise LG.LedgerError(f"{row}: pass_count {passes} is not reprogram_passes + reuse_passes {parts}")
    if budget == "B0" and passes:
        raise LG.LedgerError(f"{row}: the resident deployment schedules {passes} passes")
    out = {
        "row": row, "budget": budget, "plan": plan,
        "record": src, "record_key": record_key, "record_sha256": record_sha,
        "geometry": doc.get("geometry"), "geometry_source": f"{src}#geometry",
        "cores": dep["declared_count"], "cores_source": f"{src}#deployment/declared_count",
        "passes": passes, "passes_source": f"{src}#record/schedule/pass_count",
        "reprogram_passes": sched.get("reprogram_passes"), "reuse_passes": sched.get("reuse_passes"),
        "most_divided_segment_passes": dep["max_passes"],
        "most_divided_segment_passes_source": f"{src}#deployment/max_passes",
        "stages": dep.get("neural_stages"),
        "schedule_check_passed": k11.get("passed"), "schedule_check_non_vacuous": k11.get("non_vacuous"),
        "schedule_check_source": f"{src}#K11",
        "correct": k9["correct"], "images": k9["images"], "correct_source": f"{src}#K9/correct",
        "accuracy_pct": 100.0 * k9["correct"] / k9["images"],
        "static_check_failed": (doc.get("static_program_check") or {}).get("theorem_1_failed"),
    }
    return out, doc


def drawn_rows(rows, variants):
    ""
    drawn, held = [], list(variants)
    for r in rows:
        alt = None
        if r.get("schedule_check_passed") is False:
            alt = next((v for v in held if v.get("cores") == r["cores"] and v.get("geometry") == r["geometry"]
                        and v.get("schedule_check_passed") and v["status"] == r["status"]), None)
        if alt is None:
            drawn.append(r)
            continue
        held.remove(alt)
        alt = dict(alt, replaces=r["row"],
                   replaces_why="the default-schedule row's own schedule check does not pass (no segment divided)")
        drawn.append(alt)
        held.append(r)
    return drawn, held


def add_shares(rows):
    ""
    ref = rows[0]
    for r in rows:
        r["share_den"] = None
        if "cores" not in r or r.get("geometry") != ref["geometry"]:
            continue
        den = round(ref["cores"] / r["cores"])
        if den >= 1 and -(-ref["cores"] // den) == r["cores"]:
            r["share_den"] = den
            r["share_source"] = (f"derived: ceil({ref['cores']} / {den}) = {r['cores']}, the declared ladder's rule "
                                 f"(ledger:budgetladder): {r['cores_source']} over {ref['cores_source']}")


def load_planned() -> dict:
    ""
    global PLANNED, PLANNED_SHA256
    if LG.sha256(RELEASE_PLANNED) != RELEASE_PLANNED_SHA256:
        raise LG.LedgerError(f"{LG.rel(RELEASE_PLANNED)}: sha256 {LG.sha256(RELEASE_PLANNED)} is not the pinned "
                             f"{RELEASE_PLANNED_SHA256}")
    form = LG.load_json(RELEASE_PLANNED).get("format")
    found = {}
    for e in (LG.ledger()["entries"] if COLLECTED is not None else ()):
        srcs = e.get("sources") or []
        if not any(LG.source_path(s) == COLLECTED for s in srcs):
            continue
        for s in srcs:
            p = LG.source_path(s)
            if p == COLLECTED or p.suffix != ".json" or not p.is_file():
                continue
            try:
                doc = LG.load_json(p)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and form and doc.get("format") == form:
                found.setdefault(p, s.get("sha256"))
    if len(found) > 1:
        raise LG.LedgerError("PLANNED_RECORD: the ledger names " + ", ".join(LG.rel(p) for p in found)
                             + f" beside {LG.rel(COLLECTED)}, records of the plan's form {form!r}: expected 1")
    if found:
        (p, want), = found.items()
        if not want or LG.sha256(p) != want:
            raise LG.LedgerError(f"{LG.rel(p)}: sha256 {LG.sha256(p)} is not the one the ledger holds ({want})")
        PLANNED, PLANNED_SHA256 = p, want
    else:
        PLANNED, PLANNED_SHA256 = RELEASE_PLANNED, RELEASE_PLANNED_SHA256
    return LG.load_json(PLANNED)


def planned_rows(planned: dict, member: str, executed) -> list:
    ""
    entry = planned["members"].get(member)
    if entry is None or not planned["gate_holds"].get(member):
        return []
    src = LG.rel(PLANNED)
    taken = {(r.get("cores"), r.get("geometry")) for r in executed}
    out = []
    for i, p in enumerate(entry["points"]):
        if (p["cores"], entry["geometry"]) in taken:
            continue
        at = f"{src}#members/{member}/points/{i}"
        out.append({
            "row": f"{member}.planned.d{p['d']}", "budget": f"1/{p['d']}", "plan": p["plan"], "status": "planned",
            "geometry": entry["geometry"], "d": p["d"], "share": p["share"], "share_den": p["d"],
            "share_source": f"{at}/d (the declared ladder, {src}#declared_ladder)",
            "cores": p["cores"], "cores_source": f"{at}/cores",
            "passes": p.get("passes"), "passes_source": f"{at}/passes",
            "verdict": p["verdict"], "verdict_source": f"{at}/verdict",
            "most_divided_segment_passes": p.get("most_divided_segment_passes"),
            "segments": p.get("segments"),
            "template_row": p["template_row"],
            "source": src, "source_sha256": PLANNED_SHA256,
        })
    return out


def ledger_tables() -> list:
    ""
    n = len(TABLE_REL.parts)
    out = {}
    for e in LG.ledger()["entries"]:
        for s in e.get("sources") or []:
            p = LG.source_path(s)
            if tuple(p.parts[-n:]) == TABLE_REL.parts and s.get("sha256"):
                out.setdefault(p.parents[n - 1], s["sha256"])
    return list(out.items())


def record_tables() -> list:
    ""
    import _family as FAM
    n = len(TABLE_REL.parts)
    out = {}
    for f in FAM.registry().get("families") or []:
        for r in ((f.get("members") or {}).get("figure_records") or []):
            p = LG.REPO / r["path"]
            if tuple(p.parts[-n:]) == TABLE_REL.parts:
                out.setdefault(p.parents[n - 1], r["sha256"])
    return list(out.items())


def ladder_runs() -> list:
    ""
    global LADDER_RUNS
    LADDER_RUNS = tuple(ledger_tables() or record_tables())
    return list(LADDER_RUNS)


def ladder_run(runs=None):
    ""
    global LADDER_RUN, COLLECTED, COLLECTED_SHA256
    resident = {}
    for member, _label, suffix, ladder, _others in MEMBERS:
        resident[member] = row_record((f"rbcheck{suffix}", f"rbcount{suffix}"), f"{member}.{ladder[0]}")[2]
    held = []
    for run, pin in (ladder_runs() if runs is None else runs):
        table = run / "tables" / "t2" / "ladder_rows.jsonl"
        if not table.is_file():
            continue
        got = LG.sha256(table)
        if got != pin:
            raise LG.LedgerError(f"{LG.rel(table)}: sha256 {got} is not the pinned {pin}")
        with open(table, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        owners = [next((m for m in resident if t["row"].startswith(m + ".")), None) for t in rows]
        if rows and all(m is not None and resident[m] and (t.get("K10") or {}).get("b0_result_sha256") == resident[m]
                        for t, m in zip(rows, owners)):
            held.append((run, pin))
    if len(held) != 1:
        raise LG.LedgerError(f"LADDER_RUN: {len(held)} ladder runs of record were compared against the resident records "
                             "the ledger's sources name ("
                             + ", ".join(f"{m} {str(s)[:12]}" for m, s in resident.items()) + "), expected 1")
    LADDER_RUN, COLLECTED_SHA256 = held[0]
    COLLECTED = LADDER_RUN / "tables" / "t2" / "ladder_rows.jsonl"
    return held[0]


def load_collected() -> list:
    ""
    got = LG.sha256(COLLECTED)
    if got != COLLECTED_SHA256:
        raise LG.LedgerError(f"{LG.rel(COLLECTED)}: sha256 {got} is not the pinned {COLLECTED_SHA256}")
    with open(COLLECTED, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collected_rows(table: list, planned: dict, member: str, ref: dict, taken) -> list:
    ""
    entry = planned["members"].get(member) or {}
    tsrc, psrc = LG.rel(COLLECTED), LG.rel(PLANNED)
    out = []
    for line, t in enumerate(table, 1):
        row = t["row"]
        if not row.startswith(member + "."):
            continue

        def same(what, *vals):
            if any(v != vals[0] for v in vals[1:]):
                raise LG.LedgerError(f"{row} ({tsrc} line {line}): {what} differ: {vals}")

        if t.get("status") != "complete" or t.get("smoke"):
            raise LG.LedgerError(f"{row} ({tsrc} line {line}): not a complete row of record "
                                 f"(status {t.get('status')!r}, smoke {t.get('smoke')!r})")
        budget = row[len(member) + 1:].split(".")[0]
        path = LADDER_RUN / "rows" / row / "result.json"
        if not path.is_file():
            raise LG.LedgerError(f"{row} ({tsrc} line {line}): no record {LG.rel(path)}")
        twins = [(i, p) for i, p in enumerate(entry.get("points") or []) if p["cores"] == t["planned_cores"]]
        if len(twins) != 1:
            raise LG.LedgerError(f"{row}: {len(twins)} planned points of {member} at {t['planned_cores']} cores in "
                                 f"{psrc}, expected 1")
        i, p = twins[0]
        at = f"{psrc}#members/{member}/points/{i}"
        r, doc = record_fields(row, budget, p["plan"], path, None, LG.sha256(path))
        k10, own = t["K10"], doc.get("K10") or {}
        same("the record's row name and the table's", doc.get("row"), row)
        same("core geometry (record, planned ladder)", r["geometry"], entry["geometry"])
        same("cores (table, record, table's planned, planned ladder)", t["cores"], r["cores"], t["planned_cores"],
             p["cores"])
        same("scheduled passes (table, record, table's planned, planned ladder)", t["passes"], r["passes"],
             t["planned_passes"], p["passes"])
        same("passes of the most divided segment (table, record)", t["max_passes"], r["most_divided_segment_passes"])
        same("correct images (table, record)", t["K9_correct"], r["correct"])
        same("images (table, record, resident row)", t["K9_images"], r["images"], ref["images"])
        same("the compared row (table, record, the member's resident row)", k10["against"], own.get("against"),
             ref["row"])
        same("the resident record's sha256 (table, record, ledger)", k10["b0_result_sha256"],
             own.get("b0_result_sha256"), ref["record_sha256"])
        for field in ("outputs", "outputs_differ", "pred_differ", "passed"):
            same(f"K10/{field} (table, record)", k10[field], own.get(field))
        if math.ceil(ref["cores"] / p["d"]) != r["cores"]:
            raise LG.LedgerError(f"{row}: {r['cores']} cores is not ceil({ref['cores']} / {p['d']})")
        if (r["cores"], r["geometry"]) in taken:
            raise LG.LedgerError(f"{row}: a second executed row at {r['cores']} cores of {r['geometry']}")
        r.update(
            status="equal" if k10["outputs_differ"] == 0 and k10["passed"] is True else "differs",
            differing=k10["outputs_differ"], outputs=k10["outputs"], differing_source=f"{tsrc}#{row}/K10",
            share_den=p["d"],
            share_source=f"{at}/d (the declared ladder, {psrc}#declared_ladder)",
            d=p["d"], share=p["share"],
            planned_twin={"row": f"{member}.planned.d{p['d']}", "cores": p["cores"], "passes": p["passes"],
                          "source": at},
            collected={"table": tsrc, "table_sha256": COLLECTED_SHA256, "line": line, "ledger_key": None,
                       "basis": COLLECTED_BASIS})
        out.append(r)
    return out


def main():
    networks, keys_used = [], []
    images = LG.value("eimages")
    ladder_run()
    planned = load_planned()
    table = load_collected()
    stray = [t["row"] for t in table if not any(t["row"].startswith(m[0] + ".") for m in MEMBERS)]
    if stray:
        raise LG.LedgerError(f"{LG.rel(COLLECTED)}: rows of no member of this figure: {stray}")
    for member, label, suffix, ladder, others in MEMBERS:
        count_key, check_key = f"rbcount{suffix}", f"rbcheck{suffix}"
        count, check = LG.value(count_key), LG.value(check_key)
        rows = [build_row(member, t, count_key, check_key, check["form"]) for t in ladder]
        variants = [build_row(member, t, count_key, check_key, check["form"]) for t in others]
        rows, variants = drawn_rows(rows, variants)
        add_shares(rows)

        rows += collected_rows(table, planned, member, rows[0], {(r.get("cores"), r.get("geometry")) for r in rows})


        rows = sorted(rows + planned_rows(planned, member, rows),
                      key=lambda r: (-(r.get("cores") or 0), r["status"] == "planned"))
        ref = rows[0]
        if ref.get("status") != "reference":
            raise LG.LedgerError(f"{member}: the resident deployment's record is missing")
        for r in rows + variants:
            if "correct" in r and r["status"] != "pending" and r["correct"] != ref["correct"]:
                raise LG.LedgerError(f"{r['row']}: count {r['correct']} differs from the resident {ref['correct']}")
        if count["form"] != "pending" and int(str(count["value"]).replace(",", "")) != ref["correct"]:
            raise LG.LedgerError(f"{count_key}: {count['value']} is not the record's K9/correct {ref['correct']}")
        if int(str(images["value"]).replace(",", "")) != ref["images"]:
            raise LG.LedgerError(f"eimages: {images['value']} is not the record's K9/images {ref['images']}")
        networks.append({
            "member": member, "label": label, "resident_cores": ref["cores"],
            "count": count, "check": check,
            "accuracy_pct": ref["accuracy_pct"],
            "accuracy_source": (f"derived: 100 x {count['display']} / {images['display']} (ledger:{count_key} over "
                                f"ledger:eimages; {ref['correct_source']} over "
                                f"{ref['correct_source'].replace('K9/correct', 'K9/images')})"),
            "rows": rows, "variants": variants,
        })
        keys_used += [count_key, check_key]

    drawn = [r for n in networks for r in n["rows"] if r["status"] == "equal"]
    drawn_keys = [n["check"]["key"] for n in networks if any(r["status"] == "equal" for r in n["rows"])]
    per_row = sorted({r["outputs"] for r in drawn})
    in_ledger = [r for r in drawn if r.get("record_key")]
    in_table = [r for r in drawn if "collected" in r]
    where = (f"in the comparison tables the check keys cite (ledger:{', ledger:'.join(drawn_keys)})" if not in_table
             else f"{len(in_ledger)} rows in the comparison tables the check keys cite (K10/<row>; "
                  f"ledger:{', ledger:'.join(drawn_keys)}) and {len(in_table)} rows in the collected table of the "
                  f"dense ladder (<row>/K10; {LG.rel(COLLECTED)}, sha256 {COLLECTED_SHA256}; not ledger keys yet)")
    data = {
        "name": NAME,
        "built": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "networks": networks,
        "compared": {
            "rows_drawn": len(drawn),
            "differing_drawn": sum(r["differing"] for r in drawn),
            "outputs_per_row": per_row[0] if len(per_row) == 1 else None,
            "outputs_per_row_source": (
                f"derived: K10/<row>/outputs_differ = {sum(r['differing'] for r in drawn)} of K10/<row>/outputs = "
                f"{per_row[0] if len(per_row) == 1 else per_row} on each of the {len(drawn)} compared budget rows drawn "
                f"(one value on every row), {where}; the same fields summed over every budget row of Table III "
                f"are ledger:routebinvariance"),
            "keys": drawn_keys,
            "rows": [{"row": r["row"], "differing": r["differing"], "outputs": r["outputs"],
                      "source": r["differing_source"]} for r in drawn],
            "outputs_drawn": sum(r["outputs"] for r in drawn),
            "rows_ledger": len(in_ledger), "rows_collected": len(in_table),
            "counts": "outputs_per_row: the output codes compared on one budget row (images x classes), one value "
                      "on every drawn row; differing_drawn and outputs_drawn: the sums over the rows_drawn rows",
        },
        "all_budget_rows": LG.value("routebinvariance"),
        "images": images,
        "ladder": LG.value("budgetladder"),
        "ladder_variants": LG.value("bfivevariants"),
        "planned": {
            "record": LG.rel(PLANNED), "record_sha256": PLANNED_SHA256,
            "declared_ladder": planned["declared_ladder"],
            "gate_holds": planned["gate_holds"],
            "rows_added": {n["member"]: sum(r["status"] == "planned" for r in n["rows"]) for n in networks},
            "what": planned["what"],
        },
        "collected": {
            "table": LG.rel(COLLECTED), "table_sha256": COLLECTED_SHA256, "rows_root": LG.rel(LADDER_RUN / "rows"),
            "rows_added": {n["member"]: [r["row"] for r in n["rows"] if "collected" in r] for n in networks},
            "ledger_keys": None,
            "what": "Executed rows of the dense ladder (step LADDER-T2), collected in the table: each deployed and "
                    "compared with its member's resident row. Not ledger keys yet: their source is the collected "
                    "table and each row's own record under rows_root, no ledger key names either.",
        },
        "gaps": [
            "Collected rows (the rows with a `collected` field): executed rows of the dense ladder that are not "
            "ledger keys yet. Their source is the collected table (`collected.table`, sha256 pinned in the builder) "
            "and each row's own record; `record_key` is null and `record_sha256` is the record's sha256 as read, "
            "which no ledger entry holds. The check keys' printed values (rbcheck*) do not count these rows.",
            "Planned rows (status planned): the planner alone at the declared dense ladder of core budgets "
            "(mapping and pass schedule from the toolchain at the head of record), not executed: no accuracy, no "
            "output comparison, no zero-difference mark; at the budgets of the executed rows the planned cores and "
            "passes equal the executed records' (the gate in the ladder record).",
            "ViT-Lite-7/4, operand cores: planned rows under the floor plan only (the drawn row's plan); under the "
            "default schedule the planner's segmentation is not the builder's (planned and executed pass counts "
            "differ at the half budget), so no default-schedule row is planned.",
            "Latency and energy estimate per budget row: the deployment records of the ladder rows state none "
            "(the cost records price only the study pick's rows), so the lower trace is the pass count alone.",
            "ViT-Lite-7/4, operand cores: no eighth-budget row in the ledger's records (resident and half budget "
            "only; the quarter and eighth budgets under the floor plan are collected rows); "
            "the half-budget row under the default schedule divides no segment (its own schedule check does not "
            "pass), so the half-budget row under the floor plan is drawn.",
            "DenseNet-BC k12, PyramidNet-110, ViT-Lite-7/4: no study-pick row in the records.",
        ],
    }
    path = LG.write(NAME, data, F3 / "data")
    print(f"wrote {LG.rel(path)}")
    for n in networks:
        print(f"  {n['label']:24s} {n['count']['display']:>6s} of {n['rows'][0]['images']}  check: {n['check']['display']}"
              f" [{n['check']['form']}]")
        for r in n["rows"] + n["variants"]:
            tag = (("planned" if r["status"] == "planned" else "collect" if "collected" in r else "ladder ")
                   if r in n["rows"] else "variant")
            print(f"    {tag} {r['row']:34s} cores {r.get('cores')!s:>6s} share 1/{r.get('share_den')!s:>4s} "
                  f"passes {r.get('passes')!s:>5s} (most divided segment {r.get('most_divided_segment_passes')!s:>5s}, "
                  f"schedule check {r.get('schedule_check_passed')}) correct {r.get('correct')} "
                  f"differing {r.get('differing')} of {r.get('outputs')} [{r['status']}]"
                  + (f" in place of {r['replaces']}" if r.get("replaces") else ""))
    print("  compared:", json.dumps(data["compared"]))


if __name__ == "__main__":
    main()

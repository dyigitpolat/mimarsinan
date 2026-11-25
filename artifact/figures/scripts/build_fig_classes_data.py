""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
F2_SCRIPTS = F3.parent / "figures_v2" / "scripts"
sys.path.insert(0, str(F2_SCRIPTS))
import _ledger as LG
from build_fig_toolchain_1b_data import decision_classes
import build_fig_invariance_data as INV

NAME = "fig_classes"
PAIR_KEY = "rbcheckrone"


DRAWN = 3
DRAWN_FIRST = ("placement", "pass partition and order", "weights kept on chip")
SHORT = {
    "cost": [("placement", "Placement", "placement"),
             ("splitting", "Splitting", "splitting"),
             ("pass partition and order", "Pass order", "passes"),
             ("weights kept on chip", "Weights on chip", "resident"),
             ("batch", "Batch", "batch"),
             ("core geometry", "Core geometry", "geometry")],
    "comp": [("weight and operand grids", "Weight bits", "grid"),
             ("converter precision", "Converter bits", "converter"),
             ("neuron model", "Neuron model", "neuron")],
    "adm": [("write endurance against a service-life floor", "Write endurance", "endurance"),
            ("carry-buffer size", "Carry-buffer size", "buffer")],
}
CLASSES = [
    {"id": "cost", "role": "cost_only", "name": "Cost-only", "effect": "Cost"},
    {"id": "comp", "role": "changing", "name": "Computation", "effect": "Outputs"},
    {"id": "adm", "role": "admission", "name": "Acceptance", "effect": "Accepted programs"},
]


def _ladder(check_key: str):
    ""
    hits = [m for m in INV.MEMBERS if f"rbcheck{m[2]}" == check_key]
    if len(hits) != 1:
        raise LG.LedgerError(f"{len(hits)} members of fig_invariance have the check key {check_key}, expected 1")
    member, _label, suffix, ladder, others = hits[0]
    count_key, form = f"rbcount{suffix}", LG.value(check_key)["form"]
    rows = [INV.build_row(member, t, count_key, check_key, form) for t in ladder]
    variants = [INV.build_row(member, t, count_key, check_key, form) for t in others]
    rows, _held = INV.drawn_rows(rows, variants)
    INV.add_shares(rows)
    return rows, count_key


PASS_MEASURE = "scheduled passes of the whole deployment (record/schedule/pass_count, the measure of fig_invariance)"
SINGLE_PASS = "Single pass"


def _passes(row: dict) -> dict:
    ""
    out = {"value": row["passes"], "source": row["passes_source"], "key": row["record_key"],
           "record_sha256": row["record_sha256"], "measure": PASS_MEASURE}
    if row["passes"] == 0:
        if row["most_divided_segment_passes"] != 1:
            raise LG.LedgerError(f"{row['row']}: no scheduled pass, but {row['most_divided_segment_passes']} passes "
                                 f"in its most divided segment; the label {SINGLE_PASS!r} does not hold")
        out["label"] = SINGLE_PASS
        out["label_basis"] = (f"no scheduled pass ({row['passes_source']} = 0) and one pass in every segment "
                              f"({row['most_divided_segment_passes_source']} = 1)")
    return out


def _schedule(row: dict) -> dict:
    ""
    path = LG.REPO / row["record"]
    if LG.sha256(path) != row["record_sha256"]:
        raise LG.LedgerError(f"{row['record']}: not the record fig_invariance's row was read from")
    rec, rel = LG.load_json(path), row["record"]
    dep, sch = rec["deployment"], rec["record"]["schedule"]
    if sch["pass_count"] != row["passes"]:
        raise LG.LedgerError(f"{rel}: record/schedule/pass_count is not the row's passes")
    per_segment = {str(k): v for k, v in sorted(dep["passes_per_segment"].items(), key=lambda kv: int(kv[0]))}
    if row["passes"] and sum(per_segment.values()) != row["passes"]:
        raise LG.LedgerError(f"{rel}: the passes of the segments do not add to the scheduled passes")
    stated = sorted(k for k in list(dep) + list(sch) if "per_pass" in k or "per_stage" in k)
    return {"weight_write_passes": {"value": sch["reprogram_passes"],
                                    "source": f"{rel}#record/schedule/reprogram_passes"},
            "segments_scheduled": {"value": dep["segments_scheduled"],
                                   "source": f"{rel}#deployment/segments_scheduled"},
            "passes_per_segment": {"value": per_segment, "source": f"{rel}#deployment/passes_per_segment"},
            "fullest_pass_cores": {"value": dep["physical_cores_max_stage"],
                                   "source": f"{rel}#deployment/physical_cores_max_stage"},
            "per_pass_cores": {"value": None, "fields_found": stated,
                               "note": "the record states no cores and no layers per pass (deployment and "
                                       "record/schedule hold no per-pass field); every pass is drawn as the "
                                       "declared cores"},
            "record_sha256": row["record_sha256"]}


def _deployment(row: dict, key: str) -> dict:
    ""
    if row["record_key"] != key:
        raise LG.LedgerError(f"{row['row']}: its record is named by {row['record_key']}, not by {key}")
    return {"row": row["row"], "budget": row["budget"], "geometry": row["geometry"],
            "schedule": _schedule(row),
            "cores": {"value": row["cores"], "source": row["cores_source"], "key": key,
                      "record_sha256": row["record_sha256"]},
            "passes": _passes(row),
            "theorem_1_failed": {"value": row["static_check_failed"],
                                 "source": row["cores_source"].split("#")[0] + "#static_program_check/theorem_1_failed"}}


def _network_display(label: str) -> str:
    ""
    name, sep, width = label.rpartition(" w ")
    if not sep:
        return label
    return name if float(width) == 1.0 else f"{name}, width {width}"


def _network(check_key: str, row: str) -> dict:
    ""
    seen = []
    for src in LG.sources(check_key):
        path = LG.source_path(src)
        if path.name != "table.json" or path in seen:
            continue
        seen.append(path)
        for r in LG.load_json(path).get("rows", []):
            if r.get("row") == row:
                return {"value": r["label"], "display": _network_display(r["label"]), "images": r.get("images"),
                        "source": f"{LG.rel(path)} rows[row={row}]/label", "key": check_key}
    raise LG.LedgerError(f"no row {row} in the tables {check_key} cites")


def _whole(value: float) -> str:
    ""
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def main():
    dc = decision_classes()
    classes = []
    for c in CLASSES:
        listed = dc[c["id"]]
        keys = []
        for wording, short, picture in SHORT[c["id"]]:
            if wording not in listed["value"]:
                raise LG.LedgerError(f"{wording!r} is no longer in the draft's enumeration {listed['value']}")
            keys.append({"label": short, "picture": picture, "wording": wording, "source": listed["source"]})
        order = sorted(keys, key=lambda k: (k["wording"] not in DRAWN_FIRST,))
        for k in keys:
            k["drawn"] = k in order[:DRAWN]
        missing = [w for w in listed["value"] if w not in {k["wording"] for k in keys}]
        if missing:
            raise LG.LedgerError(f"the draft lists {missing} in class {c['id']}; the figure has no label for them")
        classes.append({**c, "keys": keys, "source": listed["source"], "quote": listed["quote"]})

    rows, count_key = _ladder(PAIR_KEY)
    resident = rows[0]
    if resident.get("status") != "reference":
        raise LG.LedgerError(f"{PAIR_KEY}: the resident deployment's record is missing")
    shared = [r for r in rows[1:] if r.get("share_den")]
    if not shared:
        raise LG.LedgerError(f"{PAIR_KEY}: no budget row on the resident's core geometry")
    reduced = max(shared, key=lambda r: r["share_den"])
    if reduced["status"] != "equal" or reduced["differing"] != 0:
        raise LG.LedgerError(f"{reduced['row']}: its output codes are not shown equal to the resident's "
                             f"({reduced['status']}); the figure draws a zero-difference pair only")
    a, b = _deployment(resident, PAIR_KEY), _deployment(reduced, PAIR_KEY)
    for d in (a, b):
        if d["theorem_1_failed"]["value"] is not False:
            raise LG.LedgerError(f"the static check of {d['row']} does not establish Theorem 1")
    if "label" not in a["passes"] or "label" in b["passes"]:
        raise LG.LedgerError(f"the pair is not a single-pass deployment against a scheduled one: "
                             f"{a['passes']['value']} and {b['passes']['value']} scheduled passes")
    net = _network(PAIR_KEY, b["row"])
    ratios = {
        "cores": {"value": reduced["share_den"], "display": _whole(reduced["share_den"]), "key": "budgetladder",
                  "source": f"the divisor d of the ladder's rule; {reduced['share_source']}"},
    }
    outputs = {"value": reduced["outputs"], "source": f"{reduced['differing_source']}/outputs", "key": PAIR_KEY}
    differ = {"value": reduced["differing"], "source": f"{reduced['differing_source']}/outputs_differ",
              "key": PAIR_KEY}
    data = {
        "figure": NAME,
        "panel": "Fig. F4 (theory.tex, fig:classes)",
        "built_by": "records/figures_v3/scripts/build_fig_classes_data.py",
        "classes": classes,
        "counts": LG.values("costonlyrows", "provenrows"),
        "pair": {"key": LG.value(PAIR_KEY), "count_key": count_key, "network": net, "images": net["images"],
                 "read_by": "figures_v3/scripts/build_fig_invariance_data.py build_row (the reader of fig_invariance)",
                 "deployments": [a, b], "ratios": ratios, "outputs": outputs, "outputs_differ": differ},
    }
    out = LG.write(NAME, data, F3 / "data")
    print(f"wrote {out}: classes {[len(c['keys']) for c in classes]} keys; "
          f"{a['cores']['value']} cores, {a['passes']['label']} ({a['passes']['value']} scheduled) against "
          f"{b['cores']['value']} cores in {b['passes']['value']} scheduled passes "
          f"({ratios['cores']['display']}x fewer cores); {differ['value']} of {outputs['value']} "
          f"output codes differ")


if __name__ == "__main__":
    main()
